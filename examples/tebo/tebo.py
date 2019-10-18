import numpy as np
import mxnet as mx
from mxgpy.lvsgp_new import LVSGP, GaussianSampler
#from mxgpy.kernel.rbf import RBF
from mxgpy.util import positive_transform_reverse_numpy, positive_transform_numpy
from mxgpy.kernel import rbf


class RBF(rbf.RBF):
    class Gluon_K(mx.gluon.HybridBlock):
        def __init__(self, input_dim, ARD, dtype, prefix=None, params=None):
            super(RBF.Gluon_K, self).__init__(prefix=prefix, params=params)
            self.input_dim = input_dim
            self.pick_id = mx.nd.array([0]*input_dim[0]+[1]*input_dim[1])
            self.ARD = ARD
            with self.name_scope():
                self.lengthscale = self.params.get('lengthscale')
                self.variance = self.params.get('variance')

        def hybrid_forward(self, F, x, x2=None, **kwargs):
            lengthscale = rbf.positive_transform(F, kwargs['lengthscale'])
            lengthscale = F.take(lengthscale, self.pick_id)
            variance = rbf.positive_transform(F, kwargs['variance'])
            if x2 is None:
                xsc = rbf._rescale_data(F, x, lengthscale)
                # Inner product matrix => amat
                amat = F.linalg.syrk(xsc, transpose=False, alpha=1.)
                # Matrix of squared distances times (-1/2) => amat
                dg_a = rbf.extract_diagonal(F, amat)/-2.
                amat = F.broadcast_add(amat, F.reshape(dg_a, shape=(1, -1)))
                amat = F.broadcast_add(amat, F.reshape(dg_a, shape=(-1, 1)))
                return rbf._kern_gaussian_pointwise(F, amat, variance)
            else:
                x1sc = rbf._rescale_data(F, x, lengthscale)
                x2sc = rbf._rescale_data(F, x2, lengthscale)
                # Inner product matrix => amat
                amat = F.linalg.gemm2(
                    x1sc, x2sc, transpose_a=False, transpose_b=True, alpha=1.)
                # Matrix of squared distances times (-1/2) => amat
                dg1 = rbf._sum_squares_symbol(F, x1sc, axis=1)/-2.
                amat = F.broadcast_add(amat, dg1)
                dg2 = F.reshape(
                    rbf._sum_squares_symbol(F, x2sc, axis=1), shape=(1, -1)) / -2.
                amat = F.broadcast_add(amat, dg2)
                return rbf._kern_gaussian_pointwise(F, amat, variance)

    def __init__(self, input_dim, ARD, variance=1., lengthscale=[1., 1.],
                 dtype=None, name='rbf'):
        self.ARD = ARD
        assert not ARD
        super(RBF, self).__init__(input_dim=input_dim, dtype=dtype, name=name)
        self.variance_init = positive_transform_reverse_numpy(variance,
                                                              self.dtype)
        lengthscale = np.array(lengthscale)
        self.lengthscale_init = positive_transform_reverse_numpy(lengthscale,
                                                                 self.dtype)

    def _initialize_kernel_params(self, params):
        from mxnet.initializer import Constant
        params.get('lengthscale', shape=(2,), dtype=self.dtype, init=Constant(self.lengthscale_init), allow_deferred_init=True)
        params.get('variance', shape=(1,), dtype=self.dtype,
                         init=Constant(self.variance_init),
                         allow_deferred_init=True)


def normalize_Y(Y, indexD, target_task):
    max_idx = np.max(indexD)
    Y_mean = np.zeros(max_idx+1)
    Y_std = np.zeros(max_idx+1)
    for i in range(max_idx+1):
        Y_mean[i] = Y[indexD == i].mean()
        Y_std[i] = Y[indexD == i].std() + 1e-8
        Y[indexD == i] = (Y[indexD == i] - Y_mean[i])/Y_std[i]
    return Y, Y_mean[:, None], Y_std[:, None]


def unnormalize_Y(Y, Y_mean, Y_std, indexD):
    Y2 = Y.copy()
    max_idx = np.max(indexD)
    for i in range(max_idx+1):
        Y2[indexD == i] = Y[indexD == i]*Y_std[i] + Y_mean[i]
    return Y2


def initialize_model(X, Y, indexD, Q, Q_h, M, M_init, nSamples, ctx, target_task, sgp_lengthscale=1., optimize=True):
    import GPy
    from GPy.models.sparse_gp_regression_md import SparseGPRegressionMD
    from GPy.models import BayesianGPLVM
    # initialize sparse GP
    kern = GPy.kern.RBF(Q, lengthscale=sgp_lengthscale)
    Mc = M_init[1]
    m_sgp = SparseGPRegressionMD(X, Y, indexD, kernel=kern, num_inducing=Mc)
    m_sgp.likelihood.variance[:] = Y.var()*0.01
    if optimize:
        m_sgp.likelihood.variance.fix()
        m_sgp.optimize(max_iters=100)
        m_sgp.likelihood.variance.unfix()
        m_sgp.optimize(max_iters=1000)
    print(m_sgp)

    # initialize gplvm
    Mr = M_init[0]
    kern_row = GPy.kern.RBF(Q_h)
    m_lvm = BayesianGPLVM(m_sgp.posterior.mean.copy().T, Q_h, kernel=kern_row,
                          num_inducing=Mr)
    m_lvm.likelihood.variance[:] = m_lvm.Y.var()*0.01
    if optimize:
        m_lvm.optimize(max_iters=10000, messages=1)
    print(m_lvm)

    # initialize lvsgp
    H_mean = m_lvm.X.mean.values.copy()
    H_var = m_lvm.X.variance.values.copy()
    H_var[:] = 1e-2
    t = np.hstack([H_mean[indexD], X])

    Z = t[np.random.permutation(t.shape[0])[:M]]

    qH_sampler = GaussianSampler(nSamples, H_mean[:-1], H_var[:-1], np.float64, prefix='qH_')
    tH_sampler = GaussianSampler(nSamples, H_mean[-1:], H_var[-1:], np.float64, prefix='tH_')
    # tH_sampler = MultiVariateGaussianSampler(nSamples, H_mean[-1], np.eye(Q_h)*0.1, np.ones((Q_h,))*1e-2, np.float64, prefix='tH_')
    # tH_sampler = MultiVariateGaussianSampler(nSamples, H_mean[-1], np.zeros(( Q_h, Q_h)), np.ones((Q_h,))*1e-2, np.float64, prefix='tH_')
    qH_sampler.initialize(ctx=ctx)
    tH_sampler.initialize(ctx=ctx)

    m = LVSGP(X, Y, indexD, kern=RBF((Q_h,Q), ARD=True), latent_dim=Q_h, Z=Z, qH_sampler=qH_sampler,
              tH_sampler=tH_sampler, targetD=target_task, num_inducing=M, nSamples=nSamples, ctx=ctx)

    m._params.get('noise_var').set_data(positive_transform_reverse_numpy(np.array([Y.var()*0.01]), np.float64))
    # m._params.get('noise_var').grad_req = 'null'
    m.kern._params['rbf_lengthscale'].set_data(positive_transform_reverse_numpy(np.hstack([m_lvm.kern.lengthscale.values, m_sgp.kern.lengthscale.values]), np.float64))
    return m


def make_new_suggestion(X, Y, indexD, m, Q, Q_h, target_task, n_random_searches, bounds_arr, tebo_RBF=False, use_matern=True):
    h_sample = np.vstack([m.qH_sampler(mx.nd.array([0]))[0][0].asnumpy(), m.tH_sampler(mx.nd.array([0]))[0][0].asnumpy()])
    X_qs = np.hstack([h_sample[indexD], X])
    noise_var = positive_transform_numpy(m._params.get('noise_var').data().asnumpy(), np.float64)+1e-8
    kern_lengthscale = positive_transform_numpy(m.collect_params()['kern_lengthscale'].data().asnumpy(), np.float64)
    if tebo_RBF:
        kern_lengthscale = np.array([kern_lengthscale[0]]*Q_h + [kern_lengthscale[1]]*Q)
    kern_variance = positive_transform_numpy(m.collect_params()['kern_variance'].data().asnumpy(), np.float64)

    from mxgpy.post_sampling import GP_cont_approx
    approx_n_samples = 5000
    approx = GP_cont_approx(kern_lengthscale, kern_variance, X_qs, Y, float(noise_var), approx_n_samples, is_matern=use_matern, matern_lambda=5/2.)

    f = approx.draw_sample_f()

    def obj_func(x):
        X_t = np.empty((1, Q+Q_h))
        X_t[0, :Q_h] = h_sample[target_task]
        X_t[0, Q_h:] = x
        return f(mx.nd.array(X_t, dtype=np.float64)).asnumpy()

    def obj_fprim(x):
        X_t = np.empty((1, Q+Q_h))
        X_t[0, :Q_h] = h_sample[target_task]
        X_t[0, Q_h:] = x
        x = mx.nd.array(X_t, dtype=np.float64)
        x.attach_grad()
        with mx.autograd.record():
            l = f(x)
            l.backward()
        return x.grad[0, Q_h:].asnumpy()

    bounds = [(bounds_arr[i,0], bounds_arr[i,1]) for i in range(bounds_arr.shape[0])]

    from scipy.optimize import fmin_l_bfgs_b
    x_opt = []
    y_opt = []
    X_tar = X[indexD == target_task]
    for i_iter in range(n_random_searches):
        if i_iter < X_tar.shape[0]:
            x_init = X_tar[i_iter]
        else:
            x_init = np.random.rand(bounds_arr.shape[0])*(bounds_arr[:, 1]-bounds_arr[:, 0])+bounds_arr[:, 0]
        res = fmin_l_bfgs_b(obj_func, x_init, obj_fprim, bounds=bounds, maxiter=1000)
        x_opt.append(res[0])
        y_opt.append(res[1])
    x_choice = x_opt[np.argmin(y_opt)]
    return x_choice[None,:]


def evaluate_augment_dataset(f, x_choice, target_task, X, Y, indexD, Y_mean, Y_std, normalizeY=True):
    # Evaluate the target function
    y_choice = f(x_choice)

    if normalizeY:
        Y = unnormalize_Y(Y, Y_mean, Y_std, indexD)

    # Augment the training data
    X = np.vstack([X, x_choice])
    Y = np.vstack([Y, y_choice])
    indexD = np.hstack([indexD,[target_task]])

    if normalizeY:
        Y, Y_mean, Y_std = normalize_Y(Y, indexD, target_task)
    return X, Y, indexD, Y_mean, Y_std


def initialize_model_from_existing(m, X, Y, indexD, Q, Q_h, M, nSamples, ctx, target_task, init_lengthscale=None):
    qH_sampler = m.qH_sampler
    tH_sampler = m.tH_sampler
    noise_var = positive_transform_numpy(m._params.get('noise_var').data().asnumpy(), np.float64)
    if init_lengthscale is None:
        kern_lengthscale = positive_transform_numpy(m.collect_params()['rbf_lengthscale'].data().asnumpy(), np.float64)
    else:
        kern_lengthscale = positive_transform_numpy(init_lengthscale, np.float64)
    kern_variance = positive_transform_numpy(m.collect_params()['rbf_variance'].data().asnumpy(), np.float64)
    Z_pre = m.collect_params()['lvsgp_inducing_inputs'].data().asnumpy()
    # h_sample = np.vstack([qH_sampler(mx.nd.array([0]))[0][0].asnumpy(), tH_sampler(mx.nd.array([0]))[0][0].asnumpy()])
    # t = np.hstack([h_sample[indexD], X])
    # Z_pre = t[np.hstack([np.random.permutation((indexD!=target_task).sum())[:M],np.where(indexD==target_task)[0]])]

    qH_sampler_new = GaussianSampler(nSamples, qH_sampler.collect_params()['qH_mean'].data().asnumpy(),
                                     positive_transform_numpy(qH_sampler.collect_params()['qH_var'].data().asnumpy(), np.float64),
                                     np.float64, prefix='qH_')
    tH_sampler_new = GaussianSampler(nSamples, tH_sampler.collect_params()['tH_mean'].data().asnumpy(),
                                     positive_transform_numpy(tH_sampler.collect_params()['tH_var'].data().asnumpy(), np.float64),
                                     np.float64, prefix='tH_')
    # tH_sampler_new = MultiVariateGaussianSampler(nSamples, tH_sampler.collect_params()['tH_mean'].data().asnumpy(),
    #                                  tH_sampler.collect_params()['tH_cov_W'].data().asnumpy(),
    #                                  positive_transform_numpy(tH_sampler.collect_params()['tH_cov_diag'].data().asnumpy(), np.float64),
    #                                  np.float64, prefix='tH_')
    qH_sampler_new.initialize(ctx=ctx)
    tH_sampler_new.initialize(ctx=ctx)

    m1 = LVSGP(X, Y, indexD, kern=RBF((Q_h,Q), ARD=True), latent_dim=Q_h, Z=Z_pre, qH_sampler=qH_sampler_new,
              tH_sampler=tH_sampler_new, targetD=target_task, num_inducing=Z_pre.shape[0], nSamples=nSamples,ctx=ctx)

    m1._params.get('noise_var').set_data(positive_transform_reverse_numpy(noise_var, np.float64))
    # m1._params.get('noise_var').grad_req = 'null'
    m1.kern._params['rbf_lengthscale'].set_data(positive_transform_reverse_numpy(kern_lengthscale, np.float64))
    m1.kern._params['rbf_variance'].set_data(positive_transform_reverse_numpy(kern_variance, np.float64))
    return m1


def save_plot_target_task(target_task, m, filename):
    from matplotlib import pyplot
    d = target_task

    fig = pyplot.figure(figsize=(10,5))
    ax = fig.gca()

    xx =np.linspace(0,1,100)
    xt = np.empty((100,3))
    t,_ = m.tH_sampler(mx.nd.array([0]))

    mean = []
    var = []
    for i in range(t.shape[0]):
        xt[:,2] = xx
        xt[:,:2] = t[i].asnumpy()[0]
        ym, yv = m.predict(mx.nd.array(xt,dtype=np.float64))
        mean.append(ym.asnumpy()[:,0])
        var.append(yv.asnumpy()[:,0])
    mean = np.vstack(mean)
    var = np.vstack(var)
    ym = mean.mean(axis=0)
    yv = (np.square(ym[None,:]-mean)+var).mean(axis=0)
    yv = np.sqrt(yv)

    ax.plot(xx, ym,'-b')
    ax.plot(xx, ym-2*yv,'--b')
    ax.plot(xx, ym+2*yv,'--b')
    ax.plot(X[indexD==target_task],Y[indexD==target_task],'og')
    ax.plot(xx, forrester(a[target_task],b[target_task])(xx),'-r')
    fig.savefig(filename)


def save_model(X, Y, indexD, m, filename, Y_mean, Y_std):
    m.collect_params().save(filename+'.params')
    Y = unnormalize_Y(Y, Y_mean, Y_std)
    import pickle
    with open(filename+'.pkl', 'bw') as f:
        data = {
            'X': X,
            'Y': Y,
            'indexD': indexD,
        }
        pickle.dump(data, f)


def load_model(Q, Q_h, M, nSamples, ctx, target_task, M_init, filename, normalizeY=True):
    import pickle
    with open(filename+'.pkl', 'rb') as f:
        data = pickle.load(f)
        X = data['X']
        Y = data['Y']
        indexD = data['indexD']
    if normalizeY:
        Y, Y_mean, Y_std = normalize_Y(Y, indexD, target_task)
    m = initialize_model(X, Y, indexD, Q, Q_h, M, M_init, nSamples, ctx, target_task, optimize=False)
    m.collect_params().load(filename+'.params', ctx=ctx)
    return m
