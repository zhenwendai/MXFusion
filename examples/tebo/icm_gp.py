import numpy as np
from GPy.models.sparse_gp_regression_md import SparseGPRegressionMD
from GPy.models import BayesianGPLVM
import GPy
import pickle
import mxnet as mx
from mxgpy.lvsgp_new import LVSGP, GaussianSampler, MultiVariateGaussianSampler
from mxgpy.kernel.rbf import RBF
from mxgpy.util import positive_transform_reverse_numpy
from matplotlib import pyplot


def initialize_model(nTask, X, Y, indexD, Q, M, nSamples, ctx):
    K = GPy.kern.RBF(Q)
    icm = GPy.util.multioutput.ICM(input_dim=Q, num_outputs=nTask, kernel=K, W_rank=nTask)
    m_co = GPy.models.GPCoregionalizedRegression([X[indexD==i].copy() for i in range(nTask)],[Y[indexD==i].copy() for i in range(nTask)], kernel=icm, W_rank=nTask)
    print(m_co)

    # initialize sparse GP
    kern = GPy.kern.RBF(Q)
    Mc = M_init[1]
    m_sgp = SparseGPRegressionMD(X, Y, indexD, kernel=kern, num_inducing=Mc)
    m_sgp.likelihood.variance[:] = Y.var()*0.01
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
    m_lvm.optimize(max_iters=10000, messages=1)
    print(m_lvm)

    # initialize lvsgp
    H_mean = m_lvm.X.mean.values.copy()
    H_var = m_lvm.X.variance.values.copy()
    H_var[:] = 1e-2
    t = np.hstack([H_mean[indexD], X])
    Z = t[np.random.permutation(t.shape[0])[:M]]

    qH_sampler = GaussianSampler(nSamples, H_mean[:-1], H_var[:-1], np.float64, prefix='qH_')
    tH_sampler = MultiVariateGaussianSampler(nSamples, H_mean[-1:], np.zeros((1, Q_h, Q_h)), np.ones((1,Q_h))*1e-2, np.float64, prefix='tH_')
    qH_sampler.initialize(ctx=ctx)
    tH_sampler.initialize(ctx=ctx)

    m = LVSGP(X, Y, indexD, kern=RBF(Q_h+Q, ARD=True), latent_dim=Q_h, Z=Z, qH_sampler=qH_sampler,
              tH_sampler=tH_sampler, targetD=target_task, num_inducing=M, nSamples=nSamples, ctx=ctx)

    m._params.get('noise_var').set_data(positive_transform_reverse_numpy(np.array([Y.var()*0.01])), np.float64)
    m.kern._params['rbf_lengthscale'].set_data(positive_transform_reverse_numpy(np.hstack([m_lvm.kern.lengthscale.values,m_lvm.kern.lengthscale.values,m_sgp.kern.lengthscale.values]), np.float64))
    return m


def make_new_suggestion(X, Y, indexD, m, Q, Q_h, target_task, n_random_searches, bounds_arr):
    h_sample = np.vstack([m.qH_sampler(mx.nd.array([0]))[0][0].asnumpy(), m.tH_sampler(mx.nd.array([0]))[0][0].asnumpy()])
    X_qs = np.hstack([h_sample[indexD], X])
    noise_var = np.log1p(np.exp(m._params.get('noise_var').data().asnumpy()))
    kern_lengthscale = np.log1p(np.exp(m.collect_params()['rbf_lengthscale'].data().asnumpy()))
    kern_variance = np.log1p(np.exp(m.collect_params()['rbf_variance'].data().asnumpy()))

    from mxgpy.post_sampling import GP_cont_approx
    approx_n_samples = 1000
    approx = GP_cont_approx(kern_lengthscale, kern_variance, X_qs, Y, float(noise_var), approx_n_samples)

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

    bounds = [(bounds_arr[i,0], bounds_arr[i,1])for i in range(bounds_arr.shape[0])]

    from scipy.optimize import fmin_l_bfgs_b
    x_opt = []
    y_opt = []
    for i_iter in range(n_random_searches):
        x_init = np.random.rand(bounds_arr.shape[0])*(bounds_arr[:, 1]-bounds_arr[:, 0])+bounds_arr[:, 0]
        res = fmin_l_bfgs_b(obj_func, x_init, obj_fprim, bounds=bounds, maxiter=1000)
        x_opt.append(res[0])
        y_opt.append(res[1])
    x_choice = x_opt[np.argmin(y_opt)]
    return x_choice


def evaluate_augment_dataset(f, x_choice, target_task, X, Y, indexD):
    # Evaluate the target function
    y_choice = f(x_choice)

    # Augment the training data
    X = np.vstack([X, x_choice])
    Y = np.vstack([Y, y_choice])
    indexD = np.hstack([indexD,[target_task]])
    return X, Y, indexD


def initialize_model_from_existing(m, X, Y, indexD, Q, Q_h, M, nSamples, ctx):
    qH_sampler = m.qH_sampler
    tH_sampler = m.tH_sampler
    noise_var = np.log1p(np.exp(m._params.get('noise_var').data().asnumpy()))
    kern_lengthscale = np.log1p(np.exp(m.collect_params()['rbf_lengthscale'].data().asnumpy()))
    kern_variance = np.log1p(np.exp(m.collect_params()['rbf_variance'].data().asnumpy()))
    Z_pre = m.collect_params()['lvsgp_inducing_inputs'].data().asnumpy()
    qH_sampler_new = GaussianSampler(nSamples, qH_sampler.collect_params()['qH_mean'].data().asnumpy(),
                                     np.log1p(np.exp(qH_sampler.collect_params()['qH_var'].data().asnumpy())),
                                     np.float64, prefix='qH_')
    tH_sampler_new = MultiVariateGaussianSampler(nSamples, tH_sampler.collect_params()['tH_mean'].data().asnumpy(),
                                     tH_sampler.collect_params()['tH_cov_W'].data().asnumpy(),
                                     np.log1p(np.exp(tH_sampler.collect_params()['tH_cov_diag'].data().asnumpy())),
                                     np.float64, prefix='tH_')
    qH_sampler_new.initialize(ctx=ctx)
    tH_sampler_new.initialize(ctx=ctx)

    m1 = LVSGP(X, Y, indexD, kern=RBF(Q_h+Q, ARD=True), latent_dim=Q_h, Z=Z_pre, qH_sampler=qH_sampler_new,
              tH_sampler=tH_sampler_new, targetD=target_task, num_inducing=M, nSamples=nSamples,ctx=ctx)

    m1._params.get('noise_var').set_data(positive_transform_reverse_numpy(noise_var, np.float64))
    m1.kern._params['rbf_lengthscale'].set_data(positive_transform_reverse_numpy(kern_lengthscale, np.float64))
    m1.kern._params['rbf_variance'].set_data(positive_transform_reverse_numpy(kern_variance, np.float64))
    return m1


def save_plot_target_task(target_task, m, filename):
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


def save_model(X, Y, indexD, m, filename):
    m.collect_params().save(filename+'.params')
    import pickle
    with open(filename+'.pkl', 'bw') as f:
        data = {
            'X': X,
            'Y': Y,
            'indexD': indexD,
        }
        pickle.dump(data, f)


def load_model(Q, Q_h, M, nSamples, ctx, X, Y, indexD, filename):
    # import pickle
    with open(filename+'.pkl', 'rb') as f:
        data = pickle.load(f)
        X = data['X']
        Y = data['Y']
        indexD = data['indexD']
    m = initialize_model(X, Y, indexD, Q, Q_h, M, nSamples, ctx)
    m.collect_params().load(filename+'.params')
    return m
    # plot_target_task(target_task, m)
