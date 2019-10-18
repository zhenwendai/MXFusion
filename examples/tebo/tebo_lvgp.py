import numpy as np
import mxnet as mx
from mxgpy.lvgp import LVGP, GaussianSampler, MultiVariateGaussianSampler
#from mxgpy.kernel.rbf import RBF
from mxgpy.util import positive_transform_reverse_numpy, positive_transform_numpy

from .tebo import normalize_Y, unnormalize_Y, make_new_suggestion, evaluate_augment_dataset, save_plot_target_task, save_model

from . import tebo
from mxgpy.kernel import rbf
from mxgpy.kernel import matern


def initialize_model(X, Y, indexD, Q, Q_h, M_init, nSamples, ctx, target_task, sgp_lengthscale=1., optimize=True, tebo_RBF=False, use_matern=False):
    import GPy
    from GPy.models.sparse_gp_regression_md import SparseGPRegressionMD
    from GPy.models import BayesianGPLVM
    # initialize sparse GP
    if use_matern:
        kern = GPy.kern.Matern52(Q, lengthscale=sgp_lengthscale, ARD=True)
    else:
        kern = GPy.kern.RBF(Q, lengthscale=sgp_lengthscale, ARD=True)
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
    if use_matern:
        kern_row = GPy.kern.Matern52(Q_h)
    else:
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

    qH_sampler = GaussianSampler(nSamples, H_mean[:-1], H_var[:-1], np.float64, prefix='qH_')
    tH_sampler = GaussianSampler(nSamples, H_mean[-1:], H_var[-1:], np.float64, prefix='tH_')
    # tH_sampler = MultiVariateGaussianSampler(nSamples, H_mean[-1], np.eye(Q_h)*0.1, np.ones((Q_h,))*1e-2, np.float64, prefix='tH_')
    # tH_sampler = MultiVariateGaussianSampler(nSamples, H_mean[-1], np.zeros(( Q_h, Q_h)), np.ones((Q_h,))*1e-2, np.float64, prefix='tH_')
    qH_sampler.initialize(ctx=ctx)
    tH_sampler.initialize(ctx=ctx)

    if tebo_RBF:
        kernel = tebo.RBF((Q_h,Q), ARD=True, name='kern')
    elif use_matern:
        kernel = matern.Matern52(Q_h+Q, ARD=True, name='kern')
    else:
        kernel = rbf.RBF(Q_h+Q, ARD=True, name='kern')
    m = LVGP(X, Y, indexD, kern=kernel, latent_dim=Q_h, qH_sampler=qH_sampler,
              tH_sampler=tH_sampler, targetD=target_task, nSamples=nSamples, ctx=ctx)

    m._params.get('noise_var').set_data(positive_transform_reverse_numpy(np.array([Y.var()*0.01]), np.float64))
    # m._params.get('noise_var').grad_req = 'null'
    if tebo_RBF:
        m.kern._params['kern_lengthscale'].set_data(positive_transform_reverse_numpy(np.hstack([m_lvm.kern.lengthscale.values, m_sgp.kern.lengthscale.values]), np.float64))
    else:
        m.kern._params['kern_lengthscale'].set_data(positive_transform_reverse_numpy(np.hstack([m_lvm.kern.lengthscale.values]*Q_h+[ m_sgp.kern.lengthscale.values]), np.float64))
    return m

def initialize_model_from_existing(m, X, Y, indexD, Q, Q_h, nSamples, ctx, target_task, init_lengthscale=None, tebo_RBF=False, use_matern=False):
    qH_sampler = m.qH_sampler
    tH_sampler = m.tH_sampler
    noise_var = positive_transform_numpy(m._params.get('noise_var').data().asnumpy(), np.float64)
    if init_lengthscale is None:
        kern_lengthscale = positive_transform_numpy(m.collect_params()['kern_lengthscale'].data().asnumpy(), np.float64)
    else:
        kern_lengthscale = positive_transform_numpy(init_lengthscale, np.float64)
    kern_variance = positive_transform_numpy(m.collect_params()['kern_variance'].data().asnumpy(), np.float64)
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

    if tebo_RBF:
        kernel = tebo.RBF((Q_h,Q), ARD=True, name='kern')
    elif use_matern:
        kernel = matern.Matern52(Q_h+Q, ARD=True, name='kern')
    else:
        kernel = rbf.RBF(Q_h+Q, ARD=True, name='kern')
    m1 = LVGP(X, Y, indexD, kern=kernel, latent_dim=Q_h, qH_sampler=qH_sampler_new,
              tH_sampler=tH_sampler_new, targetD=target_task,  nSamples=nSamples,ctx=ctx)

    m1._params.get('noise_var').set_data(positive_transform_reverse_numpy(noise_var, np.float64))
    # m1._params.get('noise_var').grad_req = 'null'
    m1.kern._params['kern_lengthscale'].set_data(positive_transform_reverse_numpy(kern_lengthscale, np.float64))
    m1.kern._params['kern_variance'].set_data(positive_transform_reverse_numpy(kern_variance, np.float64))
    return m1


def load_model(Q, Q_h, M, nSamples, ctx, target_task, M_init, filename, normalizeY=True, tebo_RBF=True):
    import pickle
    with open(filename+'.pkl', 'rb') as f:
        data = pickle.load(f)
        X = data['X']
        Y = data['Y']
        indexD = data['indexD']
    if normalizeY:
        Y, Y_mean, Y_std = normalize_Y(Y, indexD, target_task)
    m = initialize_model(X, Y, indexD, Q, Q_h, M, M_init, nSamples, ctx, target_task, optimize=False, tebo_RBF=tebo_RBF)
    m.collect_params().load(filename+'.params', ctx=ctx)
    return m


def make_initial_suggestion(X, Y, indexD, m, Q, Q_h, n_random_searches, bounds_arr, tebo_RBF=False, use_matern=False):
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

    newh_sample = np.random.randn(Q_h)

    f = approx.draw_sample_f()

    def obj_func(x):
        X_t = np.empty((1, Q+Q_h))
        X_t[0, :Q_h] = newh_sample
        X_t[0, Q_h:] = x
        return f(mx.nd.array(X_t, dtype=np.float64)).asnumpy()

    def obj_fprim(x):
        X_t = np.empty((1, Q+Q_h))
        X_t[0, :Q_h] = newh_sample
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
    for i_iter in range(n_random_searches):
        x_init = np.random.rand(bounds_arr.shape[0])*(bounds_arr[:, 1]-bounds_arr[:, 0])+bounds_arr[:, 0]
        res = fmin_l_bfgs_b(obj_func, x_init, obj_fprim, bounds=bounds, maxiter=1000)
        x_opt.append(res[0])
        y_opt.append(res[1])
    x_choice = x_opt[np.argmin(y_opt)]
    return x_choice[None,:]


def draw_function_sample(X, Y, indexD, m, Q, Q_h, target_task, n_random_searches, bounds_arr, tebo_RBF=False, use_matern=False):
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
    return obj_func, f, h_sample
