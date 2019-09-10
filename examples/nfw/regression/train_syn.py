import os
import torch
import argparse
import numpy as np

from torch.backends import cudnn
from nfw.toy_KL import DistributionFit
from nfw.regression.config import load_config


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train synctheic functions.')
    parser.add_argument('json')
    parser.add_argument('run_name')
    parser.add_argument('seed', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_2nd', type=float)
    parser.add_argument('--sigma2_trans', type=float)
    parser.add_argument('--sigma2_rev', type=float)
    parser.add_argument('--wd', type=float)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--random_dim', type=int)
    parser.add_argument('--mlp_hidden')
    parser.add_argument('--model_hidden')
    args = parser.parse_args()
    print(args)

    random_seed = args.seed
    run_name = args.run_name

    conf = load_config(args.json)
    if args.lr is not None:
        conf['lr'] = args.lr
    if args.lr_2nd is not None:
        conf['lr_2nd'] = args.lr_2nd
    if args.sigma2_trans is not None:
        conf['sigma2_trans'] = args.sigma2_trans
    if args.sigma2_rev is not None:
        conf['sigma2_rev'] = args.sigma2_rev
    if args.wd is not None:
        conf['wd'] = args.wd
    if args.num_epochs is not None:
        conf['num_epochs'] = args.num_epochs
    if args.random_dim is not None:
        conf['random_dim'] = args.random_dim
    if args.mlp_hidden is not None:
        conf['mlp_hidden'] = tuple(int(i) for i in args.mlp_hidden.split('.'))
    if args.model_hidden is not None:
        conf['model_hidden'] = tuple(int(i) for i in args.model_hidden.split('.'))

    device = torch.device('cuda')
    dtype = torch.double if conf['dtype'] == 'float64' else torch.float
    cudnn.benchmark = True

    Q = conf['random_dim']
    latent_dim = conf['vae_hidden']
    output_dim = 2
    H = conf['model_hidden']

    # Prepare data loaders
    if conf['dataset'] == 'function1':
        cov = torch.Tensor([[1, 0.9], [0.9, 1]]).to(device).to(dtype)
        chol = torch.cholesky(cov)
        log_det = torch.log(torch.diag(chol)).sum()

        def test_log_pdf(x):
            xt = torch.cat([x[:, 0:1], x[:, 1:2] + x[:, 0:1]**2 + 1], -1)
            Linva = torch.triangular_solve(xt[:, :, None], chol[None, :, :], upper=False)[0].sum(-1)
            logL = - (Linva**2).sum(-1)/2 - np.log(2*np.pi) - log_det
            return logL
    elif conf['dataset'] == 'function2':
        mean1 = torch.Tensor([-2, 0]).to(device).to(dtype)
        mean2 = torch.Tensor([2, 0]).to(device).to(dtype)

        def test_log_pdf(x):
            log_p1 = torch.sum((x-mean1)**2/(-2) - np.log(2*np.pi)/2, -1, keepdim=True)
            log_p2 = torch.sum((x-mean2)**2/(-2) - np.log(2*np.pi)/2, -1, keepdim=True)
            logL = torch.cat([log_p1, log_p2], -1)
            logL = torch.logsumexp(logL, dim=-1) - np.log(2)
            return logL
    elif conf['dataset'] == 'function3':
        cov1 = torch.Tensor([[2, 1.8], [1.8, 2]]).to(device).to(dtype)
        chol1 = torch.cholesky(cov1)
        log_det1 = torch.log(torch.diag(chol1)).sum()
        cov2 = torch.Tensor([[2, -1.8], [-1.8, 2]]).to(device).to(dtype)
        chol2 = torch.cholesky(cov2)
        log_det2 = torch.log(torch.diag(chol2)).sum()

        def test_log_pdf(x):
            Linva1 = torch.triangular_solve(x[:, :, None], chol1[None, :, :], upper=False)[0].sum(-1)
            Linva2 = torch.triangular_solve(x[:, :, None], chol2[None, :, :], upper=False)[0].sum(-1)

            logL = - (Linva1**2).sum(-1)/2 - np.log(2*np.pi) - log_det1

            log_p1 = torch.sum((Linva1**2)/-2, -1, keepdim=True) - np.log(2*np.pi) - log_det1
            log_p2 = torch.sum((Linva2**2)/-2, -1, keepdim=True) - np.log(2*np.pi) - log_det2
            logL = torch.cat([log_p1, log_p2], -1)
            logL = torch.logsumexp(logL, dim=-1) - np.log(2)
            return logL

    net = DistributionFit(test_log_pdf, output_dim, Q, conf['mlp_hidden'], conf['num_samples'],
                          conf['sigma2_trans'], conf['sigma2_rev'], device, dtype).to(device).to(dtype)

    print('Training phase 1.')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                 lr=conf['lr'])
    if conf['constrain_trans_rev']:
        net.sigma2_rev.requires_grad = False
    # Model training
    for i in range(conf['num_epochs']):
        accu_loss = 0
        for j in range(1000):
            optimizer.zero_grad()
            logL = net()
            loss = -logL
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            accu_loss += float(loss.data.cpu())
        train_loss = accu_loss/1000
        print("Train loss: %f." % (train_loss, ))

    print('Training phase 2.')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                 lr=conf['lr_2nd'])
    net.sigma2_rev.requires_grad = True
    for i in range(conf['num_epochs_2nd']):
        accu_loss = 0
        for j in range(1000):
            optimizer.zero_grad()
            logL = net()
            loss = -logL
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            accu_loss += float(loss.data.cpu())
        train_loss = accu_loss/1000
        print("Train loss: %f." % (train_loss, ))

    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'run_name': run_name,
        'config': conf
    }
    torch.save(state, os.path.join(conf['expdir'], run_name+'.t7'))
