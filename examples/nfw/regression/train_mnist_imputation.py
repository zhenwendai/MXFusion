import json
import os
import torch
import argparse
import pickle

from torch.backends import cudnn
from torch import nn
import numpy as np

from nfw.vae_comb import VAE
from nfw.regression.config import load_config
from nfw.toy_KL import DistributionFit
from nfw.toy_KL_mf import DistributionFit_MF

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train VAE with NFW.')
    parser.add_argument('json')
    parser.add_argument('run_name')
    parser.add_argument('digit', type=int)
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

    digit = args.digit
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
    cudnn.benchmark = True

    # Prepare data loaders
    with open(os.path.join(conf['datadir'], 'mnist.pkl'), 'rb') as f:
        d = pickle.load(f)
    data = d['train_data']
    data = data.reshape(data.shape[0], -1)
    data_test = d['test_data']
    data_test = data_test.reshape(data_test.shape[0], -1)
    label_test = d['test_label']
    np.random.seed(3)
    idx = np.random.permutation(data_test.shape[0])
    label = label_test[idx]
    img_idx = idx[np.where(label == digit)[0][0]]
    test_img = data_test[img_idx:img_idx+1]
    test_binarized_img = (np.random.rand(*test_img.shape) < test_img)*1
    test_binarized_img = torch.Tensor(test_binarized_img).to(device)

    Q = conf['random_dim']
    latent_dim = conf['vae_hidden']
    output_dim = data.shape[1]
    H = conf['model_hidden'][0]

    decoder = nn.Sequential(nn.Linear(latent_dim, H),
                            nn.ReLU(),
                            nn.Linear(H, H),
                            nn.ReLU(),
                            nn.Linear(H, output_dim)).to(device)

    vae = VAE(decoder, data.shape[-1], latent_dim, Q, conf['mlp_hidden'], conf['num_samples'],
              conf['sigma2_trans'], conf['sigma2_rev'], device).to(device)

    trained_model = torch.load(os.path.join(conf['expdir'], conf['model_name']))
    vae.load_state_dict(trained_model['net'])

    def test_log_pdf(x):
        y = test_binarized_img
        log_prob = vae.decoder(x)
        log_prob_masked = log_prob.reshape(-1, 28, 28)[:, :12, :].reshape(-1, 12*28)
        y_masked = y.reshape(-1, 28, 28)[:, :12, :].reshape(-1, 12*28)
        logL = -torch.nn.functional.softplus((1-2*y_masked)*log_prob_masked).sum(-1)
        logL += ((x)**2/-2 - np.log(2*np.pi)/2).sum(-1)
        return logL

    # Fit with NFW
    net = DistributionFit(test_log_pdf, latent_dim, Q, (300, 300), conf['num_samples'],
                          conf['sigma2_trans'], conf['sigma2_rev'], device).to(device)

    print('Fitting NFW.')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                 lr=conf['lr'], amsgrad=True)
    # Model training
    for i in range(conf['num_epochs']):
        accu_loss = 0
        for j in range(1000):
            optimizer.zero_grad()
            logL = net()
            loss = -logL
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            with torch.autograd.no_grad():
                accu_loss += loss
        train_loss = float(accu_loss.data.cpu())/1000
        print("Train loss: %f." % (train_loss, ))

    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'run_name': run_name,
        'config': conf
    }
    torch.save(state, os.path.join(conf['expdir'], run_name+'_nfw.t7'))

    # Fit with MF
    net = DistributionFit_MF(test_log_pdf, latent_dim,  conf['num_samples'], device).to(device)

    print('Fitting MF.')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                 lr=conf['lr'], amsgrad=True)
    # Model training
    for i in range(conf['num_epochs']):
        accu_loss = 0
        for j in range(1000):
            optimizer.zero_grad()
            logL = net()
            loss = -logL
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            with torch.autograd.no_grad():
                accu_loss += loss
        train_loss = float(accu_loss.data.cpu())/1000
        print("Train loss: %f." % (train_loss, ))

    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'run_name': run_name,
        'config': conf
    }
    torch.save(state, os.path.join(conf['expdir'], run_name+'_mf.t7'))
