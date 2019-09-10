import json
import os
import torch
import argparse
import pandas as pd

from torch.backends import cudnn
import numpy as np

from nfw.bnn_mf import BNN_MF
from nfw.regression.util import eval_test_logL
from nfw.regression.config import load_config

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train BNN with NFW.')
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

    # Prepare data loaders
    if conf['dataset'] == 'housing':
        data = np.loadtxt(os.path.join(conf['datadir'], 'housing.data'))
    elif conf['dataset'] == 'kin8nm':
        data = np.loadtxt(os.path.join(conf['datadir'], 'regression-datasets-kin8nm.csv'), delimiter=',')
    elif conf['dataset'] == 'power':
        df = pd.read_csv(os.path.join(conf['datadir'], 'power.csv'))
        data = np.array(df)
    elif conf['dataset'] == 'concrete':
        df = pd.read_csv(os.path.join(conf['datadir'], 'concrete.csv'))
        data = np.array(df)
    elif conf['dataset'] == 'energy':
        df = pd.read_csv(os.path.join(conf['datadir'], 'energy2.csv'))
        data = np.array(df)
    elif conf['dataset'] == 'naval':
        data = np.loadtxt(os.path.join(conf['datadir'], 'naval.txt'))
        data = data[:, [0,1,2,3,4,5,6,7,9,10,12,13,14,15,17]]
    elif conf['dataset'] == 'redwine':
        df = pd.read_csv(os.path.join(conf['datadir'], 'winequality-red.csv'), delimiter=';')
        data = np.array(df)

    np.random.seed(random_seed)
    idx = np.random.permutation(data.shape[0])
    if conf['validation']:
        train = data[idx][:int(data.shape[0]*0.8)]
        test = data[idx][int(data.shape[0]*0.8):int(data.shape[0]*0.9)]
    else:
        train = data[idx][:int(data.shape[0]*0.9)]
        test = data[idx][int(data.shape[0]*0.9):]

    train_mean = train.mean(0)
    train_std = train.std(0)
    # if conf['dataset'] == 'housing':
    #     train_std[-1] = 1
    train = (train-train_mean)/train_std
    test = (test-train_mean)/train_std

    device = torch.device('cuda')
    cudnn.benchmark = True
    model_units = (train.shape[1]-1,) + conf['model_hidden'] + (1,)
    net = BNN_MF(model_units, conf['num_samples'], device).to(device)

    test_data = torch.Tensor(test).to(device)
    train_data = torch.Tensor(train).to(device)

    # Model training
    # Training while constraining the variance of auxiliary term
    print('Training phase 1.')
    # net.sigma2_rev.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                 lr=conf['lr'], amsgrad=True)
    net.train()
    for i in range(conf['num_epochs']):
        accu_loss = 0
        for j in range(conf['epoch_size']):
            optimizer.zero_grad()
            logL = net(train_data[:, :-1], train_data[:, -1:])
            loss = -logL
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            with torch.autograd.no_grad():
                accu_loss += loss

        train_logL = float(accu_loss.data.cpu())/conf['epoch_size']/train_data.shape[0]
        print("Train loss: %f." % train_logL)

    # Training without constraints
    print('Training phase 2.')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                 lr=conf['lr_2nd'], amsgrad=True)
    for i in range(conf['num_epochs_2nd']):
        accu_loss = 0
        for j in range(conf['epoch_size']):
            optimizer.zero_grad()
            logL = net(train_data[:, :-1], train_data[:, -1:])
            loss = -logL
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            with torch.autograd.no_grad():
                accu_loss += loss

        train_logL = float(accu_loss.data.cpu())/conf['epoch_size']/train_data.shape[0]
        print("Train loss: %f." % train_logL)

    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'run_name': run_name,
        'config': conf
    }
    torch.save(state, os.path.join(conf['expdir'], run_name+'.t7'))

    test_logL = eval_test_logL(net, test_data[:, :-1], test_data[:, -1:], train_std[-1], 1000)
    with open('ntl_'+run_name+'.json', 'w') as outfile:
        result = {'test_ntl': test_logL, 'config': conf}
        json.dump(result, outfile)
