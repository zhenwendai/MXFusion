import sys
import json
import os
import torch
import argparse
from copy import deepcopy

from torch.backends import cudnn
import numpy as np

from nfw.bnn import BNN
from nfw.regression.util import eval_test_logL
from nfw.regression.config import load_config

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train BNN with NFW.')
    parser.add_argument('json')
    parser.add_argument('run_name')
    parser.add_argument('seed', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--wd', type=float)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--random_dim', type=int)
    parser.add_argument('--mlp_hidden')
    args = parser.parse_args()
    print(args)

    random_seed = args.seed
    run_name = args.run_name

    conf = load_config(args.json)
    if args.lr is not None:
        conf['lr'] = args.lr
    if args.wd is not None:
        conf['wd'] = args.wd
    if args.num_epochs is not None:
        conf['num_epochs'] = args.num_epochs
    if args.random_dim is not None:
        conf['random_dim'] = args.random_dim
    if args.mlp_hidden is not None:
        conf['mlp_hidden'] = tuple(int(i) for i in args.mlp_hidden.split('.'))

    # Prepare data loaders
    data = np.loadtxt(os.path.join(conf['datadir'], 'housing.data'))

    np.random.seed(random_seed)
    idx = np.random.permutation(data.shape[0])
    train = data[idx][:int(data.shape[0]*0.8)]
    val = data[idx][int(data.shape[0]*0.8):int(data.shape[0]*0.9)]
    test = data[idx][int(data.shape[0]*0.9):]

    train_mean = train.mean(0)
    train_std = train.std(0)
    train = (train-train_mean)/train_std
    val = (val-train_mean)/train_std
    test = (test-train_mean)/train_std

    device = torch.device('cuda')
    cudnn.benchmark = True
    net = BNN((train.shape[1]-1, 50, 1), conf['random_dim'], conf['mlp_hidden'], conf['num_samples'], device).to(device)

    test_data = torch.Tensor(test).to(device)
    val_data = torch.Tensor(val).to(device)
    train_data = torch.Tensor(train).to(device)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                lr=conf['lr'], momentum=conf['momentum'], weight_decay=conf['wd'])

    # Model training
    net.train()

    val_logL_best = -np.inf
    best_params = None
    for i in range(conf['num_epochs']):
        accu_loss = 0
        for j in range(conf['epoch_size']):
            optimizer.zero_grad()
            logL = net(train_data[:, :-1], train_data[:, -1:])
            loss = -logL
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            accu_loss += float(loss.data.cpu())/train_data.shape[0]

        train_logL = accu_loss/conf['epoch_size']
        val_logL = eval_test_logL(net, val_data[:, :-1], val_data[:, -1:], train_std[-1], 1000)
        print("Train loss: %f, test logL: %f." % (train_logL, val_logL))
        if val_logL > val_logL_best:
            val_logL_best = val_logL
            best_params = deepcopy(net.state_dict())

    net.load_state_dict(best_params)
    test_logL = eval_test_logL(net, test_data[:, :-1], test_data[:, -1:], train_std[-1], 1000)
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'run_name': run_name,
        'config': conf
    }
    torch.save(state, os.path.join(conf['expdir'], run_name+'.t7'))

    with open('ntl_'+run_name+'.json', 'w') as outfile:
        result = {'test_ntl': test_logL, 'val_ntl': val_logL_best, 'config': conf}
        json.dump(result, outfile)
