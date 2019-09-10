import json
import os
import torch
import argparse
import pickle
import gzip

from torch.backends import cudnn
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from nfw.vae import VAE, Encoder
from nfw.regression.config import load_config

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train VAE with NFW.')
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
    elif conf['dataset'] == 'mnist':
        # with open(os.path.join(conf['datadir'], 'mnist.pkl'), 'rb') as f:
        #     d = pickle.load(f)
        # data = d['train_data']
        # data = data.reshape(data.shape[0], -1)
        # data_test = d['test_data']
        # data_test = data_test.reshape(data_test.shape[0], -1)
        data = np.loadtxt(os.path.join(conf['datadir'], 'binarized_mnist_train.amat'))
        data_val = np.loadtxt(os.path.join(conf['datadir'], 'binarized_mnist_valid.amat'))
        data_test = np.loadtxt(os.path.join(conf['datadir'], 'binarized_mnist_test.amat'))
        data = np.vstack([data, data_val])
    elif conf['dataset'] == 'fashion':
        with gzip.open(os.path.join(conf['datadir'], 'train-images-idx3-ubyte.gz'), 'rb') as imgpath:
            data = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(-1, 784)
            data = data.astype(np.float32)/255
            data = ((data > 0.5)*1).astype(np.uint8)
        with gzip.open(os.path.join(conf['datadir'], 't10k-images-idx3-ubyte.gz'), 'rb') as imgpath:
            data_test = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(-1, 784)
            data_test = data_test.astype(np.float32)/255
            data_test = ((data_test > 0.5)*1).astype(np.uint8)

    device = torch.device('cuda')
    cudnn.benchmark = True

    Q = conf['random_dim']
    latent_dim = conf['vae_hidden']
    output_dim = data.shape[1]
    H = conf['model_hidden']

    decoder = nn.Sequential(nn.Linear(latent_dim, H),
                            nn.ReLU(),
                            nn.Linear(H, H),
                            nn.ReLU(),
                            nn.Linear(H, output_dim)).to(device)

    encoder = Encoder(output_dim, H, Q).to(device)

    vae = VAE(decoder, encoder, latent_dim, Q, conf['mlp_hidden'], conf['num_samples'], conf['sigma2_trans'],
              conf['sigma2_rev'], device).to(device)

    train_dataset = TensorDataset(torch.Tensor(data))
    train_loader = DataLoader(train_dataset, batch_size=conf['train_batch_size'],
                              shuffle=True, num_workers=1)

    print('Training phase 1.')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, vae.parameters()),
                                 lr=conf['lr'])
    vae.sigma2_trans.requires_grad = False
    if conf['constrain_trans_rev']:
        vae.sigma2_rev.requires_grad = False
    # Model training
    vae.train()
    for i in range(conf['num_epochs']):
        accu_loss = 0
        for batch_idx, (data_batch,) in enumerate(train_loader):
            optimizer.zero_grad()
            binarized_data = data_batch.to(device, non_blocking=True)
            logL = vae(binarized_data.float())
            loss = -logL
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            accu_loss += float(loss.data.cpu())

        train_loss = accu_loss/batch_idx
        print("Train loss: %f." % (train_loss,))

    print('Training phase 2.')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, vae.parameters()),
                                 lr=conf['lr_2nd'])
    for i in range(conf['num_epochs_2nd']):
        accu_loss = 0
        for batch_idx, (data_batch,) in enumerate(train_loader):
            optimizer.zero_grad()
            binarized_data = data_batch.to(device, non_blocking=True)
            logL = vae(binarized_data.float())
            loss = -logL
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            accu_loss += float(loss.data.cpu())

        train_loss = accu_loss/batch_idx
        print("Train loss: %f." % (train_loss,))

    test_dataset = TensorDataset(torch.Tensor(data_test))
    test_loader = DataLoader(test_dataset, batch_size=conf['train_batch_size'],
                             shuffle=False, num_workers=1)

    test_logL = []
    for batch_idx, (data_batch,) in enumerate(test_loader):
        binarized_data = data_batch.to(device, non_blocking=True)
        logL = vae.comp_test_logL(binarized_data.float(), 1000)
        test_logL.append(logL.cpu().numpy())
    test_logL = np.hstack(test_logL)
    avg_test_logL = float(test_logL.mean())

    # Save checkpoint.
    print('Saving..')
    state = {
        'net': vae.state_dict(),
        'run_name': run_name,
        'config': conf
    }
    torch.save(state, os.path.join(conf['expdir'], run_name+'.t7'))

    with open('ntl_'+run_name+'.json', 'w') as outfile:
        result = {'test_ntl': avg_test_logL, 'config': conf}
        json.dump(result, outfile)
