import json
from copy import copy

default_config = {
    'dataset': 'housing',
    'random_dim': 20,
    'vae_hidden': 10,
    'num_samples': 100,
    'model_hidden': (50,),
    'mlp_hidden': (200, 400),
    'sigma2_trans': -9,
    'sigma2_rev': -11,
    # initial learning rate
    'lr': 1e-8,
    # weight decay for the classification layer
    'wd': 1e-4,
    # The momentum of SGD
    'momentum': 0.9,
    # number of epochs
    'num_epochs': 1000,
    'num_epochs_2nd': 1000,
    'lr_2nd': 1e-5,
    # number of epochs before first lr decrease
    'epoch_size': 1000,
    # Experiment Folder
    'expdir': './exp',
    # Data Folder
    'datadir': './housing',
    'train_batch_size': 128,
    'validation': False,
    'constrain_trans_rev': False,
    'dtype': 'float32'
}


def load_config(file_name):
    config = copy(default_config)
    with open(file_name, 'r') as f:
        loaded_config = json.load(f)
        if 'mlp_hidden' in loaded_config:
            loaded_config['mlp_hidden'] = tuple(int(i) for i in loaded_config['mlp_hidden'].split(','))
        if 'model_hidden' in loaded_config:
            loaded_config['model_hidden'] = tuple(int(i) for i in loaded_config['model_hidden'].split(','))
        config.update(loaded_config)
    return config
