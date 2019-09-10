import torch
import numpy as np


def eval_test_logL(net, test_data_x, test_data_y, y_std, num_samples):
    with torch.autograd.no_grad():
        mean, var = net.predict(test_data_x, num_samples=num_samples)
        logL = (test_data_y - mean)**2/(-2*var) - torch.log(2*np.pi*var)/2 - np.log(y_std)
        logL = float((torch.logsumexp(logL, dim=0) - np.log(num_samples)).mean())
    return logL
