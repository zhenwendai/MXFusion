import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .mlp import MLP


def take_params(var, params, weight_name, bias_name, offset, dim_in, dim_out):
    params[weight_name] = var[:, offset: offset+dim_in*dim_out].reshape((-1, dim_in, dim_out))
    offset += dim_in*dim_out
    params[bias_name] = var[:, offset: offset+dim_out]
    offset += dim_out
    return offset


class BNN(nn.Module):
    def __init__(self, num_units, Q, mlp_units, num_samples, device):
        super(BNN, self).__init__()
        self.device = device
        self.num_samples = num_samples
        self.num_units = num_units
        self.num_layers = len(num_units)-1
        self.param_size = self._total_param_size()
        self.Q = Q
        self.trans_net = MLP((self.Q,) + mlp_units + (self.param_size,))

    def _total_param_size(self):
        offset = 0
        for i in range(self.num_layers-1):
            dim_in, dim_out = self.num_units[i], self.num_units[i+1]
            offset += dim_in*dim_out+dim_out
        dim_in, dim_out = self.num_units[self.num_layers-1], self.num_units[self.num_layers]
        offset += dim_in*dim_out+dim_out+1
        return offset

    def _construct_params(self, var):
        params = {}
        offset = 0
        for i in range(self.num_layers-1):
            dim_in, dim_out = self.num_units[i], self.num_units[i+1]
            offset = take_params(var, params, 'weight_'+str(i), 'bias_'+str(i), offset, dim_in, dim_out)

        dim_in, dim_out = self.num_units[self.num_layers-1], self.num_units[self.num_layers]
        offset = take_params(var, params, 'weight_'+str(self.num_layers-1)+'_mean',
                             'bias_'+str(self.num_layers-1)+'_mean', offset, dim_in, dim_out)
        params['noise_var'] = var[:, offset: offset+1]
        offset += 1
        return params

    def _eval_mlp(self, x, params):
        for i in range(self.num_layers-1):
            weight = params['weight_'+str(i)]
            bias = params['bias_'+str(i)]
            x = torch.matmul(x, weight)+bias[:, None, :]
            x = torch.relu(x)
        weight = params['weight_'+str(self.num_layers-1)+'_mean']
        bias = params['bias_'+str(self.num_layers-1)+'_mean']
        mean = torch.matmul(x, weight)+bias[:, None, :]
        logvar = params['noise_var']
        return mean, F.softplus(logvar)+1e-8

    def compute_loglikelihood(self, mean, var, target):
        logL = ((target - mean)**2/(-2*var[:, None, :]) - torch.log(2*np.pi*var[:, None, :])/2).sum(-1)
        return logL

    def forward(self, x, target):
        logL = []
        z_sample = torch.randn(self.num_samples, self.Q, device=self.device, requires_grad=True)

        p_sample, jacob = self.trans_net.forward_and_jacobian(z_sample)

        jacob_sq = torch.matmul(jacob.transpose(-2, -1), jacob)

        with torch.autograd.no_grad():
            diag_max = float(torch.diagonal(jacob_sq, dim1=-2, dim2=-1).max())

        L = torch.cholesky(jacob_sq+torch.eye(jacob.shape[-1], device=self.device)[None, :, :]*diag_max*1e-5)
        logdet_r = torch.log(torch.diagonal(L, dim1=-2, dim2=-1).abs()).sum(-1)

        p = self._construct_params(p_sample)
        mean, var = self._eval_mlp(x[None, :, :], p)

        log_p = self.compute_loglikelihood(mean, var, target[None, :, :]).sum(-1)

        log_p_theta = ((p_sample)**2/-2 - np.log(2*np.pi)/2).sum(-1)

        log_z = ((z_sample)**2/-2 - np.log(2*np.pi)/2).sum(-1)

        logL = log_p + log_p_theta + logdet_r - log_z
        logL = torch.logsumexp(logL, dim=0) - np.log(self.num_samples)
        return logL

    def predict(self, x, num_samples):

        with torch.autograd.no_grad():
            z_sample = torch.randn(num_samples, self.Q, device=self.device, requires_grad=True)
            p_sample = self.trans_net(z_sample)
            p = self._construct_params(p_sample)
            mean, var = self._eval_mlp(x[None, :, :], p)
        return mean, var
