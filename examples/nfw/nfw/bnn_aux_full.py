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


class BNN_AUX(nn.Module):
    def __init__(self, num_units, Q, mlp_units, num_samples, sigma2_trans, sigma2_rev, device,
                 noise_var=-7, z_var=-5):
        super(BNN_AUX, self).__init__()
        self.device = device
        self.num_samples = num_samples
        self.num_units = num_units
        self.num_layers = len(num_units)-1
        self.param_size = self._total_param_size()
        self.Q = Q
        self.trans_net = MLP((self.Q,) + mlp_units + (self.param_size,))
        self.reverse_net = MLP(tuple(list((self.Q,) + mlp_units + (self.param_size,))[::-1]))
        self.sigma2_trans = nn.Parameter(torch.Tensor(self.param_size,))
        self.sigma2_trans.data.fill_(sigma2_trans)
        self.sigma2_rev = nn.Parameter(torch.Tensor(self.Q,))
        self.sigma2_rev.data.fill_(sigma2_rev)
        self.noise_var = nn.Parameter(torch.Tensor(1,))
        self.noise_var.data.fill_(noise_var)
        self.z_var = nn.Parameter(torch.Tensor(1,))
        self.z_var.data.fill_(z_var)

    def _total_param_size(self):
        offset = 0
        for i in range(self.num_layers-1):
            dim_in, dim_out = self.num_units[i], self.num_units[i+1]
            offset += dim_in*dim_out+dim_out
        dim_in, dim_out = self.num_units[self.num_layers-1], self.num_units[self.num_layers]
        offset += dim_in*dim_out+dim_out
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
        return params

    def _eval_mlp(self, x, params, noise_var):
        for i in range(self.num_layers-1):
            weight = params['weight_'+str(i)]
            bias = params['bias_'+str(i)]
            x = torch.matmul(x, weight)+bias[:, None, :]
            x = torch.relu(x)
        weight = params['weight_'+str(self.num_layers-1)+'_mean']
        bias = params['bias_'+str(self.num_layers-1)+'_mean']
        mean = torch.matmul(x, weight)+bias[:, None, :]
        return mean, F.softplus(noise_var)

    def compute_loglikelihood(self, mean, var, target):
        logL = ((target - mean)**2/(-2*var) - torch.log(2*np.pi*var)/2).sum(-1)
        return logL

    def forward(self, x, target):
        z_var = F.softplus(self.z_var)
        z_sample = torch.sqrt(z_var)*torch.randn(self.num_samples, self.Q, device=self.device, requires_grad=True)
        p_sample = self.trans_net(z_sample)
        epsilon = torch.randn(self.num_samples, self.param_size, device=self.device, requires_grad=True)
        sigam2_trans = F.softplus(self.sigma2_trans)
        p_sample += torch.sqrt(sigam2_trans) * epsilon

        p = self._construct_params(p_sample)
        mean, var = self._eval_mlp(x[None, :, :], p, self.noise_var)
        log_p = self.compute_loglikelihood(mean, var, target[None, :, :]).sum(-1)
        log_p_theta = ((p_sample)**2/-2 - np.log(2*np.pi)/2).sum(-1)

        log_z = ((z_sample)**2/(-2*z_var) - torch.log(2*np.pi*z_var)/2).sum(-1)
        log_epsilon = ((epsilon)**2/-2 - np.log(2*np.pi)/2 - torch.log(sigam2_trans)/2).sum(-1)

        aux_mean = self.reverse_net(p_sample)
        sigma2_rev = F.softplus(self.sigma2_rev)
        log_aux = ((z_sample - aux_mean)**2/(-2*sigma2_rev) - torch.log(2*np.pi*sigma2_rev)/2).sum(-1)

        logL = log_p + log_p_theta + log_aux - log_z - log_epsilon
        logL = logL.mean(0)
        # print(float((log_p+log_p_theta).sum().cpu()), float(log_aux.sum().cpu()))
        # logL = torch.logsumexp(logL, dim=0) - np.log(self.num_samples)
        return logL

    def predict(self, x, num_samples):

        with torch.autograd.no_grad():
            z_var = F.softplus(self.z_var)
            z_sample = torch.sqrt(z_var)*torch.randn(num_samples, self.Q, device=self.device, requires_grad=True)
            p_sample = self.trans_net(z_sample)
            epsilon = torch.randn(num_samples, self.param_size, device=self.device, requires_grad=True)
            sigma_trans = torch.sqrt(F.softplus(self.sigma2_trans))
            p_sample += sigma_trans * epsilon
            p = self._construct_params(p_sample)
            mean, var = self._eval_mlp(x[None, :, :], p, self.noise_var)
        return mean, var
