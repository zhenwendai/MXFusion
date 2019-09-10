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


class DistributionFit(nn.Module):
    def __init__(self, log_pdf, num_dim, Q, mlp_units, num_samples, sigma2_trans, sigma2_rev, device, dtype=torch.float):
        super(DistributionFit, self).__init__()
        self.dtype = dtype
        self.log_pdf = log_pdf
        self.num_dim = num_dim
        self.device = device
        self.num_samples = num_samples
        self.Q = Q
        self.trans_net = MLP((self.Q,) + mlp_units + (self.num_dim,))
        self.reverse_net = MLP(tuple(list((self.Q,) + mlp_units + (self.num_dim,))[::-1]))
        self.sigma2_trans = nn.Parameter(torch.Tensor(self.num_dim,))
        self.sigma2_trans.data.fill_(sigma2_trans)
        self.sigma2_rev = nn.Parameter(torch.Tensor(1,))
        self.sigma2_rev.data.fill_(sigma2_rev)

    def compute_loglikelihood(self, mean, var, target):
        logL = ((target - mean)**2/(-2*var) - torch.log(2*np.pi*var)/2).sum(-1)
        return logL

    def forward(self):

        z_sample = torch.randn(self.num_samples, self.Q, device=self.device, requires_grad=True, dtype=self.dtype)
        p_sample = self.trans_net(z_sample)
        epsilon = torch.randn(self.num_samples, self.num_dim, device=self.device, requires_grad=True, dtype=self.dtype)
        sigam2_trans = F.softplus(self.sigma2_trans)
        p_sample_noisy = p_sample + torch.sqrt(sigam2_trans) * epsilon

        log_p = self.log_pdf(p_sample_noisy)
        log_z = ((z_sample)**2/-2 - np.log(2*np.pi)/2).sum(-1)

        # regularization term
        sigma2_rev = F.softplus(self.sigma2_rev)
        aux_mean, jacob = self.reverse_net.forward_and_jacobian(p_sample)
        reg_term = (((aux_mean - z_sample)**2 + (sigam2_trans*jacob**2).sum(-1))/(-2*sigma2_rev)
                    - torch.log(sigma2_rev)/2 + torch.log(sigam2_trans)/2 + 0.5).sum(-1)

        logL = log_p - log_z + reg_term
        logL = logL.mean(0)
        # logL = torch.logsumexp(logL, dim=0) - np.log(self.num_samples)
        return logL

    def draw_samples(self, num_samples):

        with torch.autograd.no_grad():
            z_sample = torch.randn(num_samples, self.Q, device=self.device, requires_grad=True, dtype=self.dtype)
            p_sample = self.trans_net(z_sample)
            epsilon = torch.randn(num_samples, self.num_dim, device=self.device, requires_grad=True, dtype=self.dtype)
            sigma_trans = torch.sqrt(F.softplus(self.sigma2_trans))
            p_sample += sigma_trans * epsilon
        return p_sample
