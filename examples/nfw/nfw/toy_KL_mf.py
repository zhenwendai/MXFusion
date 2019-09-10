import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DistributionFit_MF(nn.Module):
    def __init__(self, log_pdf, num_dim, num_samples, device, dtype=torch.float):
        super(DistributionFit_MF, self).__init__()
        self.log_pdf = log_pdf
        self.num_dim = num_dim
        self.device = device
        self.dtype = dtype
        self.num_samples = num_samples
        self.p_mean = nn.Parameter(torch.Tensor(num_dim,))
        self.p_mean.data.fill_(0)
        self.p_var = nn.Parameter(torch.Tensor(num_dim,))
        self.p_var.data.fill_(-5)

    def forward(self):
        p_var = F.softplus(self.p_var)
        epsilon = torch.randn(self.num_samples, self.num_dim, device=self.device, requires_grad=True)
        p_sample = self.p_mean + torch.sqrt(p_var)*epsilon
        log_p = self.log_pdf(p_sample)
        log_epsilon = ((epsilon)**2/-2 - np.log(2*np.pi)/2 - torch.log(p_var)/2).sum(-1)
        logL = log_p - log_epsilon
        logL = logL.mean(0)
        return logL

    def draw_samples(self, num_samples):

        with torch.autograd.no_grad():
            p_var = F.softplus(self.p_var)
            epsilon = torch.randn(self.num_samples, self.num_dim, device=self.device, requires_grad=True)
            p_sample = self.p_mean + torch.sqrt(p_var)*epsilon
        return p_sample
