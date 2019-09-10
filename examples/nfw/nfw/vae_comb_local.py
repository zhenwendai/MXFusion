import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .mlp import MLP


class VAE(nn.Module):
    def __init__(self, decoder, out_dim, latent_dim, Q, mlp_units, num_samples, sigma2_trans,
                 sigma2_rev, device):
        super(VAE, self).__init__()
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.device = device
        self.num_samples = num_samples
        self.Q = Q
        self.trans_net = MLP((Q+out_dim,) + mlp_units + (latent_dim,))
        self.reverse_net = MLP(tuple(list((Q,) + mlp_units + (latent_dim,))[::-1]))
        self.sigma2_trans = nn.Parameter(torch.Tensor(1,))
        self.sigma2_trans.data.fill_(sigma2_trans)
        self.sigma2_rev = nn.Parameter(torch.Tensor(1,))
        self.sigma2_rev.data.fill_(sigma2_rev)

    def compute_loglikelihood(self, x, y):
        log_prob = self.decoder(x)
        logL = -F.softplus((1-2*y)*log_prob).sum(-1)
        logL += ((x)**2/-2 - np.log(2*np.pi)/2).sum(-1)
        return logL

    def forward(self, y):
        N = y.shape[0]

        z_sample = torch.randn(self.num_samples, N, self.Q, device=self.device, requires_grad=True)
        z_sample_expand = torch.cat([z_sample, y[None, :, :].expand(self.num_samples, -1, -1)], dim=-1)

        x_sample = self.trans_net(z_sample_expand)
        epsilon = torch.randn(self.num_samples, N, self.latent_dim, device=self.device, requires_grad=True)
        sigam2_trans = F.softplus(self.sigma2_trans)
        x_sample_noisy = x_sample + torch.sqrt(sigam2_trans) * epsilon

        log_p = self.compute_loglikelihood(x_sample_noisy, y[None, :, :])
        log_z = ((z_sample)**2/(-2) - np.log(2*np.pi)/2).sum(-1)

        # regularization term
        sigma2_rev = F.softplus(self.sigma2_rev)
        aux_mean, jacob = self.reverse_net.forward_and_jacobian(x_sample)
        reg_term = (((aux_mean - z_sample)**2 + (sigam2_trans*jacob**2).sum(-1))/(-2*sigma2_rev)
                    - torch.log(sigma2_rev)/2 + torch.log(sigam2_trans)/2 + 0.5).sum(-1)

        logL = log_p - log_z + reg_term
        logL = logL.mean(0)
        # logL = torch.logsumexp(logL, dim=0) - np.log(self.num_samples)
        return logL.mean(0)

    def comp_test_logL(self, y, num_samples):

        with torch.autograd.no_grad():
            N = y.shape[0]

            z_sample = torch.randn(num_samples, N, self.Q, device=self.device, requires_grad=True)
            z_sample_expand = torch.cat([z_sample, y[None, :, :].expand(num_samples, -1, -1)], dim=-1)

            x_sample = self.trans_net(z_sample_expand)
            epsilon = torch.randn(num_samples, N, self.latent_dim, device=self.device, requires_grad=True)
            sigam2_trans = F.softplus(self.sigma2_trans)
            x_sample_noisy = x_sample + torch.sqrt(sigam2_trans) * epsilon

            log_z = ((z_sample)**2/(-2) - np.log(2*np.pi)/2).sum(-1)
            log_epsilon = ((epsilon)**2/-2 - np.log(2*np.pi)/2 - torch.log(sigam2_trans)/2).sum(-1)

            log_p = self.compute_loglikelihood(x_sample_noisy, y[None, :, :])

            aux_mean = self.reverse_net(x_sample_noisy)
            sigma2_rev = F.softplus(self.sigma2_rev)
            log_aux = ((z_sample - aux_mean)**2/(-2*sigma2_rev) - torch.log(2*np.pi*sigma2_rev)/2).sum(-1)

            logL = log_p + log_aux - log_z - log_epsilon
            logL = torch.logsumexp(logL, dim=0) - np.log(num_samples)
        return logL
