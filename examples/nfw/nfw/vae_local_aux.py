import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .mlp import MLP


class Encoder(nn.Module):
    def __init__(self, output_dim, H, latent_dim, var_delta=-10):
        super(Encoder, self).__init__()
        self.var_delta = var_delta

        self.layer = nn.Sequential(nn.Linear(output_dim, H),
                                   nn.ReLU(),
                                   nn.Linear(H, H),
                                   nn.ReLU())
        self.mean_out = nn.Linear(H, latent_dim)
        self.var_out = nn.Linear(H, latent_dim)

    def forward(self, x):
        out = self.layer(x)
        mean = self.mean_out(out)
        var = F.softplus(self.var_out(out) + self.var_delta)
        return mean, var


class VAE(nn.Module):
    def __init__(self, decoder, encoder, out_dim, num_dim, Q, mlp_units, num_samples, sigma2_trans, sigma2_rev, device, dtype=torch.float):
        super(VAE, self).__init__()
        self.decoder = decoder
        self.encoder = encoder
        self.out_dim = out_dim
        self.num_dim = num_dim
        self.device = device
        self.dtype = dtype
        self.num_samples = num_samples
        self.Q = Q
        self.trans_net = MLP((self.Q,) + mlp_units + (self.num_dim,))
        self.reverse_net = MLP(tuple(list((self.Q,) + mlp_units + (self.num_dim,))[::-1]))
        self.sigma2_trans = nn.Parameter(torch.Tensor(self.num_dim,))
        self.sigma2_trans.data.fill_(sigma2_trans)
        self.sigma2_rev = nn.Parameter(torch.Tensor(1,))
        self.sigma2_rev.data.fill_(sigma2_rev)
        #
        self.trans_x_var = MLP((out_dim,) + mlp_units + (num_dim,))
        self.reverse_x = Encoder(num_dim, 200, num_dim, 0)

    def compute_loglikelihood(self, x, y):
        log_prob = self.decoder(x)
        logL = -F.softplus((1-2*y)*log_prob).sum(-1)
        logL += ((x)**2/-2 - np.log(2*np.pi)/2).sum(-1)
        return logL

    def _compute_logL(self, y, num_samples):
        N = y.shape[0]

        z_mean, z_var = self.encoder(y)
        z_sample = z_mean + torch.sqrt(z_var)*torch.randn(num_samples, N, self.Q, device=self.device, requires_grad=True, dtype=self.dtype)
        x_sample = self.trans_net(z_sample)
        epsilon = torch.randn(num_samples, N, self.num_dim, device=self.device, requires_grad=True, dtype=self.dtype)
        sigam2_trans = F.softplus(self.sigma2_trans)
        x_sample_noisy = x_sample + torch.sqrt(sigam2_trans) * epsilon

        sigam2_x_var = F.softplus(self.trans_x_var(y)-15)
        epsilon_x_var = torch.randn(num_samples, N, self.num_dim, device=self.device, requires_grad=True, dtype=self.dtype)
        x_sample_extra = x_sample_noisy + torch.sqrt(sigam2_x_var) * epsilon_x_var

        log_p = self.compute_loglikelihood(x_sample_extra, y[None, :, :])

        log_z = ((z_sample-z_mean)**2/(-2*z_var) - torch.log(2*np.pi*z_var)/2).sum(-1)

        # regularization term
        sigma2_rev = F.softplus(self.sigma2_rev)
        aux_mean, jacob = self.reverse_net.forward_and_jacobian(x_sample)
        reg_term = (((aux_mean - z_sample)**2 + (sigam2_trans*jacob**2).sum(-1))/(-2*sigma2_rev)
                    - torch.log(sigma2_rev)/2 + torch.log(sigam2_trans)/2 + 0.5).sum(-1)

        # extra
        log_q_x = ((epsilon_x_var)**2/-2 - np.log(2*np.pi)/2 - torch.log(sigam2_x_var)/2).sum(-1)
        rx_mean, rx_var = self.reverse_x(x_sample_extra)
        log_q_x_aux = ((x_sample_noisy - rx_mean)**2/(-2*rx_var) - torch.log(2*np.pi*rx_var)/2).sum(-1)

        logL = log_p - log_z + reg_term - log_q_x + log_q_x_aux
        return logL

    def forward(self, y):
        logL = self._compute_logL(y, self.num_samples)
        logL = logL.mean(0)
        return logL.mean(0)

    def comp_test_logL(self, y, num_samples):

        with torch.autograd.no_grad():

            N = y.shape[0]

            z_mean, z_var = self.encoder(y)
            z_sample = z_mean + torch.sqrt(z_var)*torch.randn(num_samples, N, self.Q, device=self.device, requires_grad=True, dtype=self.dtype)
            x_sample = self.trans_net(z_sample)
            epsilon = torch.randn(num_samples, N, self.num_dim, device=self.device, requires_grad=True, dtype=self.dtype)
            sigam2_trans = F.softplus(self.sigma2_trans)
            x_sample_noisy = x_sample + torch.sqrt(sigam2_trans) * epsilon

            sigam2_x_var = F.softplus(self.trans_x_var(y)-10)
            epsilon_x_var = torch.randn(num_samples, N, self.num_dim, device=self.device, requires_grad=True, dtype=self.dtype)
            x_sample_extra = x_sample_noisy + torch.sqrt(sigam2_x_var) * epsilon_x_var

            log_z = ((z_sample-z_mean)**2/(-2*z_var) - torch.log(2*np.pi*z_var)/2).sum(-1)
            log_epsilon = ((epsilon)**2/-2 - np.log(2*np.pi)/2 - torch.log(sigam2_trans)/2).sum(-1)

            log_p = self.compute_loglikelihood(x_sample_extra, y[None, :, :])

            aux_mean = self.reverse_net(x_sample)
            sigma2_rev = F.softplus(self.sigma2_rev)
            log_aux = ((z_sample - aux_mean)**2/(-2*sigma2_rev) - torch.log(2*np.pi*sigma2_rev)/2).sum(-1)

            # extra
            log_q_x = ((epsilon_x_var)**2/-2 - np.log(2*np.pi)/2 - torch.log(sigam2_x_var)/2).sum(-1)
            rx_mean, rx_var = self.reverse_x(x_sample_extra)
            log_q_x_aux = ((x_sample_noisy - rx_mean)**2/(-2*rx_var) - torch.log(2*np.pi*rx_var)/2).sum(-1)

            logL = log_p + log_aux - log_z - log_epsilon - log_q_x + log_q_x_aux
            logL = torch.logsumexp(logL, dim=0) - np.log(num_samples)

        return logL
