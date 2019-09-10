import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, num_units, activation='relu'):
        super(MLP, self).__init__()
        self.num_units = num_units
        self.num_layers = len(num_units)-1
        self.activation = activation

        for i in range(self.num_layers):
            dim_in, dim_out = self.num_units[i], self.num_units[i+1]
            setattr(self, 'weight_'+str(i), nn.Parameter(torch.Tensor(dim_out, dim_in)))
            setattr(self, 'bias_'+str(i), nn.Parameter(torch.Tensor(dim_out,)))
            nn.init.xavier_normal_(getattr(self, 'weight_'+str(i)), 1.)
            getattr(self, 'bias_'+str(i)).data.fill_(0.)

    def forward(self, x):
        for i in range(self.num_layers):
            weight = getattr(self, 'weight_'+str(i))
            bias = getattr(self, 'bias_'+str(i))
            x = F.linear(x, weight, bias)
            if i < self.num_layers-1:
                if self.activation == 'tanh':
                    x = torch.tanh(x)
                elif self.activation == 'relu':
                    x = torch.relu(x)
        return x

    def forward_and_jacobian(self, x):
        jacob = None
        for i in range(self.num_layers):
            weight = getattr(self, 'weight_'+str(i))
            bias = getattr(self, 'bias_'+str(i))
            x = F.linear(x, weight, bias)
            if i < self.num_layers-1:
                if self.activation == 'tanh':
                    x = torch.tanh(x)
                    dact = 1-x**2
                elif self.activation == 'relu':
                    dact = (x > 0).to(x.dtype)
                    x = torch.relu(x)
                dlayer = dact.reshape(dact.shape+(1,)) * weight
            else:
                dlayer = weight

            if jacob is None:
                jacob = dlayer
            else:
                jacob = torch.matmul(dlayer, jacob)
        return x, jacob
