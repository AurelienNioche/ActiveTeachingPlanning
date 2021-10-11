import os
import torch
import torch.distributions as dist
from torch import nn


def safe_log(z):
    return torch.log(z + 1e-7)


class Planar(nn.Module):

    """
    Planar flow.
    z = f(x) = x + u h(wᵀx + b)
    """

    def __init__(self, dim):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(1, dim))
        self.bias = nn.Parameter(torch.empty(1))
        self.scale = nn.Parameter(torch.empty(1, dim))
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.uniform_(self.weight, -0.01, 0.01)
        nn.init.uniform_(self.scale, -0.01, 0.01)
        nn.init.uniform_(self.bias, -0.01, 0.01)

    def forward(self, x):

        activation = self.tanh(x@self.weight.t() + self.bias)

        f = x + self.scale * activation

        psi = (1 - activation ** 2) * self.weight
        det_grad = 1 + psi@self.scale.t()
        ld = safe_log(det_grad.abs()).squeeze(-1)

        return f, ld


class NormalizingFlows(nn.Module):

    """
    Adapted from https://github.com/ex4sperans/variational-inference-with-normalizing-flows

    [Rezende and Mohamed, 2015]

    """

    BKP_DIR = "bkp"

    def __init__(self, dim, flow_length):
        super().__init__()

        self.transforms = nn.ModuleList([
            Planar(dim) for _ in range(flow_length)
        ])

        self.dim = dim
        self.flow_length = flow_length

        self.mu = nn.Parameter(torch.zeros(dim).uniform_(-0.01, 0.01))
        self.log_var = nn.Parameter(torch.zeros(dim).uniform_(-0.01, 0.01))

    def sample_base_dist(self, batch_size):
        std = torch.exp(0.5 * self.log_var)
        eps = torch.randn((batch_size, self.dim))
        return self.mu + eps * std

    def log_prob_base_dist(self, x):
        std = torch.exp(0.5 * self.log_var)
        return dist.Normal(self.mu, std).log_prob(x).sum(axis=-1)

    def forward(self, x):

        log_prob_base_dist = self.log_prob_base_dist(x)

        log_det = torch.zeros(x.shape[0])

        for i in range(self.flow_length):
            x, ld = self.transforms[i](x)
            log_det += ld

        return x, log_prob_base_dist, log_det

    def save(self, name):

        path = os.path.join(self.BKP_DIR, name)
        os.makedirs(self.BKP_DIR, exist_ok=True)
        torch.save(self.state_dict(),
                   path+"_state_dict.p")
        torch.save(dict(dim=self.dim,
                        flow_length=self.flow_length,
                        flow_type=self.flow_model.__name__),
                   path+"_attr.p")

    @classmethod
    def load(cls, name):
        path = os.path.join(cls.BKP_DIR, name)
        model = cls(**torch.load(path+"_attr.p"))
        model.load_state_dict(torch.load(path+"_state_dict.p"))
        return model

    def inverse(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        for flow in self.transforms[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples):
        z = self.sample_base_dist(n_samples)
        x, _ = self.inverse(z)
        return x
