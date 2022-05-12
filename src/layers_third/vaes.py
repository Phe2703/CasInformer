import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.distributions as distributions
import torch.optim as optim
import numpy as np


class PlanarFlow(nn.Module):
    def __init__(self, dim):
        """Instantiates one step of planar flow.

        Reference:
        Variational Inference with Normalizing Flows
        Danilo Jimenez Rezende, Shakir Mohamed
        (https://arxiv.org/abs/1505.05770)

        Args:
            dim: input dimensionality.
        """
        super(PlanarFlow, self).__init__()

        self.u = nn.Parameter(torch.randn(1, dim))
        self.w = nn.Parameter(torch.randn(1, dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        def m(x):
            return F.softplus(x) - 1.
        def h(x):
            return torch.tanh(x)
        def h_prime(x):
            return 1. - h(x)**2

        inner = (self.w * self.u).sum()
        u = self.u + (m(inner) - inner) * self.w / self.w.norm()**2
        activation = (self.w * x).sum(dim=1, keepdim=True) + self.b
        x = x + u * h(activation)
        psi = h_prime(activation) * self.w
        log_det = torch.log(torch.abs(1. + (u * psi).sum(dim=1, keepdim=True)))

        return x, log_det

class RadialFlow(nn.Module):
    def __init__(self, dim):
        """Instantiates one step of radial flow.

        Reference:
        Variational Inference with Normalizing Flows
        Danilo Jimenez Rezende, Shakir Mohamed
        (https://arxiv.org/abs/1505.05770)

        Args:
            dim: input dimensionality.
        """
        super(RadialFlow, self).__init__()

        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1, dim))
        self.d = dim

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        def m(x):
            return F.softplus(x)
        def h(r):
            return 1. / (a + r)
        def h_prime(r):
            return -h(r)**2

        a = torch.exp(self.a)
        b = -a + m(self.b)
        r = (x - self.c).norm(dim=1, keepdim=True)
        tmp = b * h(r)
        x = x + tmp * (x - self.c)
        log_det = (self.d - 1) * torch.log(1. + tmp) + torch.log(1. + tmp + b * h_prime(r) * r)

        return x, log_det

class HouseholderFlow(nn.Module):
    def __init__(self, dim):
        """Instantiates one step of householder flow.
        
        Reference:
        Improving Variational Auto-Encoders using Householder Flow
        Jakub M. Tomczak, Max Welling
        (https://arxiv.org/abs/1611.09630)

        Args:
            dim: input dimensionality.
        """
        super(HouseholderFlow, self).__init__()

        self.v = nn.Parameter(torch.randn(1, dim))
        self.d = dim

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        outer = self.v.t() * self.v
        v_sqr = self.v.norm()**2
        H = torch.eye(self.d, device=x.device) - 2. * outer / v_sqr
        # H = torch.eye(self.d).cuda() - 2. * outer / v_sqr
        x = torch.mm(H, x.t()).t()
        
        return x, 0

class NiceFlow(nn.Module):
    def __init__(self, dim, mask, final=False):
        """Instantiates one step of NICE flow.

        Reference:
        NICE: Non-linear Independent Components Estimation
        Laurent Dinh, David Krueger, Yoshua Bengio
        (https://arxiv.org/abs/1410.8516)

        Args:
            dim: input dimensionality.
            mask: mask that determines active variables.
            final: True if the final step, False otherwise.
        """
        super(NiceFlow, self).__init__()

        self.final = final
        if final:
            self.scale = nn.Parameter(torch.zeros(1, dim))
        else:
            self.mask = mask
            self.coupling = nn.Sequential(
                nn.Linear(dim//2, dim*5), nn.ReLU(), 
                nn.Linear(dim*5, dim*5), nn.ReLU(), 
                nn.Linear(dim*5, dim//2))

    def forward(self, x):
        if self.final:
            x = x * torch.exp(self.scale)
            log_det = torch.sum(self.scale)
            
            return x, log_det
        else:
            [B, W] = list(x.size())
            x = x.reshape(B, W//2, 2)
            
            if self.mask:
                on, off = x[:, :, 0], x[:, :, 1]
            else:
                off, on = x[:, :, 0], x[:, :, 1]
            
            on = on + self.coupling(off)

            if self.mask:
                x = torch.stack((on, off), dim=2)
            else:
                x = torch.stack((off, on), dim=2)
            
            return x.reshape(B, W), 0

class Flow(nn.Module):
    def __init__(self, dim, type, length):
        """Instantiates a chain of flows.

        Args:
            dim: input dimensionality.
            type: type of flow.
            length: length of flow.
        """
        super(Flow, self).__init__()

        if type == 'planar':
            self.flow = nn.ModuleList([PlanarFlow(dim) for _ in range(length)])
        elif type == 'radial':
            self.flow = nn.ModuleList([RadialFlow(dim) for _ in range(length)])
        elif type == 'householder':
            self.flow = nn.ModuleList([HouseholderFlow(dim) for _ in range(length)])
        elif type == 'nice':
            self.flow = nn.ModuleList([NiceFlow(dim, i//2, i==(length-1)) for i in range(length)])
        else:
            self.flow = nn.ModuleList([])

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        # [B, _] = list(x.size())
        B = x.size(0)
        log_det = torch.zeros(B, 1, device=x.device)
        # log_det = torch.zeros(B, 1).cuda()
        for i in range(len(self.flow)):
            x, inc = self.flow[i](x)
            log_det = log_det + inc

        return x, log_det

class GatedLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        """Instantiates a gated MLP layer.

        Args:
            in_dim: input dimensionality.
            out_dim: output dimensionality.
        """
        super(GatedLayer, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.gate = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid())

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            transformed x.
        """
        return self.linear(x) * self.gate(x)

class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, gate):
        """Instantiates an MLP layer.

        Args:
            in_dim: input dimensionality.
            out_dim: output dimensionality.
            gate: whether to use gating mechanism.
        """
        super(MLPLayer, self).__init__()

        if gate:
            self.layer = GatedLayer(in_dim, out_dim)
        else:
            self.layer = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            transformed x.
        """
        return self.layer(x)

class StaticFlowVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, layer = 2, gate = 0, flow = 'none', length = 2):
        """Instantiates a VAE.

        Args:
            layer: number of hidden layers.
            in_dim: input dimensionality.
            hidden_dim: hidden dimensionality.
            latent_dim: latent dimensionality.
            gate: whether to use gating mechanism.
            flow: type of the flow (None if do not use flow).
            length: length of the flow.
        """
        super(StaticFlowVAE, self).__init__()

        self.latent_dim = latent_dim
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

        self.encoder = nn.ModuleList(
            [MLPLayer(in_dim, hidden_dim, gate)] + \
            [MLPLayer(hidden_dim, hidden_dim, gate) for _ in range(layer - 1)])
        self.flow = Flow(latent_dim, flow, length)
        self.decoder = nn.ModuleList(
            [MLPLayer(latent_dim, hidden_dim, gate)] + \
            [MLPLayer(hidden_dim, hidden_dim, gate) for _ in range(layer - 1)] + \
            [nn.Linear(hidden_dim, in_dim)])

    def encode(self, x):
        """Encodes input.

        Args:
            x: input tensor (B x D).
        Returns:
            mean and log-variance of the gaussian approximate posterior.
        """
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
        return self.mean(x), self.log_var(x)

    def transform(self, mean, log_var):
        """Transforms approximate posterior.

        Args:
            mean: mean of the gaussian approximate posterior.
            log_var: log-variance of the gaussian approximate posterior.
        Returns:
            transformed latent codes and the log-determinant of the Jacobian.
        """
        std = torch.exp(.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)

        return self.flow(z)

    def decode(self, z):
        """Decodes latent codes.

        Args:
            z: latent codes.
        Returns:
            reconstructed input.
        """
        for i in range(len(self.decoder)):
            z = self.decoder[i](z)
        return z

    def sample(self, size):
        """Generates samples from the prior.

        Args:
            size: number of samples to generate.
        Returns:
            generated samples.
        """
        z = torch.randn(size, self.latent_dim).cuda()
        return self.decode(z)
        if self.dataset == 'mnist':
            return torch.sigmoid(self.decode(z))
        else:
            return self.decode(z)

    def reconstruction_loss(self, x, x_hat):
        """Computes reconstruction loss.

        Args:
            x: original input (B x D).
            x_hat: reconstructed input (B x D).
        Returns: 
            sum of reconstruction loss over the minibatch.
        """
        return nn.MSELoss(reduction='none')(x_hat, x).sum(dim=1, keepdim=True)
        if self.dataset == 'mnist':
            return nn.BCEWithLogitsLoss(reduction='none')(x_hat, x).sum(dim=1, keepdim=True)
        else:
            return nn.MSELoss(reduction='none')(x_hat, x).sum(dim=1, keepdim=True)

    def latent_loss(self, mean, log_var, log_det):
        """Computes KL loss.

        Args:
            mean: mean of the gaussian approximate posterior.
            log_var: log-variance of the gaussian approximate posterior.
            log_det: log-determinant of the Jacobian.
        Returns: sum of KL loss over the minibatch.
        """
        kl = -.5 * torch.sum(1. + log_var - mean.pow(2) - log_var.exp(), dim=1, keepdim=True)
        return kl - log_det

    def loss(self, x, x_hat, mean, log_var, log_det):
        """Computes overall loss.

        Args:
            x: original input (B x D).
            x_hat: reconstructed input (B x D).
            mean: mean of the gaussian approximate posterior.
            log_var: log-variance of the gaussian approximate posterior.
            log_det: log-determinant of the Jacobian.
        Returns:
            sum of reconstruction and KL loss over the minibatch.
        """
        return self.reconstruction_loss(x, x_hat) + self.latent_loss(mean, log_var, log_det) 

    def reparameterize(self, mean, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            average loss over the minibatch.
        """
        mean, log_var = self.encode(x)
        # z, log_det = self.reparameterize(mean, log_var)
        z, log_det = self.transform(mean, log_var)
        x_hat = self.decode(z)
        
        return x_hat, self.loss(x, x_hat, mean, log_var, log_det).mean()
    
    # def get_loss(self, x_hat, x, mean, log_var):
    #     mse = F.mse_loss(x_hat, x, reduction='sum')
    #     kld_e = mean.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
    #     kld = torch.sum(kld_e).mul_(-0.5)
    #     return mse + kld
    
    # def forward(self, x):
    #     mean, log_var = self.encode(x)
    #     z = self.reparameterize(mean, log_var)
    #     x_hat = self.decode(z)
    #     return x_hat, self.get_loss(x_hat, x, mean, log_var)


# def logit_transform(x, constraint=0.9, reverse=False):
#     '''Transforms data from [0, 1] into unbounded space.

#     Restricts data into [0.05, 0.95].
#     Calculates logit(alpha+(1-alpha)*x).

#     Args:
#         x: input tensor.
#         constraint: data constraint before logit.
#         reverse: True if transform data back to [0, 1].
#     Returns:
#         transformed tensor and log-determinant of Jacobian from the transform.
#         (if reverse=True, no log-determinant is returned.)
#     '''
#     if reverse:
#         x = 1. / (torch.exp(-x) + 1.)    # [0.05, 0.95]
#         x *= 2.             # [0.1, 1.9]
#         x -= 1.             # [-0.9, 0.9]
#         x /= constraint     # [-1, 1]
#         x += 1.             # [0, 2]
#         x /= 2.             # [0, 1]
#         return x, 0
#     else:
#         [B, C, D] = list(x.size())
        
#         # dequantization
#         noise = distributions.Uniform(0., 1.).sample((B, C, D))
#         x = (x * 255. + noise) / 256.
        
#         # restrict data
#         x *= 2.             # [0, 2]
#         x -= 1.             # [-1, 1]
#         x *= constraint     # [-0.9, 0.9]
#         x += 1.             # [0.1, 1.9]
#         x /= 2.             # [0.05, 0.95]

#         # logit data
#         logit_x = torch.log(x) - torch.log(1. - x)

#         # log-determinant of Jacobian from the transform
#         pre_logit_scale = torch.tensor(
#             np.log(constraint) - np.log(1. - constraint))
#         log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) \
#             - F.softplus(-pre_logit_scale)

#         return logit_x, torch.sum(log_diag_J, dim=(1, 2)).mean()