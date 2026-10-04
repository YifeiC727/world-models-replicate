"""Vision (V), memory (M), and controller (C) from World Models."""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):
    """Paper-faithful 64x64 convolutional VAE.

    TensorFlow's default VALID convolutions give spatial sizes 31, 14, 6, 2.
    The asymmetric transposed-convolution decoder maps 1, 5, 13, 30, 64.
    """

    def __init__(self, latent_size: int = 32):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2), nn.ReLU(), nn.Flatten(),
        )
        self.mu = nn.Linear(1024, latent_size)
        self.logvar = nn.Linear(1024, latent_size)
        self.decoder_input = nn.Linear(latent_size, 1024)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (1024, 1, 1)),
            nn.ConvTranspose2d(1024, 128, 5, 2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, 2), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(.5 * logvar)

    def decode(self, z):
        return self.decoder(self.decoder_input(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def loss(recon, x, mu, logvar, beta: float = 1.0, kl_tolerance: float = .5):
        reconstruction = F.mse_loss(recon, x, reduction="sum") / x.shape[0]
        kl_per_item = -.5 * torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=1)
        kl = torch.clamp(kl_per_item, min=kl_tolerance * mu.shape[1]).mean()
        return reconstruction + beta * kl, {"reconstruction": reconstruction, "kl": kl}


class MDNRNN(nn.Module):
    """LSTM predicting a diagonal Gaussian mixture for the next latent state."""

    def __init__(self, latent_size=32, action_size=3, hidden_size=256, mixtures=5,
                 discrete_actions=True):
        super().__init__()
        self.latent_size, self.action_size = latent_size, action_size
        self.hidden_size, self.mixtures = hidden_size, mixtures
        self.discrete_actions = discrete_actions
        self.rnn = nn.LSTM(latent_size + action_size, hidden_size, batch_first=True)
        # PyTorch stores separate input/hidden gate biases, whereas the TF 1.x
        # reference has one. Their sum is equivalent, so freeze the redundant
        # hidden bias at zero to match the paper's trainable parameter count.
        with torch.no_grad():
            self.rnn.bias_hh_l0.zero_()
        self.rnn.bias_hh_l0.requires_grad_(False)
        # The reference implementation predicts an independent K-component
        # mixture for every scalar latent dimension.
        self.head = nn.Linear(hidden_size, latent_size * mixtures * 3)

    def forward(self, z, actions, hidden=None):
        if self.discrete_actions:
            actions = F.one_hot(actions.long(), self.action_size).float()
        else:
            actions = actions.float()
            if actions.ndim == 2:
                actions = actions.unsqueeze(-1)
        out, hidden = self.rnn(torch.cat((z, actions), dim=-1), hidden)
        raw = self.head(out).view(*z.shape[:2], self.latent_size, 3 * self.mixtures)
        logits, mu, log_sigma = torch.split(raw, self.mixtures, dim=-1)
        return logits, mu, log_sigma.clamp(-7, 7), hidden

    def nll(self, z, actions, target):
        logits, mu, log_sigma, _ = self(z, actions)
        target = target.unsqueeze(-1)
        component = -.5 * (((target-mu)/log_sigma.exp()).square()
                            + 2*log_sigma + math.log(2*math.pi))
        per_dimension = -torch.logsumexp(F.log_softmax(logits, -1) + component, -1)
        return per_dimension.mean()

    def step_hidden(self, z, action, hidden=None):
        """Update memory for one environment step; return flat LSTM h."""
        if self.discrete_actions:
            a = torch.as_tensor([[action]], device=z.device)
        else:
            a = torch.as_tensor(action, dtype=torch.float32, device=z.device).view(1, 1, -1)
        _, _, _, hidden = self(z[:, None], a, hidden)
        return hidden, hidden[0][-1]


class LinearController(nn.Module):
    def __init__(self, latent_size=16, hidden_size=128, action_size=3, use_memory=True):
        super().__init__()
        self.use_memory = use_memory
        self.linear = nn.Linear(latent_size + (hidden_size if use_memory else 0), action_size)

    def forward(self, z, h=None):
        x = torch.cat((z, h), -1) if self.use_memory else z
        return self.linear(x)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def set_flat_parameters(self, values):
        values = torch.as_tensor(values, dtype=torch.float32)
        offset = 0
        with torch.no_grad():
            for p in self.parameters():
                n = p.numel()
                p.copy_(values[offset:offset+n].view_as(p))
                offset += n
        if offset != len(values):
            raise ValueError(f"expected {offset} controller parameters, got {len(values)}")
