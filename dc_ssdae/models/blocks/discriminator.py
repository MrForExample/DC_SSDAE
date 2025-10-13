

# Adapted from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py

from typing import Optional

import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch.nn.utils import spectral_norm


def gan_weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    m._w_init = True


def get_conv(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs), n_power_iterations=1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_fn="batch_norm", trajectory=False, flatten=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        self.norm_fn = norm_fn
        self.time_proj = None

        if norm_fn == "batch_norm":
            norm_layer = nn.BatchNorm2d
        elif norm_fn == "act_norm":
            norm_layer = ActNorm
        elif norm_fn == "group_norm":
            norm_layer = lambda d: nn.GroupNorm(32, d, eps=1e-5, affine=True)
        elif norm_fn == "ada":
            self.time_proj = Timesteps(128, True, 0)
            self.time_embedding = TimestepEmbedding(128, 128, act_fn="silu")

            norm_layer = lambda d: AdaLayerNorm2d(d, 128)
        else:
            raise ValueError(f"Unknown norm layer: {norm_fn}")

        # no need to use bias as BatchNorm2d has affine parameters
        use_bias = norm_fn != "act_norm"
        self.trajectory = trajectory
        if trajectory:
            input_nc *= 2

        kw = 4
        padw = 1
        sequence = [
            get_conv(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                get_conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            get_conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        self.flatten = flatten
        if flatten:
            self.post_proj = nn.Sequential(nn.Linear(ndf * nf_mult, 1), nn.Sigmoid())
        else:
            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.ModuleList(sequence)

        self.apply(gan_weights_init)

    def forward(self, x, xt=None, t=None):
        """Standard forward."""

        if self.trajectory:
            x = torch.cat([x, xt], dim=1)

        if self.time_proj is not None:
            t_emb = self.time_proj(t)
            t_emb = self.time_embedding(t_emb)

        x = x.clamp(-1.0, 1.0)

        for l in self.main:
            if isinstance(l, AdaLayerNorm2d):
                x = l(x, t_emb)
            else:
                x = l(x)

        if self.flatten:
            x = x.mean(dim=[-1, -2])
            x = self.post_proj(x)

        return x.clamp(min=-10, max=10)


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True, allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            std = flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError("Initializing ActNorm in reverse direction is disabled by default. Use allow_reverse_init=True to enable.")
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class AdaLayerNorm2d(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        tdim: int,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(tdim, embedding_dim * 2)
        self.norm = nn.GroupNorm(32, embedding_dim, eps=norm_eps, affine=norm_elementwise_affine)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        temb = self.linear(self.silu(temb))

        shift, scale = temb[:, :, None, None].chunk(2, dim=1)

        x = self.norm(x) * (1 + scale) + shift
        return x
