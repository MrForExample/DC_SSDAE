import math

import torch
import torch.nn as nn

from ..mutils.train_utils import init_weights
from .blocks.diag_gauss import DiagonalGaussianDistribution


class SwishActivation(nn.Module):
    # Custom implementation of Swish activation function to ensure correct reproduction
    def forward(self, x):
        return x * torch.sigmoid(x)


class VQGDownsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)  # pylint: disable=E1102
        return x


class VQGResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_dim,
        out_channels=None,
        kernel_size=3,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        normalize=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        out_channels = in_dim if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = nn.GroupNorm(32, in_dim, eps=1e-6) if normalize else torch.nn.Identity()
        self.act1 = SwishActivation()
        self.conv1 = torch.nn.Conv2d(in_dim, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
            self.temb_act = SwishActivation()
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6) if normalize else torch.nn.Identity()
        self.act2 = SwishActivation()
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        if self.in_dim != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_dim, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_dim, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = self.act1(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.temb_act(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.act1(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_dim != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class VQGAttnBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim

        self.norm = nn.GroupNorm(32, in_dim, eps=1e-6)
        self.q = torch.nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class VQEncoder(nn.Module):
    def __init__(
        self,
        *,
        base_dim,
        z_dim,
        num_res_blocks,
        in_dim=3,
        ch_mult=(1, 2, 4, 8),
        attn_levels=(),
        dropout=0.0,
        resamp_with_conv=True,
        double_z=False,
        checkpoint=None,
    ):
        super().__init__()
        self.base_dim = base_dim
        self.temb_ch = 0
        self.num_levels = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.patch_size = 2 ** (len(ch_mult) - 1)

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_dim, self.base_dim, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_levels):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = base_dim * in_ch_mult[i_level]
            block_out = base_dim * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(VQGResnetBlock(in_dim=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if i_level in attn_levels:
                    attn.append(VQGAttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_levels - 1:
                down.downsample = VQGDownsample(block_in, resamp_with_conv)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = VQGResnetBlock(in_dim=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = VQGAttnBlock(block_in)
        self.mid.block_2 = VQGResnetBlock(in_dim=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # end
        self.double_z = double_z
        out_ch = 2 * z_dim if double_z else z_dim
        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)
        self.act_out = SwishActivation()
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
        self.out_proj = nn.Conv2d(out_ch, out_ch, 1)

        self.init_weights(checkpoint=checkpoint)

    def init_weights(self, ckpt_module="encoder", **kwargs):
        init_weights(self, ckpt_module=ckpt_module, **kwargs)

    def forward(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        # timestep embedding
        temb = None

        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_levels):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_levels - 1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = self.act_out(h)
        hb = self.conv_out(h)
        hb = self.out_proj(hb)

        if self.double_z:
            hb = DiagonalGaussianDistribution(hb, deterministic=False)
        else:
            hb = DiagonalGaussianDistribution(torch.cat([hb, torch.zeros_like(hb)], axis=1), deterministic=True)

        return hb

    ### Initialize with a standard architecture ###

    @staticmethod
    def get_config(z_dim, patch_size, **vq_kwargs):
        return {
            **dict(
                z_dim=z_dim,
                ch_mult=[1, 2] + [4] * int(math.log2(patch_size) - 1),
                double_z=True,
                base_dim=128,
                num_res_blocks=2,
                dropout=0.0,
            ),
            **vq_kwargs,
        }

    @classmethod
    def make(cls, z_dim, patch_size, encoder_checkpoint):
        cfg = cls.get_config(z_dim, patch_size=patch_size, checkpoint=encoder_checkpoint)
        return cls(**cfg)
