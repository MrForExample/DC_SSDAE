# Code modified based on: https://github.com/dc-ai-projects/DC-Gen/blob/main/dc_gen/aecore/models/dc_ae.py 
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from .nn.act import build_act
from .nn.norm import build_norm
from .nn.ops import (
    AdaptiveInputConvLayer,
    AdaptiveOutputConvLayer,
    ChannelAttentionResBlock,
    ChannelDuplicatingPixelShuffleUpSampleLayer,
    ConvLayer,
    ConvPixelShuffleUpSampleLayer,
    ConvPixelUnshuffleDownSampleLayer,
    EfficientViTBlock,
    GLUMBConv,
    GLUResBlock,
    IdentityLayer,
    InterpolateConvUpSampleLayer,
    OpSequential,
    PixelUnshuffleChannelAveragingDownSampleLayer,
    ResBlock,
    ResidualBlock,
    SoftmaxCrossAttention,
)
from .nn.utils import get_submodule_weights
from .blocks.diag_gauss import DiagonalGaussianDistribution

__all__ = ["DCAE", "dc_ae_f32c32", "dc_ae_f64c128", "dc_ae_f128c512"]

@dataclass
class EncoderConfig:
    in_channels: int = 3
    latent_channels: int = 32
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = "ResBlock"
    norm: Any = "trms2d"
    act: str = "silu"
    downsample_block_type: str = "ConvPixelUnshuffle"
    downsample_match_channel: bool = True
    downsample_shortcut: Optional[str] = "averaging"
    out_norm: Optional[str] = None
    out_act: Optional[str] = None
    out_block_type: str = "ConvLayer"
    out_shortcut: Optional[str] = "averaging"
    # We don't seem to need double_latent with KL loss, since as observed in experiments the mean and std converge to 0, 1 respectively as training progresses
    # But keeping it here for faster convergence at the beginning of training
    double_latent: bool = True 
    pretrained_path: Optional[str] = None
    pretrained_source: str = "dc-ae"

def dc_ae_fc(pretrained_path: Optional[str] = None, f=64, c=128) -> EncoderConfig:
    if f == 8:
        cfg_str = (
            f"latent_channels={c} "
            "block_type=ResBlock out_shortcut=null "
            "width_list=[128,256,512,512] depth_list=[0,5,10,4] "
            "pretrained_source=dc-ae"
        )
    elif f == 16:
        cfg_str = (
            f"latent_channels={c} "
            "block_type=ResBlock out_shortcut=null "
            "width_list=[128,256,512,512,1024] depth_list=[0,5,10,4,4] "
            "pretrained_source=dc-ae"
        )
    elif f == 32:
        cfg_str = (
            f"latent_channels={c} "
            "block_type=ResBlock out_shortcut=null "
            "width_list=[128,256,512,512,1024,1024] depth_list=[0,5,10,4,4,4] "
            "pretrained_source=dc-ae"
        )
    elif f == 64:
        cfg_str = (
            f"latent_channels={c} "
            "block_type=ResBlock out_shortcut=null "
            "width_list=[128,256,512,512,1024,1024,1024] depth_list=[0,5,10,4,4,4,4] "
            "pretrained_source=dc-ae"
        )
    elif f == 128:
        cfg_str = (
            f"latent_channels={c} "
            "block_type=ResBlock out_shortcut=null "
            "width_list=[128,256,512,512,1024,1024,1024,1024] depth_list=[0,5,10,4,4,4,4,4] "
            "pretrained_source=dc-ae"
        )
    else:
        raise NotImplementedError

    cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
    cfg: EncoderConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(EncoderConfig), cfg))
    cfg.pretrained_path = pretrained_path
    return cfg

def build_block(
    block_type: str, in_channels: int, out_channels: int, norm: Optional[str], act: Optional[str]
) -> nn.Module:
    cfg = block_type.split("@")
    block_name = cfg[0]
    if block_name == "ResBlock":
        assert in_channels == out_channels
        main_block = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
        )
        block = ResidualBlock(main_block, IdentityLayer())
    elif block_name == "GLUResBlock":
        assert in_channels == out_channels
        main_block = GLUResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            gate_kernel_size=int(cfg[1]),
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
        )
        block = ResidualBlock(main_block, IdentityLayer())
    elif block_name == "ChannelAttentionResBlock":
        assert in_channels == out_channels
        main_block = ChannelAttentionResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
            channel_attention_operation=cfg[1],
            channel_attention_position=int(cfg[2]) if len(cfg) > 2 else 2,
        )
        block = ResidualBlock(main_block, IdentityLayer())
    elif block_name == "GLUMBConv":
        assert in_channels == out_channels
        main_block = GLUMBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            expand_ratio=float(cfg[1]),
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act, act, None),
        )
        block = ResidualBlock(main_block, IdentityLayer())
    elif block_name == "EViTGLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=())
    elif block_name == "EViTNormQKGLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(
            in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=(), norm_qk=True
        )
    elif block_name == "EViTS5GLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=(5,))
    elif block_name == "ViTGLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(
            in_channels, norm=norm, act_func=act, context_module="SoftmaxAttention", local_module="GLUMBConv"
        )
    elif block_name == "SoftmaxCrossAttention":
        block = SoftmaxCrossAttention(
            q_in_channels=in_channels,
            kv_in_channels=int(cfg[1]),
            out_channels=out_channels,
            norm=(None, norm),
        )
    else:
        raise ValueError(f"block_name {block_name} is not supported")
    return block


def build_stage_main(
    width: int, depth: int, block_type: str | list[str], norm: str, act: str, input_width: int
) -> list[nn.Module]:
    assert isinstance(block_type, str) or (isinstance(block_type, list) and depth == len(block_type))
    stage = []
    for d in range(depth):
        current_block_type = block_type[d] if isinstance(block_type, list) else block_type
        block = build_block(
            block_type=current_block_type,
            in_channels=width if d > 0 else input_width,
            out_channels=width,
            norm=norm,
            act=act,
        )
        stage.append(block)
    return stage


def build_downsample_block(block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str]) -> nn.Module:
    if block_type == "Conv":
        block = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_bias=True,
            norm=None,
            act_func=None,
        )
    elif block_type == "ConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for downsampling")
    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for downsample")
    return block


def build_upsample_block(block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str]) -> nn.Module:
    if block_type == "ConvPixelShuffle":
        block = ConvPixelShuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    elif block_type == "InterpolateConv":
        block = InterpolateConvUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for upsampling")
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelShuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    return block


def build_encoder_project_in_block(in_channels: int, out_channels: int, factor: int, downsample_block_type: str):
    if factor == 1:
        block = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
        )
    elif factor == 2:
        block = build_downsample_block(
            block_type=downsample_block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None
        )
    else:
        raise ValueError(f"downsample factor {factor} is not supported for encoder project in")
    return block


def build_encoder_project_out_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    norm: Optional[str],
    act: Optional[str],
    shortcut: Optional[str],
):
    block = [build_norm(norm), build_act(act)]
    if block_type == "ConvLayer":
        block.append(
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_bias=True,
                norm=None,
                act_func=None,
            )
        )
    elif block_type == "AdaptiveOutputConvLayer":
        block.append(
            AdaptiveOutputConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_bias=True,
            )
        )
    else:
        raise ValueError(f"encoder project out block type {block_type} is not supported")
    block = OpSequential(block)

    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=1
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for encoder project out")
    return block


def build_decoder_project_in_block(block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str]):
    if block_type == "ConvLayer":
        block = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
        )
    elif block_type == "AdaptiveInputConvLayer":
        block = AdaptiveInputConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=True,
        )
    else:
        raise ValueError(f"decoder project in block type {block_type} is not supported")
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelShuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=1
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for decoder project in")
    return block


def build_decoder_project_out_block(
    in_channels: int, out_channels: int, factor: int, upsample_block_type: str, norm: Optional[str], act: Optional[str]
):
    layers: list[nn.Module] = [
        build_norm(norm, in_channels),
        build_act(act),
    ]
    if factor == 1:
        layers.append(
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_bias=True,
                norm=None,
                act_func=None,
            )
        )
    elif factor == 2:
        layers.append(
            build_upsample_block(
                block_type=upsample_block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None
            )
        )
    else:
        raise ValueError(f"upsample factor {factor} is not supported for decoder project out")
    return OpSequential(layers)

class DCEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        num_stages = len(cfg.width_list)
        self.num_stages = num_stages
        assert len(cfg.depth_list) == num_stages
        assert len(cfg.width_list) == num_stages
        assert isinstance(cfg.block_type, str) or (
            isinstance(cfg.block_type, list) and len(cfg.block_type) == num_stages
        )
        assert isinstance(cfg.norm, str) or (isinstance(cfg.norm, list) and len(cfg.norm) == num_stages)
        
        self.patch_size = 2 ** (num_stages - 1)

        self.project_in = build_encoder_project_in_block(
            in_channels=cfg.in_channels,
            out_channels=cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1],
            factor=1 if cfg.depth_list[0] > 0 else 2,
            downsample_block_type=cfg.downsample_block_type,
        )

        self.stages: list[OpSequential] = []
        for stage_id, (width, depth) in enumerate(zip(cfg.width_list, cfg.depth_list)):
            block_type = cfg.block_type[stage_id] if isinstance(cfg.block_type, list) else cfg.block_type
            norm = cfg.norm[stage_id] if isinstance(cfg.norm, list) else cfg.norm
            stage = build_stage_main(
                width=width, depth=depth, block_type=block_type, norm=norm, act=cfg.act, input_width=width
            )

            if stage_id < num_stages - 1 and depth > 0:
                downsample_block = build_downsample_block(
                    block_type=cfg.downsample_block_type,
                    in_channels=width,
                    out_channels=cfg.width_list[stage_id + 1] if cfg.downsample_match_channel else width,
                    shortcut=cfg.downsample_shortcut,
                )
                stage.append(downsample_block)
            self.stages.append(OpSequential(stage))
        self.stages = nn.ModuleList(self.stages)

        self.project_out = build_encoder_project_out_block(
            block_type=cfg.out_block_type,
            in_channels=cfg.width_list[-1],
            out_channels=2 * cfg.latent_channels if cfg.double_latent else cfg.latent_channels,
            norm=cfg.out_norm,
            act=cfg.out_act,
            shortcut=cfg.out_shortcut,
        )

        self.load_model(cfg.pretrained_path)
        
    @classmethod
    def make(cls, z_dim, patch_size, encoder_checkpoint=None):
        cfg = dc_ae_fc(pretrained_path=encoder_checkpoint, f=patch_size, c=z_dim)
        return cls(cfg)
    
    def load_model(self, pretrained_path):
        if pretrained_path is not None and os.path.isfile(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)["state_dict"]
            self.load_state_dict(get_submodule_weights(state_dict, "encoder."))

    def get_trainable_modules(self) -> nn.Module:
        trainable_modules = nn.ModuleDict({})

        for name, module in self.named_children():
            if name in ["project_in", "stages", "project_out"]:
                trainable_modules[name] = module
            else:
                raise ValueError(f"module {name} is not supported")
        return trainable_modules

    def train(self, mode=True):
        self.training = mode
        for name, module in self.named_children():
            if name in ["project_in", "stages", "project_out"]:
                module.train(mode)
            else:
                raise ValueError(f"module {name} is not supported")
        return self

    def convert_sync_batchnorm(self):
        names = []
        for name, _ in self.named_children():
            names.append(name)
        for name in names:
            if name in ["project_in", "stages", "project_out"]:
                setattr(self, name, nn.SyncBatchNorm.convert_sync_batchnorm(getattr(self, name)))
            else:
                raise ValueError(f"module {name} is not supported")

    def forward(
        self, x: torch.Tensor, latent_channels: Optional[int | list[int]] = None
    ) -> torch.Tensor | list[torch.Tensor]:
        x = self.project_in(x)
        for stage in self.stages:
            if len(stage.op_list) == 0:
                continue
            for block in stage.op_list:
                x = block(x)
        if latent_channels is not None: # Structured Latent Space as in paper: DC-AE 1.5
            assert isinstance(self.project_out, OpSequential) and len(self.project_out.op_list) == 1
            if isinstance(latent_channels, int):
                x = self.project_out.op_list[0](x, out_channels=latent_channels)
            elif isinstance(latent_channels, list) and all(
                isinstance(latent_channels_, int) for latent_channels_ in latent_channels
            ):
                x = [
                    self.project_out.op_list[0](x, out_channels=latent_channels_)
                    for latent_channels_ in latent_channels
                ]
            else:
                raise ValueError(f"latent_channels {latent_channels} is not supported")
        else:
            x = self.project_out(x)
            
        if self.cfg.double_latent:
            x = DiagonalGaussianDistribution(x, deterministic=False)
        else:
            x = DiagonalGaussianDistribution(torch.cat([x, torch.zeros_like(x)], axis=1), deterministic=True)

        return x
