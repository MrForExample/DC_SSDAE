import re
from os import PathLike
from pathlib import Path
from typing import Mapping, Optional, Union

import torch
import torch.nn as nn
import yaml
from safetensors.torch import load_file as safe_load_file

from ...flow import FlowMatchingTrainer, FMEulerSampler
from ...eqm import EquilibriumMatchingTrainer, EqMEulerSampler
from ...mutils.torch_utils import freeze_model
from ...mutils.train_utils import init_weights
from ..blocks.diag_gauss import DiagonalGaussianDistribution
from ..model_utils import TrainStepResult
from ..vq_encoder import VQEncoder
from ..dc_encoder import DCEncoder
from .uvit import UViTDecoder


class DC_SSDAE(nn.Module):
    def __init__(
        self,
        encoder: Optional[Mapping | nn.Module] = None,
        encoder_checkpoint: Optional[str] = None,
        encoder_train: bool = False,
        decoder: Optional[Mapping] = None,
        trainer: Optional[Mapping] = None,
        sampler: Optional[Mapping] = None,
        trainer_type:str = "FM",
        encoder_type: str = "vq",
        checkpoint: Optional[str] = None,
    ):
        super().__init__()

        ### Equilibrium Matching (EqM) or Flow Matching (FM) ###
        if trainer_type == "EqM":
            self.trainer = EquilibriumMatchingTrainer(**(trainer or {}))
            self.sampler = EqMEulerSampler(**(sampler or {}))
            use_time_embedding = False
        elif trainer_type == "FM":
            self.trainer = FlowMatchingTrainer(**(trainer or {}))
            self.sampler = FMEulerSampler(**(sampler or {}))
            use_time_embedding = True
        else:
            raise ValueError(f"Invalid trainer_type: {trainer_type}. Must be 'EqM' or 'FM'.")
            
        ### Submodules ###
        self.encoder_train = encoder_train
        self.encoder, z_dim = self.make_encoder(encoder, encoder_type, encoder_checkpoint)
        self.decoder = UViTDecoder.make(decoder, z_dim=z_dim, use_time_embedding=use_time_embedding)

        ## Weights init ###
        self.init_weights(checkpoint=checkpoint)

    def make_encoder(self, encoder, encoder_type, encoder_checkpoint):
        z_dim = None
        if not isinstance(encoder, nn.Module):
            # Check if matches pattern f?c? with regex
            assert isinstance(encoder, str)
            enc_cfg_re = r"^f(\d+)c(\d+)$"
            enc_cfg_match = re.match(enc_cfg_re, encoder)
            if enc_cfg_match:
                patch_size = int(enc_cfg_match.group(1))
                z_dim = int(enc_cfg_match.group(2))
                if encoder_type == "vq":
                    enc_cls = VQEncoder
                elif encoder_type == "dc":
                    enc_cls = DCEncoder
                else:
                    raise ValueError(f"Invalid encoder type: {encoder_type}")
                encoder = enc_cls.make(z_dim=z_dim, patch_size=patch_size, encoder_checkpoint=encoder_checkpoint)
            else:
                raise ValueError(f"Invalid encoder config: {encoder}")

        if not self.encoder_train:
            freeze_model(encoder)

        return encoder, z_dim

    def init_weights(self, method="kaiming_normal", **kwargs):
        init_weights(self, method=method, **kwargs)

    def encode(self, x) -> DiagonalGaussianDistribution:
        return self.encoder(x)

    def decode(
        self,
        z: torch.Tensor,
        steps: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fn_kwargs = {"z": z}

        B, _, zH, zW = z.shape
        H, W = zH * self.encoder.patch_size, zW * self.encoder.patch_size

        ret = self.sampler.sample(
            self.decoder,
            self.trainer,
            shape=(B, self.decoder.out_dim, H, W),
            steps=steps,
            fn_kwargs=fn_kwargs,
            noise=noise,
        )

        return ret

    def forward(
        self,
        gt_x: torch.Tensor,
        steps: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, TrainStepResult]:
        # Encoder
        encoded = None
        if z is None:
            if self.encoder_train:
                encoded = self.encode(gt_x)
            else:
                with torch.no_grad():
                    encoded = self.encode(gt_x)
            z = encoded.sample() if self.training else encoded.mode()

        # Decoder
        if not self.training:
            return self.decode(z, steps=steps, noise=noise)
        else:
            # Use decoder to get a diffusion reconstruction loss
            diff_loss, (x_t, noise, noise_t, v_pred) = self.trainer.loss(self.decoder, x=gt_x, fn_kwargs={"z": z})

            # Compute auxiliary losses
            x0_pred = self.trainer.step(x_t, v_pred, noise_t)

            losses = {"diffusion": diff_loss}
            if encoded is not None and self.encoder_train:
                losses["kl"] = encoded.kl().mean().to(diff_loss.device)

            return TrainStepResult(
                x0_gt=gt_x,
                x0_pred=x0_pred,
                xt=x_t,
                t=noise_t,
                z=z,
                noise=noise,
                losses=losses,
            )

    def get_last_layer_weight(self):
        return self.decoder.conv_out.weight

    ### Loading / Checkpointing ###

    def load(
        self,
        weights: Union[str, Path, Mapping],
        strict: bool = True,
        freeze=False,
        eval=None,
    ):
        if not isinstance(weights, Mapping):
            weights = safe_load_file(weights)
        self.load_state_dict(weights, strict=strict)

        if eval or (eval is None and freeze):
            self.eval()
        if freeze:
            self.requires_grad_(False)
        return self

    @classmethod
    def build(cls, config, checkpoint=None, freeze=True, eval=True):
        """Build the model from a config name."""
        if isinstance(config, (str, PathLike)):
            with open(config, "r") as yaml_file:
                model_args = yaml.safe_load(yaml_file)["ssdd"]
        elif isinstance(config, Mapping):
            model_args = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}. Expected model size, path, or a mapping.")

        model = cls(**model_args)

        if checkpoint:
            model.load(checkpoint)
        if eval:
            model.eval()
        if freeze:
            model.requires_grad_(False)
        return model
