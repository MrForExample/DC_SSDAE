

import math
from typing import Any, Optional

import lpips as lpips_lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
import transformers

from ...mutils.torch_utils import Frozen, ensure_1d_tokens, freeze_model, unwrap
from ...mutils.train_utils import init_weights
from ..blocks.discriminator import NLayerDiscriminator

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class DinoEncoder(nn.Module):
    def __init__(self, cache_dir=None):
        super().__init__()

        self.out_dim = 768
        self.base_patch_size = 14
        self.base_res = 224

        self.model = transformers.AutoModel.from_pretrained("facebook/dinov2-base", cache_dir=cache_dir)
        freeze_model(self)

    def rescale_and_process_image(self, x, target_n_tokens):
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)((x + 1) / 2)
        _, _, h, w = x.shape
        r = math.sqrt(target_n_tokens * self.base_patch_size**2 / (h * w))
        H, W = round(h * r), round(w * r)
        x = torch.nn.functional.interpolate(x, (H, W), mode="bicubic")
        return x

    def forward(self, x, target_n_tokens=None):
        if self.training:
            self.eval()
        x = self.rescale_and_process_image(x, target_n_tokens)
        z = self.model(x).last_hidden_state[:, 1:]  # Remove CLS token
        return z


class REPALoss(nn.Module):
    # https://arxiv.org/pdf/2410.06940#page=23

    def __init__(self, unet, n_layers=2, i_extract=4, cache_dir=None, accelerator=None):
        super().__init__()
        if hasattr(unet, "mid_dim"):
            features_dim = unet.mid_dim
        else:
            features_dim = unet.mid_block.attention.inner_dim

        self.features_extractor = Frozen(
            DinoEncoder(cache_dir=cache_dir),
            accelerator,
            allow_grad=False,
        )

        # Create features MLP
        self.repa_mlp = nn.Sequential()
        self.repa_loss = nn.CosineSimilarity(dim=2, eps=1e-5)
        for i in range(n_layers):
            in_dim = features_dim
            out_dim = in_dim
            if i == n_layers - 1:
                out_dim = self.features_extractor.module.out_dim
            self.repa_mlp.append(nn.Linear(in_dim, out_dim))
            if i != n_layers - 1:
                self.repa_mlp.append(nn.SiLU())

        # Register hook to get miiddle features
        if hasattr(unet.mid_block, "transformer_blocks"):
            tformer_blocks = unet.mid_block.transformer_blocks
        else:
            tformer_blocks = unet.mid_block.attention.transformer_blocks

        i_extract = min(i_extract, len(tformer_blocks) // 2)
        tformer_blocks[i_extract].register_forward_hook(self._hook_repa)

    def _hook_repa(self, module, input, output):
        if self.training:
            self._repa_layer_output = output

    def forward(self, x_gt):
        # Extract and project repa extracted features
        repa_val = self._repa_layer_output
        repa_val = ensure_1d_tokens(repa_val)
        repa_val = self.repa_mlp(repa_val)

        # Compute loss with a reference model
        with torch.no_grad():
            repa_ref = self.features_extractor(x_gt, target_n_tokens=repa_val.shape[1])

        assert repa_val.shape == repa_ref.shape, f"Invalid shape {repa_val.shape} != {repa_ref.shape}"

        self._repa_layer_output = None
        with torch.autocast("cuda", enabled=False):
            return 1 - self.repa_loss(repa_val.to(torch.float32), repa_ref.to(torch.float32)).mean()


class SSDDLosses(nn.Module):
    def __init__(
        self,
        ae: nn.Module,
        repa: Optional[dict] = None,
        lpips: bool = True,
        checkpoint=None,
        accelerator: Optional[Any] = None,
    ):
        super().__init__()
        self.accelerator = accelerator

        # REPA loss
        self.repa_loss = None
        if repa is not None:
            ae = unwrap(ae, unw_ema=True)
            self.repa_loss = REPALoss(ae.decoder, accelerator=accelerator, **repa)

        # LPIPS
        self.lpips_loss = None
        if lpips:
            self.lpips_loss = Frozen(lpips_lib.LPIPS(net="vgg"), accelerator=accelerator)

        init_weights(self, method="kaiming_normal", checkpoint=checkpoint, ckpt_args=dict(default_load="aux_losses"))

    def forward(self, x_BCWH, x0_pred, target_x=None):
        losses = {}
        if target_x is None:
            target_x = x_BCWH

        # REPA loss
        if self.repa_loss is not None:
            losses["repa"] = self.repa_loss(x_BCWH)

        # LPIPS loss
        if self.lpips_loss is not None:
            losses["lpips"] = self.lpips_loss(target_x, x0_pred).mean()

        return losses

    def __deepcopy__(self, memo):
        return None


class GanLoss(nn.Module):
    def __init__(self, start_iter=0, discriminator=None):
        super().__init__()

        self.gan_model = NLayerDiscriminator(
            input_nc=3,
            flatten=True,
            **(discriminator or {}),
        )
        self.start_iter = start_iter

    def forward(self, x_gt, x_pred, xt, t, n_train_steps, existing_losses=None, step=None):
        with torch.autocast("cuda", enabled=False):
            losses = {}
            if step == "disc_loss":
                assert existing_losses is not None, "Existing losses must be provided for GAN discriminator loss calculation"
                # Add adversarial loss
                if n_train_steps >= self.start_iter:
                    logits_fake = self.gan_model(x_pred, xt, t)

                    losses["gan_disc"] = F.binary_cross_entropy(logits_fake, torch.ones_like(logits_fake))
            elif step == "train":
                # Train GAN model
                logits_real = self.gan_model(x_gt.detach(), xt.detach(), t)
                logits_fake = self.gan_model(x_pred.detach(), xt.detach(), t)

                loss_real = F.binary_cross_entropy(logits_real, torch.ones_like(logits_real))
                loss_fake = F.binary_cross_entropy(logits_fake, torch.zeros_like(logits_fake))

                losses["gan_train"] = (loss_real + loss_fake) / 2

                if n_train_steps < self.start_iter:
                    losses["gan_train"] = 0 * losses["gan_train"]
            else:
                raise ValueError(f"Unknown GAN step: {step}")
        return losses
