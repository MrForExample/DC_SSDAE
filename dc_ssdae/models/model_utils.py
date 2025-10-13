from dataclasses import dataclass
from typing import Dict

import torch
from einops import rearrange

####################################################################
# Data classes
####################################################################


@dataclass
class TrainStepResult:
    x0_gt: torch.Tensor = None
    x0_pred: torch.Tensor = None
    xt: torch.Tensor = None
    t: torch.Tensor = None
    z: torch.Tensor = None
    noise: torch.Tensor = None
    losses: Dict[str, torch.Tensor] = None


def get_causal_mask(seq_len, device=None, as_float=True):
    mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).tril(diagonal=0)
    if as_float:
        mask = to_float_mask(mask)
    return mask


def get_full_attention_mask(seq_len, device=None, as_float=True):
    mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)
    if as_float:
        mask = to_float_mask(mask)
    return mask


def to_float_mask(bool_mask):
    # Ensure & expand dimensions
    if bool_mask.dim() == 2:
        bool_mask = bool_mask.unsqueeze(0).unsqueeze(0)
    elif bool_mask.dim() == 3:
        bool_mask = bool_mask.unsqueeze(1)
    else:
        assert bool_mask.ndim == 4

    # Convert to floats
    float_mask = torch.zeros(bool_mask.shape, device=bool_mask.device)
    float_mask.masked_fill_(~bool_mask.to(torch.bool), torch.finfo(float_mask.dtype).min)
    return float_mask


def hide_matriochka_entries(x, n_matriochka, masked_with=0):
    # x dimensions: (B, N, ...)
    mask = torch.arange(x.size(1), device=x.device) < n_matriochka[..., None]
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    return x * mask + masked_with * (~mask)


def patchify_linear(x_BCWH, patch_size):
    B, C, W, H = x_BCWH.shape
    assert W % patch_size == 0 and H % patch_size == 0
    assert C in [1, 3]
    assert W == H

    x_BWHC = x_BCWH.permute(0, 2, 3, 1)
    x_BWHC = x_BWHC.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    x_BNC = x_BWHC.contiguous().view(B, -1, patch_size * patch_size * C)
    return x_BNC


def patchify_2d(x_BCWH, patch_size):
    return rearrange(x_BCWH, "B C (H P1) (W P2) -> B (P1 P2 C) H W", P1=patch_size, P2=patch_size)


def unpatchify_2d(x_BCWH, patch_size):
    return rearrange(x_BCWH, "B (P1 P2 C) H W -> B C (H P1) (W P2)", P1=patch_size, P2=patch_size)


def random_orthonormal_matrix(N, M=None, device="cpu", dtype=torch.float32):
    if isinstance(N, torch.Tensor):
        A, (N, M) = N.T, N.shape
    else:
        A = torch.randn(M, N, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    return Q.T
