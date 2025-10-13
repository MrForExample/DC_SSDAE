import collections
from inspect import signature
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

def build_kwargs_from_config(config: dict, target_func: Callable) -> dict[str, Any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs

def get_same_padding(kernel_size: int | tuple[int, ...]) -> int | tuple[int, ...]:
    if isinstance(kernel_size, (tuple, list)):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2

def resize(
    x: torch.Tensor,
    size: Optional[Any] = None,
    scale_factor: Optional[list[float]] = None,
    mode: str = "bicubic",
    align_corners: Optional[bool] = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")
    
def get_submodule_weights(weights: collections.OrderedDict, prefix: str):
    submodule_weights = collections.OrderedDict()
    len_prefix = len(prefix)
    for key, weight in weights.items():
        if key.startswith(prefix):
            submodule_weights[key[len_prefix:]] = weight
    return submodule_weights
    

def list_sum(x: list) -> Any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])

def val2list(x: list | tuple | Any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]

def val2tuple(x: list | tuple | Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)