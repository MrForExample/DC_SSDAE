

import logging
from copy import deepcopy

import torch
from torch.nn import Module

from ...mutils.torch_utils import Ref, freeze_model

logger = logging.getLogger(__name__)


class EMA(Module):
    def __init__(self, model: Module, decay: float = 0.999, start_iter=0):
        super().__init__()

        # Doesn't include persistent buffers
        for name, buffer in model.named_buffers():
            if not hasattr(buffer, "_ema"):
                print(f"[WARNING] Model buffers behavior should be defined using the '_ema' parameter. No _ema key for the buffer {name}. Will default to 'ingore'.")
            elif buffer._ema in [False, "ignore"]:
                pass
            else:
                raise RuntimeError(f"Buffer {name}: unexpected value for _ema key, got {buffer._ema}.")

        self.model = Ref(model)
        self.ema_model = deepcopy(model)
        freeze_model(self.ema_model)

        self.decay = decay
        self.start_iter = start_iter

        # Put this in a buffer so that it gets included in the state dict
        self.register_buffer("num_updates", torch.tensor(0))

    def __repr__(self) -> str:
        return f"EMA(ema_model={self.ema_model.__class__.__name__}, decay={self.decay}, start_iter={self.start_iter})"

    @torch.no_grad()
    def update(self) -> None:
        num_updates = self.num_updates.item()  # pylint: disable=no-member
        if self.num_updates < self.start_iter:  # pylint: disable=no-member
            decay = 0.0
        else:
            n = num_updates - self.start_iter
            decay = min(self.decay, (1 + n) / (10 + n))

        model_params = dict(self.model.named_parameters())
        ema_params = dict(self.ema_model.named_parameters())

        for name, ema_p in ema_params.items():
            model_p = model_params[name]
            if model_p.requires_grad:
                ema_p.sub_((1 - decay) * (ema_p - model_p))
        self.num_updates += 1  # pylint: disable=no-member

    @staticmethod
    def uses_ema(m: Module):
        for module in m.modules():
            if isinstance(module, EMA):
                return True
        return False

    @staticmethod
    def update_ema_modules(model: Module):
        for module in model.modules():
            if isinstance(module, EMA):
                module.update()
        return model

    ### Go through ###
    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)


class EMAWrapper(Module):
    def __init__(self, model: Module, decay: float = 0.999, start_iter=0, eval_ema=True):
        super().__init__()
        self.model = model
        self.eval_ema = eval_ema
        self.ema = EMA(model, decay=decay, start_iter=start_iter)

    def update_ema(self) -> None:
        self.ema.update()

    def forward(self, *args, **kwargs):
        if self.training or not self.eval_ema:
            return self.model(*args, **kwargs)
        else:
            return self.ema(*args, **kwargs)
