from enum import Enum
from pathlib import Path
from typing import Optional

import lpips
import torch
import torcheval.metrics as t_metrics
from dreamsim import dreamsim
from skimage.metrics import structural_similarity
from torch_fidelity.feature_extractor_inceptionv3 import (
    URL_INCEPTION_V3,
    FeatureExtractorInceptionV3,
)
from torcheval.metrics.toolkit import sync_and_compute
from torchmetrics import Metric as TMMetric

from ..mutils.main_utils import TaskState, download_if_not_exists, smaller_p2_greater_than

####################################################################
# Utilities
####################################################################


# Global directory used by metrics to store models & avoid copy when clonning
METRIC_MODELS = {}


def compute_parallel(metric):
    if isinstance(metric, WrappedMetric):
        return compute_parallel(metric.metric)
    if TaskState().accelerator.num_processes > 1:
        if isinstance(metric, t_metrics.Metric):
            return sync_and_compute(metric)
        elif isinstance(metric, TMMetric):
            return metric.compute()
        else:
            raise ValueError(f"Cannot sync metric of type {type(metric)}")
    else:
        return metric.compute()


class Orderings(Enum):
    MAX = "max"
    MIN = "min"
    NONE = "none"


class FIDExtractorModel(FeatureExtractorInceptionV3):
    def __init__(self, cfg):
        inception_path = Path(cfg.cache_dir) / Path(URL_INCEPTION_V3).name
        with TaskState().accelerator.main_process_first():
            download_if_not_exists(str(inception_path), URL_INCEPTION_V3)
        super().__init__("FIDModel", ["2048"], str(inception_path))
        self.eval()

    def forward(self, x):
        x = x.clip(0, 1)
        x = torch.round(x * 255).to(torch.uint8)
        (x,) = super().forward(x)
        return x


class MetricResult:
    PREFIXS = {
        Orderings.MAX: "max_",
        Orderings.MIN: "min_",
        Orderings.NONE: "",
    }

    def __init__(self, value=None):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.value = value

    def format_value(self, value, name=None, ordering=None, sep=" | "):
        ordering = self.PREFIXS[ordering] if ordering is not None else ""
        name = name + "=" if name is not None else ""
        return f"{ordering}{name}{value:.4g}"

    def display(self, name=None, ordering=None, sep=" | "):
        return self.format_value(self.value, name=name, ordering=ordering, sep=sep)

    def is_strictly_better_than(self, other, ordering: Orderings):
        if ordering == Orderings.MAX:
            return self.value > other.value
        elif ordering == Orderings.MIN:
            return self.value < other.value
        elif ordering == Orderings.NONE:
            return False
        else:
            raise ValueError(f"Unknown ordering: {ordering}")

    def __repr__(self):
        return self.display()

    def to_json(self):
        return self.value


class WrappedMetric:
    """Wraps an existing metric to use it with the MetricsManager.
    Gives the following:
    - `update` can receive any arguments, and will pass the correct ones to the metric.
    - will define `ordering` attribute
    """

    def __init__(self, metric, metric_args, ordering=Orderings.NONE):
        self.metric = metric
        self.ordering = ordering
        self.metric_args = metric_args

    def update(self, **kwargs):
        for a in self.metric_args:
            if a not in kwargs:
                raise ValueError(f"Missing argument {a} for metric {self}")
            elif kwargs[a] is None:
                raise ValueError(f"Argument {a} for metric {self} cannot be None")
        args = [kwargs[a] for a in self.metric_args]
        return self.metric.update(*args)

    def reset(self):
        return self.metric.reset()

    def to(self, device):
        return self.metric.to(device)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.metric}, args={self.metric_args}, ordering={self.ordering})"


####################################################################
# General / Image Metrics
####################################################################


class AnyVectMPError(t_metrics.Metric):
    ordering = Orderings.MIN

    def __init__(self, device=None, scale=1.0, pow=2, reduction="mean"):
        # Set pow to 2 for MSE (Mean Squared Error), to 1 for MAE (Mean Absolute Error)
        super().__init__(device=device)

        self.scale = scale
        self.loss_sum = None
        self.num_samples = None
        self.pow = pow
        self.reduction = reduction
        self._add_state("loss_sum", torch.tensor(0.0, device=self.device))
        self._add_state("num_samples", torch.tensor(0, device=self.device))

    def update(self, x_gt, x_pred, **ignored):
        if self.reduction == "sum":
            self.loss_sum += ((x_gt - x_pred).abs() ** self.pow).sum()
        elif self.reduction == "mean":
            self.loss_sum += ((x_gt - x_pred).abs() ** self.pow).mean() * len(x_gt)
        else:
            raise ValueError(f"{self.reduction}")
        self.num_samples += len(x_gt)

    def compute(self):
        return self.loss_sum * self.scale / self.num_samples

    def merge_state(self, metrics):
        for m in metrics:
            self.loss_sum += m.loss_sum
            self.num_samples += m.num_samples
        return self


class LPIPSMetric(t_metrics.Metric):
    ordering = Orderings.MIN

    def __init__(self, data_range, net="alex", device=None):
        super().__init__(device=device)
        if "lpips" not in METRIC_MODELS:
            METRIC_MODELS["lpips"] = lpips.LPIPS(net=net).to(device)
        self.data_range = data_range

        self._add_state("loss_sum", torch.tensor(0.0, device=self.device))
        self._add_state("num_samples", torch.tensor(0, device=self.device))

    def update(self, x_gt, x_pred, **ignored):
        x_gt = self._img_to_in(x_gt)
        x_pred = self._img_to_in(x_pred)
        self.loss_sum += METRIC_MODELS["lpips"](x_gt, x_pred).sum()
        self.num_samples += len(x_gt)

    def _img_to_in(self, x):
        assert 0 <= x.min() and x.max() <= self.data_range
        # Pad to power of 2 channels
        B, C, H, W = x.shape
        assert C == 3
        W2, H2 = smaller_p2_greater_than(W), smaller_p2_greater_than(H)
        x = torch.nn.functional.pad(x, (0, W2 - W, 0, H2 - H), mode="constant", value=0)

        x = x / self.data_range * 2.0 - 1.0
        x = torch.clip(x, -1.0, 1.0)
        return x

    def compute(self):
        return self.loss_sum / self.num_samples

    def merge_state(self, metrics):
        for m in metrics:
            self.loss_sum += m.loss_sum.to(self.device)
            self.num_samples += m.num_samples.to(self.device)
        return self


class DreamSim(t_metrics.Metric):
    ordering = Orderings.MIN

    def __init__(self, cache_dir, device=None):
        super().__init__(device=device)
        if "dreamsim" not in METRIC_MODELS:
            loss, _ = dreamsim(pretrained=True, cache_dir=cache_dir)
            METRIC_MODELS["dreamsim"] = loss.to(device)

        self._add_state("loss_sum", torch.tensor(0.0, device=self.device))
        self._add_state("num_samples", torch.tensor(0, device=self.device))

    def update(self, x_gt, x_pred, **ignored):
        x_gt = self._img_to_in(x_gt)
        x_pred = self._img_to_in(x_pred)
        self.loss_sum += METRIC_MODELS["dreamsim"](x_gt, x_pred).sum()
        self.num_samples += len(x_gt)

    def _img_to_in(self, x):
        assert 0 <= x.min() and x.max() <= 1
        # Pad to power of 2 channels
        B, C, H, W = x.shape
        assert C == 3
        W2, H2 = smaller_p2_greater_than(W), smaller_p2_greater_than(H)
        x = torch.nn.functional.pad(x, (0, W2 - W, 0, H2 - H), mode="constant", value=0)
        return x

    def compute(self):
        return self.loss_sum / self.num_samples

    def merge_state(self, metrics):
        for m in metrics:
            self.loss_sum += m.loss_sum.to(self.device)
            self.num_samples += m.num_samples.to(self.device)
        return self


class CorrectedStructuralSimilarity(t_metrics.StructuralSimilarity):
    """Fix SSIM implementation"""

    ordering = Orderings.MAX

    def __init__(self, data_range, device=None) -> None:
        super().__init__(device=device)

        self.data_range = data_range

    def update(self, x_gt: torch.Tensor, x_pred: torch.Tensor, **ingored):
        """
        Update the metric state with new input.
        Ensure that the two sets of images have the same value range (ex. [-1, 1], [0, 1]).

        Args:
            x_gt (Tensor): A batch of the first set of images of shape [N, C, H, W].
            x_pred (Tensor): A batch of the second set of images of shape [N, C, H, W].

        """
        assert x_gt.max() - x_gt.min() <= self.data_range
        assert x_pred.max() - x_pred.min() <= self.data_range

        if x_gt.shape != x_pred.shape:
            raise RuntimeError("The two sets of images must have the same shape.")
        # convert to fp32, mostly for bf16 types
        x_gt = x_gt.to(dtype=torch.float32)
        x_pred = x_pred.to(dtype=torch.float32)

        batch_size = x_gt.shape[0]

        for idx in range(batch_size):
            mssim = structural_similarity(
                x_gt[idx].detach().cpu().numpy(),
                x_pred[idx].detach().cpu().numpy(),
                channel_axis=0,
                data_range=self.data_range,
            )
            self.mssim_sum += mssim

        self.num_images += batch_size

        return self


class FIDMetric(t_metrics.FrechetInceptionDistance):
    """
    Overloaded FID metric using the correct model for feature extraction, and float64 states to ensure precise FID computation.
    """

    ordering = Orderings.MIN

    def __init__(self, cfg, device: Optional[torch.device] = None) -> None:
        super(t_metrics.FrechetInceptionDistance, self).__init__(device=device)  # Skip parent constructor

        # Set the model and put it in evaluation mode
        model = FIDExtractorModel(cfg)
        self.model = model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        # Initialize state variables used to compute FID
        # Changed: use float64 for precision
        feature_dim = 2048
        self._add_state("real_sum", torch.zeros(feature_dim, device=device, dtype=torch.float64))
        self._add_state("real_cov_sum", torch.zeros((feature_dim, feature_dim), device=device, dtype=torch.float64))
        self._add_state("fake_sum", torch.zeros(feature_dim, device=device, dtype=torch.float64))
        self._add_state("fake_cov_sum", torch.zeros((feature_dim, feature_dim), device=device, dtype=torch.float64))
        self._add_state("num_real_images", torch.tensor(0, device=device).int())
        self._add_state("num_fake_images", torch.tensor(0, device=device).int())

    def update(self, x_gt=None, x_pred=None, **ignored):
        if x_gt is not None:
            super().update(x_gt, is_real=True)
        if x_pred is not None:
            super().update(x_pred, is_real=False)


####################################################################
# Managing sets of metrics
####################################################################


class MetricsManager:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.metrics = {}

        self.metrics["MSE"] = AnyVectMPError(scale=1000, pow=2.0, device=device)
        self.metrics["MAE"] = AnyVectMPError(scale=1, pow=1.0, device=device)
        self.metrics["LPIPS"] = LPIPSMetric(data_range=1.0, device=device)
        self.metrics["PSNR"] = WrappedMetric(t_metrics.PeakSignalNoiseRatio(data_range=1.0, device=device), ["x_gt", "x_pred"], Orderings.MAX)
        self.metrics["SSIM"] = CorrectedStructuralSimilarity(data_range=1.0, device=device)
        self.metrics["dreamsim"] = DreamSim(cache_dir=cfg.cache_dir, device=device)
        self.metrics["FID"] = FIDMetric(cfg, device=device)

        self.bests = {}
        self.last_m_vals = {}

    def reset(self):
        for m in self.metrics.values():
            m.reset()

    @torch.no_grad()
    def update(self, **metric_args):
        for m in self.metrics.values():
            m.update(**metric_args)

    def to(self, device):
        for m in self.metrics.values():
            m.to(device)

    @torch.no_grad()
    def compute(self, store_best=False):
        m_vals = {}
        for m_name, m in self.metrics.items():
            v = compute_parallel(m)
            if not isinstance(v, MetricResult):
                v = MetricResult(v)
            m_vals[m_name] = v.value
            self.last_m_vals[m_name] = v
            if store_best:
                if m_name not in self.bests:
                    self.bests[m_name] = v
                elif not self.bests[m_name].is_strictly_better_than(v, m.ordering):
                    self.bests[m_name] = v
        return m_vals

    def metrics_as_str(self, m_vals=None):
        if not m_vals:
            m_vals = self.last_m_vals
        return "[[" + " | ".join(v.display(name=name) for name, v in m_vals.items()) + "]]"

    def bests_as_str(self):
        return "[[" + " | ".join(v.display(name=name, ordering=self.metrics[name].ordering) for name, v in self.bests.items()) + "]]"

    def state_dict(self):
        return {
            "bests": self.bests,
            "last_m_vals": self.last_m_vals,
        }

    def load_state_dict(self, state_dict):
        self.bests = state_dict["bests"]
        self.last_m_vals = state_dict["last_m_vals"]
