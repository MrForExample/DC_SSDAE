import hydra
import lpips
import torch
from omegaconf import DictConfig

from dc_ssdae.tasks import AutoencodingTasks


# Patching lpips loss to avoid NaN issues during training by increasing eps from 1e-10 to 1e-8
def _normalize_tensor(in_feat, eps=1e-8):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True) + eps)
    return in_feat / (norm_factor + eps)


lpips.normalize_tensor = _normalize_tensor


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    task = AutoencodingTasks(cfg)
    task()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
