import hydra
import lpips
import torch
import sys
from omegaconf import DictConfig

from dc_ssdae.tasks import AutoencodingTasks

# Patching lpips loss to avoid NaN issues during training by increasing eps from 1e-10 to 1e-8
def _normalize_tensor(in_feat, eps=1e-8):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True) + eps)
    return in_feat / (norm_factor + eps)

lpips.normalize_tensor = _normalize_tensor

def get_hydra_decorator():
    """Parse custom args for config_path and config_name."""
    config_path = "./config"
    config_name = "main"
    
    # Parse custom arguments
    args_to_remove = []
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith("--config_path="):
            config_path = arg.split("=", 1)[1]
            args_to_remove.append(i)
        elif arg.startswith("--config_name="):
            config_name = arg.split("=", 1)[1]
            args_to_remove.append(i)
    
    # Remove processed arguments
    for idx in reversed(args_to_remove):
        sys.argv.pop(idx)
    
    return hydra.main(version_base=None, config_path=config_path, config_name=config_name)

@get_hydra_decorator()
def main(cfg: DictConfig):
    task = AutoencodingTasks(cfg)
    task()

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
