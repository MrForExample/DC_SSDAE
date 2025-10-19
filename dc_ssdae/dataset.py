import os
from typing import Any

import torch
import torchvision.transforms.v2 as transforms
from torchvision.datasets.folder import ImageFolder

from .mutils.main_utils import TaskState


class ImageNet(ImageFolder):
    def __init__(
        self,
        root: str,
        split: str = "train",
        **kwargs: Any,
    ) -> None:
        assert split in ["train", "validation"]
        self.root = root
        self.split = split

        super().__init__(self.split_folder, **kwargs)
        self.root = root

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


def make_transform(is_train_split, im_size, aug_scale):
    interpolation = transforms.InterpolationMode.LANCZOS if is_train_split else transforms.InterpolationMode.BILINEAR
    t_list = []

    if is_train_split and aug_scale:
        t_list.append(transforms.RandomResize(im_size, int(im_size * aug_scale), interpolation=interpolation))
    else:
        t_list.append(transforms.Resize(im_size, interpolation=interpolation))

    if is_train_split:
        t_list.append(transforms.RandomCrop(im_size))
        t_list.append(transforms.RandomHorizontalFlip())
    else:
        t_list.append(transforms.CenterCrop(im_size))

    t_list.extend(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    return transforms.Compose(t_list)


def make_dataset_and_loader(is_train, *, imagenet_root, im_size, batch_size, aug_scale=None, limit=None):
    transform = make_transform(is_train, im_size=im_size, aug_scale=aug_scale)
    dataset = ImageNet(os.path.join('..', '..', '..', imagenet_root), "train" if is_train else "validation", transform=transform)
    if limit is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(min(limit, len(dataset)))))

    num_proc = TaskState().accelerator.num_processes
    gpu_batch_size = batch_size // num_proc
    assert gpu_batch_size * num_proc == batch_size, f"Batch size {batch_size} not divisible by number of processes {num_proc}"

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=gpu_batch_size,
        shuffle=is_train,
        num_workers=10,
        persistent_workers=True,
        pin_memory=True,
    )

    return dataset, loader


def load_imagenet(ds_cfg):
    train_dataset, train_loader = make_dataset_and_loader(True, **ds_cfg)
    test_dataset, test_loader = make_dataset_and_loader(False, **ds_cfg)

    return (train_dataset, test_dataset), (train_loader, test_loader)
