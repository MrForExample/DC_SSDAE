

import dataclasses
import functools
import logging
import os
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping

from .optional_import import UndefinedObject, is_imported, optional_import

accelerate = optional_import("accelerate")
AcceleratorState = optional_import("accelerate.state", "AcceleratorState")
np = optional_import("numpy")
requests = optional_import("requests")
torch = optional_import("torch")
nn = optional_import("torch", "nn")


###############################################################
# General utilities
###############################################################


class TaskState:
    CKPT_IGNORE = ["accelerator", "cfg", "models", "optimizer"]
    single_instance = None

    def __new__(cls, **kwargs):
        if cls.single_instance is None:
            cls.single_instance = super().__new__(cls)
            cls.single_instance._init_vars()
        cls.single_instance.register_vars(**kwargs)
        return cls.single_instance

    def print(self, *args, **kwargs):
        if self.accelerator is not None:
            self.accelerator.print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def _init_vars(self):
        # Ensure some attributes exist
        self.cfg = None  # pylint: disable=W0201
        self.accelerator = None  # pylint: disable=W0201
        self.logger = None  # pylint: disable=W0201
        self.models = {}  # pylint: disable=W0201
        self.registered_models = []

    def register_vars(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __deepcopy__(self, memo):
        return self  # Can't be copied, only reconstructed

    def _make_state_dict(self, obj):
        if isinstance(obj, Mapping):
            return {k: self._make_state_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_state_dict(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._make_state_dict(v) for v in obj)
        if hasattr(obj, "state_dict"):
            return obj.state_dict()
        if isinstance(obj, (int, float, str, type(None))):
            return obj
        raise ValueError(f"Cannot make state dict from object of type {type(obj)}")

    def _load_state_dict(self, obj, state_dict):
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                if k in state_dict:
                    obj[k] = self._load_state_dict(v, state_dict[k])
        elif isinstance(obj, list):
            assert len(obj) == len(state_dict)
            for i, v in enumerate(obj):
                obj[i] = self._load_state_dict(v, state_dict[i])
        elif isinstance(obj, tuple):
            assert len(obj) == len(state_dict)
            return type(obj)([self._load_state_dict(v, state_dict[i]) for i, v in enumerate(obj)])
        elif hasattr(obj, "load_state_dict"):
            obj.load_state_dict(state_dict)
        else:
            assert type(obj) is type(state_dict), f"Cannot load state dict into object of different type {type(obj)} != {type(state_dict)}"
        return obj

    def state_dict(self):
        # Won't save any field in CKPT_IGNORE or starting with '_'
        save_fields = {k: v for k, v in self.__dict__.items() if k not in self.CKPT_IGNORE and not k.startswith("_")}
        sd = self._make_state_dict(save_fields)
        return sd

    def load_state_dict(self, state_dict):
        self._load_state_dict(self.__dict__, state_dict)


if not is_imported(accelerate):
    UpAccelerator = UndefinedObject(accelerate)
else:

    class UpAccelerator(accelerate.Accelerator):
        """Custom accelerator class that adds some utilities for logging and data preparation."""

        main_log = logging.getLogger("main")

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.main_log.setLevel(logging.DEBUG)

        ##### Logging #####

        def print(self, *args, sep=" ", level="info", show=True):  # pylint: disable=arguments-differ
            if self.is_main_process and show:
                msg = sep.join([str(a) for a in args])
                if level == "info":
                    self.main_log.info(msg)
                elif level == "debug":
                    self.main_log.debug(msg)
                elif level == "warning":
                    self.main_log.warning(msg)
                elif level == "error":
                    self.main_log.error(msg)
                elif level == "critical":  # Error + raises an exception
                    self.main_log.critical(msg)
                    raise RuntimeError(msg)
                else:
                    raise ValueError(f"{level=} unknown")

        def info(self, *args, **kwargs):
            self.print(*args, level="info", **kwargs)

        def debug(self, *args, **kwargs):
            self.print(*args, level="debug", **kwargs)

        def warning(self, *args, **kwargs):
            self.print(*args, level="warning", **kwargs)

        def error(self, *args, **kwargs):
            self.print(*args, level="error", **kwargs)

        def critical(self, *args, **kwargs):
            self.print(*args, level="critical", **kwargs)

        def print_nolog(self, *args, **kwargs):
            if self.is_main_process:
                print(*args, **kwargs)

        ##### Utilities #####

        def prepare_test_data(self, dataloader, even_batches=False):
            if dataloader is None:
                return dataloader
            return accelerate.data_loader.prepare_data_loader(
                dataloader,
                device=self.device,
                put_on_device=True,
                even_batches=even_batches,
            )

        @contextmanager
        def sync_ctx(self):
            self.wait_for_everyone()
            try:
                yield self.is_main_process
            finally:
                self.wait_for_everyone()

        def __deepcopy__(self, memo):
            return self  # Can't be copied, should be unique


###############################################################
# Logging & formatting
###############################################################


def format_memory(mem):
    if mem >= 2**40:
        return f"{mem / (2**40):.1f}T"
    if mem >= 2**30:
        return f"{mem / (2**30):.1f}G"
    if mem >= 2**20:
        return f"{mem / (2**20):.1f}M"
    if mem >= 2**10:
        return f"{mem / (2**10):.1f}K"
    return f"{mem:.1f}b"


def format_time(t):
    hours, t = divmod(t, 3600)
    minutes, seconds = divmod(t, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


###############################################################
# Types & math utils
###############################################################


def smaller_p2_greater_than(n):
    k = 1
    while k < n:
        k *= 2
    return k


###############################################################
# Files utils
###############################################################


def download_if_not_exists(path: str, url: str):
    """
    Downloads the file from the given URL if it does not already exist at the specified path.

    Args:
        path (str): The local file path to check/download.
        url (str): The URL to download the file from.
    """
    if os.path.exists(path):
        print(f"File already exists: {path}")
        return

    print(f"Downloading {url} to {path}...")

    response = requests.get(url, stream=True)  # pylint: disable=missing-timeout
    response.raise_for_status()  # Raise an error for bad responses

    with open(path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"Download complete: {path}")


def ensure_path(path, parent=False):
    path = Path(path)
    if parent:
        path = path.parent
    path.mkdir(parents=True, exist_ok=True)


####################################################################
# Dataclasses utils
####################################################################


def soft_dataclass_dict(data):
    field_names = [f.name for f in dataclasses.fields(data.__class__)]
    return {field: getattr(data, field) for field in field_names}


###############################################################
# Iterators, container, functions utils
###############################################################


def weak_method_lru(maxsize=128, typed=False):
    'LRU Cache decorator that keeps a weak reference to "self"'

    def wrapper(func):
        @functools.lru_cache(maxsize, typed)
        def _func(_self, *args, **kwargs):
            return func(_self(), *args, **kwargs)

        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            return _func(weakref.ref(self), *args, **kwargs)

        return inner

    return wrapper


def split_dict(d, keys):
    d1 = {k: v for k, v in d.items() if k in keys}
    d2 = {k: v for k, v in d.items() if k not in keys}
    return d1, d2
