

import importlib


class MissingModule:
    def __init__(self, name, import_error=None):
        self._mod_name = name
        self._mod_import_error = import_error

    def _raise_error(self):
        raise ImportError(f"Module '{self._mod_name}' is not installed. Please install it to use this feature. Original error: {self._mod_import_error}")

    def __getattr__(self, item):
        return self._raise_error()

    def __call__(self, *args, **kwargs):
        return self._raise_error()


class UndefinedObject(MissingModule):
    def __init__(self, missing_module):
        self._missing_module = missing_module

    def _raise_error(self):
        self._missing_module._raise_error()


def optional_import(module: str, name: str = None, package: str = None):
    if package is None:
        package = module
    try:
        module = importlib.import_module(module)
        return module if name is None else getattr(module, name)
    except ImportError as e:
        return MissingModule(package, import_error=e)


def is_imported(module):
    return not isinstance(module, MissingModule)


def optional_getattr(module, name):
    if isinstance(module, MissingModule):
        return UndefinedObject(module)
    return getattr(module, name)
