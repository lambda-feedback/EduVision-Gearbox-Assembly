import importlib


class LazyModule:
    """
    This class is response for lazy loading of heavy imports
    """
    def __init__(self, name):
        self._name = name
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = importlib.import_module(self._name)
        return self._module

    def __getattr__(self, item):
        return getattr(self._load(), item)

