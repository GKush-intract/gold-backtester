from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

from ..strategy import Strategy

_PACKAGE = __name__


def get_strategy_registry() -> dict[str, type]:
    """Discover all Strategy subclasses in this package. name -> class.
    Adding a strategy file here requires zero changes elsewhere."""
    registry: dict[str, type] = {}
    pkg_dir = Path(__file__).parent
    for mod in pkgutil.iter_modules([str(pkg_dir)]):
        if mod.name.startswith("_"):
            continue
        module = importlib.import_module(f"{_PACKAGE}.{mod.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Strategy) and obj is not Strategy and obj.__module__ == module.__name__:
                registry[obj.name] = obj
    return registry
