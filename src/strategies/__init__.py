from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

from ..strategy import Strategy

_PACKAGE = __name__


def get_strategy_registry() -> dict[str, type]:
    """Discover all Strategy subclasses in this package (and generated/). name -> class.
    Adding a strategy file here requires zero changes elsewhere."""
    registry: dict[str, type] = {}
    pkg_dir = Path(__file__).parent
    scan = [(str(pkg_dir), _PACKAGE), (str(pkg_dir / "generated"), f"{_PACKAGE}.generated")]
    for path, pkg in scan:
        for mod in pkgutil.iter_modules([path]):
            if mod.name.startswith("_"):
                continue
            module = importlib.import_module(f"{pkg}.{mod.name}")
            module = importlib.reload(module)  # generated files change during a session
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Strategy) and obj is not Strategy and obj.__module__ == module.__name__:
                    registry[obj.name] = obj
    return registry
