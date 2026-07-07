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
    scan = [(str(pkg_dir), _PACKAGE, False),
            (str(pkg_dir / "generated"), f"{_PACKAGE}.generated", True)]
    for path, pkg, is_generated in scan:
        for mod in pkgutil.iter_modules([path]):
            if mod.name.startswith("_"):
                continue
            try:
                module = importlib.import_module(f"{pkg}.{mod.name}")
                if is_generated:  # generated files change during a session
                    module = importlib.reload(module)
            except Exception:
                if not is_generated:
                    raise
                continue  # one broken generated file must not brick the app
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Strategy) and obj is not Strategy and obj.__module__ == module.__name__:
                    registry[obj.name] = obj
    return registry
