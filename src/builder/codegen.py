from __future__ import annotations

import datetime as dt
import importlib
import inspect
import json
import re
from pathlib import Path
from typing import Optional

from .interview import DEFAULT_MODEL

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED_DIR = REPO_ROOT / "src" / "strategies" / "generated"

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _engine_contract() -> str:
    """The full strategy API contract embedded into every codegen prompt."""
    strategy_src = (REPO_ROOT / "src" / "strategy.py").read_text()
    example_src = (REPO_ROOT / "src" / "strategies" / "ma_crossover.py").read_text()
    return f"""\
You write Python strategy files for a gold (XAUUSD) backtesting engine.

## The engine contract (src/strategy.py) — your class MUST subclass Strategy:
```python
{strategy_src}
```

## A complete worked example (src/strategies/ma_crossover.py):
```python
{example_src}
```

## Hard rules
1. Output ONE complete Python file in a single ```python fenced block. No other code blocks.
2. Subclass Strategy. Set a unique `name` (use the spec's name). Implement `on_bar(self, ctx)`.
3. EVERY tunable from the spec goes in the `params` dict: name -> (type, default, min, max, help),
   type is "int" | "float" | "bool". Read values via self.p["name"]. Never hardcode tunables.
4. Use ctx.history / ctx.bar only — never index past ctx.index (no look-ahead).
5. Sizing: use ctx.size_for_risk(risk_pct, entry, stop) unless the spec says otherwise.
6. If the spec needs higher timeframes, set `htf_timeframes = ["h1", ...]` and read ctx.htf["h1"]
   (a resampled OHLCV DataFrame; only use rows with index <= ctx.bar["time"]).
7. Imports allowed: stdlib, numpy, pandas, and `from ...strategy import Strategy`
   (THREE dots — the file lives in src/strategies/generated/, two levels below src/).
8. Keep on_bar fast: O(lookback) per bar, no prints, no file/network access.
"""


def extract_code(text: str) -> str:
    """The Python file from a model response: last fenced block, or bare code."""
    blocks = _CODE_BLOCK_RE.findall(text)
    if blocks:
        return blocks[-1]
    stripped = text.strip()
    if stripped.startswith(("from ", "import ", '"""', "#")):
        return text if text.endswith("\n") else text + "\n"
    raise ValueError("No Python code block found in model response")


def strategy_filename(name: str, now: Optional[dt.datetime] = None) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "strategy"
    now = now or dt.datetime.now()
    return f"gen_{slug}_{now:%Y%m%d_%H%M}.py"


def write_strategy_file(code: str, name: str, path: Optional[Path] = None) -> Path:
    """Write (or overwrite) the strategy file. Pass `path` to overwrite the session file."""
    if path is None:
        path = GENERATED_DIR / strategy_filename(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)
    return path


def load_strategy_class(path: Path):
    """Import (or reload) the generated module and return its Strategy subclass."""
    importlib.invalidate_caches()
    from ..strategy import Strategy
    mod_name = f"src.strategies.generated.{path.stem}"
    module = importlib.import_module(mod_name)
    module = importlib.reload(module)
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Strategy) and obj is not Strategy and obj.__module__ == module.__name__:
            return obj
    raise ValueError(f"No Strategy subclass found in {path.name}")


def _call(client, model: str, user_content: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        cache_control={"type": "ephemeral"},
        system=_engine_contract(),
        messages=[{"role": "user", "content": user_content}],
    )
    if getattr(response, "stop_reason", None) == "max_tokens":
        raise ValueError("Model response was truncated (hit max_tokens) — "
                         "the strategy may be too complex for one file; try simplifying the spec.")
    text = "".join(b.text for b in response.content if b.type == "text")
    return extract_code(text)


def generate_strategy(client, spec: dict, model: str = DEFAULT_MODEL) -> str:
    return _call(client, model, "Implement this strategy spec as a complete file:\n\n"
                 + json.dumps(spec, indent=2))


def revise_strategy(client, code: str, request: str, model: str = DEFAULT_MODEL) -> str:
    return _call(client, model,
                 f"Here is the current strategy file:\n```python\n{code}\n```\n\n"
                 f"Apply this change and return the FULL updated file:\n{request}")


def repair_strategy(client, code: str, error: str, model: str = DEFAULT_MODEL) -> str:
    return _call(client, model,
                 f"This strategy file failed validation.\n```python\n{code}\n```\n\n"
                 f"Error:\n```\n{error}\n```\n\nFix it and return the FULL corrected file.")
