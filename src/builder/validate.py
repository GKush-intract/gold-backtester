from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .codegen import (GENERATED_DIR, REPO_ROOT, generate_strategy, repair_strategy,
                      write_strategy_file)
from .interview import DEFAULT_MODEL

DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "XAUUSD_m5_5y.csv"
_SENTINEL = "RESULT_JSON:"


def validate_strategy(path: Path, csv_path: Path = DEFAULT_CSV,
                      bars: int = 2000, timeout: int = 60) -> dict:
    """ast-parse + import + smoke-backtest the file in a subprocess.
    Returns {"ok": bool, "error": str|None, "num_trades": int|None, "warning": str|None}."""
    cmd = [sys.executable, "-m", "src.builder.validate",
           str(path), str(csv_path), str(bars)]
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "num_trades": None, "warning": None,
                "error": f"Validation timed out after {timeout}s — "
                         "likely an infinite loop in on_bar()."}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_SENTINEL):
            return json.loads(line[len(_SENTINEL):])
    return {"ok": False, "num_trades": None, "warning": None,
            "error": "Validator crashed before reporting.\n"
                     f"stdout: {proc.stdout[-1000:]}\nstderr: {proc.stderr[-2000:]}"}


def generate_validated(client, spec: dict, model: str = DEFAULT_MODEL,
                       csv_path: Path = DEFAULT_CSV, bars: int = 2000,
                       max_attempts: int = 3, log=None):
    """Generate -> validate -> repair loop. Returns (path, result, code).
    The same file is overwritten on every repair. `log` is an optional callable
    for UI status lines."""
    log = log or (lambda msg: None)
    log("Generating strategy code…")
    code = generate_strategy(client, spec, model=model)
    path = write_strategy_file(code, spec["name"])
    result = None
    for attempt in range(1, max_attempts + 1):
        log(f"Validating (attempt {attempt}/{max_attempts})…")
        result = validate_strategy(path, csv_path=csv_path, bars=bars)
        if result["ok"] or attempt == max_attempts:
            break
        log("Validation failed — asking Claude to fix it…")
        code = repair_strategy(client, code, result["error"], model=model)
        write_strategy_file(code, spec["name"], path=path)
    return path, result, code


def _main() -> None:
    """Subprocess entry: python -m src.builder.validate <file> <csv> <bars>"""
    import ast
    import traceback

    out = {"ok": False, "error": None, "num_trades": None, "warning": None}
    try:
        path = Path(sys.argv[1])
        csv_path, bars = sys.argv[2], int(sys.argv[3])

        ast.parse(path.read_text())                       # 1) syntax

        import importlib
        import inspect
        from src.strategy import Strategy                  # 2) import + instantiate
        importlib.invalidate_caches()
        module = importlib.import_module(f"src.strategies.generated.{path.stem}")
        module = importlib.reload(module)
        cls = next(obj for _, obj in inspect.getmembers(module, inspect.isclass)
                   if issubclass(obj, Strategy) and obj is not Strategy
                   and obj.__module__ == module.__name__)
        strat = cls()

        from src.data_loader import load_ohlc              # 3) smoke backtest
        from src.engine import BacktestConfig
        from src import runner
        data = load_ohlc(csv_path).tail(bars)
        res = runner.run_backtest(BacktestConfig(), strat, data)
        out["ok"] = True
        out["num_trades"] = int(len(res.trades))
        if out["num_trades"] == 0:
            out["warning"] = ("Smoke run produced 0 trades — the logic may be too "
                              "restrictive, or it may just need more data.")
    except Exception:
        out["error"] = traceback.format_exc()
    print(_SENTINEL + json.dumps(out))


if __name__ == "__main__":
    _main()
