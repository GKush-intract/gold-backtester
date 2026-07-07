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
            try:
                parsed = json.loads(line[len(_SENTINEL):])
            except ValueError:
                continue  # malformed sentinel (e.g. printed by generated code) — keep scanning
            if isinstance(parsed, dict):
                return parsed
    return {"ok": False, "num_trades": None, "warning": None,
            "error": "Validator crashed before reporting.\n"
                     f"stdout: {proc.stdout[-1000:]}\nstderr: {proc.stderr[-2000:]}"}


def generate_validated(client, spec: dict, model: str = DEFAULT_MODEL,
                       csv_path: Path = DEFAULT_CSV, bars: int = 2000,
                       max_attempts: int = 3, log=None) -> tuple[Path | None, dict, str]:
    """Generate -> validate -> repair loop. Returns (path, result, code).
    The same file is overwritten on every repair. `log` is an optional callable
    for UI status lines. `path` is None when generation itself failed (e.g.
    truncated response or missing code block)."""
    log = log or (lambda msg: None)
    log("Generating strategy code…")
    try:
        code = generate_strategy(client, spec, model=model)
    except ValueError as e:
        return None, {"ok": False, "error": str(e), "num_trades": None, "warning": None}, ""
    path = write_strategy_file(code, spec["name"])
    result = None
    for attempt in range(1, max_attempts + 1):
        log(f"Validating (attempt {attempt}/{max_attempts})…")
        result = validate_strategy(path, csv_path=csv_path, bars=bars)
        if result["ok"] or attempt == max_attempts:
            break
        log("Validation failed — asking Claude to fix it…")
        try:
            code = repair_strategy(client, code, result["error"], model=model)
        except ValueError as e:
            result["error"] = (result["error"] or "") + f"\n\nRepair attempt failed: {e}"
            break
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

        from src.builder.codegen import load_strategy_class  # 2) import + instantiate
        strat = load_strategy_class(path)()

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
