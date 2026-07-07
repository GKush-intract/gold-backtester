# Natural-Language Strategy Builder — Design

**Date:** 2026-07-07
**Status:** Approved

## Purpose

A new Streamlit page where the user describes a trading strategy in plain
English. Claude (via the Anthropic API) interviews the user to pin down every
detail — buying condition, selling/exit condition, TP/SL strategy, risk
management, and indicator construction (e.g. "EFI: raw or EMA-smoothed? what
period? which timeframe?") — then generates a Python strategy file, validates
it, backtests it on the existing XAUUSD data, and lets the user fine-tune
parameters without further API calls.

## Decisions made

| Decision | Choice |
|---|---|
| UI location | New page in the existing Streamlit app (`pages/1_Strategy_Builder.py`) |
| API key | `ANTHROPIC_API_KEY` environment variable only |
| File storage | Persist in `src/strategies/generated/`, one file per strategy |
| Spec gate | Structured spec card must be confirmed before codegen |
| Run mode | Auto-run after generation, with auto-repair loop (max 3 attempts) |
| Architecture | Two-phase: interview → confirmed spec → codegen (not agent loop, not template) |
| Default model | `claude-sonnet-4-6`, sidebar model picker |

## Architecture

```
pages/
  1_Strategy_Builder.py     # chat UI page (thin — UI/session-state only)
src/builder/
  __init__.py
  interview.py              # Anthropic chat loop + finalize_spec tool schema
  codegen.py                # codegen prompt, code extraction, file naming/writing
  validate.py               # subprocess import + smoke backtest + repair loop
src/strategies/generated/
  __init__.py               # package marker; generated files land here
src/ui_results.py           # results rendering factored out of app.py, shared by both pages
```

Streamlit's `pages/` convention makes `app.py` the home page and adds the
builder page to the nav with zero changes to the existing flow.

Changes to existing code (both targeted, behavior-preserving):

1. `src/strategies/__init__.py::get_strategy_registry()` also scans the
   `generated/` subpackage, so generated strategies appear in the main
   backtester dropdown automatically.
2. `app.py`'s results block (metrics row, equity/drawdown chart, R-histogram,
   trade log, downloads) moves to `src/ui_results.py::render_results(res, m,
   summary, cfg)`; `app.py` calls it.

New dependency: `anthropic` (added to `requirements.txt`).

## Component: interview (`src/builder/interview.py`)

- Holds the Anthropic client (key from `ANTHROPIC_API_KEY`; page shows setup
  instructions if missing).
- System prompt: Claude is a trading-strategy analyst for THIS engine. It is
  given the engine's capability envelope so it never specs unsupported
  features and instead offers the nearest supported alternative:
  - Supported: market entry filled next bar open, absolute SL/TP, trailing
    stop (price distance), risk-% sizing (`ctx.size_for_risk`), leverage cap,
    HTF views via `htf_timeframes`, single open position at a time,
    long and short.
  - Not supported: limit/stop entry orders, partial closes, multiple
    concurrent positions, per-trade pyramiding.
- Interview style: one focused question at a time; must cover buy condition,
  sell/exit condition, TP/SL strategy, risk management & sizing, timeframe(s),
  and full construction details of every indicator mentioned.
- Tool `finalize_spec` (forced structured output when Claude judges the spec
  complete). Schema fields:
  - `name`, `description`, `timeframe`, `htf_timeframes[]`
  - `buy_condition`, `sell_condition`, `tp_sl_strategy`, `risk_management`
    (prose, precise)
  - `parameters[]`: `{name, type(int|float|bool), default, min, max, help}` —
    every tunable value must be a parameter, never a hardcoded constant
  - `notes` (assumptions/limitations)
- UI renders the spec as a card with sections + **Generate code** button.
  Continued chat revises the spec; each revision re-emits the card.

## Component: codegen (`src/builder/codegen.py`)

- Single API call. System prompt embeds:
  - full source of `src/strategy.py` (the contract),
  - `src/strategies/ma_crossover.py` as a worked example,
  - rules: subclass `Strategy`; unique `name`; all tunables in `params` with
    `(type, default, min, max, help)` matching the spec's `parameters`;
    use `ctx.history` only (no look-ahead); declare `htf_timeframes` when the
    spec uses them; imports limited to stdlib/numpy/pandas + relative
    `..strategy`.
- Response contract: one complete Python file in a single fenced code block;
  extractor takes the last/only block.
- File naming: `gen_<slug>_<YYYYMMDD_HHMM>.py` in `src/strategies/generated/`.
  That file is the session's working file; logic revisions overwrite it.

## Component: validate (`src/builder/validate.py`)

Runs in a subprocess with a 60s timeout so generated code cannot hang or
crash the Streamlit process:

1. `ast.parse` (syntax),
2. import + instantiate the Strategy subclass,
3. smoke backtest on the last ~2,000 bars of `data/raw/XAUUSD_m5_5y.csv`.

On failure: traceback + current code go back to Claude for a corrected file;
max 3 repair attempts, each overwriting the file. Zero trades on the smoke
run is a warning, not a failure. Subprocess timeout is reported to Claude as
"likely infinite loop". Generated code is always visible in an expander —
it is arbitrary Python executed locally, same trust level as any strategy
file in the repo.

## Run & fine-tune (page behavior)

- After validation passes, a full backtest runs automatically.
- Sidebar: data controls (CSV file, date range, resample; defaults:
  `XAUUSD_m5_5y.csv`, last 2 years) and account/cost controls mirroring
  `app.py`; results render via shared `ui_results.py`.
- Fine-tuning tiers:
  - **Params:** sidebar auto-renders inputs from the generated `params`
    schema (same widget loop as `app.py`). Re-run is instant, no API call.
  - **Logic:** continued chat sends current file + request to Claude, which
    returns the updated full file → same validate/repair/re-run pipeline.
- Session state tracks: message history, current spec, working file path,
  last backtest result.

## Error handling

| Failure | Behavior |
|---|---|
| No `ANTHROPIC_API_KEY` | Page shows setup instructions, chat disabled |
| API error (rate limit/overload) | Message in chat + retry button |
| Repair loop exhausted (3×) | Show last traceback + code; user directs fix via chat |
| Subprocess timeout | Fail attempt, report "likely infinite loop" to Claude |
| Smoke run has 0 trades | Warning banner, proceed anyway |

## Testing (mocked API, no network)

- Code-block extraction from model responses (single block, extra prose,
  malformed).
- File naming/slugging.
- Registry discovers strategies in `generated/`.
- Validation subprocess: passes a known-good strategy, fails a deliberately
  broken one, times out an infinite loop.
- `finalize_spec` schema validation.
- Existing test suite stays green (notably `app.py` refactor to
  `ui_results.py` is behavior-preserving).

## Out of scope (YAGNI)

- Auto-optimization/param sweeps, multi-symbol support, strategy versioning
  UI, deleting/managing generated files from the UI, streaming token output,
  replayer integration.
