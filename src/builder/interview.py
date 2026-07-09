from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

DEFAULT_MODEL = "claude-sonnet-4-6"
MODEL_CHOICES = ["claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5"]

SPEC_REQUIRED_KEYS = ["name", "description", "timeframe", "buy_condition",
                      "sell_condition", "tp_sl_strategy", "risk_management", "parameters"]

FINALIZE_SPEC_TOOL = {
    "name": "finalize_spec",
    "description": (
        "Record the completed strategy specification. Call this ONLY once every section is "
        "pinned down with the user: buy condition, sell/exit condition, TP/SL strategy, risk "
        "management & sizing, timeframe(s), and full construction details of every indicator. "
        "Every tunable value must appear in `parameters` — never hardcode a number the user "
        "might want to tune."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Short unique strategy name"},
            "description": {"type": "string", "description": "One-paragraph summary"},
            "timeframe": {"type": "string", "description": "Base bar timeframe, e.g. m5, m15, h1"},
            "htf_timeframes": {"type": "array", "items": {"type": "string"},
                                "description": "Higher-timeframe views needed, e.g. ['h1']"},
            "buy_condition": {"type": "string", "description": "Precise long-entry rule"},
            "sell_condition": {"type": "string",
                                "description": "Precise exit rule and/or short-entry rule"},
            "tp_sl_strategy": {"type": "string", "description": "Stop-loss and take-profit logic"},
            "risk_management": {"type": "string",
                                 "description": "Position sizing, risk % per trade, leverage"},
            "parameters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": ["int", "float", "bool"]},
                        "default": {},
                        "min": {},
                        "max": {},
                        "help": {"type": "string"},
                    },
                    "required": ["name", "type", "default", "min", "max", "help"],
                },
            },
            "notes": {"type": "string", "description": "Assumptions and limitations"},
        },
        "required": SPEC_REQUIRED_KEYS,
    },
}

_ENGINE_CAPABILITIES = """\
## Engine capabilities (spec ONLY within these)
- Market entries filled at the NEXT bar open (plus spread/slippage costs)
- Absolute stop-loss and take-profit prices per position
- Trailing stop: a fixed price distance the engine trails intrabar
- Risk-percent sizing helper: lots = (equity * risk_pct) / (|entry - stop| * 100)
- Leverage cap on notional; sizes are in lots (1 lot = 100 oz), min step 0.01
- Higher-timeframe views (h1, h4, ...) resampled from the base data
- ONE open position at a time; long and short both supported
- Partial closes / scaling out: ctx.close(reason=..., fraction=0.5) closes that fraction \
at the next bar open; the remainder keeps its SL/TP/trailing stop and R is pro-rated
- Full bar history up to the current bar (no look-ahead)

## NOT supported — if the user asks, offer the nearest supported alternative
- Limit/stop entry orders (only market at next open)
- Pyramiding / adding to an open position, multiple concurrent positions
- Tick data (bars only), multi-symbol strategies\
"""

SYSTEM_PROMPT = f"""\
You are a trading-strategy analyst for a gold (XAUUSD) backtesting engine. Your job is to \
interview the user about the strategy they describe until every detail needed to implement \
it is pinned down, then call the finalize_spec tool.

{_ENGINE_CAPABILITIES}

## Interview rules
- Ask ONE focused question at a time. Keep questions short and concrete.
- You MUST cover: buy condition, sell/exit condition, TP/SL strategy, risk management & \
sizing, base timeframe, and the exact construction of every indicator mentioned \
(e.g. for Elder Force Index: raw or EMA-smoothed? what period? which timeframe?).
- Suggest sensible defaults so the user can just say "yes" (e.g. "EFI is usually smoothed \
with a 13-period EMA — use that?").
- Every tunable value (periods, multipliers, risk %, session hours...) must become an entry \
in `parameters` with a sensible default, min, max and a one-line help string.
- When everything is pinned down, give a one-line summary and call finalize_spec. If the \
user later asks for changes, ask what's needed and call finalize_spec again with the \
revised spec.
"""

_PARAM_KEYS = FINALIZE_SPEC_TOOL["input_schema"]["properties"]["parameters"]["items"]["required"]


def get_client(env_file: Optional[Path] = None) -> Optional[anthropic.Anthropic]:
    """Client from ANTHROPIC_API_KEY (repo-root .env file, or shell env), or None if
    the key is missing. override=True: .env is re-read on every Streamlit rerun, so
    editing the key there takes effect on the next message — no server restart."""
    load_dotenv(env_file or _ENV_FILE, override=True)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    return anthropic.Anthropic()


def _validate_spec(spec: dict) -> list[str]:
    missing = []
    for k in SPEC_REQUIRED_KEYS:
        if k == "parameters":
            if not isinstance(spec.get(k), list):
                missing.append(k)
        else:
            if not spec.get(k):
                missing.append(k)
    for p in (spec.get("parameters", []) if isinstance(spec.get("parameters"), list) else []):
        if not isinstance(p, dict):
            missing.append("parameters[] must be objects")
            continue
        for k in _PARAM_KEYS:
            if k not in p:
                missing.append(f"parameters[].{k}")
    return missing


def _tool_turn(client, messages: list, model: str, system: str, tool: dict,
               validate_fn, ok_ack: str, reject_label: str) -> tuple[str, dict | None]:
    """Shared chat-turn core for the creation and revision interviews. Appends the
    assistant turn (and tool_result answers for any tool calls) to `messages` itself —
    callers only append user turns. Returns (display_text, tool_payload_or_None)."""
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        cache_control={"type": "ephemeral"},
        system=system,
        tools=[tool],
        tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        messages=messages,
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    if getattr(response, "stop_reason", None) == "max_tokens":
        text += "\n\n(Response was truncated — say 'continue' to let me finish.)"

    assistant_blocks = []
    tool_uses = []
    for b in response.content:
        if b.type == "text":
            assistant_blocks.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            assistant_blocks.append({"type": "tool_use", "id": b.id, "name": b.name,
                                     "input": b.input})
            tool_uses.append(b)

    if not tool_uses:
        if text:
            messages.append({"role": "assistant", "content": text})
        return text, None

    messages.append({"role": "assistant", "content": assistant_blocks})

    payload = None
    results = []
    for i, tu in enumerate(tool_uses):
        if tu.name != tool["name"] or i > 0:
            results.append({"type": "tool_result", "tool_use_id": tu.id, "is_error": True,
                            "content": f"Ignored — only one {tool['name']} call per turn."})
            continue
        missing = validate_fn(tu.input)
        if missing:
            results.append({"type": "tool_result", "tool_use_id": tu.id, "is_error": True,
                            "content": f"{reject_label} — missing/invalid fields: "
                                       f"{', '.join(missing)}. Ask the user for these."})
            text += (f"\n\n({reject_label} — missing fields: {', '.join(missing)}. "
                     "Please continue the conversation to fill them in.)")
        else:
            payload = dict(tu.input)
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": ok_ack})
    messages.append({"role": "user", "content": results})
    return text, payload


def run_interview_turn(client, messages: list, model: str = DEFAULT_MODEL) -> tuple[str, dict | None]:
    """One creation-interview turn. Returns (display_text, spec_or_None)."""
    return _tool_turn(client, messages, model, SYSTEM_PROMPT, FINALIZE_SPEC_TOOL,
                      _validate_spec, "Spec recorded and shown to the user as a card.",
                      "Spec rejected")


FINALIZE_REVISION_TOOL = {
    "name": "finalize_revision",
    "description": (
        "Record the agreed revision plan for the existing strategy. Call this ONLY when the "
        "requested change is fully pinned down — ask clarifying questions first if anything is "
        "ambiguous (indicator variant, periods, timeframes, how it interacts with the existing "
        "logic, which new tunable parameters to expose). The plan is handed to a separate "
        "implementer AFTER the user confirms it; do not write code yourself."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string",
                        "description": "One-paragraph description of the revision"},
            "changes": {"type": "array", "items": {"type": "string"},
                        "description": "Concrete, complete change list for the implementer. "
                                       "Every new tunable value must appear as a new params "
                                       "entry with type/default/min/max/help."},
        },
        "required": ["summary", "changes"],
    },
}


def _validate_revision(plan: dict) -> list[str]:
    missing = []
    if not plan.get("summary"):
        missing.append("summary")
    changes = plan.get("changes")
    if not isinstance(changes, list) or not changes:
        missing.append("changes")
    return missing


def _revision_system(code: str, spec: dict | None) -> str:
    spec_txt = json.dumps(spec, indent=2) if spec else "(no saved spec)"
    return f"""\
You are a trading-strategy analyst helping revise an EXISTING strategy for a gold (XAUUSD) \
backtesting engine. Discuss the user's requested change; when anything is ambiguous ask ONE \
focused clarifying question at a time; once the change is fully pinned down give a one-line \
summary and call the finalize_revision tool. Do NOT write code — the confirmed plan goes to a \
separate implementer.

{_ENGINE_CAPABILITIES}

## Revision rules
- Small unambiguous tweaks ("change the EMA default to 21") need no questions — call \
finalize_revision immediately.
- Ambiguous requests ("add a trend filter") DO need pinning down: which indicator, what \
period, which timeframe, how it gates entries/exits, and which new tunable parameters to \
expose (with default/min/max).
- If the user asks for something the engine cannot do, say so and offer the nearest \
supported alternative.

## Current strategy spec
{spec_txt}

## Current strategy code
```python
{code}
```
"""


def run_revision_turn(client, messages: list, code: str, spec: dict | None = None,
                      model: str = DEFAULT_MODEL) -> tuple[str, dict | None]:
    """One revision-interview turn about an existing strategy. Returns
    (display_text, revision_plan_or_None). Same history-ownership contract as
    run_interview_turn."""
    return _tool_turn(client, messages, model, _revision_system(code, spec),
                      FINALIZE_REVISION_TOOL, _validate_revision,
                      "Revision plan recorded and shown to the user for confirmation.",
                      "Revision plan rejected")
