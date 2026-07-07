from __future__ import annotations

import os
from typing import Optional

import anthropic

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

SYSTEM_PROMPT = """\
You are a trading-strategy analyst for a gold (XAUUSD) backtesting engine. Your job is to \
interview the user about the strategy they describe until every detail needed to implement \
it is pinned down, then call the finalize_spec tool.

## Engine capabilities (spec ONLY within these)
- Market entries filled at the NEXT bar open (plus spread/slippage costs)
- Absolute stop-loss and take-profit prices per position
- Trailing stop: a fixed price distance the engine trails intrabar
- Risk-percent sizing helper: lots = (equity * risk_pct) / (|entry - stop| * 100)
- Leverage cap on notional; sizes are in lots (1 lot = 100 oz), min step 0.01
- Higher-timeframe views (h1, h4, ...) resampled from the base data
- ONE open position at a time; long and short both supported
- Full bar history up to the current bar (no look-ahead)

## NOT supported — if the user asks, offer the nearest supported alternative
- Limit/stop entry orders (only market at next open)
- Partial closes / scaling out, pyramiding, multiple concurrent positions
- Tick data (bars only), multi-symbol strategies

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


def get_client() -> Optional[anthropic.Anthropic]:
    """Client from ANTHROPIC_API_KEY, or None if the key is missing."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    return anthropic.Anthropic()


def _validate_spec(spec: dict) -> list[str]:
    missing = [k for k in SPEC_REQUIRED_KEYS if not spec.get(k)]
    for p in spec.get("parameters", []):
        for k in ("name", "type", "default", "min", "max", "help"):
            if k not in p:
                missing.append(f"parameters[].{k}")
    return missing


def run_interview_turn(client, messages: list, model: str = DEFAULT_MODEL):
    """One assistant turn. Returns (text, spec_or_None, raw_content). `messages` is the
    raw Anthropic-format history; when spec is not None the caller must pass raw_content
    to ack_spec() so the tool call is answered before the next turn."""
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=[FINALIZE_SPEC_TOOL],
        messages=messages,
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    spec = next((b.input for b in response.content
                 if b.type == "tool_use" and b.name == "finalize_spec"), None)
    if spec is not None:
        missing = _validate_spec(spec)
        if missing:
            return (text + f"\n\n(Spec rejected — missing fields: {', '.join(missing)}. "
                    "Please continue the conversation to fill them in.)"), None, response.content
        return text, dict(spec), response.content
    return text, None, response.content


def ack_spec(messages: list, response_content) -> None:
    """Append the assistant tool-call turn + a tool_result ack so the chat can continue."""
    blocks = []
    for b in response_content:
        if b.type == "text":
            blocks.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    messages.append({"role": "assistant", "content": blocks})
    tool_use_id = next(b.id for b in response_content if b.type == "tool_use")
    messages.append({"role": "user", "content": [{
        "type": "tool_result", "tool_use_id": tool_use_id,
        "content": "Spec recorded and shown to the user as a card.",
    }]})
