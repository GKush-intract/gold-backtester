from __future__ import annotations

import datetime as dt
from pathlib import Path

import anthropic
import streamlit as st

from src import runner
from src.builder import interview
from src.builder.diagnose import DIAG_REQUEST, build_digest
from src.builder.codegen import (GENERATED_DIR, load_spec, load_strategy_class,
                                 repair_strategy, revise_strategy, save_spec,
                                 write_strategy_file)
from src.builder.validate import DEFAULT_CSV, generate_validated, validate_strategy
from src.data_loader import load_ohlc, resample
from src.engine import BacktestConfig
from src.metrics import compute_metrics
from src.ui_results import render_param_inputs, render_results
from src.ui_replay import render_replay

st.set_page_config(page_title="Strategy Builder", layout="wide")
st.title("🤖 Strategy Builder")

# ---------------- Session state ----------------
ss = st.session_state
ss.setdefault("b_messages", [])      # Anthropic-format history (mutated by run_interview_turn)
ss.setdefault("b_display", [])       # [(role, text)] for rendering
ss.setdefault("b_spec", None)
ss.setdefault("b_path", None)        # Path of the session's strategy file
ss.setdefault("b_code", None)
ss.setdefault("b_run_requested", False)
ss.setdefault("b_rev_messages", [])   # revision-interview history (per loaded strategy)
ss.setdefault("b_rev_plan", None)     # confirmed-pending revision plan card
ss.setdefault("b_last", None)         # last backtest: (res, m, summary, cfg, elapsed, data)

client = interview.get_client()
if client is None:
    st.error("`ANTHROPIC_API_KEY` is not set. Put it in a `.env` file in the project root "
             "(see `.env.example`) or export it, then restart:\n\n"
             "```bash\necho 'ANTHROPIC_API_KEY=sk-ant-...' > .env\nstreamlit run app.py\n```")
    st.stop()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Model")
    model = st.selectbox("Claude model", interview.MODEL_CHOICES, index=0)

    st.header("Data")
    files = sorted(p.name for p in Path("data/raw").glob("*.csv"))
    default_ix = files.index(DEFAULT_CSV.name) if DEFAULT_CSV.name in files else 0
    csv_name = st.selectbox("CSV file", files, index=default_ix) if files else None
    start = st.date_input("Start", dt.date(2026, 1, 1))
    end = st.date_input("End", dt.date(2026, 3, 31))
    resample_to = st.selectbox("Resample to (optional)",
                               ["(none)", "m5", "m15", "m30", "h1", "h4"], index=0)

    st.header("Account & costs")
    balance = st.number_input("Opening balance", 100.0, 1e9, 10_000.0, step=1000.0)
    spread = st.number_input("Spread (price units)", 0.0, 10.0, 0.30)
    max_leverage = st.number_input("Max leverage (×)", 0.0, 500.0, 20.0)

    gen_files = sorted(f.name for f in GENERATED_DIR.glob("gen_*.py"))
    if gen_files:
        st.header("Strategy file")
        sel = st.selectbox("Load existing strategy", ["(current session)"] + gen_files)
        if sel != "(current session)" and st.button("📂 Load for editing"):
            lpath = GENERATED_DIR / sel
            try:
                lcls = load_strategy_class(lpath)
                ss.b_path = lpath
                ss.b_code = lpath.read_text()
                ss.b_spec = load_spec(lpath) or {"name": lcls.name}
                ss.b_rev_messages = []
                ss.b_rev_plan = None
                ss.b_last = None
                ss.b_display.append(("assistant",
                    f"Loaded `{sel}` (**{lcls.name}**). Describe a change below — I'll ask "
                    "questions if needed and show you a revision plan to confirm before "
                    "touching the code. Or just tune params here and hit Run."))
                st.rerun()
            except Exception as e:
                st.warning(f"Could not load {sel}: {e}")

    param_values = {}
    strat_cls = None
    if ss.b_path is not None:
        try:
            strat_cls = load_strategy_class(ss.b_path)
            st.header(f"Parameters — {strat_cls.name}")
            param_values = render_param_inputs(strat_cls, key_prefix="gen_")
            if st.button("▶ Run backtest", type="primary"):
                ss.b_run_requested = True
        except Exception as e:  # file mid-repair or deleted
            ss.b_run_requested = False
            st.warning(f"Strategy not loadable yet: {e}")


def _csv_path() -> Path:
    return Path("data/raw") / csv_name if csv_name else DEFAULT_CSV


def run_backtest_now():
    data = load_ohlc(str(_csv_path()), start=str(start), end=str(end))
    if resample_to != "(none)":
        data = resample(data, resample_to)
    cfg = BacktestConfig(opening_balance=balance, spread=spread, max_leverage=max_leverage)
    strat = strat_cls(**param_values)
    t0 = dt.datetime.now()
    res = runner.run_backtest(cfg, strat, data)
    elapsed = (dt.datetime.now() - t0).total_seconds()
    m, summary = compute_metrics(res)
    ss.b_last = (res, m, summary, cfg, elapsed, data)


# ---------------- Chat history ----------------
st.caption("Describe your strategy in plain words — I'll ask questions until every detail "
           "(entries, exits, TP/SL, risk, indicator settings) is pinned down, then write, "
           "validate and backtest the code. Fine-tune params in the sidebar afterwards.")

for role, text in ss.b_display:
    with st.chat_message(role):
        st.markdown(text)

# ---------------- Spec card ----------------
if ss.b_spec is not None and ss.b_path is None:
    spec = ss.b_spec
    with st.container(border=True):
        st.subheader(f"📋 Spec: {spec['name']}")
        st.markdown(spec["description"])
        c1, c2 = st.columns(2)
        c1.markdown(f"**🟢 Buy condition**\n\n{spec['buy_condition']}")
        c2.markdown(f"**🔴 Sell / exit condition**\n\n{spec['sell_condition']}")
        c1.markdown(f"**🎯 TP/SL strategy**\n\n{spec['tp_sl_strategy']}")
        c2.markdown(f"**🛡 Risk management**\n\n{spec['risk_management']}")
        tf = spec.get("timeframe", "m5")
        htf = ", ".join(spec.get("htf_timeframes", [])) or "none"
        st.markdown(f"**Timeframe:** {tf} &nbsp;&nbsp; **HTF views:** {htf}")
        if spec.get("parameters"):
            st.markdown("**Tunable parameters:** " +
                        ", ".join(f"`{p['name']}`={p['default']}" for p in spec["parameters"]))
        if spec.get("notes"):
            st.caption(spec["notes"])
        st.caption("Not right? Just keep chatting below to revise the spec.")

        if st.button("⚡ Generate code & run backtest", type="primary"):
            with st.status("Building strategy…", expanded=True) as status:
                path, result, code = generate_validated(
                    client, spec, model=model, csv_path=_csv_path(), log=st.write)
                ss.b_code = code
                if result["ok"]:
                    ss.b_path = path
                    save_spec(path, spec)  # sidecar lets future sessions reload intent
                    ss.b_rev_messages = []
                    ss.b_rev_plan = None
                    if result.get("warning"):
                        st.warning(result["warning"])
                    status.update(label=f"✅ {path.name} validated "
                                        f"({result['num_trades']} smoke trades)", state="complete")
                    ss.b_run_requested = True
                    st.rerun()
                else:
                    if path is not None:
                        path.unlink(missing_ok=True)  # don't accumulate broken gen_*.py files
                    status.update(label="❌ Validation failed", state="error")
                    st.code(result["error"] or "unknown error")
                    st.info("Keep chatting below to revise the spec, then generate again.")

# ---------------- Revision plan card ----------------
if ss.b_rev_plan is not None and ss.b_path is not None:
    plan = ss.b_rev_plan
    with st.container(border=True):
        st.subheader("🔧 Revision plan")
        st.markdown(plan["summary"])
        for ch in plan.get("changes", []):
            st.markdown(f"- {ch}")
        st.caption("Not right? Keep chatting below to refine the plan — nothing changes "
                   "until you apply.")
        if st.button("✅ Apply changes & re-run", type="primary"):
            request = (plan["summary"] + "\n\nApply exactly these changes:\n"
                       + "\n".join(f"- {c}" for c in plan.get("changes", [])))
            old_code = ss.b_code
            ok = False
            try:
                with st.status("Revising strategy…", expanded=True) as status:
                    st.write("Asking Claude for the updated file…")
                    code = revise_strategy(client, ss.b_code, request, model=model)
                    write_strategy_file(code, ss.b_spec["name"], path=ss.b_path)
                    st.write("Validating…")
                    result = validate_strategy(ss.b_path, csv_path=_csv_path())
                    attempts = 1
                    while not result["ok"] and attempts < 3:
                        st.write("Validation failed — asking Claude to fix it…")
                        code = repair_strategy(client, code, result["error"], model=model)
                        write_strategy_file(code, ss.b_spec["name"], path=ss.b_path)
                        result = validate_strategy(ss.b_path, csv_path=_csv_path())
                        attempts += 1
                    ok = result["ok"]
                    if ok:
                        ss.b_code = code
                        status.update(label="✅ Revision validated", state="complete")
                        ss.b_display.append(("assistant",
                                             "Applied the plan — re-running the backtest."))
                        ss.b_run_requested = True
                    else:
                        status.update(label="❌ Revision failed — previous version restored",
                                      state="error")
                        ss.b_display.append(("assistant",
                            "Applying the plan failed validation, so I restored the previous "
                            "working version. Last error:\n```\n"
                            + (result["error"] or "unknown")[-1500:] + "\n```\n"
                            "Refine the plan below and apply again."))
            except anthropic.APIError as e:
                st.error(f"Claude API error: {e} — click Apply again to retry.")
            except ValueError as e:
                st.error(f"Code generation problem: {e} — click Apply again to retry.")
            finally:
                if not ok:
                    write_strategy_file(old_code, ss.b_spec["name"], path=ss.b_path)
            if ok:
                ss.b_rev_plan = None
                ss.b_rev_messages = []   # next revision interviews against the NEW code
            st.rerun()

# ---------------- Generated code + results ----------------
if ss.b_code:
    with st.expander(f"📄 Generated code ({ss.b_path.name if ss.b_path else 'not validated'})"):
        st.code(ss.b_code, language="python")

if ss.b_run_requested and ss.b_path is not None and strat_cls is not None:
    ss.b_run_requested = False
    with st.spinner("Backtesting…"):
        run_backtest_now()

# ---------------- Last backtest: results, replay, opt-in diagnosis ----------------
if ss.b_last is not None:
    res, m, summary, cfg, elapsed, bt_data = ss.b_last
    render_results(res, m, summary, cfg, elapsed, len(bt_data))

    with st.expander("🎬 Trade replay — step through entries & exits"):
        render_replay(res, bt_data)

    if ss.b_path is not None:
        if st.button("🩺 Diagnose with AI",
                     help="Sends a compact digest of these results + price context to Claude "
                          "so it can critique the strategy. Nothing is sent unless you click."):
            ss.b_display.append(("user", "🩺 Diagnose why the strategy is underperforming "
                                          "(backtest digest attached)"))
            ss.b_rev_messages.append({"role": "user",
                                      "content": DIAG_REQUEST + build_digest(res, m, bt_data)})
            try:
                with st.spinner("Diagnosing…"):
                    text, plan = interview.run_revision_turn(
                        client, ss.b_rev_messages, ss.b_code, spec=ss.b_spec, model=model)
                if plan is not None:
                    ss.b_rev_plan = plan
                if text:
                    ss.b_display.append(("assistant", text))
            except (anthropic.APIError, ValueError) as e:
                ss.b_rev_messages.pop()
                ss.b_display.pop()
                st.error(f"Diagnosis failed: {e} — click Diagnose again to retry.")
            else:
                st.rerun()

def _pop_dangling_turn(prompt: str) -> None:
    """Undo the just-appended user turn after an API failure so resending doesn't duplicate it."""
    for hist in (ss.b_messages, ss.b_rev_messages):
        if hist and hist[-1] == {"role": "user", "content": prompt}:
            hist.pop()
    if ss.b_display and ss.b_display[-1] == ("user", prompt):
        ss.b_display.pop()


# ---------------- Chat input ----------------
placeholder = ("Describe your strategy (e.g. 'Buy when Elder Force Index turns positive…')"
               if ss.b_path is None else
               "Describe a change — I'll confirm a plan with you before editing the code")
if prompt := st.chat_input(placeholder):
    ss.b_display.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        if ss.b_path is None:
            # interview / spec-revision mode. run_interview_turn owns ALL history
            # mutation (assistant turn + tool_result acks) — only append user turns.
            ss.b_messages.append({"role": "user", "content": prompt})
            with st.spinner("Thinking…"):
                text, spec = interview.run_interview_turn(client, ss.b_messages, model=model)
            if spec is not None:
                ss.b_spec = spec
            if text:
                ss.b_display.append(("assistant", text))
            st.rerun()
        else:
            # revision-interview mode: discuss -> plan card -> user confirms -> apply.
            # run_revision_turn owns ALL history mutation — only append user turns.
            ss.b_rev_messages.append({"role": "user", "content": prompt})
            with st.spinner("Thinking…"):
                text, plan = interview.run_revision_turn(
                    client, ss.b_rev_messages, ss.b_code, spec=ss.b_spec, model=model)
            if plan is not None:
                ss.b_rev_plan = plan
            if text:
                ss.b_display.append(("assistant", text))
            st.rerun()
    except anthropic.RateLimitError:
        _pop_dangling_turn(prompt)
        st.error("Rate limited by the API — wait a moment and resend your message.")
    except anthropic.APIStatusError as e:
        _pop_dangling_turn(prompt)
        st.error(f"Claude API error ({e.status_code}) — resend to retry.")
    except anthropic.APIConnectionError:
        _pop_dangling_turn(prompt)
        st.error("Network error reaching the Claude API — check your connection and resend.")
    except ValueError as e:
        _pop_dangling_turn(prompt)
        st.error(f"Code generation problem: {e}")
