"""
Veritas demo: a 4-agent LangGraph pipeline (Retrieve -> Generate -> Critique -> Judge)
that checks an LLM answer against retrieved evidence and declines when it can't ground it.

Run locally (project root, venv active):   streamlit run app.py
"""

import json
import os
import random
import sys
import threading
from datetime import date

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

st.set_page_config(page_title="Veritas", page_icon="🔍", layout="centered")

DAILY_CAP = int(os.environ.get("VERITAS_DAILY_CAP", "30"))      # runs per day, all visitors
SESSION_CAP = int(os.environ.get("VERITAS_SESSION_CAP", "5"))   # runs per browser session

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Streamlit Cloud: copy secrets into environment variables (pipeline.py reads os.environ)
try:
    for _k in ("GROQ_API_KEY", "VERITAS_DAILY_CAP", "VERITAS_SESSION_CAP"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets file locally; .env is used instead

DAILY_CAP = int(os.environ.get("VERITAS_DAILY_CAP", "30"))      # runs per day, all visitors
SESSION_CAP = int(os.environ.get("VERITAS_SESSION_CAP", "5"))   # runs per browser session


@st.cache_resource
def _usage():
    return {"day": None, "count": 0, "lock": threading.Lock()}


def _consume_daily() -> bool:
    u = _usage()
    with u["lock"]:
        today = date.today().isoformat()
        if u["day"] != today:
            u["day"], u["count"] = today, 0
        if u["count"] >= DAILY_CAP:
            return False
        u["count"] += 1
        return True


@st.cache_resource(show_spinner="Loading retrieval index and embedding model...")
def _graph():
    from pipeline import build_graph
    return build_graph()


@st.cache_data
def _examples():
    path = os.path.join("data", "examples.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    path = os.path.join("data", "truthfulqa_subset.json")
    if os.path.exists(path):
        with open(path) as f:
            rows = json.load(f)
        random.Random(1).shuffle(rows)
        return [r["question"] for r in rows[:15]]
    return []


@st.cache_data
def _summary():
    path = os.path.join("results", "summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _pct(x):
    return "n/a" if x is None else f"{x:.1%}"


st.title("🔍 Veritas")
st.write(
    "Ask a factual question. Four agents answer it: **Retrieve** supporting passages, "
    "**Generate** an answer, **Critique** it against the evidence, and **Judge** whether to "
    "accept it, revise it, or decline."
)

with st.expander("What this demo can and can't do", expanded=False):
    st.markdown(
        "- The evidence index is built from **TruthfulQA's own reference answers** (800 questions). "
        "For a question outside that set, retrieval returns unrelated passages, the Critic will "
        "usually mark the answer unsupported, and the pipeline will usually **decline**. That is "
        "expected here. It is not a general-purpose fact checker.\n"
        "- Accuracy numbers come from an **LLM judge** scoring answers against TruthfulQA's "
        "reference answers. The judge is noisy, and a few dataset references are questionable.\n"
        "- The pipeline declines a large share of questions. Read the declined rate next to the "
        "hallucination rate, not instead of it."
    )

s = _summary()
if s:
    st.subheader(f"Measured on {s['n']} TruthfulQA questions")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline hallucination rate", _pct(s["baseline_rate"]))
    c2.metric("Pipeline (declines count as truthful)", _pct(s["pipeline_rate_all"]))
    c3.metric("Questions declined", _pct(s["declined_rate"]))
    st.caption(
        f"Fairer comparison on the {s['answered']} questions the pipeline chose to answer: "
        f"baseline {_pct(s['baseline_rate_same_answered'])} vs pipeline {_pct(s['pipeline_rate_answered'])}. "
        "Scored by an LLM judge."
    )

st.divider()

with st.sidebar:
    st.header("Confidence threshold")
    min_grounding = st.slider(
        "Minimum grounding score", 0, 5, 0,
        help="Combines the Critic's SUPPORTED verdict and its CONFIDENCE into 0-5 "
             "(NO/HIGH=0 ... YES/HIGH=5). Answers scoring below the minimum are declined. 0 = off.",
    )
    st.caption(
        "0 = trust the Judge as is. Higher = stricter: fewer answers, fewer unsupported ones. "
        "The offline sweep shows the trade-off on the measured set."
    )
    st.divider()
    st.caption(f"Public demo limits: {SESSION_CAP} runs per session, {DAILY_CAP} per day for everyone.")

examples = _examples()
choice = st.selectbox("Try an example question", ["(type your own)"] + examples)
default_q = "" if choice == "(type your own)" else choice
question = st.text_input("Question", value=default_q, max_chars=300,
                         placeholder="e.g. What happens if you eat watermelon seeds?")
go = st.button("Run pipeline", type="primary", disabled=not question.strip())

if go:
    if not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not set. Add it to .env locally, or under Secrets in the app settings.")
        st.stop()
    st.session_state.setdefault("runs", 0)
    if st.session_state["runs"] >= SESSION_CAP:
        st.warning("Session limit reached. Reload the page to start a new session.")
        st.stop()
    if not _consume_daily():
        st.warning("The daily demo limit has been reached. Please try again tomorrow.")
        st.stop()
    st.session_state["runs"] += 1

    from thresholds import parse_critique
    from traced_run import run_traced

    graph = _graph()
    TITLES = {"retrieve": "1 · Retrieve", "generate": "2 · Generate",
              "critique": "3 · Critique", "judge": "4 · Judge"}

    def draw(node, update, ms):
        with st.expander(f"{TITLES.get(node, node)}  ·  {ms:.0f} ms", expanded=True):
            if node == "retrieve":
                for p in update.get("retrieved_context") or []:
                    st.markdown(f"> {p[:300]}{'...' if len(p) > 300 else ''}")
            elif node == "generate":
                st.write(update.get("generated_answer") or "(empty answer)")
                st.caption("Unchecked model answer.")
            elif node == "critique":
                p = parse_critique(update.get("critique_raw"))
                st.code(update.get("critique_raw") or "(empty)", language=None)
                st.caption(f"Parsed: supported={p['supported']}, confidence={p['confidence']}, "
                           f"grounding score={p['grounding_score']}")
            elif node == "judge":
                st.markdown(f"**Action:** `{update.get('judge_action')}`")
                st.write(update.get("final_answer") or "(empty)")

    try:
        with st.spinner("Running the four agents..."):
            out = run_traced(graph, question.strip(), min_grounding=min_grounding,
                             log_path=os.environ.get("VERITAS_LOG", "logs/trace.jsonl"), on_step=draw)
    except Exception as e:
        msg = str(e)
        if "429" in msg or "rate_limit" in msg.lower():
            st.error("The model provider's rate limit was reached. Please try again later.")
        else:
            st.error(f"The pipeline failed: {type(e).__name__}")
        st.stop()

    d = out["decision"]
    with st.container(border=True):
        st.subheader("Final answer")
        st.write(out["final_answer"] or "(empty)")
        if d["gated"]:
            st.warning(f"Declined by the confidence threshold: {d['reason']}.")
        st.caption(f"Action: {d['action']}  ·  grounding score: {d['grounding_score']}  ·  "
                   f"total {out['total_ms']:.0f} ms  ·  run {out['run_id']}")