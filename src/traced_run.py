"""
Structured tracing for the Veritas LangGraph pipeline.

Wraps the graph from pipeline.py, runs it one node at a time, and writes one JSON
line per step (plus one summary line per run) to logs/trace.jsonl:

    {"type":"step","run_id":"..","node":"critique","latency_ms":812.4,"grounding_score":4,...}
    {"type":"run","run_id":"..","total_ms":2310.7,"action":"REVISE","gated":false,...}

Usage (from the project root, venv active):
    python src/traced_run.py --question "What happens if you eat watermelon seeds?"
    python src/traced_run.py --batch 10               # first 10 questions of the subset
    python src/traced_run.py --batch 10 --min-grounding 4
    python src/traced_run.py --summary                # latency + action stats from the log

Token counts are not logged because the pipeline nodes don't return them.
"""

import argparse
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thresholds import DECLINE_TEXT, gate, parse_critique  # noqa: E402

LOG_PATH = os.environ.get("VERITAS_LOG", "logs/trace.jsonl")


def make_state(question: str, row: dict = None) -> dict:
    """Initial PipelineState. Reference fields are only filled for dataset questions."""
    row = row or {}
    return {
        "question": question,
        "best_answer": row.get("best_answer", ""),
        "correct_answers": row.get("correct_answers", []),
        "incorrect_answers": row.get("incorrect_answers", []),
        "retrieved_context": None,
        "generated_answer": None,
        "critique_raw": None,
        "critique_supported": None,
        "judge_action": None,
        "final_answer": None,
    }


def _write(log_path: str, record: dict) -> None:
    if not log_path:
        return
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _step_fields(node: str, update: dict) -> dict:
    """The few fields worth logging per node (no full text blobs)."""
    if node == "retrieve":
        ctx = update.get("retrieved_context") or []
        return {"n_passages": len(ctx)}
    if node == "generate":
        return {"answer_chars": len(update.get("generated_answer") or "")}
    if node == "critique":
        p = parse_critique(update.get("critique_raw"))
        return {"supported": p["supported"], "confidence": p["confidence"],
                "grounding_score": p["grounding_score"]}
    if node == "judge":
        return {"action": update.get("judge_action"),
                "answer_chars": len(update.get("final_answer") or "")}
    return {}


def run_traced(app, question: str, row: dict = None, min_grounding: int = 0,
               log_path: str = LOG_PATH, on_step=None) -> dict:
    """Run the compiled graph on one question, logging every step.

    on_step(node, update, latency_ms) is called after each node finishes (used by the
    Streamlit app to draw the trace live). Returns a dict with the final state, the
    gate decision and timings. Raises whatever the graph raised, after logging it.
    """
    run_id = uuid.uuid4().hex[:12]
    t_start = last = time.perf_counter()
    state = make_state(question, row)

    try:
        for chunk in app.stream(state, stream_mode="updates"):
            for node, update in chunk.items():
                now = time.perf_counter()
                latency_ms = round((now - last) * 1000, 1)
                last = now
                state.update(update)
                _write(log_path, {
                    "type": "step", "run_id": run_id,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "node": node, "latency_ms": latency_ms,
                    **_step_fields(node, update),
                })
                if on_step:
                    on_step(node, update, latency_ms)
    except Exception as e:
        _write(log_path, {
            "type": "run", "run_id": run_id, "ts": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(e).__name__}: {str(e)[:300]}",
            "total_ms": round((time.perf_counter() - t_start) * 1000, 1),
            "question_chars": len(question),
        })
        raise

    decision = gate(state.get("judge_action") or "UNKNOWN", state.get("critique_raw"), min_grounding)
    final_answer = DECLINE_TEXT if decision["gated"] else state.get("final_answer")
    total_ms = round((time.perf_counter() - t_start) * 1000, 1)

    _write(log_path, {
        "type": "run", "run_id": run_id, "ts": datetime.now(timezone.utc).isoformat(),
        "total_ms": total_ms, "judge_action": state.get("judge_action"),
        "action": decision["action"], "gated": decision["gated"],
        "gate_reason": decision["reason"], "min_grounding": min_grounding,
        "grounding_score": decision["grounding_score"], "question_chars": len(question),
    })
    return {"run_id": run_id, "state": state, "decision": decision,
            "final_answer": final_answer, "total_ms": total_ms}


def _pct(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def summarize(log_path: str = LOG_PATH) -> None:
    if not os.path.exists(log_path):
        print(f"No log at {log_path}. Run with --batch or --question first.")
        return
    steps, runs = {}, []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") == "step":
                steps.setdefault(r["node"], []).append(r["latency_ms"])
            elif r.get("type") == "run":
                runs.append(r)

    ok = [r for r in runs if "error" not in r]
    failed = len(runs) - len(ok)
    print(f"Runs: {len(runs)}  ok: {len(ok)}  failed: {failed}")
    print(f"\n{'node':<10} {'n':>5} {'p50 ms':>9} {'p95 ms':>9}")
    for node in ("retrieve", "generate", "critique", "judge"):
        v = steps.get(node, [])
        if v:
            print(f"{node:<10} {len(v):>5} {_pct(v, .5):>9.0f} {_pct(v, .95):>9.0f}")
    totals = [r["total_ms"] for r in ok]
    if totals:
        print(f"{'total':<10} {len(totals):>5} {_pct(totals, .5):>9.0f} {_pct(totals, .95):>9.0f}")
    acts = {}
    for r in ok:
        acts[r["action"]] = acts.get(r["action"], 0) + 1
    print("\nFinal actions:", acts)
    print("Gated by threshold:", sum(1 for r in ok if r.get("gated")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--question", help="run one question")
    ap.add_argument("--batch", type=int, help="run N questions from data/truthfulqa_subset.json")
    ap.add_argument("--seed", type=int, default=0, help="random seed for --batch sampling")
    ap.add_argument("--min-grounding", type=int, default=0, help="0-5; 0 disables the gate")
    ap.add_argument("--log", default=LOG_PATH)
    ap.add_argument("--summary", action="store_true", help="print stats from the log and exit")
    args = ap.parse_args()

    if args.summary:
        summarize(args.log)
        return
    if not args.question and not args.batch:
        ap.error("give --question, --batch N, or --summary")

    from pipeline import build_graph  # loads FAISS + embedding model; needs GROQ_API_KEY

    graph = build_graph()
    jobs = []
    if args.question:
        jobs.append((args.question, None))
    if args.batch:
        with open("data/truthfulqa_subset.json") as f:
            rows = json.load(f)
        random.Random(args.seed).shuffle(rows)
        jobs += [(r["question"], r) for r in rows[: args.batch]]

    for i, (q, row) in enumerate(jobs, 1):
        try:
            out = run_traced(graph, q, row, args.min_grounding, args.log)
            print(f"[{i}/{len(jobs)}] {out['decision']['action']:<7} {out['total_ms']:>7.0f} ms  {q[:60]}")
        except Exception as e:
            print(f"[{i}/{len(jobs)}] FAILED  {type(e).__name__}: {str(e)[:120]}")
            if "429" in str(e) or "rate_limit" in str(e).lower():
                print("Rate limit reached. Stopping; rerun later.")
                break
    print(f"\nLog written to {args.log}.  Run: python src/traced_run.py --summary")


if __name__ == "__main__":
    main()
