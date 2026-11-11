"""
Day 12-13: Full pipeline wired as an actual LangGraph StateGraph.

Nodes: retrieve -> generate -> critique -> judge
State is shared and passed between nodes explicitly, which is what
makes this a real LangGraph pipeline rather than sequential scripts.
"""

import json
import re
import os
import time
from typing import TypedDict, List, Optional

import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm
from langgraph.graph import StateGraph, END

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
GENERATOR_MODEL = "openai/gpt-oss-20b"
CRITIC_MODEL = "openai/gpt-oss-120b"
JUDGE_MODEL = "openai/gpt-oss-20b"

CORPUS_PATH = "data/corpus.json"
INDEX_PATH = "data/faiss_index.bin"
INPUT_PATH = "data/truthfulqa_subset.json"
OUTPUT_PATH = "results/langgraph_pipeline_output.json"

TOP_K = 3

# --- Load retrieval assets once ---
print("Loading corpus, FAISS index, embedding model...")
with open(CORPUS_PATH) as f:
    corpus = json.load(f)
faiss_index = faiss.read_index(INDEX_PATH)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# --- Shared state passed between graph nodes ---
class PipelineState(TypedDict):
    question: str
    best_answer: str
    correct_answers: List[str]
    incorrect_answers: List[str]
    retrieved_context: Optional[List[str]]
    generated_answer: Optional[str]
    critique_raw: Optional[str]
    critique_supported: Optional[bool]
    judge_action: Optional[str]
    final_answer: Optional[str]


# --- Node: Retriever ---
def retrieve_node(state: PipelineState) -> PipelineState:
    q_embedding = embed_model.encode([state["question"]], convert_to_numpy=True)
    _, indices = faiss_index.search(q_embedding, TOP_K)
    retrieved = [corpus[i]["passage"] for i in indices[0]]
    return {**state, "retrieved_context": retrieved}


# --- Node: Generator ---
def generate_node(state: PipelineState) -> PipelineState:
    resp = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {"role": "system", "content": "Answer the question concisely and factually in 1-2 sentences."},
            {"role": "user", "content": state["question"]},
        ],
        temperature=0.0,
        max_completion_tokens=1500,
        reasoning_effort="low",
    )
    answer = resp.choices[0].message.content.strip()
    return {**state, "generated_answer": answer}


# --- Node: Critic ---
CRITIC_PROMPT = """You are a strict fact-checking Critic. Your job is to check whether
a generated answer is actually SUPPORTED by the given retrieved context — not whether
it sounds plausible, and not using your own outside knowledge.

Question: {question}

Retrieved context (this is the ONLY source you may use to judge support):
{context}

Generated answer to check: {generated_answer}

Carefully check every factual claim in the generated answer against the retrieved context.

Respond in this exact format:
SUPPORTED: <YES or NO>
UNSUPPORTED_CLAIMS: <list any specific claims not backed by the context, or "None">
CONFIDENCE: <LOW, MEDIUM, or HIGH>"""


def critique_node(state: PipelineState) -> PipelineState:
    context_str = "\n".join(f"- {c}" for c in state["retrieved_context"])
    prompt = CRITIC_PROMPT.format(
        question=state["question"],
        context=context_str,
        generated_answer=state["generated_answer"],
    )
    resp = client.chat.completions.create(
        model=CRITIC_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_completion_tokens=600,
        reasoning_effort="low",
    )
    raw = resp.choices[0].message.content.strip()
    raw_upper = raw.upper()
    supported = "SUPPORTED: YES" in raw_upper
    return {**state, "critique_raw": raw, "critique_supported": supported}


# --- Node: Judge ---
JUDGE_PROMPT = """You are the final Judge in a fact-checking pipeline. You receive a
generated answer along with a Critic's assessment of whether it's grounded in retrieved
context.

Question: {question}

Generated answer: {generated_answer}

Critic's assessment:
{critique}

Decide one of three actions:
- ACCEPT: the answer is well-grounded, keep it as-is
- REVISE: the answer has some unsupported claims mixed with valid content — rewrite it
  to include ONLY the grounded, supported information, removing unsupported additions
- REJECT: the answer is fundamentally unreliable — respond with "I don't have enough
  grounded information to answer this confidently."

Respond in this exact format:
ACTION: <ACCEPT, REVISE, or REJECT>
FINAL_ANSWER: <the final answer text after applying your action>"""


def judge_node(state: PipelineState) -> PipelineState:
    prompt = JUDGE_PROMPT.format(
        question=state["question"],
        generated_answer=state["generated_answer"],
        critique=state["critique_raw"],
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_completion_tokens=1500,
        reasoning_effort="low",
    )
    raw = resp.choices[0].message.content.strip()

    m = re.search(r"ACTION\W*(ACCEPT|REVISE|REJECT)", raw, re.IGNORECASE)
    action = m.group(1).upper() if m else "UNKNOWN"
    if action == "UNKNOWN":
        print(f"\nUNKNOWN judge output for '{state['question'][:50]}': {raw[:300]!r}")

    final_answer = state["generated_answer"]
    fm = re.search(r"FINAL_ANSWER\W*(.+)", raw, re.IGNORECASE | re.DOTALL)
    if fm:
        final_answer = fm.group(1).strip()

    return {**state, "judge_action": action, "final_answer": final_answer}


# --- Build the graph ---
def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critique", critique_node)
    graph.add_node("judge", judge_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critique")
    graph.add_edge("critique", "judge")
    graph.add_edge("judge", END)

    return graph.compile()


def main():
    with open(INPUT_PATH) as f:
        data = json.load(f)

    app = build_graph()

    # Resume support
    existing = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            prev = json.load(f)
        existing = {r["question"]: r for r in prev if r.get("final_answer")}
        print(f"Resuming: {len(existing)} already processed.")

    results = []
    for row in tqdm(data, desc="Running LangGraph pipeline"):
        if row["question"] in existing:
            results.append(existing[row["question"]])
            continue

        initial_state: PipelineState = {
            "question": row["question"],
            "best_answer": row["best_answer"],
            "correct_answers": row["correct_answers"],
            "incorrect_answers": row["incorrect_answers"],
            "retrieved_context": None,
            "generated_answer": None,
            "critique_raw": None,
            "critique_supported": None,
            "judge_action": None,
            "final_answer": None,
        }

        try:
            final_state = app.invoke(initial_state)
            results.append(final_state)
        except Exception as e:
            print(f"Error on '{row['question'][:50]}...': {e}")
            results.append({**initial_state, "final_answer": None})

        # periodic save so a crash doesn't lose everything
        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

        time.sleep(0.2)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {OUTPUT_PATH}")
    from collections import Counter
    actions = [r.get("judge_action") for r in results if r.get("judge_action")]
    print(f"Action distribution: {Counter(actions)}")


if __name__ == "__main__":
    main()