# Veritas

A four-agent [LangGraph](https://langchain-ai.github.io/langgraph/) pipeline that checks an LLM's answer against retrieved evidence before returning it, and declines when it cannot ground the answer.

**Live demo:** https://veritas-jsgpqg8bzqqmkftnlpbgmt.streamlit.app/

## How it works

```
question -> Retrieve -> Generate -> Critique -> Judge -> final answer (or decline)
```

| Agent | What it does | Model |
|---|---|---|
| Retrieve | Finds supporting passages with FAISS over `all-MiniLM-L6-v2` embeddings | sentence-transformers |
| Generate | Drafts an answer | gpt-oss-20b (Groq) |
| Critique | Checks the draft against the passages: `SUPPORTED` yes/no, unsupported claims, `CONFIDENCE` | gpt-oss-120b (Groq) |
| Judge | Chooses ACCEPT, REVISE (rewrite using only supported claims) or REJECT (decline) | gpt-oss-20b (Groq) |

The Critic's verdict and confidence are combined into a 0-5 **grounding score** (NO/HIGH = 0 ... YES/HIGH = 5). An optional minimum-score threshold declines any answer below it. The gate can only turn an answer into a decline, never the reverse.

Every run is traced to `logs/trace.jsonl` (one JSON line per step: node, latency, parsed critique fields; no question or answer text is stored).

## Results

Evaluated on all 800 questions of [TruthfulQA](https://github.com/sylinrl/TruthfulQA) (generation split). Answers were scored by an LLM judge against TruthfulQA's reference answers (HALLUCINATED / TRUTHFUL; a decline counts as truthful).

| | Baseline (same generator, no retrieval or checking) | Pipeline |
|---|---|---|
| Hallucination rate, all 800 questions (declines count as truthful) | 40.6% | 5.9% |
| Questions declined | 0% | 48.1% |
| Hallucination rate on the 415 questions the pipeline chose to answer | 22.7% | 11.3% |

The last row is the fairer comparison: it compares both systems on the same 415 questions rather than crediting the pipeline for declining. Even there, hallucinations roughly halve, but the pipeline answers about half as many questions.

Latency (24 logged runs, laptop calling Groq's free tier): p50 1.5 s, p95 2.3 s end to end.

### Confidence threshold sweep

Raising the minimum grounding score trades coverage for fewer unsupported answers (offline, same 800 questions):

| Minimum score | Questions answered | Pipeline hallucination rate | Baseline on the same questions |
|---|---|---|---|
| 0 (default) | 415 (51.9%) | 11.3% | 22.7% |
| 1 | 135 (16.9%) | 1.5% | 3.7% |
| 2-4 | 128 (16.0%) | 0.8% | 3.1% |
| 5 | 127 (15.9%) | 0.8% | 3.1% |

Caveats: at the stricter settings only 127-135 questions are answered, so the rates rest on a handful of items (about 1-2 hallucinations). The Critic rarely produces scores 2-4, so those rows are identical. The questions it is most confident about are also ones the baseline already gets right most of the time.

## Limitations (read these before trusting the numbers)

- **The evidence corpus is TruthfulQA's own reference answers.** Retrieval is therefore far easier than in a real setting, and the results are optimistic. For questions outside that set the Critic usually marks the answer unsupported and the pipeline declines. This is a demonstration of the mechanism, not a general fact checker.
- **"Supported by the evidence" is not the same as "true".** The pipeline will remove a correct detail if the retrieved passages do not contain it.
- **The judge is an LLM.** It is noisy, and a few dataset references are questionable. Numbers are single runs, not averaged, and have no confidence intervals.
- **Declining half the questions is a real cost.** Read the decline rate next to the hallucination rate.
- Latency was measured on a laptop against a free API tier and will vary.

## Run it locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo 'GROQ_API_KEY=your_key' > .env
streamlit run app.py
```

Needs a free [Groq](https://console.groq.com) API key. The public demo caps usage at 5 runs per session and 30 per day.

## Stack

Python, LangGraph, FAISS, sentence-transformers, Groq (gpt-oss-20b / gpt-oss-120b), Streamlit.
