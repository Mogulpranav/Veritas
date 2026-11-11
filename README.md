# Veritas

A four-agent LangGraph pipeline (Retrieve, Generate, Critique, Judge) that checks an LLM
answer against retrieved evidence and declines when it cannot ground it.

- Retrieve: FAISS + all-MiniLM-L6-v2 over a small evidence corpus
- Generate: gpt-oss-20b (Groq)
- Critique: gpt-oss-120b (Groq), scores how well the evidence supports the answer
- Judge: gpt-oss-20b, accepts / revises / rejects

The evidence index is built from TruthfulQA's own reference answers, so this demonstrates the
mechanism and is not a general fact checker. Accuracy figures come from an LLM judge and are
shown in the app. The app needs a `GROQ_API_KEY` secret to run.
