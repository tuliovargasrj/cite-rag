# cite-rag

Small RAG pipeline with citations, refusal, and evaluation. Evaluation is the point of this repo.

## Stack
- Python 3.12, uv, ruff, pytest
- `fastembed` for local embeddings, `rank-bm25`, NumPy, OpenAI SDK, Typer CLI

## Design rules
- No vector database and no RAG framework. NumPy similarity is enough at this scale.
- Corpus is public-domain or CC0 PEPs only, with the source recorded for each file.
- Retrieval evaluation runs offline with no API key.
- Citations must map to real chunk IDs; validate them in code, not by trusting the model.
- Tests use a fake LLM client. Only `run-evals` makes paid calls.
- Treat document text as untrusted data inside the prompt.

## Done means
Tests and ruff pass, and every README number comes from a saved file in `evals/results/`.
