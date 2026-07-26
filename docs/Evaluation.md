# Evaluation

The committed corpus in `tests/fixtures/evaluation/` contains owner-supplied
English Machine Learning lectures. Its manifest pins each PDF's SHA-256 hash,
page count, retrieval question, and expected citation page.

The ordinary test suite evaluates the corpus with deterministic lexical vectors
to keep it offline, reproducible, and free of model downloads:

```bash
uv run pytest tests/test_evaluation_corpus.py
```

Run semantic retrieval with the configured local BGE-M3 model after its first
download:

```bash
uv run --group dev python scripts/evaluate_retrieval.py
```

The command writes its detailed citations and pass/fail result to
`data/evaluation/retrieval.json`. A passing case returns at least one retrieved
chunk from the expected source page within `RETRIEVAL_TOP_K` results.

## Latest local result

The BGE-M3 CPU evaluation completed on 2026-07-26 with all 6 expected
document-page citations retrieved (`6 passed`, `0 failed`).
