# Scripts

Scripts provide repeatable local evaluation, benchmark, and demonstration commands. They are
development utilities and do not change application-layer behavior.

## Available Scripts

| Script | Purpose |
| --- | --- |
| `evaluate_retrieval.py` | Evaluates BGE-M3 retrieval quality against the corpus fixture. |
| `benchmark_local_runtime.py` | Benchmarks the Ollama model for latency and throughput. |
| `train_mini_gpt.py` | Demonstrates Chapters 2 & 5 concepts: tokenization, sliding window datasets, pre-training loop with AdamW, loss curves, and text generation before/after training. |
| `fine_tune_gpt2.py` | Demonstrates Chapter 6 concepts: instruction prompt formatting, loading pre-trained GPT-2 weights into the custom architecture, and running a fine-tuning loop on small QA pairs. |
| `train_scholar_gpt.py` | Localizes cached GPT-2 124M with response-only instruction tuning, validation, early stopping, and resumable checkpoints. |
| `evaluate_scholar_gpt.py` | Scores a ScholarGPT checkpoint against held-out planner, cited-material, document-map, mission-plan, explanation, assessment, and verification contracts. |
| `evaluate_missions.py` | Runs the eight deterministic Phase 1 mission scenarios and writes the redacted release-gate report. |
| `evaluate_reviews.py` | Runs the nine deterministic Phase 2 learner-model and review-memory scenarios. |

## Running the LLM-from-Scratch Demos

```bash
# Pre-training demo (runs in seconds, no GPU required)
uv run python scripts/train_mini_gpt.py

# Fine-tuning demo (downloads GPT-2 weights on first run)
uv run python scripts/fine_tune_gpt2.py
```

## Training ScholarGPT

The repository's `.venv` can run the complete offline training and evaluation
workflow:

```powershell
.\.venv\Scripts\python.exe scripts\train_scholar_gpt.py --offline
.\.venv\Scripts\python.exe scripts\evaluate_scholar_gpt.py `
  --checkpoint data\scholar_gpt.pt --require-score 8
```

The training set includes the exact mission-plan object contract, cited
explanation and assessment objects, and citation-bearing quiz and flashcard
arrays. The evaluator validates chunk IDs against the supplied held-out chunk;
it does not accept a syntactically valid but uncited response.

## Phase 1 mission evaluator

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_missions.py
```

This uses the real bounded LangGraph mission adapter with deterministic fake
capabilities. It covers first-pass proficiency, cited remediation, optional
failure continuation, save/reload/resume, explicit completion, the session
limit, version-2 migration, and ledger tampering. A non-zero exit means a
release gate failed. Report version 2 includes per-scenario evidence and
document-isolation details: applicable factual outputs must be cited to the
selected document, every recorded tool call must use that document, and every
normal scenario must verify its ledger. Non-factual scenarios are explicitly
not applicable to those checks rather than passing vacuously.

Phase 2 review memory evaluation:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_reviews.py
```

This produces a versioned report for redaction, deterministic scheduling,
confidence decay, transfer weighting, profile round trips, deletion and
session detachment, equivalence consent, and one-document review dispatch.
