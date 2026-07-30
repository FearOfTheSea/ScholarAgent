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
| `evaluate_scholar_gpt.py` | Scores a ScholarGPT checkpoint against six held-out application contracts. |

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
  --checkpoint data\scholar_gpt.pt --require-score 6
```
