# Local Benchmark

The benchmark measures the default `qwen3:1.7b` model on the local Ollama
runtime with a 2048-token context, one warm-up, and three non-streaming
generation samples. It records wall-clock latency, Ollama-reported generation
throughput, aggregate Ollama-process RSS, and GPU memory when `nvidia-smi` is
available.

```bash
uv run --group dev python scripts/benchmark_local_runtime.py
```

The raw result is written to `data/benchmarks/qwen3_1.7b.json`. The benchmark
does not call a paid API or send prompts outside `OLLAMA_URL`.

## Latest result

Recorded on 2026-07-26 with `qwen3:1.7b`, a 2048-token context, one warm-up,
and three 128-token samples:

| Measure | Result |
| --- | --- |
| Median wall-clock response time | 3.57 s |
| Median generation throughput | 38.68 tokens/s |
| Ollama-reported loaded-model VRAM | 1397.72 MiB |
| Ollama-process RSS before / after | 115.14 / 115.55 MiB |
| GPU telemetry via `nvidia-smi` | Unavailable on this installation |

The VRAM value comes from Ollama's local process-status API. The benchmark
excludes its initial model-load warm-up and uses no remote inference service.
