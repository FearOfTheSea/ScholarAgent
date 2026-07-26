"""Benchmark the configured local Ollama model without cloud inference."""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil

ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT_DIRECTORY / "data" / "benchmarks" / "qwen3_1.7b.json"
DEFAULT_PROMPT = (
    "In one sentence, explain why validation data is used in machine learning."
)


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Memory visible from the benchmark process."""

    ollama_rss_mib: float | None
    gpu_memory_mib: int | None
    gpu_telemetry_available: bool


@dataclass(frozen=True, slots=True)
class Sample:
    """One non-streaming local generation measurement."""

    wall_seconds: float
    generated_tokens: int | None
    evaluation_seconds: float | None
    tokens_per_second: float | None


def main() -> None:
    """Run warm-up and measured local Ollama generation requests."""
    arguments = _parse_arguments()
    with httpx.Client(base_url=arguments.base_url.rstrip("/"), timeout=120.0) as client:
        _require_model(client, arguments.model)
        for _ in range(arguments.warmups):
            _generate(client, arguments)

        model_vram_mib = _model_vram_mib(client, arguments.model)
        memory_before = _memory_snapshot()
        samples = tuple(_generate(client, arguments) for _ in range(arguments.samples))
        memory_after = _memory_snapshot()

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": arguments.model,
        "base_url": arguments.base_url,
        "context_length": arguments.context_length,
        "maximum_tokens": arguments.maximum_tokens,
        "warmups": arguments.warmups,
        "samples": [asdict(sample) for sample in samples],
        "median_wall_seconds": statistics.median(
            sample.wall_seconds for sample in samples
        ),
        "median_tokens_per_second": _median_tokens_per_second(samples),
        "ollama_reported_model_vram_mib": model_vram_mib,
        "memory_before": asdict(memory_before),
        "memory_after": asdict(memory_after),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "qwen3:1.7b"))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--maximum-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _require_model(client: httpx.Client, model_name: str) -> None:
    try:
        response = client.get("/api/tags")
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise RuntimeError("The local Ollama service is unavailable.") from error
    payload = _payload(response)
    models = payload.get("models")
    if not isinstance(models, list) or not any(
        isinstance(model, dict) and model.get("name") == model_name for model in models
    ):
        raise RuntimeError(f"The local Ollama model '{model_name}' is unavailable.")


def _generate(client: httpx.Client, arguments: argparse.Namespace) -> Sample:
    started_at = time.perf_counter()
    try:
        response = client.post(
            "/api/generate",
            json={
                "model": arguments.model,
                "prompt": arguments.prompt,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "num_ctx": arguments.context_length,
                    "num_predict": arguments.maximum_tokens,
                    "temperature": 0.2,
                },
            },
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise RuntimeError("The local Ollama generation request failed.") from error
    payload = _payload(response)
    wall_seconds = time.perf_counter() - started_at
    generated_tokens = _optional_int(payload.get("eval_count"))
    evaluation_nanoseconds = _optional_int(payload.get("eval_duration"))
    evaluation_seconds = (
        evaluation_nanoseconds / 1_000_000_000
        if evaluation_nanoseconds is not None
        else None
    )
    tokens_per_second = (
        generated_tokens / evaluation_seconds
        if generated_tokens is not None and evaluation_seconds not in (None, 0.0)
        else None
    )
    return Sample(
        wall_seconds=wall_seconds,
        generated_tokens=generated_tokens,
        evaluation_seconds=evaluation_seconds,
        tokens_per_second=tokens_per_second,
    )


def _model_vram_mib(client: httpx.Client, model_name: str) -> float | None:
    """Return Ollama's loaded-model VRAM value when its local API supplies one."""
    try:
        response = client.get("/api/ps")
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    models = _payload(response).get("models")
    if not isinstance(models, list):
        return None
    for model in models:
        if not isinstance(model, dict) or model.get("name") != model_name:
            continue
        size_vram = _optional_int(model.get("size_vram"))
        return round(size_vram / (1024 * 1024), 2) if size_vram is not None else None
    return None


def _memory_snapshot() -> MemorySnapshot:
    ollama_rss_bytes = 0
    for process in psutil.process_iter(["name", "memory_info"]):
        try:
            name = process.info["name"]
            memory_info = process.info["memory_info"]
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if isinstance(name, str) and name.lower().startswith("ollama") and memory_info:
            ollama_rss_bytes += memory_info.rss
    gpu_memory_mib = _gpu_memory_mib()
    return MemorySnapshot(
        ollama_rss_mib=round(ollama_rss_bytes / (1024 * 1024), 2)
        if ollama_rss_bytes
        else None,
        gpu_memory_mib=gpu_memory_mib,
        gpu_telemetry_available=gpu_memory_mib is not None,
    )


def _gpu_memory_mib() -> int | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        default_windows_path = Path(
            "C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe"
        )
        executable = (
            str(default_windows_path) if default_windows_path.exists() else None
        )
    if executable is None:
        return None
    completed = subprocess.run(
        [executable, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        return None
    values = [
        int(value.strip()) for value in completed.stdout.splitlines() if value.strip()
    ]
    return max(values) if values else None


def _payload(response: httpx.Response) -> dict[str, Any]:
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Ollama returned an invalid response payload.")
    return payload


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _median_tokens_per_second(samples: tuple[Sample, ...]) -> float | None:
    values = [
        sample.tokens_per_second for sample in samples if sample.tokens_per_second
    ]
    return statistics.median(values) if values else None


if __name__ == "__main__":
    main()
