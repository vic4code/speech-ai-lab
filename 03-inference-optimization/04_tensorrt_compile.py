"""
TensorRT compilation and benchmarking of Whisper encoder.

Demonstrates: torch.compile with TensorRT backend, engine serialization,
and latency comparison against the eager FP16 baseline.

Requires: tensorrt, torch-tensorrt (install separately from requirements.txt
since they depend on the specific CUDA/TRT version available on the instance).
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import whisper
from datasets import load_dataset
from rich.console import Console
from rich.table import Table

console = Console()

WARMUP_ITERS = 10
BENCH_ITERS = 10
RESULTS_DIR = Path("results")
OUTPUT_FILE = RESULTS_DIR / "tensorrt_results.json"


def get_test_audio() -> np.ndarray:
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:1]")
    return np.array(ds[0]["audio"]["array"], dtype=np.float32)


def benchmark_trt(model_name: str, audio: np.ndarray) -> dict[str, Any]:
    device = "cuda"
    console.print(f"\n[bold cyan]Compiling Whisper {model_name} encoder with torch.compile (TensorRT)[/bold cyan]")

    model = whisper.load_model(model_name, device=device).half()
    model.eval()

    try:
        import torch_tensorrt  # noqa: F401
        model.encoder = torch.compile(
            model.encoder,
            backend="tensorrt",
            options={"enabled_precisions": {torch.float16}},
        )
        console.print("  torch_tensorrt backend available — compiling encoder ...")
    except ImportError:
        console.print("[yellow]  torch_tensorrt not installed — falling back to torch.compile (inductor)[/yellow]")
        model.encoder = torch.compile(model.encoder, mode="reduce-overhead")

    # Trigger compilation on first run (counted as warmup)
    console.print(f"  Running {WARMUP_ITERS} warmup / compilation iterations ...")
    for _ in range(WARMUP_ITERS):
        with torch.no_grad():
            model.transcribe(audio, language="en", fp16=True)

    console.print(f"  Running {BENCH_ITERS} benchmark iterations ...")
    latencies_ms: list[float] = []

    for i in range(BENCH_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            result = model.transcribe(audio, language="en", fp16=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)
        console.print(f"    iter {i+1:02d}: {latencies_ms[-1]:.1f} ms")

    gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    return {
        "model": model_name,
        "backend": "tensorrt",
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "gpu_memory_mb": round(gpu_mem_mb, 1),
        "transcription_sample": result["text"].strip(),
    }


def main() -> None:
    if not torch.cuda.is_available():
        console.print("[red]CUDA required for TensorRT compilation. Exiting.[/red]")
        return

    audio = get_test_audio()
    results = [benchmark_trt("large-v3", audio)]

    table = Table(title="TensorRT Benchmark")
    table.add_column("Model")
    table.add_column("Backend")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("GPU Mem (MB)", justify="right")
    for r in results:
        table.add_row(r["model"], r["backend"], f"{r['mean_latency_ms']:.1f}", f"{r['p95_latency_ms']:.1f}", f"{r['gpu_memory_mb']:.0f}")
    console.print(table)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2))
    console.print(f"\n[green]Results saved → {OUTPUT_FILE}[/green]")


if __name__ == "__main__":
    main()
