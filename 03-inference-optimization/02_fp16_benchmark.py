"""
FP16 (half-precision) Whisper inference benchmark on CUDA.

Why FP16 is faster on tensor cores
------------------------------------
NVIDIA Ampere and Hopper GPUs have dedicated Tensor Core units that execute
FP16/BF16 matrix multiplications at 2× the throughput of FP32.  In Whisper's
encoder, the bulk of compute is large GEMMs (attention projections, FFN layers).
FP16 reduces the bytes fetched per weight element from 4 → 2, effectively
doubling HBM bandwidth utilization for weight loads.

What "memory-bound" means in practice
---------------------------------------
A kernel is memory-bound when the ratio of arithmetic operations to memory
accesses (arithmetic intensity) is below the hardware's "roofline" crossover
point.  For batch-size-1 transformer inference, each token requires a full
pass over the weight matrices, but very few FLOPs per byte loaded.
On an A100-80GB, the crossover is ~208 FLOPs/byte; batch-1 GEMM at typical
Whisper sequence lengths sits well below this.  Therefore latency tracks
bandwidth, not TFLOP/s — and reducing weight size (FP16, INT8) reduces
the bytes loaded, directly cutting latency.
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
BASELINE_FILE = RESULTS_DIR / "baseline_results.json"
OUTPUT_FILE = RESULTS_DIR / "fp16_results.json"

MODELS = ["tiny", "large-v3"]


def get_test_audio() -> np.ndarray:
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:1]")
    return np.array(ds[0]["audio"]["array"], dtype=np.float32)


def benchmark_fp16(model_name: str, audio: np.ndarray) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"\n[bold cyan]Benchmarking Whisper {model_name} (FP16) on {device}[/bold cyan]")

    model = whisper.load_model(model_name, device=device)
    model = model.half()  # cast all parameters to float16
    model.eval()

    torch.cuda.reset_peak_memory_stats(device)

    console.print(f"  Running {WARMUP_ITERS} warmup iterations ...")
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
        "precision": "fp16",
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "gpu_memory_mb": round(gpu_mem_mb, 1),
        "transcription_sample": result["text"].strip(),
    }


def load_baseline() -> dict[str, dict[str, Any]]:
    """Load FP32 baseline results keyed by model name."""
    if not BASELINE_FILE.exists():
        console.print(f"[yellow]Baseline file not found at {BASELINE_FILE}. Run 01_baseline_benchmark.py first.[/yellow]")
        return {}
    data = json.loads(BASELINE_FILE.read_text())
    return {r["model"]: r for r in data}


def print_comparison_table(fp16_results: list[dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> None:
    table = Table(title="FP32 vs FP16 Comparison")
    table.add_column("Model", style="bold")
    table.add_column("FP32 Mean (ms)", justify="right")
    table.add_column("FP16 Mean (ms)", justify="right")
    table.add_column("Speedup", justify="right", style="green")
    table.add_column("FP32 Mem (MB)", justify="right")
    table.add_column("FP16 Mem (MB)", justify="right")

    for r in fp16_results:
        model = r["model"]
        b = baseline.get(model, {})
        fp32_lat = b.get("mean_latency_ms", 0)
        speedup = f"{fp32_lat / r['mean_latency_ms']:.2f}×" if fp32_lat else "—"
        table.add_row(
            model,
            f"{fp32_lat:.1f}" if fp32_lat else "—",
            f"{r['mean_latency_ms']:.1f}",
            speedup,
            f"{b.get('gpu_memory_mb', 0):.0f}" if b else "—",
            f"{r['gpu_memory_mb']:.0f}",
        )
    console.print(table)


def main() -> None:
    if not torch.cuda.is_available():
        console.print("[yellow]WARNING: CUDA not available — running on CPU, timings not representative[/yellow]")

    audio = get_test_audio()
    baseline = load_baseline()
    fp16_results: list[dict[str, Any]] = []

    for model_name in MODELS:
        stats = benchmark_fp16(model_name, audio)
        fp16_results.append(stats)

    print_comparison_table(fp16_results, baseline)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(fp16_results, indent=2))
    console.print(f"\n[green]Results saved → {OUTPUT_FILE}[/green]")


if __name__ == "__main__":
    main()
