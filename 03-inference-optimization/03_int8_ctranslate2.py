"""
INT8 and mixed-precision benchmarks using faster-whisper (CTranslate2 backend).

How CTranslate2 differs from TensorRT
---------------------------------------
CTranslate2 is a framework-agnostic, CPU + CUDA inference engine that applies
static quantization (INT8) and operator fusion at the graph level.  It converts
models offline to its own binary format and executes fused kernels at runtime.
It ships as a Python wheel — zero NVIDIA toolchain setup.

TensorRT is NVIDIA's proprietary compiler that takes a neural network graph and
emits highly optimized CUDA kernels tuned to the exact GPU architecture (e.g.,
A100 SM80).  It applies layer fusion, kernel auto-tuning, and can use FP8 on
Hopper.  Build times are long (minutes) but produced engines are fastest on
the target GPU.

What operator fusion does
--------------------------
Fusing, say, LayerNorm + Projection + GELU into one kernel eliminates multiple
round-trips to HBM (write result → read it back for the next op) and reduces
kernel launch overhead.  This is especially impactful for memory-bound,
sequence-length-bounded transformer inference.

When to use which
-----------------
CTranslate2 → rapid deployment, CPU/edge, no NVIDIA dependency required.
TensorRT    → maximum throughput on a fixed GPU architecture, production serving.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from faster_whisper import WhisperModel
from jiwer import cer
from rich.console import Console
from rich.table import Table

console = Console()

WARMUP_ITERS = 10
BENCH_ITERS = 10
RESULTS_DIR = Path("results")
BASELINE_FILE = RESULTS_DIR / "baseline_results.json"
FP16_FILE = RESULTS_DIR / "fp16_results.json"
OUTPUT_FILE = RESULTS_DIR / "ctranslate2_results.json"

COMPUTE_TYPES = ["float16", "int8_float16", "int8"]
MODEL_NAME = "large-v3"  # benchmark on the full model only


def get_test_audio_with_reference() -> tuple[np.ndarray, int, str]:
    """Return audio array, sample rate, and ground-truth transcript."""
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:1]")
    sample = ds[0]
    audio = np.array(sample["audio"]["array"], dtype=np.float32)
    sr: int = sample["audio"]["sampling_rate"]
    reference: str = sample["text"].strip().lower()
    return audio, sr, reference


def benchmark_compute_type(
    compute_type: str,
    audio: np.ndarray,
    sr: int,
    reference: str,
) -> dict[str, Any]:
    console.print(f"\n[bold cyan]Benchmarking faster-whisper {MODEL_NAME} ({compute_type})[/bold cyan]")

    model = WhisperModel(MODEL_NAME, device="cuda", compute_type=compute_type)

    # Warmup
    console.print(f"  Running {WARMUP_ITERS} warmup iterations ...")
    for _ in range(WARMUP_ITERS):
        segments, _ = model.transcribe(audio, language="en")
        list(segments)  # exhaust generator

    # Benchmark
    console.print(f"  Running {BENCH_ITERS} benchmark iterations ...")
    latencies_ms: list[float] = []
    final_text = ""

    for i in range(BENCH_ITERS):
        t0 = time.perf_counter()
        segments, _ = model.transcribe(audio, language="en")
        final_text = " ".join(s.text for s in segments).strip().lower()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)
        console.print(f"    iter {i+1:02d}: {latencies_ms[-1]:.1f} ms")

    char_error_rate = cer([reference], [final_text])

    # faster-whisper does not expose torch GPU memory stats directly
    try:
        import torch
        gpu_mem_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        gpu_mem_mb = -1.0

    return {
        "model": MODEL_NAME,
        "backend": "ctranslate2",
        "compute_type": compute_type,
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 2),
        "p50_latency_ms": round(float(np.percentile(latencies_ms, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "gpu_memory_mb": gpu_mem_mb,
        "cer": round(float(char_error_rate), 4),
    }


def load_prior_results() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    baseline = fp16 = None
    if BASELINE_FILE.exists():
        data = json.loads(BASELINE_FILE.read_text())
        baseline = next((r for r in data if r["model"] == MODEL_NAME), None)
    if FP16_FILE.exists():
        data = json.loads(FP16_FILE.read_text())
        fp16 = next((r for r in data if r["model"] == MODEL_NAME), None)
    return baseline, fp16


def print_full_comparison(
    ct2_results: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
    fp16: dict[str, Any] | None,
) -> None:
    fp32_lat = baseline["mean_latency_ms"] if baseline else None

    table = Table(title=f"Full Precision Comparison — {MODEL_NAME}")
    table.add_column("Backend / Precision", style="bold")
    table.add_column("P50 (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("GPU Mem (MB)", justify="right")
    table.add_column("CER", justify="right")
    table.add_column("Speedup vs FP32", justify="right", style="green")

    def speedup(lat_ms: float) -> str:
        if fp32_lat:
            return f"{fp32_lat / lat_ms:.2f}×"
        return "—"

    def add_row(label: str, r: dict[str, Any], p50_key: str = "mean_latency_ms") -> None:
        table.add_row(
            label,
            f"{r.get('p50_latency_ms', r.get(p50_key, 0)):.1f}",
            f"{r.get('p95_latency_ms', 0):.1f}",
            f"{r.get('gpu_memory_mb', 0):.0f}",
            f"{r.get('cer', '—')}" if "cer" in r else "—",
            speedup(r["mean_latency_ms"]),
        )

    if baseline:
        add_row("PyTorch FP32", baseline)
    if fp16:
        add_row("PyTorch FP16", fp16)
    for r in ct2_results:
        add_row(f"CT2 {r['compute_type']}", r)

    console.print(table)


def main() -> None:
    audio, sr, reference = get_test_audio_with_reference()
    console.print(f"Reference transcript: [italic]{reference[:80]}...[/italic]")

    ct2_results: list[dict[str, Any]] = []
    for compute_type in COMPUTE_TYPES:
        stats = benchmark_compute_type(compute_type, audio, sr, reference)
        ct2_results.append(stats)

    baseline, fp16 = load_prior_results()
    print_full_comparison(ct2_results, baseline, fp16)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(ct2_results, indent=2))
    console.print(f"\n[green]Results saved → {OUTPUT_FILE}[/green]")


if __name__ == "__main__":
    main()
