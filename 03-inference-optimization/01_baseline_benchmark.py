"""
Baseline FP32 Whisper inference benchmark on CUDA.

What this measures
------------------
End-to-end transcription latency for Whisper tiny and large-v3 in full
FP32 precision.  We measure wall-clock time between CUDA synchronization
points so that kernel execution — not Python overhead — is captured.

Why warmup matters
------------------
The first N CUDA kernel launches carry JIT compilation overhead (PTX → SASS)
and driver setup costs that are not representative of steady-state throughput.
Running 10 un-timed warmup iterations brings the GPU into its thermal and
clock steady state and ensures the CUDA kernel cache is populated before
measurement begins.

Why ASR inference is memory-bound
----------------------------------
Transformer inference at batch-size 1 is dominated by reading weight matrices
from HBM into CUDA cores for each matrix-vector multiply.  Arithmetic
intensity (FLOPs / bytes) is too low to saturate the tensor cores, so
throughput scales with HBM bandwidth, not FLOP/s.  This is the central
motivation for quantization: INT8 weights are half the bytes of FP16,
doubling the effective bandwidth for weight loads.
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
OUTPUT_FILE = RESULTS_DIR / "baseline_results.json"

MODELS = ["tiny", "large-v3"]


def get_test_audio() -> np.ndarray:
    """Download one audio clip from librispeech_asr_dummy and return as float32 numpy array."""
    console.print("Downloading test audio from librispeech_asr_dummy ...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:1]")
    audio: np.ndarray = np.array(ds[0]["audio"]["array"], dtype=np.float32)
    console.print(f"Audio loaded: {len(audio)/ds[0]['audio']['sampling_rate']:.2f}s")
    return audio


def benchmark_model(model_name: str, audio: np.ndarray) -> dict[str, Any]:
    """Load model in FP32, run warmup then timed iterations, return latency + memory stats."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"\n[bold cyan]Benchmarking Whisper {model_name} (FP32) on {device}[/bold cyan]")

    model = whisper.load_model(model_name, device=device)
    model.eval()

    torch.cuda.reset_peak_memory_stats(device)

    # --- Warmup: not timed ---
    console.print(f"  Running {WARMUP_ITERS} warmup iterations ...")
    for _ in range(WARMUP_ITERS):
        with torch.no_grad():
            model.transcribe(audio, language="en", fp16=False)

    # --- Benchmark ---
    console.print(f"  Running {BENCH_ITERS} benchmark iterations ...")
    latencies_ms: list[float] = []

    for i in range(BENCH_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            result = model.transcribe(audio, language="en", fp16=False)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)
        console.print(f"    iter {i+1:02d}: {latencies_ms[-1]:.1f} ms")

    gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    return {
        "model": model_name,
        "precision": "fp32",
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "gpu_memory_mb": round(gpu_mem_mb, 1),
        "transcription_sample": result["text"].strip(),
    }


def print_results_table(results: list[dict[str, Any]]) -> None:
    table = Table(title="Baseline FP32 Benchmark Results")
    table.add_column("Model", style="bold")
    table.add_column("Precision")
    table.add_column("Mean Latency (ms)", justify="right")
    table.add_column("P95 Latency (ms)", justify="right")
    table.add_column("GPU Mem (MB)", justify="right")

    for r in results:
        table.add_row(
            r["model"],
            r["precision"].upper(),
            f"{r['mean_latency_ms']:.1f}",
            f"{r['p95_latency_ms']:.1f}",
            f"{r['gpu_memory_mb']:.0f}",
        )
    console.print(table)


def main() -> None:
    if not torch.cuda.is_available():
        console.print("[yellow]WARNING: CUDA not available — running on CPU, timings not representative[/yellow]")

    audio = get_test_audio()
    results: list[dict[str, Any]] = []

    for model_name in MODELS:
        stats = benchmark_model(model_name, audio)
        results.append(stats)

    print_results_table(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2))
    console.print(f"\n[green]Results saved → {OUTPUT_FILE}[/green]")


if __name__ == "__main__":
    main()
