"""
Strong-scaling experiment: measure latency as GPU count increases from 1 to N.

Strong scaling: fixed problem size (one audio clip), increasing GPU count.
Ideal: latency halves with each doubling of GPUs.
Real: communication overhead from all-reduce limits practical scaling.

Amdahl's Law: speedup ≤ 1 / (s + (1-s)/N) where s = serial fraction.
For transformer inference, the serial fraction includes tokenization,
beam search, and cross-GPU synchronization — typically 10–20%.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from rich.console import Console
from rich.table import Table

console = Console()
RESULTS_DIR = Path("results")


def run_single_gpu(model_name: str, audio: np.ndarray, num_runs: int) -> float:
    """Baseline single-GPU latency (mean ms)."""
    import whisper
    device = "cuda:0"
    model = whisper.load_model(model_name, device=device).half()
    model.eval()

    latencies: list[float] = []
    for _ in range(num_runs):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.transcribe(audio, language="en", fp16=True)
        torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(latencies))


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaling experiment")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--max-gpus", type=int, default=4)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:1]")
    audio = np.array(ds[0]["audio"]["array"], dtype=np.float32)

    available = torch.cuda.device_count()
    gpu_counts = [2**i for i in range(int(np.log2(min(args.max_gpus, available))) + 1)]

    results: list[dict] = []
    baseline_ms = None

    for n in gpu_counts:
        if n == 1:
            lat = run_single_gpu(args.model, audio, args.runs)
        else:
            console.print(f"[yellow]Tensor-parallel on {n} GPUs — run tensor_parallel.py for multi-GPU results[/yellow]")
            lat = None  # placeholder; real measurement via tensor_parallel.py

        if baseline_ms is None and lat is not None:
            baseline_ms = lat

        speedup = f"{baseline_ms / lat:.2f}×" if (lat and baseline_ms) else "—"
        results.append({"gpus": n, "mean_latency_ms": lat, "speedup": speedup})
        console.print(f"  {n} GPU(s): {lat:.1f} ms | speedup {speedup}" if lat else f"  {n} GPU(s): — (multi-GPU, see tensor_parallel.py)")

    table = Table(title=f"Scaling Experiment — {args.model}")
    table.add_column("GPUs", justify="right")
    table.add_column("Mean Latency (ms)", justify="right")
    table.add_column("Speedup", justify="right")
    for r in results:
        table.add_row(str(r["gpus"]), f"{r['mean_latency_ms']:.1f}" if r["mean_latency_ms"] else "—", r["speedup"])
    console.print(table)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "scaling_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
