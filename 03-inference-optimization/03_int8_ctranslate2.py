"""
INT8 / mixed-precision benchmark via CTranslate2 (faster-whisper).

Compute types
-------------
float16         : FP16 weights + FP16 activations (already in 02_fp16_benchmark.py,
                  repeated here as the reference point for INT8 comparison)
int8_float16    : INT8 weights + FP16 activations — production sweet spot.
                  Weight loads are 2× faster than float16; activations stay precise.
int8            : Full INT8 — smallest memory footprint, highest risk of accuracy loss.

Why int8_float16 is the sweet spot
------------------------------------
Transformer inference at batch=1 is HBM-bandwidth-bound (weight-load dominated).
INT8 halves bytes-per-weight vs float16, cutting memory bandwidth pressure.
Accumulation and activations stay in FP16, limiting numerical error.
Typical outcome: < 0.5 pp CER/WER degradation, ~2× speedup over float16.

RTF normalisation
-----------------
RTF = mean(latency_i / duration_i) over BENCH_ITERS clips — per-clip, not averaged.

Output
------
results/int8_results.json
"""

import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")

console = Console()

WARMUP = 3
BENCH_ITERS = 5
RESULTS_DIR = Path(__file__).parent.parent / "results"
MANIFEST_ZH = Path(__file__).parent.parent / "data" / "benchmark" / "zh" / "manifest.json"
MANIFEST_EN = Path(__file__).parent.parent / "data" / "benchmark" / "en" / "manifest.json"
BASELINE_FILE = RESULTS_DIR / "baseline_results.json"
FP16_FILE = RESULTS_DIR / "fp16_results.json"

COMPUTE_TYPES = ["float16", "int8_float16", "int8"]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}\nRun 00_prepare_data.py first.")
    return json.loads(path.read_text())


def normalise_zh(text: str) -> str:
    import re
    return re.sub(r"[^一-鿿㐀-䶿＀-￯]", "", text).strip()


def normalise_en(text: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9\s]", "", text).lower().strip()


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    from jiwer import wer
    return round(wer(refs, hyps) * 100, 2)


def compute_cer(refs: list[str], hyps: list[str]) -> float:
    from jiwer import cer
    return round(cer(refs, hyps) * 100, 2)


def _reset_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _peak_gpu_mb() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    return 0.0


def per_clip_rtf(latencies_ms: list[float], manifest: list[dict]) -> float:
    durs_ms = [manifest[i]["duration_s"] * 1000 for i in range(len(latencies_ms))]
    return round(float(np.mean([lat / dur for lat, dur in zip(latencies_ms, durs_ms)])), 4)


# ---------------------------------------------------------------------------
# faster-whisper CTranslate2 benchmark
# ---------------------------------------------------------------------------

def bench_ct2(
    model_name: str,
    compute_type: str,
    manifest: list[dict],
    lang: str,
) -> dict[str, Any]:
    from faster_whisper import WhisperModel

    console.print(f"\n[bold cyan]faster-whisper {model_name} {compute_type} [{lang}][/bold cyan]")

    model = WhisperModel(model_name, device="cuda", compute_type=compute_type)
    refs_raw = [m["reference"] for m in manifest]

    for m in manifest[:WARMUP]:
        segs, _ = model.transcribe(m["audio_path"], language=lang)
        list(segs)

    _reset_gpu()
    latencies: list[float] = []
    for m in manifest[:BENCH_ITERS]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        segs, _ = model.transcribe(m["audio_path"], language=lang)
        list(segs)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    all_hyps: list[str] = []
    for m in manifest:
        segs, _ = model.transcribe(m["audio_path"], language=lang)
        all_hyps.append(" ".join(s.text for s in segs).strip())

    gpu_mb = _peak_gpu_mb()
    del model
    torch.cuda.empty_cache()

    if lang == "zh":
        error = compute_cer([normalise_zh(r) for r in refs_raw], [normalise_zh(h) for h in all_hyps])
        err_key = "CER%"
    else:
        error = compute_wer([normalise_en(r) for r in refs_raw], [normalise_en(h) for h in all_hyps])
        err_key = "WER%"

    return {
        "model": f"whisper-{model_name}",
        "backend": "faster-whisper",
        "precision": compute_type,
        "lang": lang,
        "mean_ms": round(float(np.mean(latencies)), 1),
        "p50_ms": round(float(np.percentile(latencies, 50)), 1),
        "p95_ms": round(float(np.percentile(latencies, 95)), 1),
        "gpu_mb": gpu_mb,
        "rtf": per_clip_rtf(latencies, manifest),
        err_key: error,
    }


# ---------------------------------------------------------------------------
# Comparison table (stacks baseline → fp16 → int8)
# ---------------------------------------------------------------------------

def load_prior(path: Path) -> list[dict]:
    if path.exists():
        return json.loads(path.read_text())
    return []


def speedup_str(ref_rtf: float | None, rtf: float) -> str:
    if ref_rtf and rtf:
        return f"{ref_rtf / rtf:.2f}×"
    return "—"


def print_table(int8_results: list[dict]) -> None:
    baseline = load_prior(BASELINE_FILE)
    # Use FP32 baseline RTF as the reference point for speedup calculation
    fp32_map = {(r["model"], r["lang"]): r["rtf"] for r in baseline}

    table = Table(title="INT8 Quantization — Full Precision Stack (per-clip RTF)", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Lang")
    table.add_column("Precision")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("GPU MB", justify="right")
    table.add_column("RTF", justify="right")
    table.add_column("WER/CER%", justify="right")
    table.add_column("vs FP32", justify="right", style="green")

    for r in int8_results:
        fp32_rtf = fp32_map.get((r["model"], r["lang"]))
        err = r.get("WER%") or r.get("CER%") or "—"
        table.add_row(
            r["model"], r["lang"], r["precision"],
            str(r["mean_ms"]), str(r["p95_ms"]), str(r["gpu_mb"]),
            str(r["rtf"]), str(err),
            speedup_str(fp32_rtf, r["rtf"]),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.rule("[bold]INT8 / Mixed-Precision Benchmark (CTranslate2)[/bold]")

    manifest_zh = load_manifest(MANIFEST_ZH)
    manifest_en = load_manifest(MANIFEST_EN)
    results: list[dict] = []

    for model_name in ["large-v3", "large-v3-turbo"]:
        for ct in COMPUTE_TYPES:
            results.append(bench_ct2(model_name, ct, manifest_zh, "zh"))
            results.append(bench_ct2(model_name, ct, manifest_en, "en"))

    print_table(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "int8_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    console.print(f"\n[green]Results saved → {out}[/green]")


if __name__ == "__main__":
    main()
