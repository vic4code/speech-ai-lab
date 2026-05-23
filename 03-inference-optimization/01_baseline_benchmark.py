"""
Baseline FP32 inference benchmark.

Models tested
-------------
zh (Chinese, Whisper-only — Parakeet is English-only):
  - openai/whisper-large-v3          FP32, openai-whisper
  - openai/whisper-large-v3-turbo    FP32, openai-whisper

en (English, head-to-head):
  - openai/whisper-large-v3          FP32, openai-whisper
  - openai/whisper-large-v3-turbo    FP32, openai-whisper
  - nvidia/parakeet-tdt-1.1b         FP32, NeMo

Metrics
-------
  Mean / P50 / P95 latency (ms), GPU memory peak (MB), WER/CER, RTF
  WER for English, CER for Chinese.
  RTF = mean_latency_ms / audio_duration_ms

Output
------
  results/baseline_results.json
"""

import json
import re
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


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}\n"
            "Run: python 00_prepare_data.py"
        )
    return json.loads(path.read_text())


def load_audio_numpy(audio_path: str, target_sr: int = 16_000) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Normalisation for WER/CER
# ---------------------------------------------------------------------------

def normalise_zh(text: str) -> str:
    """Strip spaces and punctuation for Chinese CER."""
    text = re.sub(r"[^一-鿿㐀-䶿＀-￯]", "", text)
    return text.strip()


def normalise_en(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", "", text).lower().strip()


# ---------------------------------------------------------------------------
# WER / CER helpers
# ---------------------------------------------------------------------------

def compute_wer(refs: list[str], hyps: list[str]) -> float:
    from jiwer import wer
    return round(wer(refs, hyps) * 100, 2)


def compute_cer(refs: list[str], hyps: list[str]) -> float:
    from jiwer import cer
    return round(cer(refs, hyps) * 100, 2)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _reset_gpu(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _peak_gpu_mb(device: str) -> float:
    if device == "cuda":
        return round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    return 0.0


# ---------------------------------------------------------------------------
# Whisper (openai-whisper)
# ---------------------------------------------------------------------------

def bench_whisper(model_name: str, manifest: list[dict], lang: str) -> dict[str, Any]:
    import whisper as ow

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"\n[bold cyan]Whisper {model_name} FP32 [{lang}] on {device}[/bold cyan]")

    model = ow.load_model(model_name, device=device)
    model.eval()

    audios = [load_audio_numpy(m["audio_path"]) for m in manifest]
    refs_raw = [m["reference"] for m in manifest]

    # Warmup
    console.print(f"  warmup {WARMUP} iters …")
    for a in audios[:WARMUP]:
        with torch.no_grad():
            model.transcribe(a, language=lang, fp16=False)

    _reset_gpu(device)
    latencies: list[float] = []

    console.print(f"  benchmarking {BENCH_ITERS} iters …")
    for a in audios[:BENCH_ITERS]:
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.transcribe(a, language=lang, fp16=False)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    # Full accuracy pass over all samples
    all_hyps: list[str] = []
    for a in audios:
        with torch.no_grad():
            r = model.transcribe(a, language=lang, fp16=False)
        all_hyps.append(r["text"].strip())

    del model
    torch.cuda.empty_cache()

    avg_dur = np.mean([m["duration_s"] for m in manifest]) * 1000
    if lang == "zh":
        norm_refs = [normalise_zh(r) for r in refs_raw]
        norm_hyps = [normalise_zh(h) for h in all_hyps]
        error = compute_cer(norm_refs, norm_hyps)
        error_label = "CER%"
    else:
        norm_refs = [normalise_en(r) for r in refs_raw]
        norm_hyps = [normalise_en(h) for h in all_hyps]
        error = compute_wer(norm_refs, norm_hyps)
        error_label = "WER%"

    return {
        "model": f"whisper-{model_name}",
        "backend": "openai-whisper",
        "precision": "fp32",
        "lang": lang,
        "mean_ms": round(float(np.mean(latencies)), 1),
        "p50_ms": round(float(np.percentile(latencies, 50)), 1),
        "p95_ms": round(float(np.percentile(latencies, 95)), 1),
        "gpu_mb": _peak_gpu_mb(device),
        "rtf": round(float(np.mean(latencies)) / avg_dur, 4),
        error_label: error,
        "sample_hyp": all_hyps[0][:120],
    }


# ---------------------------------------------------------------------------
# Parakeet TDT 1.1B (NeMo) — English only
# ---------------------------------------------------------------------------

def bench_parakeet(manifest: list[dict]) -> dict[str, Any]:
    import nemo.collections.asr as nemo_asr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"\n[bold cyan]Parakeet TDT 1.1B FP32 [en] on {device}[/bold cyan]")

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-1.1b")
    model = model.to(device)
    model.eval()

    audio_paths = [m["audio_path"] for m in manifest]
    refs_raw = [m["reference"] for m in manifest]

    # Warmup
    console.print(f"  warmup {WARMUP} iters …")
    for p in audio_paths[:WARMUP]:
        with torch.no_grad():
            model.transcribe([p])

    _reset_gpu(device)
    latencies: list[float] = []

    console.print(f"  benchmarking {BENCH_ITERS} iters …")
    for p in audio_paths[:BENCH_ITERS]:
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.transcribe([p])
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    # Full accuracy pass
    with torch.no_grad():
        results = model.transcribe(audio_paths)
    all_hyps = [r.text if hasattr(r, "text") else str(r) for r in results]

    del model
    torch.cuda.empty_cache()

    avg_dur = np.mean([m["duration_s"] for m in manifest]) * 1000
    norm_refs = [normalise_en(r) for r in refs_raw]
    norm_hyps = [normalise_en(h) for h in all_hyps]
    wer = compute_wer(norm_refs, norm_hyps)

    return {
        "model": "parakeet-tdt-1.1b",
        "backend": "nemo",
        "precision": "fp32",
        "lang": "en",
        "mean_ms": round(float(np.mean(latencies)), 1),
        "p50_ms": round(float(np.percentile(latencies, 50)), 1),
        "p95_ms": round(float(np.percentile(latencies, 95)), 1),
        "gpu_mb": _peak_gpu_mb(device),
        "rtf": round(float(np.mean(latencies)) / avg_dur, 4),
        "WER%": wer,
        "sample_hyp": all_hyps[0][:120],
    }


# ---------------------------------------------------------------------------
# Pretty table
# ---------------------------------------------------------------------------

def print_table(results: list[dict]) -> None:
    table = Table(title="Baseline FP32 Benchmark", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Lang")
    table.add_column("Backend")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("GPU MB", justify="right")
    table.add_column("RTF", justify="right")
    table.add_column("WER/CER %", justify="right")

    for r in results:
        err = r.get("WER%") or r.get("CER%") or "—"
        table.add_row(
            r["model"], r["lang"], r["backend"],
            str(r["mean_ms"]), str(r["p95_ms"]),
            str(r["gpu_mb"]), str(r["rtf"]),
            str(err),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.rule("[bold]Baseline FP32 Benchmark[/bold]")

    if not torch.cuda.is_available():
        console.print("[yellow]WARNING: CUDA not available — latencies will not be representative[/yellow]")

    results: list[dict] = []

    # ----- Chinese: Whisper only -----
    manifest_zh = load_manifest(MANIFEST_ZH)
    for model_name in ["large-v3", "large-v3-turbo"]:
        r = bench_whisper(model_name, manifest_zh, lang="zh")
        results.append(r)
        console.print(f"  → CER {r.get('CER%')}%  mean {r['mean_ms']} ms")

    # ----- English: Whisper + Parakeet -----
    manifest_en = load_manifest(MANIFEST_EN)
    for model_name in ["large-v3", "large-v3-turbo"]:
        r = bench_whisper(model_name, manifest_en, lang="en")
        results.append(r)
        console.print(f"  → WER {r.get('WER%')}%  mean {r['mean_ms']} ms")

    r = bench_parakeet(manifest_en)
    results.append(r)
    console.print(f"  → WER {r.get('WER%')}%  mean {r['mean_ms']} ms")

    # ----- Output -----
    print_table(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "baseline_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    console.print(f"\n[green]Results saved → {out}[/green]")


if __name__ == "__main__":
    main()
