"""
FP16 inference benchmark — Whisper (openai-whisper amp) + faster-whisper float16.

Technique
---------
openai-whisper fp16=True  : model weights cast to float16; CUDA Tensor Cores execute
                            FP16 GEMMs at 2× throughput vs FP32. Same Python API.
faster-whisper float16    : CTranslate2 backend converts the model to float16 offline;
                            adds operator fusion on top of the precision reduction.

RTF normalisation
-----------------
Each timed latency is divided by that clip's own duration (per-clip RTF) before
averaging, so the metric is independent of clip length.

    RTF = latency_ms / clip_duration_ms   (lower is better; RTF < 1 = faster than real-time)

Output
------
results/fp16_results.json — same schema as baseline_results.json for direct comparison.
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


# ---------------------------------------------------------------------------
# Shared utilities (mirrors 01_baseline_benchmark.py)
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}\nRun 00_prepare_data.py first.")
    return json.loads(path.read_text())


def load_audio_numpy(audio_path: str, target_sr: int = 16_000) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio.astype(np.float32)


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
    rtfs = [lat / dur for lat, dur in zip(latencies_ms, durs_ms)]
    return round(float(np.mean(rtfs)), 4)


# ---------------------------------------------------------------------------
# Whisper FP16 via openai-whisper (fp16=True)
# ---------------------------------------------------------------------------

def bench_whisper_fp16(model_name: str, manifest: list[dict], lang: str) -> dict[str, Any]:
    import whisper as ow

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"\n[bold cyan]openai-whisper {model_name} FP16 [{lang}][/bold cyan]")

    model = ow.load_model(model_name, device=device)
    model.eval()

    audios = [load_audio_numpy(m["audio_path"]) for m in manifest]
    refs_raw = [m["reference"] for m in manifest]

    for a in audios[:WARMUP]:
        with torch.no_grad():
            model.transcribe(a, language=lang, fp16=True)

    _reset_gpu()
    latencies: list[float] = []
    for a in audios[:BENCH_ITERS]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.transcribe(a, language=lang, fp16=True)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    all_hyps: list[str] = []
    for a in audios:
        with torch.no_grad():
            r = model.transcribe(a, language=lang, fp16=True)
        all_hyps.append(r["text"].strip())

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
        "backend": "openai-whisper",
        "precision": "fp16",
        "lang": lang,
        "mean_ms": round(float(np.mean(latencies)), 1),
        "p50_ms": round(float(np.percentile(latencies, 50)), 1),
        "p95_ms": round(float(np.percentile(latencies, 95)), 1),
        "gpu_mb": gpu_mb,
        "rtf": per_clip_rtf(latencies, manifest),
        err_key: error,
    }


# ---------------------------------------------------------------------------
# faster-whisper float16 via CTranslate2
# ---------------------------------------------------------------------------

def bench_faster_whisper_fp16(model_name: str, manifest: list[dict], lang: str) -> dict[str, Any]:
    from faster_whisper import WhisperModel

    console.print(f"\n[bold cyan]faster-whisper {model_name} float16 [{lang}][/bold cyan]")

    model = WhisperModel(model_name, device="cuda", compute_type="float16")
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
        "precision": "float16",
        "lang": lang,
        "mean_ms": round(float(np.mean(latencies)), 1),
        "p50_ms": round(float(np.percentile(latencies, 50)), 1),
        "p95_ms": round(float(np.percentile(latencies, 95)), 1),
        "gpu_mb": gpu_mb,
        "rtf": per_clip_rtf(latencies, manifest),
        err_key: error,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def load_baseline() -> list[dict]:
    if BASELINE_FILE.exists():
        return json.loads(BASELINE_FILE.read_text())
    return []


def speedup(fp32_rtf: float | None, rtf: float) -> str:
    if fp32_rtf and rtf:
        return f"{fp32_rtf / rtf:.2f}×"
    return "—"


def print_table(fp16_results: list[dict], baseline: list[dict]) -> None:
    base_map = {(r["model"], r["lang"]): r for r in baseline}

    table = Table(title="FP32 → FP16 Comparison (per-clip RTF)", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Lang")
    table.add_column("Precision")
    table.add_column("Backend")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("GPU MB", justify="right")
    table.add_column("RTF", justify="right")
    table.add_column("WER/CER%", justify="right")
    table.add_column("vs FP32 baseline", justify="right", style="green")

    for r in fp16_results:
        base = base_map.get((r["model"], r["lang"]))
        fp32_rtf = base["rtf"] if base else None
        err = r.get("WER%") or r.get("CER%") or "—"
        table.add_row(
            r["model"], r["lang"], r["precision"], r["backend"],
            str(r["mean_ms"]), str(r["p95_ms"]), str(r["gpu_mb"]),
            str(r["rtf"]), str(err),
            speedup(fp32_rtf, r["rtf"]),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.rule("[bold]FP16 Optimization Benchmark[/bold]")

    manifest_zh = load_manifest(MANIFEST_ZH)
    manifest_en = load_manifest(MANIFEST_EN)
    results: list[dict] = []

    for model_name in ["large-v3", "large-v3-turbo"]:
        # Chinese: openai-whisper FP16 + faster-whisper float16
        results.append(bench_whisper_fp16(model_name, manifest_zh, "zh"))
        results.append(bench_faster_whisper_fp16(model_name, manifest_zh, "zh"))
        # English: same
        results.append(bench_whisper_fp16(model_name, manifest_en, "en"))
        results.append(bench_faster_whisper_fp16(model_name, manifest_en, "en"))

    baseline = load_baseline()
    print_table(results, baseline)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "fp16_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    console.print(f"\n[green]Results saved → {out}[/green]")


if __name__ == "__main__":
    main()
