"""
Step 5 — CT2 Accuracy-Speed Pareto: beam_size sweep + VAD filtering.

Two axes of optimisation on top of CT2 int8_float16:

beam_size sweep (large-v3, int8_float16)
-----------------------------------------
beam_size=1  → greedy decoding, fastest, highest WER/CER
beam_size=2  → small beam, moderate accuracy recovery
beam_size=5  → default, best accuracy

In autoregressive beam search each extra beam multiplies decoder FLOPs linearly,
so latency ∝ beam_size on a single-batch CPU-bound decoder.  On GPU the decoder
is already memory-bandwidth-bound at beam=1; larger beams batch the token
projections, so the penalty is sub-linear (typically 1.2–1.8× for beam=5 vs 1).

VAD filtering (Silero VAD via faster-whisper built-in)
-------------------------------------------------------
faster-whisper can run Silero VAD before feeding audio to Whisper.  The VAD
stamps speech segments; only those frames enter the mel spectrogram.  For clips
with leading/trailing silence or inter-sentence gaps the effective audio shrinks,
reducing encoder + decoder steps.

On the FLEURS zh set (avg 11 s, dense speech) the gain is modest.
On LibriSpeech en (avg 6.7 s, also dense) even smaller.
VAD shows its value on real-world long-form audio with significant silence.

Pareto table
------------
Plots (beam_size, VAD) → (RTF, WER/CER) so the accuracy-latency frontier is
clear and you can pick the operating point for a given SLA.

Output
------
results/ct2_beam_vad_results.json
"""

import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from faster_whisper import WhisperModel
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")

console = Console()

BENCH_ITERS = 5
WARMUP = 3
RESULTS_DIR = Path(__file__).parent.parent / "results"
MANIFEST_ZH = Path(__file__).parent.parent / "data" / "benchmark" / "zh" / "manifest.json"
MANIFEST_EN = Path(__file__).parent.parent / "data" / "benchmark" / "en" / "manifest.json"
BASELINE_FILE = RESULTS_DIR / "baseline_results.json"
INT8_FILE = RESULTS_DIR / "int8_results.json"

MODEL_NAME = "large-v3"
COMPUTE_TYPE = "int8_float16"


# ---------------------------------------------------------------------------
# Utilities
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


def per_clip_rtf(latencies_ms: list[float], manifest: list[dict]) -> float:
    durs = [manifest[i]["duration_s"] * 1000 for i in range(len(latencies_ms))]
    return round(float(np.mean([l / d for l, d in zip(latencies_ms, durs)])), 4)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def bench_ct2(
    manifest: list[dict],
    lang: str,
    beam_size: int,
    vad: bool,
) -> dict[str, Any]:
    label = f"large-v3 int8_float16 | beam={beam_size} | VAD={'on' if vad else 'off'} | {lang}"
    console.print(f"\n  [cyan]{label}[/cyan]")

    model = WhisperModel(MODEL_NAME, device="cuda", compute_type=COMPUTE_TYPE)

    audios = [load_audio_numpy(m["audio_path"]) for m in manifest]
    refs_raw = [m["reference"] for m in manifest]

    vad_kwargs: dict = {"vad_filter": True, "vad_parameters": {"min_silence_duration_ms": 500}} if vad else {}

    # Warmup
    for a in audios[:WARMUP]:
        segs, _ = model.transcribe(a, language=lang, beam_size=beam_size, **vad_kwargs)
        list(segs)

    import torch
    torch.cuda.synchronize()
    latencies: list[float] = []

    for a in audios[:BENCH_ITERS]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        segs, _ = model.transcribe(a, language=lang, beam_size=beam_size, **vad_kwargs)
        list(segs)  # consume generator
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    # Accuracy over all clips
    all_hyps: list[str] = []
    for a in audios:
        segs, _ = model.transcribe(a, language=lang, beam_size=beam_size, **vad_kwargs)
        all_hyps.append(" ".join(s.text for s in segs).strip())

    del model

    if lang == "zh":
        error = compute_cer([normalise_zh(r) for r in refs_raw], [normalise_zh(h) for h in all_hyps])
        err_key = "CER%"
    else:
        error = compute_wer([normalise_en(r) for r in refs_raw], [normalise_en(h) for h in all_hyps])
        err_key = "WER%"

    rtf = per_clip_rtf(latencies, manifest)
    console.print(f"    RTF {rtf}  mean {np.mean(latencies):.0f} ms  {err_key} {error}")
    return {
        "model": f"whisper-{MODEL_NAME}",
        "backend": "faster-whisper",
        "compute_type": COMPUTE_TYPE,
        "lang": lang,
        "beam_size": beam_size,
        "vad": vad,
        "mean_ms": round(float(np.mean(latencies)), 1),
        "p50_ms": round(float(np.percentile(latencies, 50)), 1),
        "p95_ms": round(float(np.percentile(latencies, 95)), 1),
        "rtf": rtf,
        err_key: error,
    }


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def fp32_rtf(lang: str) -> float | None:
    if not BASELINE_FILE.exists():
        return None
    rows = json.loads(BASELINE_FILE.read_text())
    for r in rows:
        if r["model"] == "whisper-large-v3" and r["lang"] == lang and r.get("precision") == "fp32":
            return r["rtf"]
    return None


def print_table(results: list[dict]) -> None:
    # Load step 3 int8_float16 as reference
    ref_rtf: dict[str, float] = {}
    if INT8_FILE.exists():
        for r in json.loads(INT8_FILE.read_text()):
            if r["model"] == "whisper-large-v3" and r.get("precision") == "int8_float16":
                ref_rtf[r["lang"]] = r["rtf"]

    table = Table(title="CT2 Accuracy-Speed Pareto — beam_size × VAD (large-v3 int8_float16)", show_lines=True)
    for col in ["Lang", "beam", "VAD", "Mean (ms)", "P95 (ms)", "RTF", "WER/CER%", "vs beam5 no-VAD", "vs FP32"]:
        table.add_column(col, justify="right" if col not in {"Lang", "VAD"} else "left")

    # baseline per lang: beam=5, vad=False
    base: dict[str, dict] = {}
    for r in results:
        if r["beam_size"] == 5 and not r["vad"]:
            base[r["lang"]] = r

    fp32: dict[str, float | None] = {lang: fp32_rtf(lang) for lang in ["zh", "en"]}

    for r in results:
        err = r.get("WER%") or r.get("CER%") or "—"
        b = base.get(r["lang"])
        vs_base = f"{b['rtf']/r['rtf']:.2f}×" if b else "—"
        fp32_val = fp32.get(r["lang"])
        vs_fp32 = f"{fp32_val/r['rtf']:.2f}×" if fp32_val else "—"
        table.add_row(
            r["lang"],
            str(r["beam_size"]),
            "✓" if r["vad"] else "✗",
            str(r["mean_ms"]),
            str(r["p95_ms"]),
            str(r["rtf"]),
            str(err),
            vs_base,
            vs_fp32,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.rule("[bold]Step 5 — CT2 Beam Size × VAD Pareto[/bold]")

    manifest_zh = load_manifest(MANIFEST_ZH)
    manifest_en = load_manifest(MANIFEST_EN)

    results: list[dict] = []

    for lang, manifest in [("zh", manifest_zh), ("en", manifest_en)]:
        for beam_size in [1, 2, 5]:
            for vad in [False, True]:
                r = bench_ct2(manifest, lang, beam_size, vad)
                results.append(r)

    print_table(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "ct2_beam_vad_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    console.print(f"\n[green]Results saved → {out}[/green]")


if __name__ == "__main__":
    main()
