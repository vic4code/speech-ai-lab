"""
Step 4 — torch.compile + TensorRT backend benchmark.

What torch.compile(backend="torch_tensorrt") does
--------------------------------------------------
torch.compile traces the model graph and hands it to the TensorRT compiler,
which applies:
  - Layer / operator fusion (e.g. LayerNorm + Linear + GELU → single kernel)
  - Kernel auto-tuning for the specific GPU architecture (SM 8.6 here)
  - FP16 / BF16 precision selection per layer
  - Memory layout optimisation (e.g. NHWC for conv)

Only the Whisper *encoder* is compiled — it is a pure feed-forward
convolutional + attention stack with static shapes, which TRT handles well.
The *decoder* is autoregressive (dynamic sequence length) and is left in eager
mode; compiling it would require dynamic shape profiles and rebuilding per
output length.

Comparison stack (all faster-whisper CTranslate2 results from previous steps
are included so the table shows the full FP32 → TRT progression):

  FP32 baseline → CT2 float16 → CT2 int8_float16 → torch.compile/TRT encoder

RTF
---
Per-clip RTF = latency_i / duration_i, averaged over BENCH_ITERS clips.

Output
------
results/tensorrt_results.json
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

WARMUP = 5     # more warmup — first pass triggers TRT engine build
BENCH_ITERS = 5
RESULTS_DIR = Path(__file__).parent.parent / "results"
MANIFEST_ZH = Path(__file__).parent.parent / "data" / "benchmark" / "zh" / "manifest.json"
MANIFEST_EN = Path(__file__).parent.parent / "data" / "benchmark" / "en" / "manifest.json"
BASELINE_FILE = RESULTS_DIR / "baseline_results.json"
INT8_FILE = RESULTS_DIR / "int8_results.json"


# ---------------------------------------------------------------------------
# Shared utilities
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
# torch.compile encoder benchmark
# ---------------------------------------------------------------------------

def bench_compile(
    model_name: str,
    manifest: list[dict],
    lang: str,
    backend: str = "torch_tensorrt",
) -> dict[str, Any]:
    import whisper as ow
    import torch_tensorrt  # registers backends on import

    device = "cuda"
    console.print(f"\n[bold cyan]torch.compile({backend}) — Whisper {model_name} FP16 [{lang}][/bold cyan]")

    # Load in FP32 — let torch.compile/TRT manage precision internally.
    # Calling .half() before compile causes dtype mismatch: the compiled
    # encoder emits FP16 but the decoder's cross-attention expects the same
    # dtype as its query projections (still FP32 in eager mode).
    model = ow.load_model(model_name, device=device)
    model.eval()

    # Compile only the encoder — static (batch=1, 80 mel bins, 3000 frames).
    # The decoder is autoregressive (dynamic sequence length) and is left in
    # eager mode; TRT would need per-length engine profiles, not worth it.
    console.print(f"  Compiling encoder with backend={backend} …")
    try:
        model.encoder = torch.compile(
            model.encoder,
            backend=backend,
            options={"enabled_precisions": {torch.float32, torch.float16}},
        )
        actual_backend = backend
    except Exception as e:
        console.print(f"  [yellow]{backend} failed ({type(e).__name__}: {e}) — falling back to inductor max-autotune[/yellow]")
        # inductor max-autotune: Triton kernel fusion + CUDA graph capture
        model.encoder = torch.compile(model.encoder, mode="max-autotune")
        actual_backend = "inductor"

    audios = [load_audio_numpy(m["audio_path"]) for m in manifest]
    refs_raw = [m["reference"] for m in manifest]

    # Warmup — first call triggers compilation / TRT engine build
    console.print(f"  Warmup {WARMUP} iters (1st iter builds engine) …")
    for i, a in enumerate(audios[:WARMUP]):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.transcribe(a, language=lang, fp16=False)
        console.print(f"    warmup {i+1}: {(time.perf_counter()-t0)*1000:.0f} ms")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    latencies: list[float] = []

    console.print(f"  Benchmarking {BENCH_ITERS} iters …")
    for a in audios[:BENCH_ITERS]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.transcribe(a, language=lang, fp16=False)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    all_hyps: list[str] = []
    for a in audios:
        with torch.no_grad():
            r = model.transcribe(a, language=lang, fp16=False)
        all_hyps.append(r["text"].strip())

    gpu_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
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
        "backend": f"torch.compile/{actual_backend}",
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
# Comparison table vs prior steps
# ---------------------------------------------------------------------------

def load_prior() -> list[dict]:
    rows = []
    for f in [BASELINE_FILE, INT8_FILE]:
        if f.exists():
            rows.extend(json.loads(f.read_text()))
    return rows


def speedup(fp32_rtf: float | None, rtf: float) -> str:
    return f"{fp32_rtf/rtf:.2f}×" if fp32_rtf else "—"


def print_table(trt_results: list[dict]) -> None:
    prior = load_prior()
    # FP32 baseline RTF per (model, lang)
    fp32_map = {
        (r["model"], r["lang"]): r["rtf"]
        for r in prior if r.get("precision") == "fp32"
    }
    # Best CT2 int8_float16 from step 3
    ct2_rows = [r for r in prior if r.get("precision") == "int8_float16"]

    table = Table(title="Full Optimization Stack — whisper-large-v3 (per-clip RTF)", show_lines=True)
    for col in ["Step", "Backend", "Precision", "Lang", "Mean (ms)", "P95 (ms)", "GPU MB", "RTF", "WER/CER%", "vs FP32"]:
        table.add_column(col, justify="right" if col in {"Mean (ms)", "P95 (ms)", "GPU MB", "RTF", "WER/CER%"} else "left")

    def row(step: str, r: dict) -> None:
        fp32_rtf = fp32_map.get((r["model"], r["lang"]))
        err = r.get("WER%") or r.get("CER%") or "—"
        table.add_row(
            step, r.get("backend", ""), r.get("precision", ""),
            r["lang"],
            str(r.get("mean_ms", r.get("mean_latency_ms", "—"))),
            str(r.get("p95_ms", r.get("p95_latency_ms", "—"))),
            str(r.get("gpu_mb", r.get("gpu_memory_mb", "—"))),
            str(r["rtf"]),
            str(err),
            speedup(fp32_rtf, r["rtf"]),
        )

    for r in prior:
        if r.get("precision") == "fp32" and r["model"] == "whisper-large-v3":
            row("1 baseline", r)
    for r in ct2_rows:
        if r["model"] == "whisper-large-v3":
            row("3 int8_float16", r)
    for r in trt_results:
        row("4 TRT compile", r)

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.rule("[bold]Step 4 — torch.compile / TensorRT Encoder[/bold]")

    manifest_zh = load_manifest(MANIFEST_ZH)
    manifest_en = load_manifest(MANIFEST_EN)
    results: list[dict] = []

    for model_name in ["large-v3", "large-v3-turbo"]:
        for manifest, lang in [(manifest_zh, "zh"), (manifest_en, "en")]:
            r = bench_compile(model_name, manifest, lang, backend="torch_tensorrt")
            results.append(r)
            err_val = r.get("WER%") or r.get("CER%")
            console.print(f"  → {r['lang']} mean {r['mean_ms']} ms  RTF {r['rtf']}  err {err_val}%")

    print_table(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "tensorrt_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    console.print(f"\n[green]Results saved → {out}[/green]")


if __name__ == "__main__":
    main()
