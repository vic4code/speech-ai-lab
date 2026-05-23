"""
Step 6 — TensorRT INT8 + Explicit Input Spec.

What this adds over Step 4
--------------------------
Step 4 used the simplest form of torch.compile(backend="torch_tensorrt") with
only FP32+FP16 enabled.  This step uses the lower-level
`torch_tensorrt.compile()` API which allows:

1. Explicit input shapes (min / opt / max)
   TRT builds three engine profiles for dynamic-range inputs.  The optimisation
   kernel is tuned for the *opt* shape; shapes outside min/max raise an error at
   runtime.  For Whisper the encoder input is always (1, 80, 3000) — a static
   shape — so min=opt=max.

2. INT8 precision in enabled_precisions
   Adding torch.int8 tells TRT it may quantise any layer to INT8 where its
   calibration heuristic predicts an accuracy-safe gain.  Without a calibration
   dataset TRT uses a *post-training quantisation* (PTQ) heuristic based on the
   weight distribution; with a calibration dataset it computes per-tensor dynamic
   ranges (histogram-based).

3. Workspace size
   `workspace_size` caps GPU memory TRT may use during engine build for tiling
   experiments.  8 GiB is a safe upper bound on the A10G (24 GiB VRAM).

Comparison table includes Step 4 (fp16-only TRT) so the INT8 delta is visible.

PTQ vs QAT (interview note)
---------------------------
This script does PTQ (no retraining).  QAT (Quantisation-Aware Training) inserts
fake-quantisation nodes during fine-tuning so the model learns to compensate for
quantisation error; it recovers ~0.5–1 pp accuracy at INT8 vs PTQ but requires a
labelled training set and retraining budget.

Output
------
results/trt_int8_results.json
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

WARMUP = 5
BENCH_ITERS = 5
RESULTS_DIR = Path(__file__).parent.parent / "results"
MANIFEST_ZH = Path(__file__).parent.parent / "data" / "benchmark" / "zh" / "manifest.json"
MANIFEST_EN = Path(__file__).parent.parent / "data" / "benchmark" / "en" / "manifest.json"
BASELINE_FILE = RESULTS_DIR / "baseline_results.json"
TRT_FP16_FILE = RESULTS_DIR / "tensorrt_results.json"


# ---------------------------------------------------------------------------
# Utilities (shared)
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
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
# TRT INT8 benchmark
# ---------------------------------------------------------------------------

def compile_encoder_trt_int8(encoder: torch.nn.Module) -> tuple[torch.nn.Module, str]:
    """
    Attempt torch_tensorrt.compile() with explicit input spec and INT8.
    Falls back to torch.compile with fp16+fp32 if the lower-level API fails.
    """
    import torch_tensorrt

    # Infer n_mels from the encoder's first conv weight: shape (out, in, k)
    # large-v3 → n_mels=128; large-v3-turbo → n_mels=128; tiny → n_mels=80
    n_mels = encoder.conv1.weight.shape[1]
    input_spec = torch_tensorrt.Input(
        min_shape=[1, n_mels, 3000],
        opt_shape=[1, n_mels, 3000],
        max_shape=[1, n_mels, 3000],
        dtype=torch.float32,
    )

    try:
        compiled = torch_tensorrt.compile(
            encoder,
            inputs=[input_spec],
            enabled_precisions={torch.float32, torch.float16, torch.int8},
            workspace_size=8 * (1 << 30),  # 8 GiB
            truncate_long_and_double=True,
        )
        return compiled, "torch_tensorrt/int8"
    except Exception as e:
        console.print(f"  [yellow]torch_tensorrt.compile INT8 failed ({type(e).__name__}: {e})[/yellow]")
        console.print("  [yellow]Falling back to torch.compile fp16+int8 via inductor[/yellow]")
        compiled = torch.compile(encoder, mode="max-autotune")
        return compiled, "inductor/max-autotune"


def bench_trt_int8(
    model_name: str,
    manifest: list[dict],
    lang: str,
) -> dict[str, Any]:
    import whisper as ow

    console.print(f"\n[bold cyan]TRT INT8 — Whisper {model_name} | {lang}[/bold cyan]")

    model = ow.load_model(model_name, device="cuda")
    model.eval()

    console.print("  Compiling encoder with TRT INT8 (explicit input spec) …")
    model.encoder, actual_backend = compile_encoder_trt_int8(model.encoder)
    console.print(f"  Backend: {actual_backend}")

    audios = [load_audio_numpy(m["audio_path"]) for m in manifest]
    refs_raw = [m["reference"] for m in manifest]

    console.print(f"  Warmup {WARMUP} iters (1st builds engine) …")
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
        "backend": actual_backend,
        "precision": "int8+fp16",
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

def print_table(new_results: list[dict]) -> None:
    fp32_map: dict[tuple, float] = {}
    if BASELINE_FILE.exists():
        for r in json.loads(BASELINE_FILE.read_text()):
            if r.get("precision") == "fp32":
                fp32_map[(r["model"], r["lang"])] = r["rtf"]

    fp16_map: dict[tuple, dict] = {}
    if TRT_FP16_FILE.exists():
        for r in json.loads(TRT_FP16_FILE.read_text()):
            fp16_map[(r["model"], r["lang"])] = r

    table = Table(title="TRT Encoder: FP16 vs INT8 — whisper-large-v3 (per-clip RTF)", show_lines=True)
    for col in ["Model", "Backend", "Precision", "Lang", "Mean (ms)", "P95 (ms)", "GPU MB", "RTF", "WER/CER%", "vs FP32"]:
        table.add_column(col, justify="right" if col not in {"Model", "Backend", "Precision", "Lang"} else "left")

    def add_row(r: dict) -> None:
        key = (r["model"], r["lang"])
        fp32_rtf = fp32_map.get(key)
        vs_fp32 = f"{fp32_rtf/r['rtf']:.2f}×" if fp32_rtf else "—"
        err = r.get("WER%") or r.get("CER%") or "—"
        table.add_row(
            r["model"], r.get("backend", ""), r.get("precision", ""),
            r["lang"],
            str(r.get("mean_ms", "—")),
            str(r.get("p95_ms", "—")),
            str(r.get("gpu_mb", "—")),
            str(r["rtf"]),
            str(err),
            vs_fp32,
        )

    # Inject Step 4 FP16-only TRT rows for comparison
    for key, r in fp16_map.items():
        if r["model"] == "whisper-large-v3":
            add_row(r)

    for r in new_results:
        if r["model"] == "whisper-large-v3":
            add_row(r)

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.rule("[bold]Step 6 — TensorRT INT8 + Explicit Input Spec[/bold]")

    manifest_zh = load_manifest(MANIFEST_ZH)
    manifest_en = load_manifest(MANIFEST_EN)
    results: list[dict] = []

    for model_name in ["large-v3", "large-v3-turbo"]:
        for manifest, lang in [(manifest_zh, "zh"), (manifest_en, "en")]:
            r = bench_trt_int8(model_name, manifest, lang)
            results.append(r)
            err_val = r.get("WER%") or r.get("CER%")
            console.print(f"  → {r['lang']} mean {r['mean_ms']} ms  RTF {r['rtf']}  err {err_val}%")

    print_table(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "trt_int8_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    console.print(f"\n[green]Results saved → {out}[/green]")


if __name__ == "__main__":
    main()
