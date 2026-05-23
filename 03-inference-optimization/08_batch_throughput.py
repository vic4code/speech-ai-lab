"""
Step 8 — Batch Throughput: memory-bound → compute-bound transition.

Why batch size matters
----------------------
At batch=1, the encoder runs a matrix-vector multiply per token.
Arithmetic intensity ≈ 2×hidden / (4×hidden) ≈ 0.5 FLOP/byte.
The A10G FP16 ridge point is 125 TFLOPS / 600 GB/s ≈ 208 FLOP/byte.
We are 400× below the ridge → memory-bound.

As batch size grows:
  AI(B) ≈ B × 0.5 FLOP/byte   (matrix-matrix instead of matrix-vector)

When AI(B) approaches the ridge, the model transitions to compute-bound and
throughput stops scaling linearly with batch — we've saturated the GPU.

This experiment quantifies that transition on the A10G with Whisper large-v3.

What we measure
---------------
1. Encoder-only batch throughput (batch 1→16)
   - Inputs: B stacked mel spectrograms (B, n_mels, 3000)
   - Metric: clips/sec = B / latency_per_batch
   - Backend: openai-whisper FP16 (encoder is pure feed-forward, batchable)

2. GPU SM utilisation via pynvml sampled during each benchmark window
   - Shows when SMs saturate (compute-bound) vs when they're stalling on DRAM

3. Roofline arithmetic intensity estimate
   - AI_estimated = 2 × B × n_tokens × n_layers × hidden² / (model_bytes × B)
   - Simplification: ignores attention FLOPs, quadratic seq term for short seqs

4. Full pipeline throughput at batch=1 (CT2 int8_float16, best from Step 3)
   - Sequential clips/sec for realistic single-instance serving estimate

Output
------
results/batch_throughput_results.json
"""

import json
import threading
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
BENCH_ITERS = 10
BATCH_SIZES = [1, 2, 4, 8, 16]
RESULTS_DIR = Path(__file__).parent.parent / "results"
MANIFEST_ZH = Path(__file__).parent.parent / "data" / "benchmark" / "zh" / "manifest.json"
MANIFEST_EN = Path(__file__).parent.parent / "data" / "benchmark" / "en" / "manifest.json"

# Whisper large-v3 architecture constants (for AI estimation)
N_LAYERS = 32
N_STATE = 1280   # hidden dim
N_HEADS = 20
N_MELS = 128
N_CTX = 1500     # encoder output tokens (3000 frames → 1500 after conv stride=2)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text())


def load_audio_numpy(audio_path: str, target_sr: int = 16_000) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio.astype(np.float32)


def audio_to_mel(model: Any, audio: np.ndarray) -> torch.Tensor:
    import whisper
    audio_t = whisper.pad_or_trim(torch.from_numpy(audio))
    return whisper.log_mel_spectrogram(audio_t, n_mels=model.dims.n_mels).to("cuda")


# ---------------------------------------------------------------------------
# GPU utilisation sampling (pynvml)
# ---------------------------------------------------------------------------

class GpuSampler:
    """Background thread sampling GPU SM utilisation every 50 ms."""

    def __init__(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
            self._available = True
        except Exception:
            self._available = False
        self._samples: list[int] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._available:
            return
        self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        if not self._available:
            return None
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return round(float(np.mean(self._samples)), 1) if self._samples else None

    def _sample_loop(self) -> None:
        while self._running:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self._samples.append(util.gpu)
            except Exception:
                pass
            time.sleep(0.05)


# ---------------------------------------------------------------------------
# Encoder batch throughput benchmark
# ---------------------------------------------------------------------------

def estimate_ai(batch: int) -> float:
    """
    Rough arithmetic intensity estimate for Whisper large-v3 encoder.
    Per encoder layer: 4 QKV proj + 4 attn out + 2 FFN = 10 matmuls of size
    (B×T, d) × (d, d).  FLOPs = 10 × 2 × B × T × d².
    Bytes loaded from DRAM = 10 × 2 × d² × 2 (FP16) + B×T×d×2 (activations).
    At large B, activation bytes dominate differently; weight bytes dominate at B=1.
    Simplified: AI ≈ B × T × (10 × 2 × d²) / (10 × 2 × d² × 2 + B×T×d×2)
    """
    flops_per_layer = 10 * 2 * batch * N_CTX * N_STATE ** 2
    weight_bytes_per_layer = 10 * 2 * N_STATE ** 2 * 2  # FP16
    act_bytes_per_layer = batch * N_CTX * N_STATE * 2
    dram_bytes = weight_bytes_per_layer + act_bytes_per_layer
    return round(N_LAYERS * flops_per_layer / (N_LAYERS * dram_bytes), 1)


def bench_encoder_batch(
    model: Any,
    mel_list: list[torch.Tensor],
    batch_size: int,
    sampler: GpuSampler,
) -> dict[str, Any]:
    # Build batched input by repeating/cycling clips
    mels = [mel_list[i % len(mel_list)] for i in range(batch_size)]
    mel_batch = torch.stack(mels, dim=0)   # (B, n_mels, 3000)

    # Warmup
    for _ in range(WARMUP):
        with torch.no_grad():
            model.encoder(mel_batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    sampler.start()
    latencies: list[float] = []
    for _ in range(BENCH_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.encoder(mel_batch)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
    gpu_util = sampler.stop()

    mean_ms = float(np.mean(latencies))
    gpu_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    throughput = round(batch_size / (mean_ms / 1000), 2)   # clips/sec
    ai = estimate_ai(batch_size)

    console.print(
        f"  batch={batch_size:2d}  mean {mean_ms:7.1f} ms  "
        f"throughput {throughput:6.2f} clips/s  "
        f"AI~{ai:5.1f} FLOP/B  "
        f"GPU util {gpu_util or '?':>4}%  "
        f"mem {gpu_mb} MB"
    )
    return {
        "batch_size": batch_size,
        "mean_ms": round(mean_ms, 1),
        "p95_ms": round(float(np.percentile(latencies, 95)), 1),
        "throughput_clips_per_sec": throughput,
        "gpu_util_pct": gpu_util,
        "gpu_mb": gpu_mb,
        "ai_flop_per_byte": ai,
    }


# ---------------------------------------------------------------------------
# Full pipeline sequential throughput (CT2 int8_float16, best from Step 3)
# ---------------------------------------------------------------------------

def bench_ct2_throughput(manifest: list[dict], lang: str, n_clips: int = 20) -> dict[str, Any]:
    from faster_whisper import WhisperModel
    console.print(f"\n[cyan]CT2 int8_float16 sequential throughput ({lang}, {n_clips} clips)[/cyan]")
    model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
    audios = [load_audio_numpy(m["audio_path"]) for m in manifest[:n_clips]]

    # Warmup
    for a in audios[:3]:
        segs, _ = model.transcribe(a, language=lang, beam_size=1)
        list(segs)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for a in audios:
        segs, _ = model.transcribe(a, language=lang, beam_size=1)
        list(segs)
    torch.cuda.synchronize()
    total_s = time.perf_counter() - t0

    throughput = round(n_clips / total_s, 2)
    total_audio_s = sum(m["duration_s"] for m in manifest[:n_clips])
    rtf = round(total_s / total_audio_s, 4)

    console.print(f"  {n_clips} clips in {total_s:.1f}s → {throughput} clips/sec  RTF {rtf}")
    del model
    return {"backend": "ct2_int8_float16_beam1", "lang": lang,
            "n_clips": n_clips, "total_s": round(total_s, 2),
            "throughput_clips_per_sec": throughput, "rtf": rtf}


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def print_table(enc_results: list[dict]) -> None:
    ridge = round(125_000 / 600, 0)  # FP16 TFLOPS / BW GB/s → FLOP/byte
    table = Table(
        title=f"Encoder Batch Throughput — whisper-large-v3 FP16 (A10G ridge ≈ {ridge:.0f} FLOP/byte)",
        show_lines=True,
    )
    for col in ["Batch", "Mean (ms)", "P95 (ms)", "Clips/sec", "Speedup vs B=1",
                "AI (FLOP/B)", "vs ridge", "GPU util %", "GPU MB"]:
        table.add_column(col, justify="right")

    base_tp = enc_results[0]["throughput_clips_per_sec"] if enc_results else 1.0
    for r in enc_results:
        speedup = round(r["throughput_clips_per_sec"] / base_tp, 2)
        vs_ridge = f"{r['ai_flop_per_byte'] / ridge * 100:.0f}%"
        table.add_row(
            str(r["batch_size"]),
            str(r["mean_ms"]),
            str(r["p95_ms"]),
            str(r["throughput_clips_per_sec"]),
            f"{speedup}×",
            str(r["ai_flop_per_byte"]),
            vs_ridge,
            str(r["gpu_util_pct"] or "—"),
            str(r["gpu_mb"]),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.rule("[bold]Step 8 — Batch Throughput: memory → compute transition[/bold]")

    import whisper as ow
    manifest_zh = load_manifest(MANIFEST_ZH)
    manifest_en = load_manifest(MANIFEST_EN)

    model = ow.load_model("large-v3", device="cuda")
    model.eval()

    mel_list_zh = [audio_to_mel(model, load_audio_numpy(m["audio_path"]))
                   for m in manifest_zh[:max(BATCH_SIZES)]]
    mel_list_en = [audio_to_mel(model, load_audio_numpy(m["audio_path"]))
                   for m in manifest_en[:max(BATCH_SIZES)]]

    sampler = GpuSampler()
    enc_results: list[dict] = []

    console.print("\n[bold cyan]Encoder batch throughput — large-v3 FP16 (zh clips)[/bold cyan]")
    for bs in BATCH_SIZES:
        try:
            r = bench_encoder_batch(model, mel_list_zh, bs, sampler)
            r["lang"] = "zh"
            enc_results.append(r)
        except torch.cuda.OutOfMemoryError:
            console.print(f"  [red]batch={bs}: OOM — stopping[/red]")
            break

    print_table(enc_results)

    del model
    torch.cuda.empty_cache()

    # Full pipeline throughput
    ct2_results: list[dict] = []
    for lang, manifest in [("zh", manifest_zh), ("en", manifest_en)]:
        ct2_results.append(bench_ct2_throughput(manifest, lang))

    all_results = {"encoder_batch": enc_results, "ct2_sequential": ct2_results}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "batch_throughput_results.json"
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    console.print(f"\n[green]Results saved → {out}[/green]")


if __name__ == "__main__":
    main()
