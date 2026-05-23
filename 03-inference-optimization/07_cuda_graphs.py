"""
Step 7 — CUDA Graphs: eliminating kernel launch overhead.

What is kernel launch overhead?
--------------------------------
Every time Python calls a CUDA kernel (matmul, LayerNorm, etc.) the CPU must:
  1. Push a command to the CUDA stream queue (~5–10 µs per kernel)
  2. Wait for GPU to pick it up and execute

For a Whisper large-v3 encoder forward pass, ~200–400 kernels launch.  At 5 µs
each that's 1–2 ms of pure CPU-side kernel-launch tax on top of the actual GPU
compute time.  At small batch sizes or on very fast GPUs this overhead is a
meaningful fraction of total latency.

CUDA Graph capture
------------------
`torch.cuda.CUDAGraph` captures the sequence of CUDA operations into a single
replayable object.  Replay issues the entire graph in ONE CPU call, reducing
kernel-launch overhead from O(n_kernels) to O(1).

Constraints:
  - All tensor shapes must be STATIC (same at capture and replay)
  - No control flow that changes the graph structure
  - No CPU↔GPU synchronisation inside the captured region

Whisper encoder is ideal: static shape (1, 80, 3000), pure feed-forward.
Decoder is NOT suitable: autoregressive, variable output length.

Three configurations benchmarked
----------------------------------
A. Eager (no compile, no graph)                    ← baseline
B. torch.compile(max-autotune)                     ← inductor Triton fusion + internal cudagraph_trees
C. torch.compile(max-autotune-no-cudagraphs)        ← Triton fusion only, no internal graph trees
   + manual torch.cuda.CUDAGraph replay

Why C uses max-autotune-no-cudagraphs, not max-autotune:
  torch.compile(max-autotune) enables inductor's cudagraph_trees internally.
  If we then open torch.cuda.graph() for outer capture, CUDA raises
  cudaErrorStreamCaptureUnsupported — graph replay inside a capture is illegal.
  max-autotune-no-cudagraphs keeps the Triton autotuning but disables the
  internal graph trees, letting us own the capture layer.
  Note: torch.compile does NOT accept mode + options simultaneously; use
  the dedicated mode string instead.

The graph is captured once on a "dummy" forward pass (identical shapes to
production) then replayed for all subsequent calls.

Interview note: why not always use CUDA graphs?
----------------------------------------------
- Dynamic shapes break the graph (must recapture per new shape)
- Memory addresses in the captured graph are fixed — input must be copied into
  the pre-allocated input tensor before each replay
- Capture has a one-time cost (~same as first compile warmup)
- Not beneficial when CPU is not the bottleneck (large batches, long sequences)

Output
------
results/cuda_graph_results.json
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
BENCH_ITERS = 10           # more iters — graph replay is fast, reduce variance
RESULTS_DIR = Path(__file__).parent.parent / "results"
MANIFEST_ZH = Path(__file__).parent.parent / "data" / "benchmark" / "zh" / "manifest.json"
MANIFEST_EN = Path(__file__).parent.parent / "data" / "benchmark" / "en" / "manifest.json"
BASELINE_FILE = RESULTS_DIR / "baseline_results.json"


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


def audio_to_mel(model: Any, audio: np.ndarray) -> torch.Tensor:
    """Convert raw audio to mel spectrogram tensor (the encoder input)."""
    import whisper
    audio_t = whisper.pad_or_trim(torch.from_numpy(audio))
    mel = whisper.log_mel_spectrogram(audio_t, n_mels=model.dims.n_mels).to("cuda")
    return mel.unsqueeze(0)  # (1, 80, 3000)


# ---------------------------------------------------------------------------
# Encoder-only latency benchmark
# ---------------------------------------------------------------------------
# We isolate the encoder because:
#   a) It is the CUDA-graph-friendly component (static shapes)
#   b) Decoder latency is dominated by autoregressive steps, not launch overhead
#   c) This cleanly shows the kernel-launch overhead reduction

def bench_encoder_only(
    model: Any,
    mel_inputs: list[torch.Tensor],
    mode: str,
) -> list[float]:
    """Return per-input encoder latencies in ms."""
    latencies = []
    # Warmup
    for mel in mel_inputs[:WARMUP]:
        with torch.no_grad():
            model.encoder(mel)
    torch.cuda.synchronize()

    for mel in mel_inputs[:BENCH_ITERS]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.encoder(mel)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def bench_with_cuda_graph(
    model: Any,
    mel_inputs: list[torch.Tensor],
) -> list[float]:
    """
    Capture encoder forward as a CUDA graph then replay it.

    Key steps:
    1. Allocate a static input tensor (same address for every replay)
    2. Warmup without graph to initialise lazy modules
    3. Capture: run one forward inside torch.cuda.graph() context
    4. Replay: copy new input into static buffer, call graph.replay()

    NOTE: the encoder must be compiled with triton.cudagraphs=False.
    torch.compile(max-autotune) enables inductor's internal cudagraph_trees;
    if we then try to capture an outer graph, CUDA raises
    cudaErrorStreamCaptureUnsupported because you cannot nest graph captures.
    Disabling internal graphs lets us own the graph capture layer ourselves.
    """
    # Static buffer — graph captures memory addresses, not values
    static_input = mel_inputs[0].clone()

    # Warmup without graph (initialises cuDNN workspace etc.)
    for mel in mel_inputs[:3]:
        static_input.copy_(mel)
        with torch.no_grad():
            _ = model.encoder(static_input)
    torch.cuda.synchronize()

    # Graph capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.no_grad():
            static_output = model.encoder(static_input)
    torch.cuda.synchronize()
    console.print("    [green]CUDA graph captured[/green]")

    # Benchmark replay
    latencies = []
    for mel in mel_inputs[:BENCH_ITERS]:
        static_input.copy_(mel)          # write new data into captured buffer
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        graph.replay()                   # single CPU call launches entire graph
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    del static_output, graph
    return latencies


def run_configs(model_name: str, manifest: list[dict], lang: str) -> list[dict]:
    import whisper as ow

    console.print(f"\n[bold cyan]CUDA Graphs — Whisper {model_name} encoder | {lang}[/bold cyan]")

    audios = [load_audio_numpy(m["audio_path"]) for m in manifest]
    refs_raw = [m["reference"] for m in manifest]

    results = []

    for config in ["eager", "compile", "compile+graph"]:
        console.print(f"\n  [yellow]Config: {config}[/yellow]")

        model = ow.load_model(model_name, device="cuda")
        model.eval()

        if config == "compile":
            # max-autotune: Triton kernel fusion + inductor's internal CUDA graphs
            console.print("  Compiling encoder (max-autotune) …")
            model.encoder = torch.compile(model.encoder, mode="max-autotune")
            dummy = torch.zeros(1, model.dims.n_mels, 3000, device="cuda")
            for _ in range(3):
                with torch.no_grad():
                    model.encoder(dummy)
            torch.cuda.synchronize()
            console.print("  Compilation done")

        elif config == "compile+graph":
            # "max-autotune-no-cudagraphs": same kernel autotuning as
            # max-autotune but disables inductor's internal cudagraph_trees,
            # so our outer torch.cuda.CUDAGraph() capture won't conflict.
            # (mode + options cannot coexist in torch.compile)
            console.print("  Compiling encoder (max-autotune-no-cudagraphs) …")
            model.encoder = torch.compile(
                model.encoder,
                mode="max-autotune-no-cudagraphs",
            )
            dummy = torch.zeros(1, model.dims.n_mels, 3000, device="cuda")
            for _ in range(3):
                with torch.no_grad():
                    model.encoder(dummy)
            torch.cuda.synchronize()
            console.print("  Compilation done")

        mel_inputs = [audio_to_mel(model, a) for a in audios]

        torch.cuda.reset_peak_memory_stats()

        if config == "compile+graph":
            latencies = bench_with_cuda_graph(model, mel_inputs)
        else:
            latencies = bench_encoder_only(model, mel_inputs, config)

        gpu_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

        # Accuracy: full transcribe in eager for the compiled/graph variants
        # (decoder is always eager; we're only measuring encoder isolation here)
        all_hyps: list[str] = []
        for a in audios:
            with torch.no_grad():
                r = model.transcribe(a, language=lang, fp16=False)
            all_hyps.append(r["text"].strip())

        if lang == "zh":
            error = compute_cer([normalise_zh(r) for r in refs_raw], [normalise_zh(h) for h in all_hyps])
            err_key = "CER%"
        else:
            error = compute_wer([normalise_en(r) for r in refs_raw], [normalise_en(h) for h in all_hyps])
            err_key = "WER%"

        mean_ms = round(float(np.mean(latencies)), 2)
        p95_ms = round(float(np.percentile(latencies, 95)), 2)
        rtf_enc = per_clip_rtf(latencies, manifest[:BENCH_ITERS])

        console.print(f"    encoder mean {mean_ms} ms  P95 {p95_ms} ms  RTF(enc) {rtf_enc}  {err_key} {error}")

        results.append({
            "model": f"whisper-{model_name}",
            "config": config,
            "lang": lang,
            "encoder_mean_ms": mean_ms,
            "encoder_p95_ms": p95_ms,
            "encoder_rtf": rtf_enc,
            "gpu_mb": gpu_mb,
            err_key: error,
        })

        del model, mel_inputs
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def print_table(all_results: list[dict]) -> None:
    table = Table(title="CUDA Graph — Encoder Latency Isolation (ms)", show_lines=True)
    for col in ["Model", "Lang", "Config", "Enc mean (ms)", "Enc P95 (ms)", "Enc RTF", "WER/CER%", "vs eager"]:
        table.add_column(col, justify="right" if col not in {"Model", "Lang", "Config"} else "left")

    # baseline per (model, lang) = eager
    eager_map: dict[tuple, float] = {}
    for r in all_results:
        if r["config"] == "eager":
            eager_map[(r["model"], r["lang"])] = r["encoder_mean_ms"]

    for r in all_results:
        key = (r["model"], r["lang"])
        base = eager_map.get(key)
        vs_eager = f"{base/r['encoder_mean_ms']:.2f}×" if base else "—"
        err = r.get("WER%") or r.get("CER%") or "—"
        table.add_row(
            r["model"], r["lang"], r["config"],
            str(r["encoder_mean_ms"]),
            str(r["encoder_p95_ms"]),
            str(r["encoder_rtf"]),
            str(err),
            vs_eager,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.rule("[bold]Step 7 — CUDA Graphs: Encoder Kernel Launch Overhead[/bold]")

    manifest_zh = load_manifest(MANIFEST_ZH)
    manifest_en = load_manifest(MANIFEST_EN)

    all_results: list[dict] = []

    for model_name in ["large-v3", "large-v3-turbo"]:
        for manifest, lang in [(manifest_zh, "zh"), (manifest_en, "en")]:
            results = run_configs(model_name, manifest, lang)
            all_results.extend(results)

    print_table(all_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "cuda_graph_results.json"
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    console.print(f"\n[green]Results saved → {out}[/green]")


if __name__ == "__main__":
    main()
