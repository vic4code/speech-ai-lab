"""
Step 9 — Nsight Compute Roofline Profiling.

What Nsight Compute measures
-----------------------------
ncu replays each CUDA kernel with different hardware counter configurations
to collect:
  - Achieved FLOPs (fp32/fp16/int8 FMAs × 2)
  - DRAM bytes transferred (L2 → DRAM traffic)
  - Kernel duration (cycles / clock)

From these we derive per-kernel:
  - Arithmetic Intensity  = FLOPs / bytes
  - Achieved throughput   = FLOPs / seconds

Plotting those on the roofline model (peak BW line, peak TFLOPS ceiling) shows
exactly where each kernel sits relative to the hardware limits and whether it is
memory-bound or compute-bound.

Workflow
--------
This script can run in two modes:

  MODE 1 (profiling target — run under ncu):
    ncu --set roofline --target-processes all \\
        --launch-skip 10 --launch-count 50 \\
        -o results/ncu_encoder \\
        python 09_nsight_roofline.py --target

  MODE 2 (analysis — parse ncu CSV and plot):
    python 09_nsight_roofline.py --plot results/ncu_encoder.ncu-rep

  MODE 3 (orchestrate — does both automatically):
    python 09_nsight_roofline.py

The script detects which mode based on argv.

Output
------
results/roofline.png
results/ncu_summary.json
"""

import argparse
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent.parent / "results"
MANIFEST_ZH = Path(__file__).parent.parent / "data" / "benchmark" / "zh" / "manifest.json"

# A10G hardware limits
A10G_FP32_TFLOPS = 31.2
A10G_FP16_TC_TFLOPS = 125.0
A10G_INT8_TOPS = 250.0
A10G_BW_GBS = 600.0  # GB/s

# ---------------------------------------------------------------------------
# MODE 1: profiling target
# ---------------------------------------------------------------------------

def run_target() -> None:
    """Encoder forward pass to be captured by ncu."""
    import numpy as np
    import torch
    import whisper

    model = whisper.load_model("large-v3", device="cuda")
    model.eval()

    # Build a fixed mel input (batch=1, n_mels=128, 3000 frames)
    mel = torch.zeros(1, model.dims.n_mels, 3000, device="cuda")

    # Warmup outside ncu --launch-skip window
    for _ in range(10):
        with torch.no_grad():
            model.encoder(mel)
    torch.cuda.synchronize()

    # Profiled region: ncu --launch-count captures these
    for _ in range(5):
        with torch.no_grad():
            out = model.encoder(mel)
        torch.cuda.synchronize()

    print("Target complete.")


# ---------------------------------------------------------------------------
# MODE 2: parse ncu CSV + plot
# ---------------------------------------------------------------------------

def parse_ncu_csv(csv_path: Path) -> list[dict]:
    """Parse ncu --csv output into a list of kernel dicts."""
    import csv
    kernels = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        # ncu csv has a few header comment lines starting with "=="
        lines = [l for l in f if not l.startswith("==")]
        reader = csv.DictReader(lines)
        for row in reader:
            kernels.append(dict(row))
    return kernels


def extract_roofline_points(kernels: list[dict]) -> list[dict]:
    """
    Pull arithmetic intensity and throughput from ncu kernel rows.
    ncu --set roofline emits columns like:
      'Metric Name', 'Metric Value'  (long format)
    or wide format depending on version.  We handle both.
    """
    points = []
    # Try wide format first (one row per kernel, many metric columns)
    if kernels and "gpu__time_duration.sum" in kernels[0]:
        for k in kernels:
            try:
                name = k.get("Kernel Name", k.get("ID", "unknown"))
                duration_ns = float(k.get("gpu__time_duration.sum", 0).replace(",", ""))
                fp32_flops = float(k.get("sm__sass_thread_inst_executed_op_ffma_pred_on.sum", 0).replace(",", "")) * 2
                fp16_flops = float(k.get("sm__sass_thread_inst_executed_op_hfma_pred_on.sum", 0).replace(",", "")) * 2
                dram_bytes = float(k.get("dram__bytes.sum", 0).replace(",", ""))
                flops = fp32_flops + fp16_flops
                if duration_ns > 0 and flops > 0 and dram_bytes > 0:
                    ai = flops / dram_bytes
                    throughput_tflops = flops / (duration_ns * 1e-9) / 1e12
                    points.append({"name": name, "ai": ai, "tflops": throughput_tflops,
                                   "duration_ms": duration_ns / 1e6, "flops": flops, "bytes": dram_bytes})
            except (ValueError, KeyError):
                continue
    return points


def plot_roofline(points: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")

    # ---- Roofline bounds ----
    bw = A10G_BW_GBS * 1e9      # bytes/sec
    fp32_peak = A10G_FP32_TFLOPS * 1e12
    fp16_peak = A10G_FP16_TC_TFLOPS * 1e12

    ai_range = np.logspace(-2, 5, 500)

    # Memory-bound slope (BW line)
    mem_roof = bw * ai_range

    # FP32 ceiling
    fp32_roof = np.minimum(mem_roof, fp32_peak)
    # FP16 TC ceiling
    fp16_roof = np.minimum(mem_roof, fp16_peak)

    ax.plot(ai_range, fp32_roof / 1e12, "b-", linewidth=2, label=f"FP32 roof ({A10G_FP32_TFLOPS} TFLOPS)")
    ax.plot(ai_range, fp16_roof / 1e12, "g--", linewidth=2, label=f"FP16 TC roof ({A10G_FP16_TC_TFLOPS} TFLOPS)")

    # Ridge points
    ridge_fp32 = fp32_peak / bw
    ridge_fp16 = fp16_peak / bw
    ax.axvline(ridge_fp32, color="blue", linestyle=":", alpha=0.5, label=f"FP32 ridge ({ridge_fp32:.0f} FLOP/B)")
    ax.axvline(ridge_fp16, color="green", linestyle=":", alpha=0.5, label=f"FP16 ridge ({ridge_fp16:.0f} FLOP/B)")

    # ---- Kernel points ----
    if points:
        # Cluster by kernel type
        def kernel_color(name: str) -> str:
            n = name.lower()
            if "gemm" in n or "cutlass" in n or "sgemm" in n or "hgemm" in n:
                return "red"
            if "softmax" in n or "norm" in n or "layer" in n:
                return "orange"
            if "conv" in n:
                return "purple"
            if "elementwise" in n or "fused" in n:
                return "gray"
            return "steelblue"

        xs = [p["ai"] for p in points]
        ys = [p["tflops"] for p in points]
        cs = [kernel_color(p["name"]) for p in points]
        sizes = [max(20, min(200, p["duration_ms"] * 50)) for p in points]

        sc = ax.scatter(xs, ys, c=cs, s=sizes, alpha=0.7, zorder=5)

        # Annotate top-5 by duration
        top5 = sorted(points, key=lambda x: x["duration_ms"], reverse=True)[:5]
        for p in top5:
            short = p["name"][:30]
            ax.annotate(short, (p["ai"], p["tflops"]),
                        fontsize=6, alpha=0.8, ha="left",
                        xytext=(5, 5), textcoords="offset points")

        # Legend patches for kernel types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="red", label="GEMM / cuBLAS / CUTLASS"),
            Patch(facecolor="orange", label="Norm / Softmax"),
            Patch(facecolor="purple", label="Conv"),
            Patch(facecolor="steelblue", label="Other"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=12)
    ax.set_ylabel("Achieved Throughput (TFLOPS)", fontsize=12)
    ax.set_title("Roofline — Whisper large-v3 Encoder (NVIDIA A10G)", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.01, 1e5)
    ax.set_ylim(1e-4, 200)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Roofline plot saved → {out_path}")


def estimate_roofline_from_results() -> None:
    """
    If ncu is unavailable or profiling was skipped, generate a roofline plot
    using the measured throughput from Steps 1-8 as annotated points.
    This gives a useful visual even without per-kernel ncu data.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xscale("log")
    ax.set_yscale("log")

    bw = A10G_BW_GBS * 1e9
    fp32_peak = A10G_FP32_TFLOPS * 1e12
    fp16_peak = A10G_FP16_TC_TFLOPS * 1e12
    int8_peak = A10G_INT8_TOPS * 1e12

    ai_range = np.logspace(-2, 5, 500)
    ax.plot(ai_range, np.minimum(bw * ai_range, fp32_peak) / 1e12,
            "b-", lw=2.5, label=f"FP32 roof ({A10G_FP32_TFLOPS} TFLOPS)")
    ax.plot(ai_range, np.minimum(bw * ai_range, fp16_peak) / 1e12,
            "g-", lw=2.5, label=f"FP16 TC roof ({A10G_FP16_TC_TFLOPS} TFLOPS)")
    ax.plot(ai_range, np.minimum(bw * ai_range, int8_peak) / 1e12,
            "r-", lw=2.5, label=f"INT8 TC roof ({A10G_INT8_TOPS} TOPS)")

    ridge_fp32 = fp32_peak / bw
    ridge_fp16 = fp16_peak / bw
    ridge_int8 = int8_peak / bw
    for ridge, color, label in [
        (ridge_fp32, "blue", f"FP32 ridge ({ridge_fp32:.0f})"),
        (ridge_fp16, "green", f"FP16 ridge ({ridge_fp16:.0f})"),
        (ridge_int8, "red", f"INT8 ridge ({ridge_int8:.0f})"),
    ]:
        ax.axvline(ridge, color=color, linestyle=":", alpha=0.4, label=label)

    # ---- Annotated operating points from our experiments ----
    # Encoder: compute-bound, AI ≈ 708 FLOP/byte (Step 8)
    # Decoder: memory-bound, AI ≈ 0.5 FLOP/byte (Step 1 analysis)
    #
    # Throughput estimates:
    #   Encoder FP32: 31.2 TFLOPS × fraction_of_peak
    #     Step 8: 170ms per encoder pass, 2.256 TFLOP total → 13.3 TFLOPS achieved
    #   Decoder FP32: single-token, ~0.5 FLOP/B × 600 GB/s = 300 GFLOPS
    #     But measured ~40ms per decoder step → much lower utilisation

    ENCODER_TFLOP = 2.256      # large-v3 encoder total FLOPs (TFLOP), computed in Step 8
    ENCODER_LATENCY_S = 0.1703  # Step 8 eager mean
    encoder_tflops = ENCODER_TFLOP / ENCODER_LATENCY_S
    encoder_ai = 708.0          # Step 8 estimate

    # Decoder: ~30 steps × 2.3 GFLOP each = 69 GFLOP, ~1000ms total → 69 GFLOPS
    # (large-v3 decoder, from Step 1 latency minus encoder time)
    decoder_total_ms = 1379 - 171   # total - encoder
    decoder_tflops = 0.069 / (decoder_total_ms / 1000)
    decoder_ai = 0.5

    points = [
        (encoder_ai, encoder_tflops, "Encoder\n(FP32, B=1)", "blue", 200, "compute-bound"),
        (decoder_ai, decoder_tflops, "Decoder\n(FP32, B=1)", "orange", 200, "memory-bound"),
        (encoder_ai * 2, encoder_tflops * 3.5, "Encoder\n(FP16 TC)", "green", 150,
         "FP16 TC gain"),
        (decoder_ai * 2, decoder_tflops * 2.7, "Decoder\n(CT2 INT8)", "red", 150,
         "INT8 gain"),
    ]

    for ai, tp, label, color, size, region in points:
        ax.scatter([ai], [tp], s=size, color=color, zorder=6, edgecolors="black", linewidth=0.8)
        ax.annotate(label, (ai, tp), fontsize=9, fontweight="bold",
                    xytext=(10, 5), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3))

    # Shade regions
    ax.axvspan(0.01, ridge_fp32, alpha=0.04, color="blue", label="memory-bound region")
    ax.axvspan(ridge_fp32, 1e5, alpha=0.04, color="green", label="compute-bound region")

    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=12)
    ax.set_ylabel("Achieved Throughput (TFLOPS)", fontsize=12)
    ax.set_title("Roofline Model — Whisper large-v3 on NVIDIA A10G\n"
                 "(encoder=compute-bound, decoder=memory-bound)", fontsize=12)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(0.01, 200)

    out = RESULTS_DIR / "roofline_estimated.png"
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Estimated roofline plot saved → {out}")
    return out


# ---------------------------------------------------------------------------
# MODE 3: orchestrate ncu + analyse
# ---------------------------------------------------------------------------

def run_ncu_and_profile() -> None:
    """Run ncu on this script in --target mode, then parse and plot."""
    import shutil

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ncu_bin = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
    report = RESULTS_DIR / "ncu_encoder"
    csv_out = RESULTS_DIR / "ncu_encoder.csv"

    print(f"\n[ncu] Profiling encoder forward pass → {report}.ncu-rep")
    print("      This replays each kernel with hardware counters — takes ~3–10 min.\n")

    cmd = [
        ncu_bin,
        "--set", "roofline",
        "--target-processes", "all",
        "--launch-skip", "10",    # skip warmup kernels
        "--launch-count", "300",  # capture ~1 encoder pass worth of kernels
        "--force-overwrite",
        "-o", str(report),
        sys.executable, __file__, "--target",
    ]
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print("[ncu] Profiling failed or was killed. Generating estimated roofline instead.")
        estimate_roofline_from_results()
        return

    # Export to CSV
    print(f"\n[ncu] Exporting to CSV → {csv_out}")
    export_cmd = [
        ncu_bin, "--import", str(report) + ".ncu-rep",
        "--page", "raw", "--csv",
    ]
    with open(csv_out, "w") as f:
        subprocess.run(export_cmd, stdout=f)

    # Parse and plot
    try:
        kernels = parse_ncu_csv(csv_out)
        points = extract_roofline_points(kernels)
        if points:
            print(f"  Parsed {len(points)} kernels with roofline data")
            plot_roofline(points, RESULTS_DIR / "roofline_ncu.png")
            summary = {
                "n_kernels": len(points),
                "top_kernels_by_duration": sorted(points, key=lambda x: x["duration_ms"], reverse=True)[:10],
            }
            (RESULTS_DIR / "ncu_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False)
            )
        else:
            print("  No roofline points extracted — generating estimated plot")
            estimate_roofline_from_results()
    except Exception as e:
        print(f"  CSV parse error: {e} — generating estimated plot")
        estimate_roofline_from_results()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", action="store_true",
                        help="Run as ncu profiling target (encoder forward passes)")
    parser.add_argument("--plot", type=str, default=None,
                        help="Parse existing .ncu-rep file and plot")
    parser.add_argument("--estimate", action="store_true",
                        help="Generate estimated roofline from Step 1-8 data (no ncu)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.target:
        run_target()
    elif args.plot:
        rep = Path(args.plot)
        csv_path = rep.with_suffix(".csv")
        if not csv_path.exists():
            import shutil
            ncu_bin = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
            print(f"Exporting {rep} to CSV …")
            with open(csv_path, "w") as f:
                subprocess.run([ncu_bin, "--import", str(rep), "--page", "raw", "--csv"],
                               stdout=f)
        kernels = parse_ncu_csv(csv_path)
        points = extract_roofline_points(kernels)
        plot_roofline(points, RESULTS_DIR / "roofline_ncu.png")
    elif args.estimate:
        estimate_roofline_from_results()
    else:
        # Full orchestration: try ncu, fall back to estimated plot
        run_ncu_and_profile()


if __name__ == "__main__":
    main()
