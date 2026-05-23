"""
Filter and clean raw audio datasets before training.

Demonstrates: duration filtering, SNR estimation, transcript normalization.
Poor-quality training examples are a leading cause of degraded WER, so
aggressive filtering before fine-tuning outperforms augmentation of noisy data.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from datasets import load_from_disk, Dataset


MIN_DURATION_S = 1.0
MAX_DURATION_S = 30.0
MIN_SNR_DB = 20.0


def estimate_snr(audio: np.ndarray, sample_rate: int) -> float:
    """Rough SNR estimate: ratio of RMS signal to RMS of lowest-energy 10% of frames."""
    frame_len = sample_rate // 10  # 100 ms frames
    frames = [audio[i : i + frame_len] for i in range(0, len(audio) - frame_len, frame_len)]
    rms = np.array([np.sqrt(np.mean(f**2)) for f in frames])
    noise_floor = np.percentile(rms, 10)
    signal_rms = np.mean(rms)
    if noise_floor == 0:
        return float("inf")
    return float(20 * np.log10(signal_rms / noise_floor))


def is_clean(example: dict[str, Any]) -> bool:
    """Return True if example passes all quality filters."""
    audio = example["audio"]
    array: np.ndarray = np.array(audio["array"], dtype=np.float32)
    sr: int = audio["sampling_rate"]

    duration = len(array) / sr
    if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
        return False

    if estimate_snr(array, sr) < MIN_SNR_DB:
        return False

    transcript: str = example.get("text", "")
    if len(transcript.strip()) < 3:
        return False

    return True


def clean(input_dir: Path, output_dir: Path) -> None:
    print(f"Loading dataset from {input_dir} ...")
    ds: Dataset = load_from_disk(str(input_dir))
    before = len(ds)

    print(f"Filtering {before:,} examples (min={MIN_DURATION_S}s, max={MAX_DURATION_S}s, SNR≥{MIN_SNR_DB}dB) ...")
    ds = ds.filter(is_clean, num_proc=4)
    after = len(ds)
    print(f"Retained {after:,} / {before:,} ({100*after/before:.1f}%)")

    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    print(f"Saved clean dataset → {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw ASR dataset")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    clean(args.input, args.output)


if __name__ == "__main__":
    main()
