"""
Data augmentation pipeline for ASR training robustness.

Demonstrates: speed perturbation, additive noise injection, SpecAugment-style
time/frequency masking. Augmentation is applied on-the-fly during training or
offline to expand the training set before fine-tuning.
"""

import argparse
import random
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from datasets import load_from_disk, Dataset


def speed_perturb(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Shift pitch-neutral tempo via librosa time-stretch."""
    return librosa.effects.time_stretch(audio, rate=rate)


def add_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise at the requested SNR level."""
    rms_signal = np.sqrt(np.mean(audio**2))
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.normal(0, rms_noise, len(audio)).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def spec_augment_time_mask(audio: np.ndarray, sr: int, max_mask_s: float = 0.2) -> np.ndarray:
    """Zero out a random time segment (time masking from SpecAugment)."""
    mask_len = int(random.uniform(0, max_mask_s) * sr)
    start = random.randint(0, max(0, len(audio) - mask_len))
    augmented = audio.copy()
    augmented[start : start + mask_len] = 0.0
    return augmented


def augment_example(example: dict[str, Any]) -> dict[str, Any]:
    """Apply a randomly chosen augmentation to a single dataset example."""
    audio = np.array(example["audio"]["array"], dtype=np.float32)
    sr: int = example["audio"]["sampling_rate"]

    choice = random.random()
    if choice < 0.33:
        rate = random.choice([0.9, 1.1])
        audio = speed_perturb(audio, sr, rate)
    elif choice < 0.66:
        snr = random.uniform(15, 30)
        audio = add_noise(audio, snr)
    else:
        audio = spec_augment_time_mask(audio, sr)

    example["audio"]["array"] = audio.tolist()
    return example


def augment(input_dir: Path, output_dir: Path) -> None:
    print(f"Loading dataset from {input_dir} ...")
    ds: Dataset = load_from_disk(str(input_dir))

    print(f"Augmenting {len(ds):,} examples ...")
    ds = ds.map(augment_example, num_proc=4)

    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    print(f"Augmented dataset saved → {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment ASR training data")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    augment(args.input, args.output)


if __name__ == "__main__":
    main()
