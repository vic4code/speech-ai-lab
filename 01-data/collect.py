"""
Collect audio datasets from HuggingFace Hub for ASR training and evaluation.

Demonstrates: HuggingFace datasets API, streaming mode for large corpora,
saving to disk in a reproducible layout.
"""

import argparse
from pathlib import Path

from datasets import load_dataset


def collect(dataset_name: str, split: str, output_dir: Path) -> None:
    """Download and save a HuggingFace audio dataset split."""
    print(f"Downloading {dataset_name} / {split} ...")
    ds = load_dataset(dataset_name, "clean", split=split, trust_remote_code=True)

    out = output_dir / dataset_name.replace("/", "_") / split
    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    print(f"Saved {len(ds):,} examples → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ASR datasets")
    parser.add_argument("--dataset", default="librispeech_asr")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    collect(args.dataset, args.split, args.output)


if __name__ == "__main__":
    main()
