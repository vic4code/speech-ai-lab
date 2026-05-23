"""
Evaluate a pretrained Whisper model on a HuggingFace ASR dataset.

Demonstrates: model loading, greedy decoding, WER/CER computation with jiwer,
and structured result logging. Run before any fine-tuning to establish a
performance baseline.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import whisper
from datasets import load_dataset
from jiwer import wer, cer
from rich.console import Console
from rich.table import Table

console = Console()


def evaluate(model_name: str, dataset_name: str, split: str, num_samples: int) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold]Loading Whisper {model_name} on {device}[/bold]")
    model = whisper.load_model(model_name, device=device)

    console.print(f"Loading {dataset_name} / {split} ({num_samples} samples) ...")
    ds = load_dataset(dataset_name, "clean", split=f"{split}[:{num_samples}]", trust_remote_code=True)

    hypotheses, references = [], []
    for example in ds:
        audio = example["audio"]["array"]
        audio_fp32 = torch.tensor(audio, dtype=torch.float32)
        result = model.transcribe(audio_fp32.numpy(), language="en", fp16=(device == "cuda"))
        hypotheses.append(result["text"].strip().lower())
        references.append(example["text"].strip().lower())

    word_error_rate = wer(references, hypotheses)
    char_error_rate = cer(references, hypotheses)

    return {
        "model": model_name,
        "dataset": dataset_name,
        "split": split,
        "num_samples": num_samples,
        "wer": round(word_error_rate, 4),
        "cer": round(char_error_rate, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline ASR evaluation")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--dataset", default="librispeech_asr")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results/baseline_eval.json"))
    args = parser.parse_args()

    results = evaluate(args.model, args.dataset, args.split, args.num_samples)

    table = Table(title="Baseline Evaluation")
    table.add_column("Model")
    table.add_column("WER", justify="right")
    table.add_column("CER", justify="right")
    table.add_row(results["model"], f"{results['wer']:.2%}", f"{results['cer']:.2%}")
    console.print(table)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    console.print(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
