"""
Download and cache benchmark datasets.

Datasets
--------
zh : FLEURS cmn_hans_cn test, 100 utterances
     Mandarin Chinese; sourced from FLORES-101; standard multilingual CER benchmark.
en : LibriSpeech test-clean, 100 utterances
     Standard English WER benchmark; official eval set for NVIDIA Parakeet.

Notes
-----
Parakeet is English-only; Chinese experiments benchmark Whisper only.
Streaming mode is used so only 100 samples are downloaded (not the full corpus).
Saves to data/benchmark/{zh,en}/ as .wav files + manifest.json.
"""

import json
import warnings
from itertools import islice
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.progress import track

warnings.filterwarnings("ignore")

console = Console()

SAMPLE_SIZE = 100
SAMPLE_RATE = 16_000
BASE_DIR = Path(__file__).parent.parent / "data" / "benchmark"


def resample_if_needed(audio_array: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == SAMPLE_RATE:
        return audio_array
    import librosa
    return librosa.resample(audio_array.astype(np.float32), orig_sr=orig_sr, target_sr=SAMPLE_RATE)


def save_split(
    lang: str,
    hf_dataset_name: str,
    hf_config: str,
    hf_split: str,
    text_field: str,
    audio_field: str = "audio",
) -> None:
    out_dir = BASE_DIR / lang
    manifest_path = out_dir / "manifest.json"

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        console.print(f"[yellow]{lang}: manifest exists with {len(existing)} samples — skipping[/yellow]")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    console.print(f"\n[bold cyan]Streaming {lang}: {hf_dataset_name}/{hf_config} split={hf_split}[/bold cyan]")
    ds_stream = load_dataset(
        hf_dataset_name,
        hf_config,
        split=hf_split,
        streaming=True,
        trust_remote_code=True,
    )

    manifest = []
    samples = list(islice(ds_stream, SAMPLE_SIZE))
    console.print(f"  Fetched {len(samples)} samples")

    for i, sample in enumerate(track(samples, description=f"  Saving {lang} audio")):
        audio_data = sample[audio_field]
        array = np.array(audio_data["array"], dtype=np.float32)
        sr = audio_data["sampling_rate"]
        array = resample_if_needed(array, sr)

        wav_path = out_dir / f"{i:04d}.wav"
        sf.write(str(wav_path), array, SAMPLE_RATE)

        manifest.append({
            "id": i,
            "audio_path": str(wav_path),
            "reference": sample[text_field].strip(),
            "duration_s": round(len(array) / SAMPLE_RATE, 3),
        })

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    total_dur = sum(m["duration_s"] for m in manifest)
    avg_dur = total_dur / len(manifest)
    console.print(
        f"  [green]Saved {len(manifest)} samples — "
        f"total {total_dur:.1f}s, avg {avg_dur:.1f}s/sample → {out_dir}[/green]"
    )


def main() -> None:
    console.rule("[bold]Benchmark Dataset Preparation[/bold]")

    # Chinese: FLEURS Mandarin test set
    save_split(
        lang="zh",
        hf_dataset_name="google/fleurs",
        hf_config="cmn_hans_cn",
        hf_split="test",
        text_field="transcription",
    )

    # English: LibriSpeech test-clean (Parakeet official eval)
    save_split(
        lang="en",
        hf_dataset_name="openslr/librispeech_asr",
        hf_config="clean",
        hf_split="test",
        text_field="text",
    )

    console.rule("[bold green]Done[/bold green]")
    console.print(f"Manifests at: {BASE_DIR}/{{zh,en}}/manifest.json")


if __name__ == "__main__":
    main()
