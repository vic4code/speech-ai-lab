"""
ASR → LLM pipeline: transcribe audio with Whisper, then summarize with Claude.

Demonstrates: chaining a local GPU model (Whisper) with a cloud API (Anthropic),
structured prompting for meeting summarization, and error handling across
the pipeline boundary.
"""

import argparse
import os
from pathlib import Path

import anthropic
import numpy as np
import soundfile as sf
import whisper
import torch
from rich.console import Console

console = Console()


def transcribe(audio_path: Path, model_name: str = "large-v3") -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"Transcribing {audio_path.name} with Whisper {model_name} on {device} ...")
    model = whisper.load_model(model_name, device=device)
    audio, _ = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    result = model.transcribe(audio, language="en", fp16=(device == "cuda"))
    return result["text"].strip()


def summarize(transcript: str, client: anthropic.Anthropic) -> str:
    console.print("Summarizing with Claude ...")
    message = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a meeting assistant. Given the transcript below, produce:\n"
                    "1. A 2-sentence executive summary\n"
                    "2. Key decisions made\n"
                    "3. Action items with owners (if mentioned)\n\n"
                    f"Transcript:\n{transcript}"
                ),
            }
        ],
    )
    return message.content[0].text


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR → LLM summarization pipeline")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--whisper-model", default="large-v3")
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ANTHROPIC_API_KEY environment variable")

    transcript = transcribe(args.audio, args.whisper_model)
    console.rule("Transcript")
    console.print(transcript)

    client = anthropic.Anthropic(api_key=api_key)
    summary = summarize(transcript, client)
    console.rule("Summary")
    console.print(summary)


if __name__ == "__main__":
    main()
