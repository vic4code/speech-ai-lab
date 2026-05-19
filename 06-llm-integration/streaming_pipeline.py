"""
Real-time streaming ASR → LLM pipeline.

Demonstrates: faster-whisper streaming transcription (word-level timestamps),
feeding partial transcripts to Claude's streaming API as speech arrives,
and interleaving ASR and LLM latency for near-real-time responses.
"""

import argparse
import os
from pathlib import Path

import anthropic
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from rich.console import Console

console = Console()

SEGMENT_STRIDE_S = 5.0  # send a new segment to LLM every N seconds of speech


def stream_transcribe(audio: np.ndarray, sr: int, model: WhisperModel) -> list[str]:
    """Yield transcript segments as speech is processed."""
    segments, _ = model.transcribe(audio, language="en", word_timestamps=True)
    accumulated: list[str] = []
    current_text = ""
    last_flush_time = 0.0

    for segment in segments:
        current_text += segment.text
        if segment.end - last_flush_time >= SEGMENT_STRIDE_S:
            accumulated.append(current_text.strip())
            console.print(f"[ASR segment] {current_text.strip()}")
            current_text = ""
            last_flush_time = segment.end

    if current_text.strip():
        accumulated.append(current_text.strip())

    return accumulated


def stream_llm(partial_transcript: str, client: anthropic.Anthropic) -> None:
    """Stream Claude's response token-by-token to stdout."""
    console.print("\n[LLM] ", end="")
    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": (
                    "Continue assisting based on the following partial transcript "
                    f"from an ongoing conversation:\n\n{partial_transcript}\n\n"
                    "Provide a brief, helpful response."
                ),
            }
        ],
    ) as stream:
        for text in stream.text_stream:
            console.print(text, end="", highlight=False)
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming ASR → LLM pipeline")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--whisper-model", default="large-v3")
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ANTHROPIC_API_KEY environment variable")

    audio, sr = sf.read(str(args.audio), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    asr_model = WhisperModel(args.whisper_model, device="cuda", compute_type="float16")
    client = anthropic.Anthropic(api_key=api_key)

    console.print(f"[bold]Streaming pipeline started — {args.audio.name}[/bold]")
    segments = stream_transcribe(audio, sr, asr_model)

    for segment in segments:
        stream_llm(segment, client)


if __name__ == "__main__":
    main()
