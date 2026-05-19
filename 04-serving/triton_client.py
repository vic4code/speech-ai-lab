"""
gRPC client for sending audio to a Triton Inference Server whisper endpoint.

Demonstrates: tritonclient usage, numpy → protobuf serialization,
async inference, and latency measurement from the client side.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import tritonclient.grpc as grpcclient
from rich.console import Console

console = Console()


def transcribe(audio_path: Path, server_url: str, model_name: str = "whisper") -> str:
    """Send audio to Triton and return the transcript."""
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono

    client = grpcclient.InferenceServerClient(url=server_url)

    audio_input = grpcclient.InferInput("audio_signal", audio.shape, "FP32")
    audio_input.set_data_from_numpy(audio)

    sr_input = grpcclient.InferInput("sample_rate", [1], "INT32")
    sr_input.set_data_from_numpy(np.array([sr], dtype=np.int32))

    outputs = [grpcclient.InferRequestedOutput("transcription")]

    t0 = time.perf_counter()
    response = client.infer(model_name=model_name, inputs=[audio_input, sr_input], outputs=outputs)
    latency_ms = (time.perf_counter() - t0) * 1000

    transcript = response.as_numpy("transcription")[0].decode("utf-8")
    console.print(f"Latency: [bold]{latency_ms:.1f} ms[/bold]")
    console.print(f"Transcript: [italic]{transcript}[/italic]")
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description="Triton gRPC ASR client")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--server", default="localhost:8001")
    parser.add_argument("--model", default="whisper")
    args = parser.parse_args()

    transcribe(args.audio, args.server, args.model)


if __name__ == "__main__":
    main()
