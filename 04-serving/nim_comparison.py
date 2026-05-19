"""
Compare Triton self-hosted serving latency against NVIDIA NIM microservice.

NIM (NVIDIA Inference Microservices) packages optimized models with a
standardized REST API, TensorRT-LLM backends, and production-ready health
checks — deployable with a single docker run command.

This script sends the same audio to both endpoints and prints a comparison table.
Set TRITON_URL and NIM_URL environment variables or use the CLI flags.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile as sf
from rich.console import Console
from rich.table import Table

console = Console()


def transcribe_triton(audio: np.ndarray, sr: int, server_url: str) -> tuple[str, float]:
    """Send to Triton gRPC and return (transcript, latency_ms)."""
    import tritonclient.grpc as gc

    client = gc.InferenceServerClient(url=server_url)

    audio_in = gc.InferInput("audio_signal", audio.shape, "FP32")
    audio_in.set_data_from_numpy(audio)
    sr_in = gc.InferInput("sample_rate", [1], "INT32")
    sr_in.set_data_from_numpy(np.array([sr], dtype=np.int32))

    t0 = time.perf_counter()
    resp = client.infer("whisper", inputs=[audio_in, sr_in], outputs=[gc.InferRequestedOutput("transcription")])
    latency = (time.perf_counter() - t0) * 1000
    transcript = resp.as_numpy("transcription")[0].decode()
    return transcript, round(latency, 2)


def transcribe_nim(audio: np.ndarray, sr: int, nim_url: str) -> tuple[str, float]:
    """Send to NIM REST endpoint and return (transcript, latency_ms)."""
    import io
    import soundfile as sf2

    buf = io.BytesIO()
    sf2.write(buf, audio, sr, format="WAV")
    buf.seek(0)

    t0 = time.perf_counter()
    resp = requests.post(
        f"{nim_url}/v1/asr",
        files={"audio": ("audio.wav", buf, "audio/wav")},
        timeout=30,
    )
    latency = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    transcript = resp.json().get("text", "")
    return transcript, round(latency, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Triton vs NIM latency comparison")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--triton", default=os.getenv("TRITON_URL", "localhost:8001"))
    parser.add_argument("--nim", default=os.getenv("NIM_URL", "http://localhost:9000"))
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    audio, sr = sf.read(str(args.audio), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    results: dict[str, list[float]] = {"triton": [], "nim": []}

    for i in range(args.runs):
        _, lat = transcribe_triton(audio, sr, args.triton)
        results["triton"].append(lat)
        _, lat = transcribe_nim(audio, sr, args.nim)
        results["nim"].append(lat)
        console.print(f"Run {i+1}: Triton {results['triton'][-1]:.1f}ms | NIM {results['nim'][-1]:.1f}ms")

    table = Table(title="Triton vs NIM")
    table.add_column("System")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    for name, lats in results.items():
        table.add_row(name.capitalize(), f"{np.mean(lats):.1f}", f"{np.percentile(lats, 95):.1f}")
    console.print(table)


if __name__ == "__main__":
    main()
