"""
FastAPI production inference server for Whisper ASR.

Exposes synchronous transcription, async job queuing, and SSE streaming.
Includes Prometheus metrics, structured logging, and graceful startup/shutdown.
"""

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .worker import transcribe_task
from .streaming import sse_transcribe

# Prometheus metrics
REQUEST_COUNT = Counter("asr_requests_total", "Total transcription requests", ["status"])
REQUEST_LATENCY = Histogram("asr_latency_seconds", "Transcription latency", buckets=[0.1, 0.5, 1, 2, 5, 10, 30])

_model: WhisperModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    global _model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = WhisperModel("large-v3", device=device, compute_type="float16" if device == "cuda" else "int8")
    yield
    _model = None


app = FastAPI(title="Speech AI Lab — ASR API", version="1.0.0", lifespan=lifespan)


def _read_audio(file_bytes: bytes) -> np.ndarray:
    buf = io.BytesIO(file_bytes)
    audio, _ = sf.read(buf, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


@app.post("/transcribe")
async def transcribe_sync(audio: UploadFile = File(...)) -> JSONResponse:
    """Synchronous transcription — waits for result before returning."""
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    file_bytes = await audio.read()
    arr = _read_audio(file_bytes)

    t0 = time.perf_counter()
    segments, info = _model.transcribe(arr, language="en")
    text = " ".join(s.text for s in segments).strip()
    latency = time.perf_counter() - t0

    REQUEST_COUNT.labels(status="success").inc()
    REQUEST_LATENCY.observe(latency)

    return JSONResponse({
        "text": text,
        "language": info.language,
        "duration_s": round(info.duration, 2),
        "latency_s": round(latency, 3),
    })


@app.post("/transcribe/async")
async def transcribe_async(audio: UploadFile = File(...)) -> JSONResponse:
    """Enqueue transcription job and return task ID immediately."""
    file_bytes = await audio.read()
    task = transcribe_task.delay(file_bytes)
    return JSONResponse({"task_id": task.id, "status": "queued"})


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str) -> JSONResponse:
    task = transcribe_task.AsyncResult(task_id)
    if task.state == "PENDING":
        return JSONResponse({"task_id": task_id, "status": "pending"})
    if task.state == "SUCCESS":
        return JSONResponse({"task_id": task_id, "status": "success", "result": task.result})
    return JSONResponse({"task_id": task_id, "status": task.state})


@app.post("/transcribe/stream")
async def transcribe_stream(audio: UploadFile = File(...)) -> StreamingResponse:
    """SSE streaming transcription — yields segments as they arrive."""
    file_bytes = await audio.read()
    arr = _read_audio(file_bytes)
    return StreamingResponse(sse_transcribe(arr, _model), media_type="text/event-stream")


@app.get("/metrics")
async def metrics() -> StreamingResponse:
    return StreamingResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "model_loaded": _model is not None})
