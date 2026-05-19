"""
Celery worker for async/batch Whisper transcription jobs.

Using a task queue decouples request acceptance from GPU execution,
enabling fair queuing, retries, and horizontal scaling of workers
independently from the API tier.
"""

import io

import numpy as np
import soundfile as sf
import torch
from celery import Celery
from faster_whisper import WhisperModel

BROKER_URL = "redis://localhost:6379/0"
RESULT_BACKEND = "redis://localhost:6379/1"

celery_app = Celery("asr_worker", broker=BROKER_URL, backend=RESULT_BACKEND)
celery_app.conf.task_routes = {"worker.transcribe_task": {"queue": "transcription"}}

_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = WhisperModel("large-v3", device=device, compute_type="float16" if device == "cuda" else "int8")
    return _model


@celery_app.task(bind=True, max_retries=3, default_retry_delay=5)
def transcribe_task(self, file_bytes: bytes) -> dict:
    """Transcribe audio bytes and return structured result."""
    try:
        buf = io.BytesIO(file_bytes)
        audio, sr = sf.read(buf, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        model = get_model()
        segments, info = model.transcribe(audio, language="en")
        text = " ".join(s.text for s in segments).strip()

        return {
            "text": text,
            "language": info.language,
            "duration_s": round(info.duration, 2),
        }
    except Exception as exc:
        raise self.retry(exc=exc)
