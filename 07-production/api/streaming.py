"""
Server-Sent Events (SSE) helpers for streaming ASR responses.

Each Whisper segment is yielded as an SSE event the moment it is decoded,
allowing the client to display partial transcription in real time without
waiting for the full audio to be processed.
"""

import json
from typing import AsyncGenerator

import numpy as np
from faster_whisper import WhisperModel


async def sse_transcribe(audio: np.ndarray, model: WhisperModel) -> AsyncGenerator[bytes, None]:
    """Yield SSE-formatted bytes for each transcript segment."""
    segments, info = model.transcribe(audio, language="en", word_timestamps=False)

    yield _sse_event("metadata", {"language": info.language, "duration_s": round(info.duration, 2)})

    full_text = []
    for segment in segments:
        text = segment.text.strip()
        full_text.append(text)
        yield _sse_event("segment", {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": text,
        })

    yield _sse_event("done", {"full_text": " ".join(full_text)})


def _sse_event(event_type: str, data: dict) -> bytes:
    payload = json.dumps(data)
    return f"event: {event_type}\ndata: {payload}\n\n".encode("utf-8")
