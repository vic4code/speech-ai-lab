# Phase 7 — Production Deployment

FastAPI async inference server, Celery workers for batch jobs, and a Grafana monitoring dashboard.

## Structure

```
07-production/
├── api/
│   ├── main.py       — FastAPI app: /transcribe (sync) and /transcribe/stream (SSE)
│   ├── worker.py     — Celery worker for async/batch transcription jobs
│   └── streaming.py  — Server-Sent Events (SSE) response helpers
└── monitoring/
    └── dashboard.py  — Prometheus metrics exporter + Grafana dashboard JSON
```

## How to Run

```bash
# Start FastAPI server
uvicorn 07-production.api.main:app --host 0.0.0.0 --port 8000

# Start Celery worker (requires Redis)
celery -A 07-production.api.worker worker --loglevel=info --concurrency=1 -Q transcription

# Full stack via Docker Compose (Phase 7)
docker-compose up
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/transcribe` | Synchronous transcription, returns JSON |
| POST | `/transcribe/async` | Enqueue job, returns task ID |
| GET | `/tasks/{task_id}` | Poll job status and result |
| GET | `/transcribe/stream` | SSE streaming transcription |
| GET | `/metrics` | Prometheus metrics |
| GET | `/health` | Health check |
