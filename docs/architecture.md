# System Architecture

## End-to-End Pipeline

```
Audio Input
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Ingestion Layer                                        │
│  FastAPI (07-production/api/main.py)                    │
│  • Sync endpoint: /transcribe                           │
│  • Async endpoint: /transcribe/async → Celery queue     │
│  • SSE endpoint: /transcribe/stream                     │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Inference Layer                                        │
│  Whisper large-v3 (faster-whisper CTranslate2)          │
│  or NVIDIA Triton (04-serving)                          │
│  • Dynamic batching: up to 8 requests                   │
│  • Compute type: int8_float16                           │
│  • Multi-instance: 1 model instance per GPU             │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  LLM Integration Layer (optional)                       │
│  Claude claude-opus-4-7 via Anthropic API               │
│  • Summarization, Q&A, RAG                              │
│  • Streaming responses via SSE                          │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Observability                                          │
│  Prometheus → Grafana (07-production/monitoring)        │
│  • ASR latency P50/P95/P99                              │
│  • GPU utilization + HBM usage                          │
│  • Request throughput + error rate                      │
└─────────────────────────────────────────────────────────┘
```

## GPU Memory Budget (single A100-80GB)

| Component | Memory |
|-----------|--------|
| Whisper large-v3 FP16 weights | ~3 GB |
| Whisper large-v3 INT8 weights | ~1.5 GB |
| KV cache (batch=8, 30s audio) | ~2 GB |
| OS + CUDA runtime | ~2 GB |
| **Available for batching headroom** | **~70 GB** |

## Scaling Strategy

- **Vertical (single GPU):** INT8 quantization + CTranslate2 → 4× throughput vs FP32
- **Horizontal (multi-GPU):** Tensor parallelism for large-v3; replicated instances behind load balancer for smaller models
- **Multi-node:** NCCL all-reduce over InfiniBand/NVLink for P2P bandwidth
