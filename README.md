# Speech AI Lab

> End-to-end ASR lifecycle portfolio — from raw audio to GPU-optimized production serving.
> Built around **OpenAI Whisper**, **NVIDIA Parakeet**, and **Claude** on AWS GPU instances.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What this is

A complete, runnable AI engineering portfolio demonstrating every stage of a production ASR system:

```
Raw Audio → Data Pipeline → Fine-Tuning → Inference Optimization → Serving → LLM Integration → Production API
```

Each phase is self-contained, documented, and benchmarked against real hardware metrics.  The Phase 3 inference optimization scripts are the centrepiece — they quantify the latency and accuracy trade-offs of FP32 → FP16 → INT8 quantization on NVIDIA GPUs with reproducible methodology.

---

## Results (Phase 3 — Inference Optimization)

> Run on AWS `g5.xlarge` (NVIDIA A10G 24 GB) · Whisper large-v3 · 5.1 s LibriSpeech clip

| Backend | Precision | P50 Latency (ms) | P95 Latency (ms) | GPU Mem (MB) | CER | Speedup |
|---------|-----------|:----------------:|:----------------:|:------------:|:---:|:-------:|
| PyTorch | FP32 | — | — | — | — | 1.00× |
| PyTorch | FP16 | — | — | — | — | —× |
| CTranslate2 | float16 | — | — | — | — | —× |
| CTranslate2 | int8\_float16 | — | — | — | — | —× |
| CTranslate2 | int8 | — | — | — | — | —× |

*Fill in after running Phase 3 scripts — see [How to Run](#how-to-run).*

---

## Repository Structure

```
speech-ai-lab/
├── 01-data/                   # Dataset collection, cleaning, augmentation
│   ├── collect.py             # HuggingFace dataset downloader
│   ├── clean.py               # Duration + SNR filter
│   └── augment.py             # Speed perturb, noise injection, SpecAugment
│
├── 02-training/               # Fine-tuning Whisper and Parakeet
│   ├── baseline_eval.py       # WER/CER on pretrained model
│   ├── lora_finetune.py       # LoRA adapters via PEFT (single GPU)
│   ├── ddp_train.py           # Multi-GPU DDP with torchrun
│   └── nemo_config.yaml       # NeMo config for Parakeet CTC
│
├── 03-inference-optimization/ # ← Start here
│   ├── 01_baseline_benchmark.py   # FP32 latency + GPU memory
│   ├── 02_fp16_benchmark.py       # FP16 comparison + speedup
│   ├── 03_int8_ctranslate2.py     # INT8 / mixed precision + CER
│   ├── 04_tensorrt_compile.py     # torch.compile TRT backend
│   └── 05_nsight_profile.sh       # Nsight Systems kernel trace
│
├── 04-serving/                # NVIDIA Triton Inference Server
│   ├── model_repository/      # Triton model config (dynamic batching)
│   ├── triton_client.py       # gRPC inference client
│   ├── load_test.sh           # perf_analyzer throughput test
│   └── nim_comparison.py      # Triton vs NIM latency comparison
│
├── 05-distributed/            # Multi-GPU scaling
│   ├── tensor_parallel.py     # Attention head sharding across GPUs
│   ├── gpu_monitor.py         # Real-time pynvml utilization table
│   └── scaling_experiment.py  # Strong-scaling latency curve
│
├── 06-llm-integration/        # ASR + LLM pipelines
│   ├── asr_llm_pipeline.py    # Whisper → Claude summarization
│   ├── rag_transcripts.py     # FAISS RAG over audio transcripts
│   └── streaming_pipeline.py  # Real-time SSE streaming
│
├── 07-production/             # Production-grade API
│   ├── api/main.py            # FastAPI: sync, async, SSE endpoints
│   ├── api/worker.py          # Celery task queue for batch jobs
│   ├── api/streaming.py       # SSE response helpers
│   └── monitoring/dashboard.py # Prometheus metrics + Grafana JSON
│
├── docs/
│   ├── architecture.md        # System architecture + GPU memory budget
│   ├── benchmark-methodology.md # Measurement protocol (warmup, sync)
│   └── interview-notes.md     # SA interview talking points
│
├── data/sample_audio/         # Drop .wav/.flac test files here
├── results/                   # Benchmark JSON outputs (git-ignored)
├── requirements.txt
└── docker-compose.yml         # Phase 7 production stack (placeholder)
```

---

## How to Run

### Prerequisites

```bash
# Python 3.10+ in a virtualenv
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

# Confirm CUDA is available
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Phase 3 — Inference Optimization (start here)

Run in order — each script loads results from the previous one for cumulative comparison.

```bash
# 1. FP32 baseline (Whisper tiny + large-v3)
python 03-inference-optimization/01_baseline_benchmark.py
# → results/baseline_results.json

# 2. FP16 comparison
python 03-inference-optimization/02_fp16_benchmark.py
# → results/fp16_results.json  (prints speedup vs FP32)

# 3. CTranslate2 INT8 / mixed precision + CER
python 03-inference-optimization/03_int8_ctranslate2.py
# → results/ctranslate2_results.json  (prints 5-row comparison table)

# 4. TensorRT compilation (requires torch-tensorrt)
python 03-inference-optimization/04_tensorrt_compile.py

# 5. Nsight Systems profiling (requires nsys CLI)
chmod +x 03-inference-optimization/05_nsight_profile.sh
./03-inference-optimization/05_nsight_profile.sh
```

### Phase 1 — Data

```bash
python 01-data/collect.py --dataset librispeech_asr --split validation
python 01-data/clean.py --input data/raw --output data/clean
python 01-data/augment.py --input data/clean/train --output data/augmented
```

### Phase 2 — Training

```bash
# Baseline WER
python 02-training/baseline_eval.py --model large-v3

# LoRA fine-tune (single GPU)
python 02-training/lora_finetune.py --model large-v3 --data data/clean/train

# Multi-GPU DDP (4 × GPU)
torchrun --nproc_per_node=4 02-training/ddp_train.py --model large-v3 --data data/clean/train
```

### Phase 4 — Triton Serving

```bash
# Start Triton server
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/04-serving/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

# Test inference
python 04-serving/triton_client.py --audio data/sample_audio/test.wav
```

### Phase 6 — LLM Integration

```bash
export ANTHROPIC_API_KEY=<your-key>

# Transcribe + summarize
python 06-llm-integration/asr_llm_pipeline.py --audio data/sample_audio/meeting.wav

# RAG over a directory of audio files
python 06-llm-integration/rag_transcripts.py \
  --audio-dir data/sample_audio \
  --query "What were the action items?"

# Streaming (real-time SSE)
python 06-llm-integration/streaming_pipeline.py --audio data/sample_audio/meeting.wav
```

### Phase 7 — Production API

```bash
# FastAPI server
uvicorn 07-production.api.main:app --host 0.0.0.0 --port 8000

# Celery worker (requires Redis)
celery -A 07-production.api.worker worker --loglevel=info -Q transcription

# Full stack (Phase 7 docker-compose — uncomment services first)
docker-compose up
```

---

## Key Concepts

### Why ASR inference is memory-bound

At batch size 1, transformer inference executes matrix-vector multiplies: one token against the full weight matrix.  Arithmetic intensity is ~0.5 FLOPs/byte — far below the A100's roofline crossover of ~156 FLOPs/byte.  **GPU TFLOP/s is irrelevant; HBM bandwidth is the bottleneck.**  Reducing weight element size (FP32 → FP16 → INT8) directly cuts latency.

### The precision trade-off

| Precision | Bytes/weight | Tensor Cores | CER impact |
|-----------|:-----------:|:------------:|:----------:|
| FP32 | 4 | No | Baseline |
| FP16 | 2 | Yes | Negligible |
| int8\_float16 | 1 (weights) | Yes | < 0.5% typical |
| INT8 | 1 | Yes | 0.5–2% typical |

`int8_float16` is the production sweet spot: INT8 weight storage + FP16 activations.

### CTranslate2 vs TensorRT

| | CTranslate2 | TensorRT |
|-|-------------|----------|
| Install | `pip install faster-whisper` | NVIDIA SDK + build step |
| Build time | Seconds | Minutes |
| Portability | Any NVIDIA GPU | Architecture-specific |
| Best for | Rapid deploy, CPU/edge | Max throughput, Triton, H100 FP8 |

Full analysis in [`docs/interview-notes.md`](docs/interview-notes.md).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ASR models | OpenAI Whisper, NVIDIA Parakeet CTC |
| Inference optimization | PyTorch, CTranslate2, TensorRT |
| Fine-tuning | HuggingFace Transformers, PEFT (LoRA), NeMo |
| Multi-GPU | PyTorch DDP, Tensor Parallelism, NCCL |
| Serving | NVIDIA Triton, FastAPI, Celery + Redis |
| LLM integration | Anthropic Claude (claude-opus-4-7) |
| Profiling | NVIDIA Nsight Systems, pynvml |
| Monitoring | Prometheus, Grafana |
| Cloud | AWS EC2 (p3 / g5 family) |

---

## Docs

- [`docs/architecture.md`](docs/architecture.md) — end-to-end system diagram, GPU memory budget, scaling strategy
- [`docs/benchmark-methodology.md`](docs/benchmark-methodology.md) — warmup rationale, CUDA sync protocol, metric definitions
- [`docs/interview-notes.md`](docs/interview-notes.md) — six NVIDIA SA interview questions with full technical answers
