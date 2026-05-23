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

### Hardware & Methodology

> GPU: NVIDIA A10G 24 GB · CUDA 12.1 · Python 3.12 · PyTorch 2.5.1  
> Benchmark: 3 warmup iters (discarded) + 5 timed iters, CUDA-synchronised wall-clock  
> RTF = mean(latency_i / duration_i) — per-clip normalised, independent of clip length  
> Data: FLEURS `cmn_hans_cn` 100 clips (zh, avg 11.0 s) · LibriSpeech test-clean 100 clips (en, avg 6.7 s)  
> Accuracy: CER (zh) and WER (en) over all 100 clips via `jiwer`  
> Full details → [`docs/benchmark-methodology.md`](docs/benchmark-methodology.md)

---

### Step 1 — Baseline FP32

#### English: Whisper vs Parakeet

| Model | Backend | Precision | Mean (ms) | P95 (ms) | GPU MB | RTF | WER% | vs large-v3 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| whisper-large-v3 | openai-whisper | FP32 | 1130 | 1947 | 6608 | 0.169 | 2.35 | 1.00× |
| whisper-large-v3-turbo | openai-whisper | FP32 | 370 | 509 | 3251 | 0.055 | 2.51 | 3.05× |
| **parakeet-tdt-1.1b** | NeMo | FP32 | **128** | **169** | 4507 | **0.019** | **1.60** | **8.80×** |

#### Chinese: Whisper only (Parakeet is English-only)

| Model | Backend | Precision | Mean (ms) | P95 (ms) | GPU MB | RTF | CER% | vs large-v3 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| whisper-large-v3 | openai-whisper | FP32 | 1379 | 2326 | 6548 | 0.125 | 4.14 | 1.00× |
| whisper-large-v3-turbo | openai-whisper | FP32 | 404 | 575 | 3251 | 0.037 | 5.40 | 3.41× |

---

### Step 2 — FP16 (openai-whisper amp + faster-whisper CTranslate2)

#### English (large-v3)

| Backend | Precision | Mean (ms) | RTF | WER% | vs FP32 |
|---|---|---:|---:|---:|---:|
| openai-whisper | FP32 baseline | 1130 | 0.169 | 2.35 | 1.00× |
| openai-whisper | **fp16** | 1283 | 0.116 | 2.51 | 1.45× |
| faster-whisper | **float16** | 637 | 0.068 | 2.57 | **2.48×** |

#### Chinese (large-v3)

| Backend | Precision | Mean (ms) | RTF | CER% | vs FP32 |
|---|---|---:|---:|---:|---:|
| openai-whisper | FP32 baseline | 1379 | 0.125 | 4.14 | 1.00× |
| openai-whisper | **fp16** | 1651 | 0.131 | 4.05 | **0.95× ⚠** |
| faster-whisper | **float16** | 761 | 0.062 | 4.08 | **2.02×** |

> **⚠ openai-whisper fp16 is slower on zh** — autoregressive decoding on long Chinese clips
> (11 s avg, ~60 decoder steps) adds FP16 cast overhead that outweighs bandwidth savings.
> faster-whisper avoids this via CTranslate2 operator fusion.

---

### Step 3 — INT8 / Mixed Precision (faster-whisper CTranslate2)

#### English (large-v3, faster-whisper)

| Precision | Mean (ms) | RTF | WER% | vs FP32 | WER Δ vs FP32 |
|---|---:|---:|---:|---:|---:|
| float16 | 599 | 0.063 | 2.57 | 2.68× | +0.22 pp |
| **int8\_float16** | **585** | **0.062** | 3.16 | **2.72×** | +0.81 pp |
| int8 | 588 | 0.062 | 2.83 | 2.71× | +0.48 pp |

#### Chinese (large-v3, faster-whisper)

| Precision | Mean (ms) | RTF | CER% | vs FP32 | CER Δ vs FP32 |
|---|---:|---:|---:|---:|---:|
| float16 | 726 | 0.059 | 4.08 | 2.12× | +0.06 pp |
| **int8\_float16** | **707** | **0.057** | **4.02** | **2.17×** | **−0.12 pp** |
| int8 | 705 | 0.057 | 4.02 | 2.18× | −0.12 pp |

---

### Step 4 — torch.compile + TensorRT Encoder

`torch.compile(backend="torch_tensorrt")` compiles the Whisper **encoder** with TRT kernel fusion + FP16 precision. The autoregressive **decoder** stays in eager mode (dynamic sequence lengths are incompatible with static TRT engine shapes).

Engine build occurs on the first warmup call (~34 s for large-v3, ~0.3 s for large-v3-turbo after caching).

#### large-v3

| Lang | Mean (ms) | P95 (ms) | GPU MB | RTF | WER/CER% | vs FP32 |
|---|---:|---:|---:|---:|---:|---:|
| zh | 1245.5 | 2112.0 | 6547 | 0.0997 | 4.14% CER | 1.25× |
| en | 1022.9 | 1769.0 | 6543 | 0.0982 | 2.78% WER | 1.72× |

#### large-v3-turbo

| Lang | Mean (ms) | P95 (ms) | GPU MB | RTF | WER/CER% | vs FP32 |
|---|---:|---:|---:|---:|---:|---:|
| zh | 342.7 | 508.0 | 3242 | 0.0282 | 5.40% CER | 1.31× |
| en | 303.9 | 434.7 | 3240 | 0.0340 | 2.51% WER | 1.62× |

> **Why TRT encoder-only is slower than CT2 INT8**: TRT fuses and tunes the encoder, but the autoregressive decoder remains the bottleneck for large-v3 (many decoder steps for long clips). CT2 INT8 quantizes the *entire* model (encoder + decoder weights) and achieves 2.17–2.72× vs TRT's 1.25–1.72×. TRT wins when the encoder is the dominant cost — e.g., encoder-heavy models or batch > 1.

---

### Full Picture — whisper-large-v3, best backend per precision

| Step | Precision | Lang | Mean (ms) | RTF | WER/CER% | Speedup |
|---|---|---|---:|---:|---:|---:|
| 1 | FP32 (openai-whisper) | en | 1130 | 0.169 | 2.35% WER | 1.00× |
| 2 | float16 (faster-whisper) | en | 637 | 0.068 | 2.57% | 2.48× |
| 3 | int8\_float16 (faster-whisper) | en | 585 | 0.062 | 3.16% | **2.72×** |
| 4 | fp16 TRT encoder (torch.compile) | en | 1023 | 0.098 | 2.78% | 1.72× |
| — | *parakeet-tdt-1.1b FP32* | en | 128 | 0.019 | 1.60% | *8.80×* |
| 1 | FP32 (openai-whisper) | zh | 1379 | 0.125 | 4.14% CER | 1.00× |
| 2 | float16 (faster-whisper) | zh | 761 | 0.062 | 4.08% | 2.02× |
| 3 | int8\_float16 (faster-whisper) | zh | 707 | 0.057 | 4.02% | **2.17×** |
| 4 | fp16 TRT encoder (torch.compile) | zh | 1246 | 0.100 | 4.14% | 1.25× |

---

### Key Findings

1. **Parakeet TDT 1.1B dominates English** — 8.8× faster than Whisper large-v3 FP32 with *better* WER (1.60% vs 2.35%). TDT decoding is O(N) vs Whisper's O(N²) autoregressive beam search.

2. **faster-whisper CT2 float16 ≈ 2.5× over FP32** with < 0.3 pp accuracy loss — the practical production baseline for Whisper.

3. **INT8 gives marginal additional gain over float16** (2.72× vs 2.48× for en): A10G memory bandwidth (600 GB/s) is already well-utilised at float16; INT8 mainly helps on bandwidth-constrained edge GPUs.

4. **openai-whisper fp16 regresses on Chinese** (0.95×): long zh clips (~11 s) produce many decoder steps; FP16 cast overhead > bandwidth savings. Always use faster-whisper for production Chinese ASR.

5. **Chinese quantization is remarkably stable**: int8\_float16 CER 4.02% = slightly *better* than FP32 4.14% — quantization noise and greedy decoding differences fall within run-to-run variance.

6. **TRT encoder-only < CT2 INT8 for this workload**: torch.compile/TRT achieves 1.25–1.72× (encoder fusion only) vs CT2's 2.17–2.72× (full model INT8). The decoder is the bottleneck at batch=1. TRT advantage emerges at batch > 1 or with encoder-only architectures.

7. **Roofline explains everything**: at batch=1, ASR arithmetic intensity ≈ 0.5 FLOP/byte, 100× below the A10G ridge (52 FLOP/byte for FP32). Every optimization is a bandwidth reduction, not a compute increase.

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
