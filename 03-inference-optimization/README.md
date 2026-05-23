# Phase 3 — Inference Optimization

End-to-end Whisper inference optimization on NVIDIA A10G: FP32 baseline → FP16 → INT8 → TensorRT compilation → beam search tuning → VAD filtering → CUDA graph capture.

All results are reproducible — run the numbered scripts in order.

---

## Hardware & Environment

| | |
|---|---|
| GPU | NVIDIA A10G 24 GB GDDR6 |
| CUDA | 13.x (torch 2.12.0+cu130) |
| Python | 3.12 |
| PyTorch | 2.12.0 |
| torch-tensorrt | 2.12.0 |
| faster-whisper | CTranslate2 backend |

### A10G Peak Throughput

| Precision | Peak | Tensor Cores |
|---|---:|:---:|
| FP32 | 31.2 TFLOPS | — |
| FP16 | 125 TFLOPS | ✓ (4×) |
| INT8 | 250 TOPS | ✓ (8×) |
| Mem BW | 600 GB/s | — |
| Ridge point (FP32) | 52 FLOP/byte | — |

### Why ASR is always memory-bandwidth bound

At batch=1, transformer inference runs matrix-vector products — one token against the full weight matrix. Arithmetic intensity ≈ **0.5 FLOP/byte**, which is 100× below the A10G ridge (52 FLOP/byte). The GPU waits for weights from DRAM, not for compute. **Every optimization that reduces bytes transferred gives a proportional speedup. Adding TFLOPS does nothing.**

### Benchmark Protocol

- **Data**: FLEURS `cmn_hans_cn` 100 clips (zh, avg 11.0 s) · LibriSpeech test-clean 100 clips (en, avg 6.7 s)
- **Warmup**: 3–5 iters discarded (Steps 4/6: first iter builds TRT engine)
- **Timed**: 5–10 CUDA-synchronised wall-clock iters
- **Accuracy**: CER (zh) and WER (en) computed over all 100 clips via `jiwer`
- **RTF**: `mean(latency_i / duration_i)` over bench iters — per-clip normalised, independent of clip length; RTF < 1 = faster-than-real-time

---

## Results

### Step 1 — FP32 Baseline

Scripts: `01_baseline_benchmark.py`

#### English — Whisper vs Parakeet

| Model | Backend | Mean (ms) | P95 (ms) | GPU MB | RTF | WER% |
|---|---|---:|---:|---:|---:|---:|
| whisper-large-v3 | openai-whisper | 1130 | 1947 | 6608 | 0.169 | 2.35 |
| whisper-large-v3-turbo | openai-whisper | 370 | 509 | 3251 | 0.055 | 2.51 |
| **parakeet-tdt-1.1b** | NeMo | **128** | **169** | 4507 | **0.019** | **1.60** |

#### Chinese — Whisper only (Parakeet is English-only)

| Model | Backend | Mean (ms) | P95 (ms) | GPU MB | RTF | CER% |
|---|---|---:|---:|---:|---:|---:|
| whisper-large-v3 | openai-whisper | 1379 | 2326 | 6548 | 0.125 | 4.14 |
| whisper-large-v3-turbo | openai-whisper | 404 | 575 | 3251 | 0.037 | 5.40 |

> **Parakeet**: TDT (Token-and-Duration Transducer) decoding is O(N) — the joint network scores all frames in parallel.  Whisper's autoregressive beam search is O(N²) in sequence length.  8.8× faster with *better* WER (1.60% vs 2.35%).  English-only.

---

### Step 2 — FP16

Script: `02_fp16_benchmark.py`

#### English (large-v3)

| Backend | Precision | Mean (ms) | RTF | WER% | vs FP32 |
|---|---|---:|---:|---:|---:|
| openai-whisper | fp32 | 1130 | 0.169 | 2.35 | 1.00× |
| openai-whisper | fp16 | 1283 | 0.116 | 2.51 | 1.45× |
| faster-whisper | **float16** | **637** | **0.068** | 2.57 | **2.48×** |

#### Chinese (large-v3)

| Backend | Precision | Mean (ms) | RTF | CER% | vs FP32 |
|---|---|---:|---:|---:|---:|
| openai-whisper | fp32 | 1379 | 0.125 | 4.14 | 1.00× |
| openai-whisper | fp16 | 1651 | 0.131 | 4.05 | **0.95× ⚠** |
| faster-whisper | **float16** | **761** | **0.062** | 4.08 | **2.02×** |

> **⚠ openai-whisper fp16 regresses on zh**: Long Chinese clips (~11 s, ~60+ decoder steps) incur an FP16 cast overhead per autoregressive step.  The cast tax exceeds bandwidth savings at this sequence length.  CTranslate2 fuses the cast into the kernel — no overhead.

---

### Step 3 — INT8 / Mixed Precision (CTranslate2)

Script: `03_int8_ctranslate2.py`

#### English (large-v3, faster-whisper)

| Precision | Mean (ms) | RTF | WER% | vs FP32 | Δ WER |
|---|---:|---:|---:|---:|---:|
| float16 | 599 | 0.063 | 2.57 | 2.68× | +0.22 pp |
| **int8\_float16** | **585** | **0.062** | 3.16 | **2.72×** | +0.81 pp |
| int8 | 588 | 0.062 | 2.83 | 2.71× | +0.48 pp |

#### Chinese (large-v3, faster-whisper)

| Precision | Mean (ms) | RTF | CER% | vs FP32 | Δ CER |
|---|---:|---:|---:|---:|---:|
| float16 | 726 | 0.059 | 4.08 | 2.12× | +0.06 pp |
| **int8\_float16** | **707** | **0.057** | **4.02** | **2.17×** | **−0.12 pp** |
| int8 | 705 | 0.057 | 4.02 | 2.18× | −0.12 pp |

> **int8\_float16 = production sweet spot**: INT8 weight storage (1 byte vs 2) + FP16 activation arithmetic.  Chinese CER actually improves by 0.12 pp — quantization noise falls within run-to-run variance.

---

### Step 4 — torch.compile + TensorRT Encoder (FP16)

Script: `04_tensorrt_compile.py`

`torch.compile(backend="torch_tensorrt")` compiles only the **encoder** (static shape `1×80×3000`).  The autoregressive **decoder** stays in eager mode — TRT requires static shapes, but decoder output length is dynamic.

Engine build: ~34 s (large-v3), ~0.3 s (large-v3-turbo). One-time cost.

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

> **Why TRT encoder < CT2 INT8**: CT2 quantises the full model including the decoder.  TRT encoder fusion removes ~20–30% of encoder time but the decoder dominates for large-v3 at batch=1.  TRT wins at batch > 1 or with Triton dynamic batching.

---

### Step 5 — CT2 Beam Size × VAD Pareto

Script: `05_ct2_beam_vad.py` · Model: large-v3 int8\_float16

**beam\_size**: number of candidate sequences the decoder maintains.  beam=1 = greedy (fastest).  On a memory-bound GPU, additional beams batch token projections → sub-linear latency penalty (1.24× for beam=5 vs beam=1, not 5×).

**VAD** (Silero, built into faster-whisper): detects speech segments before mel extraction.  Only voiced frames enter the model.  Benefit scales with fraction of silence in the audio.

#### Chinese (large-v3, int8\_float16)

| beam | VAD | Mean (ms) | P95 (ms) | RTF | CER% | vs beam5 | vs FP32 |
|---|---|---:|---:|---:|---:|---:|---:|
| **1** | **off** | **522** | **817** | **0.0425** | **4.14** | **1.24×** | **2.94×** |
| 1 | on | 541 | 848 | 0.0439 | 3.83 | 1.20× | 2.84× |
| 2 | off | 581 | 922 | 0.0470 | 4.05 | 1.12× | 2.66× |
| 2 | on | 596 | 949 | 0.0483 | 3.77 | 1.09× | 2.58× |
| 5 | off | 654 | 1057 | 0.0527 | 4.02 | 1.00× | 2.37× |
| 5 | on | 658 | 1069 | 0.0533 | 3.80 | 0.99× | 2.34× |

#### English (large-v3, int8\_float16)

| beam | VAD | Mean (ms) | P95 (ms) | RTF | WER% | vs beam5 | vs FP32 |
|---|---|---:|---:|---:|---:|---:|---:|
| **1** | **off** | **440** | **714** | **0.0450** | **2.94** | **1.20×** | **3.75×** |
| 1 | on | 465 | 753 | 0.0473 | 2.73 | 1.15× | 3.56× |
| 2 | off | 488 | 797 | 0.0501 | 3.21 | 1.08× | 3.37× |
| 2 | on | 513 | 847 | 0.0521 | 3.80 | 1.04× | 3.24× |
| 5 | off | 528 | 866 | 0.0542 | 3.26 | 1.00× | 3.11× |
| 5 | on | 557 | 914 | 0.0565 | 4.01 | 0.96× | 2.98× |

> **Key finding — beam=1 is the practical optimum for throughput**: 1.20–1.24× faster than beam=5, WER/CER penalty only +0.59 pp (en) / +0.12 pp (zh).  The best overall operating point for zh is beam=1+VAD: 2.84× vs FP32, CER drops 0.31 pp vs beam=5 no-VAD.
>
> **VAD on dense benchmarks**: Silero VAD adds 20–40 ms overhead on FLEURS/LibriSpeech (dense speech, minimal silence).  Small CER gain (+VAD improves zh CER from 4.14→3.83) but WER degrades on en (beam=5: 3.26→4.01).  VAD is most valuable on real-world audio with ≥20% silence.

---

### Step 6 — TRT INT8 + Explicit Input Spec

Script: `06_tensorrt_int8.py`

Uses `torch_tensorrt.compile()` (lower-level API vs `torch.compile`) with:
- **Explicit `Input` spec**: `min/opt/max = [1, 80, 3000]` — static shape, one engine profile
- **enabled\_precisions = {FP32, FP16, INT8}** — TRT selects INT8 per-layer where safe
- **workspace\_size = 8 GiB** — memory budget for kernel tiling search
- **PTQ** (Post-Training Quantisation) — no calibration dataset; TRT uses weight-distribution heuristics

*Results pending — see `results/trt_int8_results.json` after running the script.*

**PTQ vs QAT**:

| | PTQ | QAT |
|---|---|---|
| Retraining required | No | Yes |
| Calibration data | Optional (improves ranges) | Labelled training set |
| Accuracy recovery | Baseline INT8 | +0.5–1 pp vs PTQ |
| Use when | Fast deployment | Max accuracy at INT8 |

---

### Step 7 — CUDA Graphs

Script: `07_cuda_graphs.py`

`torch.cuda.CUDAGraph` captures the encoder forward pass as a single replayable object. Replay issues the entire sequence of kernels in **one CPU call**, eliminating O(n\_kernels × 5–10 µs) launch overhead.

For Whisper large-v3 encoder (~300 kernels): estimated 1.5–2 ms of launch overhead removed per call.

| Config | Description |
|---|---|
| eager | No compilation, no graph — baseline |
| compile | `torch.compile(mode="max-autotune")` — Triton kernel fusion |
| compile+graph | Compile + CUDA graph capture — minimum latency |

**Constraints**: static tensor shapes required; fixed memory addresses (must `copy_()` input into pre-allocated buffer before each replay); no CPU↔GPU sync inside captured region.

*Results pending — see `results/cuda_graph_results.json` after running the script.*

---

## Full Stack Comparison — whisper-large-v3

| Step | Technique | Lang | RTF | WER/CER% | vs FP32 |
|---|---|---|---:|---:|---:|
| 1 FP32 | openai-whisper | en | 0.169 | 2.35% WER | 1.00× |
| 2 FP16 | faster-whisper float16 | en | 0.068 | 2.57% | 2.48× |
| 3 INT8 | CT2 int8\_float16 | en | 0.062 | 3.16% | 2.72× |
| 4 TRT | torch.compile FP16 encoder | en | 0.098 | 2.78% | 1.72× |
| **5 beam=1** | **CT2 int8\_float16 greedy** | **en** | **0.045** | **2.94%** | **3.75×** |
| — Parakeet | NeMo FP32 | en | 0.019 | 1.60% | *8.80×* |
| 1 FP32 | openai-whisper | zh | 0.125 | 4.14% CER | 1.00× |
| 2 FP16 | faster-whisper float16 | zh | 0.062 | 4.08% | 2.02× |
| 3 INT8 | CT2 int8\_float16 | zh | 0.057 | 4.02% | 2.17× |
| 4 TRT | torch.compile FP16 encoder | zh | 0.100 | 4.14% | 1.25× |
| **5 beam=1+VAD** | **CT2 int8\_float16 greedy+VAD** | **zh** | **0.044** | **3.83%** | **2.84×** |

---

## Interview Reference Card

### Q: Why is batch=1 ASR inference memory-bound?

Matrix-vector multiply: one token × full weight matrix. AI ≈ 0.5 FLOP/byte. A10G ridge = 52 FLOP/byte. 100× below ridge → every cycle is spent waiting for DRAM, not computing.

### Q: Why doesn't FP16 give a full 2× speedup?

Latency includes decoder overhead (token sampling, logit masking, beam bookkeeping) which runs on CPU/scalar units and doesn't benefit from bandwidth reduction. Speedup is ~2.5× practical vs 2× theoretical.

### Q: Why does CT2 INT8 beat TRT encoder-only?

CT2 quantises the full model (encoder + **decoder** weights). TRT encoder fusion saves ~20–30% of encoder time, but decoder dominates at batch=1 for large models. Full-model quantisation addresses the actual bottleneck.

### Q: When does TRT beat CT2?

At batch > 1 (encoder cost amortised, decoder parallelised across items). In Triton dynamic batching. On Hopper with FP8 (CT2 doesn't support FP8). For encoder-only architectures (Whisper encoder-only speech classification).

### Q: Why is beam=5 only 1.24× slower than beam=1, not 5×?

GPU is memory-bound. Additional beams batch the token projection matmul (larger effective batch → better GPU utilisation). KV cache is shared across beams. The extra beams stay below the roofline ridge — more compute doesn't add latency.

### Q: What is a CUDA graph and when should you use it?

CUDA graph captures a stream of CUDA operations into a single replayable object. Eliminates kernel launch overhead O(n\_kernels × ~5 µs) → O(1). Use when: static input shapes, latency-critical, single-request serving, GPU execution time is already very short (fast model/small input). Don't use when: dynamic shapes, dynamic control flow, or CPU is not the bottleneck.

### Q: PTQ vs QAT — when do you choose each?

PTQ: no retraining, runs in minutes, ~0.5–2 pp accuracy loss. Use for rapid deployment when a labelled training set isn't available.  
QAT: inserts fake-quant nodes, fine-tunes end-to-end, recovers <0.3 pp vs PTQ. Use when INT8 accuracy is a hard requirement and training data is available.

---

## How to Run

```bash
cd 03-inference-optimization
uv sync   # or: pip install -r requirements.lock

# Step 0: download benchmark data (FLEURS zh + LibriSpeech en, 100 clips each)
python 00_prepare_data.py

# Step 1–7: run in order
python 01_baseline_benchmark.py     # → results/baseline_results.json
python 02_fp16_benchmark.py         # → results/fp16_results.json
python 03_int8_ctranslate2.py       # → results/int8_results.json
python 04_tensorrt_compile.py       # → results/tensorrt_results.json
python 05_ct2_beam_vad.py           # → results/ct2_beam_vad_results.json
python 06_tensorrt_int8.py          # → results/trt_int8_results.json
python 07_cuda_graphs.py            # → results/cuda_graph_results.json
```
