# Benchmark Methodology

## Hardware — NVIDIA A10G

| Property | Value |
|---|---|
| Architecture | Ampere GA102, SM 8.6 |
| SMs | 80 |
| CUDA Cores | 10,240 (128 per SM) |
| Tensor Cores | 320 (4 × 3rd-gen per SM) |
| VRAM | 24 GB GDDR6, 384-bit bus |
| GPU Boost Clock | 1,710 MHz |
| Memory Clock | 6,251 MHz (12,502 MT/s effective) |
| TDP | 300 W |

### Peak Throughput

| Precision | Compute | Notes |
|---|---|---|
| FP32 | 31.2 TFLOPS | CUDA cores, 2 FLOP/cycle/core |
| TF32 | 62.5 TFLOPS | Tensor Core (no sparsity) |
| FP16 / BF16 | **125 TFLOPS** | Tensor Core (no sparsity) — 4× FP32 |
| INT8 | **250 TOPS** | Tensor Core (no sparsity) — 8× FP32 |
| FP16 sparse | 250 TFLOPS | 2:4 structured sparsity |
| INT8 sparse | 500 TOPS | 2:4 structured sparsity |
| Memory BW | **600 GB/s** | HBM limit for all kernels |

### Roofline Model

The roofline ridge point is the arithmetic intensity (FLOP/byte) at which a kernel transitions from memory-bound to compute-bound:

```
Ridge (FLOP/byte) = Peak TFLOPS × 10¹² / Bandwidth × 10⁹
```

| Precision | Ridge (FLOP/byte) |
|---|---|
| FP32 | 52 |
| FP16 TC | 208 |
| INT8 TC | 417 |

**Batch-1 ASR inference arithmetic intensity** (Whisper large-v3):

Each weight element is loaded once from HBM and yields 2 FLOPs (FMA):
- FP32 weights: 2 FLOP / 4 bytes = **0.5 FLOP/byte** → 104× below FP32 ridge
- FP16 weights: 2 FLOP / 2 bytes = **1.0 FLOP/byte** → 208× below FP16 ridge
- INT8 weights: 2 FLOP / 1 byte  = **2.0 FLOP/byte** → 208× below INT8 ridge

**Key insight:** At batch=1, ASR inference is firmly memory-bandwidth-bound at every precision level. The GPU's Tensor Cores are idle most of the time. This has two implications:

1. Upgrading TFLOPS (e.g., A10G → A100) gives ≈ 0% speedup at batch=1.
2. Reducing weight bytes (FP32 → FP16 → INT8) cuts HBM traffic proportionally → ~linear latency reduction.
3. Batch scaling is the only path to reach the Tensor Core ridge and extract GPU FLOP/s.

---

## Measurement Protocol

All latency measurements use this protocol for reproducibility:

1. **Warmup** — 3 un-timed iterations before any measurement. Populates the GPU kernel cache (PTX → SASS JIT), stabilises boost clocks, eliminates driver init costs from results.

2. **CUDA synchronisation** — `torch.cuda.synchronize()` before and after each timed call. CUDA kernel launches are asynchronous; without sync the Python timer returns before the GPU finishes, yielding systematically under-reported latency.

3. **Iteration count** — 5 timed iterations per model × language combination. Reports mean, P50, P95.

4. **RTF (Real-Time Factor)** — per-clip normalisation:
   ```
   RTF_i = latency_i_ms / clip_duration_i_ms
   RTF   = mean(RTF_i over BENCH_ITERS)
   ```
   Per-clip RTF makes comparisons independent of clip length. RTF < 1 = faster than real-time.

5. **GPU memory** — `torch.cuda.max_memory_allocated()` after benchmark, reset before each model load with `torch.cuda.reset_peak_memory_stats()`.

6. **Accuracy** — WER (English) and CER (Chinese) computed over all 100 benchmark clips using `jiwer`. Text is normalised (strip punctuation, lowercase EN / strip non-CJK ZH) before scoring.

---

## Datasets

| Lang | Dataset | Split | Samples | Avg Duration | Benchmark Use |
|---|---|---|---|---|---|
| zh | FLEURS `cmn_hans_cn` | test | 100 | 11.0 s | Whisper zh CER |
| en | LibriSpeech `test-clean` | test | 100 | 6.7 s | Whisper + Parakeet WER |

LibriSpeech test-clean is the official eval set for NVIDIA Parakeet, enabling direct comparison with published numbers. FLEURS is accessible without credentialling and spans natural Mandarin sentences.

> **Note:** NVIDIA Parakeet (TDT / CTC / RNNT family) is English-only. Chinese experiments benchmark Whisper exclusively.

---

## Environment

| Component | Version |
|---|---|
| GPU | NVIDIA A10G 24 GB |
| CUDA | 12.1 |
| Python | 3.12.3 |
| PyTorch | 2.5.1+cu121 |
| openai-whisper | ≥ 20240930 |
| faster-whisper | ≥ 1.1.0 (CTranslate2 backend) |
| nemo_toolkit | 2.5.0 |
| Package manager | uv 0.11.x (isolated venv under `03-inference-optimization/`) |

---

## SA Interview Reference — Key Numbers

| Question | Answer |
|---|---|
| A10G peak FP16? | 125 TFLOPS (Tensor Core, no sparsity) |
| A10G memory bandwidth? | 600 GB/s |
| Why is batch-1 inference slow on big GPUs? | Arithmetic intensity ≈ 0.5 FLOP/byte; 100× below roofline → BW-bound, TFLOPS irrelevant |
| FP32 → FP16 speedup theory? | 2× (half the HBM bytes loaded) |
| FP32 → INT8 speedup theory? | 4× (quarter bytes) |
| Why int8_float16 and not pure int8? | Activations accumulate error in INT8; keeping FP16 accumulators limits CER/WER degradation to < 0.5 pp while retaining weight-load speedup |
| Parakeet vs Whisper latency gap? | ~8.8× on A10G (baseline). TDT is linear-time CTC; Whisper is O(T²) autoregressive |
| When does TensorRT beat CTranslate2? | High-throughput batched serving on fixed GPU arch; CT2 wins for rapid deployment / CPU / heterogeneous fleet |
| 2:4 sparsity benefit? | Additional 2× on top of dense precision (250 TFLOPS FP16 sparse on A10G) — requires structured weight pruning |
