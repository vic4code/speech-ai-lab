# Phase 3 — Inference Optimization

This phase benchmarks Whisper across precision levels and inference backends, quantifying the latency/accuracy trade-off for production deployment on NVIDIA GPUs.  Understanding these trade-offs is core to an NVIDIA Solutions Architect role.

---

## Results

Run the scripts in order, then fill in the table from `results/*.json`.

| Backend | Precision | P50 Latency (ms) | P95 Latency (ms) | GPU Mem (MB) | CER | Speedup vs FP32 |
|---------|-----------|-----------------|-----------------|-------------|-----|----------------|
| PyTorch | FP32 | — | — | — | — | 1.00× |
| PyTorch | FP16 | — | — | — | — | —× |
| CTranslate2 | float16 | — | — | — | — | —× |
| CTranslate2 | int8\_float16 | — | — | — | — | —× |
| CTranslate2 | int8 | — | — | — | — | —× |

*Model: Whisper large-v3 · Dataset: librispeech\_asr\_dummy · GPU: AWS instance*

---

## Key Concepts

### 1. Why ASR inference is memory-bound (not compute-bound)

At batch size 1, each transformer layer performs a matrix-vector multiply: one token vector against the full weight matrix.  The arithmetic intensity — FLOPs per byte loaded from HBM — is approximately `2 × hidden_dim / (4 × hidden_dim)` = 0.5 FLOPs/byte.  An A100-80GB has a roofline crossover at ~208 FLOPs/byte.  We are 400× below it.

**Consequence:** GPU TFLOP/s is irrelevant.  What matters is HBM bandwidth.  This is why reducing weight element size (FP32 → FP16 → INT8) directly cuts latency: fewer bytes to load per weight.

### 2. The precision trade-off: FP32 vs FP16 vs INT8

| Precision | Bytes/weight | Tensor Core support | Accuracy risk |
|-----------|-------------|-------------------|---------------|
| FP32 | 4 | No (on most GPUs) | Baseline |
| FP16 | 2 | Yes (Ampere+) | Negligible for inference |
| INT8 | 1 | Yes | CER increase ~0.5–2% typical |

FP16 halves memory bandwidth cost with essentially no accuracy degradation.  INT8 halves it again but introduces quantization error, most visible on out-of-domain vocabulary, numbers, and proper nouns.  The sweet spot for most production ASR is `int8_float16` (INT8 weights, FP16 activations), which captures most of the bandwidth benefit while keeping activations in a numerically safe range.

### 3. CTranslate2 vs TensorRT: when to use which

**CTranslate2 (faster-whisper)**
- Framework-agnostic, installs as a pip wheel
- CPU + CUDA support; also optimized for Intel CPU (MKL-DNN)
- Applies static INT8 quantization + operator fusion offline
- Engine builds in seconds; portable across NVIDIA GPus
- **Best for:** rapid deployment, mixed CPU/GPU fleets, edge devices, when NVIDIA toolchain isn't available

**TensorRT**
- NVIDIA-only; generates GPU-architecture-specific CUDA kernels
- Longer build times (minutes) but maximally optimized for the target GPU (A100 SM80 ≠ H100 SM90)
- Supports FP8 on Hopper, fine-grained sparsity, INT4
- Integrates with Triton Inference Server for production multi-model serving
- **Best for:** highest possible throughput on a fixed GPU fleet, production Triton serving, FP8 on H100

---

## How to Run

```bash
# From repo root

# Step 1: FP32 baseline (Whisper tiny + large-v3)
python 03-inference-optimization/01_baseline_benchmark.py

# Step 2: FP16 comparison
python 03-inference-optimization/02_fp16_benchmark.py

# Step 3: CTranslate2 INT8 + CER measurement
python 03-inference-optimization/03_int8_ctranslate2.py

# Step 4: TensorRT compilation (requires torch-tensorrt or nsys)
python 03-inference-optimization/04_tensorrt_compile.py

# Step 5: Nsight Systems profiling (requires nsys CLI)
chmod +x 03-inference-optimization/05_nsight_profile.sh
./03-inference-optimization/05_nsight_profile.sh
```

All results are saved to `results/` as JSON files and can be re-loaded by later scripts for cumulative comparison tables.
