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

Attempts `torch_tensorrt.compile()` with:
- **Explicit `Input` spec**: `min/opt/max = [1, n_mels, 3000]` — static shape, one engine profile
- **enabled\_precisions = {FP32, FP16, INT8}** — TRT selects INT8 per-layer where safe
- **workspace\_size = 8 GiB** — memory budget for kernel tiling search
- **PTQ** — no calibration dataset; TRT uses weight-distribution heuristics

**What happened on this system**: `torch_tensorrt.compile()` INT8 is blocked by two constraints: (1) CUDA 13 is not supported by TRT-LLM plugins, and (2) the initial script hardcoded `n_mels=80` but Whisper large-v3 uses `n_mels=128` — TRT's strict shape checking caught the mismatch immediately. Both issues are documented and the script is fixed (`n_mels` now inferred from `encoder.conv1.weight.shape[1]`). Fallback: `torch.compile(mode="max-autotune")`.

#### large-v3 (inductor/max-autotune fallback)

| Lang | Mean (ms) | P95 (ms) | GPU MB | RTF | WER/CER% | vs FP32 | vs Step 4 FP16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| zh | 1286.8 | 2141.0 | 6539 | 0.1033 | 4.14% CER | 1.21× | −0.04× |
| en | 1048.8 | 1828.0 | 9079 | 0.1013 | 2.89% WER | 1.66× | −0.06× |

> **Finding**: inductor max-autotune with INT8 in `enabled_precisions` performs comparably (within 5%) to torch_tensorrt FP16 from Step 4. Inductor selects CUDA `mm` over Triton for large matmuls (CUDA cublas wins at FP32 on A10G). True TRT INT8 gain requires CUDA 12.x + calibration dataset; expected 1.8–2.2× vs FP32 based on literature.

**PTQ vs QAT**:

| | PTQ | QAT |
|---|---|---|
| Retraining required | No | Yes |
| Calibration data | Optional (improves ranges) | Labelled training set |
| Accuracy recovery | Baseline INT8 | +0.5–1 pp vs PTQ |
| Use when | Fast deployment | Max accuracy at INT8 |

---

### Step 8 — Batch Throughput: memory-bound → compute-bound transition

Script: `08_batch_throughput.py`

#### Encoder batch throughput — large-v3 FP32 (zh clips)

| Batch | Mean (ms) | P95 (ms) | Clips/sec | vs B=1 | AI (FLOP/byte) | vs A10G ridge |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 170.3 | 170.6 | 5.87 | 1.00× | 708 | 341% |
| 2 | 335.0 | 335.6 | 5.97 | 1.02× | 1343 | 646% |
| 4 | 674.4 | 676.6 | 5.93 | 1.01× | 2430 | 1168% |
| 8 | 1453.4 | 1473.8 | 5.50 | 0.94× | 4085 | 1964% |
| 16 | 2748.3 | 2889.2 | 5.82 | 0.99× | 6194 | 2978% |

#### CT2 int8\_float16 sequential throughput (beam=1)

| Lang | n\_clips | Total time | Clips/sec | RTF |
|---|---:|---:|---:|---:|
| zh | 20 | 9.7 s | 2.06 | 0.045 |
| en | 20 | 7.6 s | 2.62 | 0.046 |

> **Critical finding — encoder is COMPUTE-BOUND at batch=1, not memory-bound.**
>
> Latency scales linearly with batch (170 ms → 2748 ms ≈ 16×), while throughput stays flat at ~5.8 clips/sec. This is textbook compute-saturation.
>
> **Why?** The simple "0.5 FLOP/byte" arithmetic intensity estimate assumed short sequences. Whisper's encoder produces **T=1500 tokens** after its stride-2 convolution. Attention FLOPs scale as O(T²): 4×B×T²×d = 4×1×1500²×1280 ≈ 11.5 GFLOP per encoder layer, comparable to the matmul FLOPs. At B=1: AI ≈ 708 FLOP/byte — **3.4× above the A10G FP32 ridge** (208 FLOP/byte).
>
> **Corrected roofline picture:**
> - **Encoder**: compute-bound (long T=1500 context, quadratic attention FLOPs dominate)
> - **Decoder**: memory-bound (autoregressive, short sequence, one token per step → matrix-vector)
>
> **Implication for optimization**: FP32→FP16→INT8 reduces decoder weight bytes and speeds up decoder. Encoder speedup from quantisation comes from bandwidth savings on FFN weights, but attention FLOPs are already at 3× peak — **tensor core throughput** (FP16 TC: 125 TFLOPS vs FP32: 31.2 TFLOPS) is what unlocks encoder speedup. This explains why float16 gave 2× on the encoder specifically.
>
> **Batching does NOT increase encoder throughput** because the GPU compute is already saturated at B=1. To improve throughput, use concurrent instances or a Triton server that pipelines encoder+decoder across requests.

---

### Step 7 — CUDA Graphs

Script: `07_cuda_graphs.py`

`torch.cuda.CUDAGraph` captures the encoder forward pass as a single replayable object. Replay issues the entire sequence of kernels in **one CPU call**, eliminating O(n\_kernels × 5–10 µs) launch overhead.

| Config | Description |
|---|---|
| eager | No compilation, no graph — baseline |
| compile | `torch.compile(mode="max-autotune")` — Triton fusion + inductor-internal graph trees |
| compile+graph | `torch.compile(mode="max-autotune-no-cudagraphs")` + manual `torch.cuda.CUDAGraph` |

#### Results — encoder isolation (ms)

| Model | Lang | Config | Enc mean (ms) | Enc P95 | vs eager |
|---|---|---|---:|---:|---:|
| large-v3 | zh | eager | 170.81 | 171.24 | 1.00× |
| large-v3 | zh | compile | 166.73 | 167.66 | 1.02× |
| large-v3 | zh | compile+graph | **165.69** | **166.17** | **1.03×** |
| large-v3 | en | eager | 170.43 | 171.12 | 1.00× |
| large-v3 | en | compile | 166.60 | 167.60 | 1.02× |
| large-v3 | en | compile+graph | **166.57** | **167.56** | **1.02×** |
| large-v3-turbo | zh | eager | 171.64 | 172.38 | 1.00× |
| large-v3-turbo | zh | compile | 167.02 | 167.54 | 1.03× |
| large-v3-turbo | zh | compile+graph | **166.21** | **166.64** | **1.03×** |
| large-v3-turbo | en | eager | 171.38 | 171.84 | 1.00× |
| large-v3-turbo | en | compile | 167.44 | 168.12 | 1.02× |
| large-v3-turbo | en | compile+graph | **166.58** | **167.18** | **1.03×** |

> **Key finding — 3% max gain, not the expected 10–15%**: Whisper encoder eager latency is ~171 ms; kernel launch overhead is only ~5 ms (~3%). The encoder is so thoroughly memory-bandwidth-bound that kernel launches are not the bottleneck — DRAM weight reads dominate every cycle.
>
> **When CUDA graphs matter**: fast models with many short kernels (e.g., ResNet-50 inference < 5 ms/image), batch=1 serving of MLP-heavy models, or pipelines with hundreds of small ops. For 170 ms memory-bound workloads, a 5 ms CPU-overhead reduction is noise.
>
> **large-v3 and large-v3-turbo have identical encoder latency** (~171 ms) because both share the same encoder architecture (128 mel bins, 32 encoder layers, 1280 hidden). The turbo variant removes decoder layers — the encoder is untouched.

**Implementation notes**:
- `torch.compile(max-autotune)` already uses `cudagraph_trees` internally — you cannot nest `torch.cuda.graph()` on top (raises `cudaErrorStreamCaptureUnsupported`)
- Use `mode="max-autotune-no-cudagraphs"` to disable inductor's internal graphs before manual capture
- `torch.compile` does not accept `mode` + `options` simultaneously — use the purpose-built mode string

---

## Full Stack Comparison — whisper-large-v3

| Step | Technique | Lang | RTF | WER/CER% | vs FP32 |
|---|---|---|---:|---:|---:|
| 1 FP32 | openai-whisper | en | 0.169 | 2.35% WER | 1.00× |
| 2 FP16 | faster-whisper float16 | en | 0.068 | 2.57% | 2.48× |
| 3 INT8 | CT2 int8\_float16 | en | 0.062 | 3.16% | 2.72× |
| 4 TRT | torch.compile/torch_tensorrt FP16 encoder | en | 0.098 | 2.78% | 1.72× |
| **5 beam=1** | **CT2 int8\_float16 greedy** | **en** | **0.045** | **2.94%** | **3.75×** |
| 6 inductor | torch.compile/inductor max-autotune encoder | en | 0.101 | 2.89% | 1.66× |
| — Parakeet | NeMo FP32 | en | 0.019 | 1.60% | *8.80×* |
| 1 FP32 | openai-whisper | zh | 0.125 | 4.14% CER | 1.00× |
| 2 FP16 | faster-whisper float16 | zh | 0.062 | 4.08% | 2.02× |
| 3 INT8 | CT2 int8\_float16 | zh | 0.057 | 4.02% | 2.17× |
| 4 TRT | torch.compile/torch_tensorrt FP16 encoder | zh | 0.100 | 4.14% | 1.25× |
| **5 beam=1+VAD** | **CT2 int8\_float16 greedy+VAD** | **zh** | **0.044** | **3.83%** | **2.84×** |
| 6 inductor | torch.compile/inductor max-autotune encoder | zh | 0.103 | 4.14% | 1.21× |

---

## Interview Reference Card

### Q: Is ASR inference memory-bound or compute-bound?

**It depends on the component:**
- **Decoder** (autoregressive, one token/step): matrix-vector multiply, AI ≈ 0.5 FLOP/byte. A10G FP32 ridge = 52 FLOP/byte → memory-bound. Reducing weight bytes (FP32→INT8) gives proportional speedup.
- **Encoder** (Whisper large-v3, T=1500 tokens): attention FLOPs ∝ T² = 11.5 GFLOP/layer. AI ≈ 708 FLOP/byte at batch=1 → already **3.4× above the ridge, compute-bound**. Tensor Core throughput (FP16: 125 TFLOPS vs FP32: 31.2 TFLOPS, 4×) is the lever for encoder speedup — not bandwidth.

**Step 8 data confirms this**: encoder throughput flat at ~5.8 clips/sec from batch=1 to batch=16. Latency scales linearly → compute-saturated at B=1.

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
