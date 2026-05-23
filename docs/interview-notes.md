# NVIDIA SA Interview — Talking Points

## 1. "Why is ASR inference memory-bound, not compute-bound?"

At batch size 1, each transformer layer executes a matrix-vector multiply: a single token activation vector against the full weight matrix.  The arithmetic intensity — FLOPs per byte of memory traffic — is approximately `2 × d_model / (4 × d_model)` = 0.5 FLOPs/byte for FP32.

The A100-80GB SXM delivers ~312 TFLOP/s (FP16 Tensor Cores) but only ~2 TB/s of HBM bandwidth.  The hardware roofline crossover — where compute becomes the bottleneck — is `312e12 / 2e12` ≈ **156 FLOPs/byte**.  We are 300× below that threshold at batch 1.

**Practical consequence:** GPU TFLOP/s rating is nearly irrelevant for real-time ASR.  What matters is HBM bandwidth.  Every optimization that reduces the bytes loaded per weight element (FP16, INT8, sparsity) directly reduces latency.  Increasing batch size shifts the workload toward compute-bound, which is why dynamic batching is essential for throughput-mode serving.

---

## 2. "Walk me through the trade-offs between FP16 and INT8 quantization."

| | FP32 | FP16 | INT8 |
|-|------|------|------|
| Bytes/weight | 4 | 2 | 1 |
| Tensor Core support (Ampere) | No | Yes | Yes |
| Dynamic range | 3.4 × 10³⁸ | 65,504 | −128 to 127 |
| Quantization error | Baseline | Negligible | ~0.5–2% CER increase |
| Typical speedup vs FP32 | 1× | 1.8–2× | 3–4× |

**FP16:** Halves memory bandwidth cost, enables Tensor Core use.  Accuracy loss is negligible for Whisper — the model was originally trained with mixed precision.  This should be the default for all production GPU ASR.

**INT8:** Halves bandwidth again vs FP16.  Quantization error accumulates at model boundaries and affects rare vocabulary, numbers, proper nouns, and code-switching.  The sweet spot is `int8_float16` (CTranslate2 terminology): INT8 for weight storage and matrix-multiply accumulation, FP16 for activations.  This captures most of the bandwidth gain while keeping activation precision high enough to avoid cascading errors through deep models.

**When INT8 fails:** Long-tail acoustic conditions (heavy accent, noise) or domain-specific vocabulary.  Always validate CER on your target domain before deploying INT8 in production.

---

## 3. "How does CTranslate2 differ from TensorRT? When would you recommend each?"

**CTranslate2 (faster-whisper)**

- Open-source, CPU + CUDA, installs as a pip wheel with no NVIDIA SDK dependency
- Converts models offline to its own binary format; applies static INT8 quantization and operator fusion
- Engine build takes seconds; the same engine runs on any NVIDIA GPU (no architecture-specific compilation)
- Also optimized for Intel/ARM CPU via MKL-DNN and OpenBLAS
- **Recommend when:** rapid deployment, mixed CPU/GPU fleets, edge devices, teams without NVIDIA toolchain expertise, or when portability across GPU generations is required

**TensorRT**

- NVIDIA-proprietary; compiles a network graph into CUDA kernels specifically tuned for the target GPU architecture (SM80 ≠ SM90 ≠ SM100)
- Applies aggressive layer fusion, kernel auto-tuning, INT4/FP8 on Hopper, fine-grained structured sparsity
- Build times measured in minutes per GPU type; engines are not portable across GPU generations
- Integrates natively with Triton Inference Server for multi-model, multi-tenant production serving
- Enables FP8 precision on H100 (additional 2× over INT8 for supported ops)
- **Recommend when:** maximum throughput on a fixed GPU fleet, production Triton serving, H100 FP8, or when the customer needs every last inference/second

**Decision heuristic for a customer conversation:** "If you're deploying today and want it running by Thursday, CTranslate2. If you're building a production system that will serve millions of users on a fixed GPU fleet and you have two weeks to tune, TensorRT + Triton."

---

## 4. "A customer wants to serve Whisper large-v3 to 10,000 concurrent users. How would you architect that?"

**Capacity math first:**

- Whisper large-v3 FP16: ~1.5 s/clip on A10G for a 5 s audio clip → ~0.7 clips/GPU/s (rough)
- 10,000 users at 1 request per 30 s average → ~333 req/s sustained peak
- Required GPUs: ~333 / 0.7 ≈ **475 A10G**s, or fewer H100s with INT8 + dynamic batching

**Architecture layers:**

1. **Edge / CDN:** Terminate TLS, audio validation, client auth.  Return 429s under overload.
2. **Message queue (Kafka or SQS):** Decouple ingestion from GPU execution.  Provides backpressure, replay, and fair queuing across users.
3. **Auto-scaling GPU worker pool:** Triton Inference Server instances on EC2 `p3` or `g5` instances.  Kubernetes HPA (Horizontal Pod Autoscaler) scales on queue depth.  Use AWS EC2 Auto Scaling with spot instances for cost; reserve on-demand for SLA-critical capacity.
4. **Dynamic batching (Triton config):** `max_queue_delay_microseconds: 5000` — wait up to 5 ms to form a batch of 8 before dispatching to GPU.  At 333 req/s this nearly always fills the batch.
5. **Model optimization:** INT8 weights + FP16 activations (`int8_float16`), CTranslate2 or TRT depending on latency SLA.  At 10k users, latency SLA is likely 2–5 s, which gives room for batch formation.
6. **Result cache (Redis):** Cache transcriptions keyed by audio SHA-256 for 1 hour.  Duplicate requests (meeting recordings, shared content) get instant responses.
7. **Observability:** Prometheus → Grafana for GPU utilization, queue depth, P95 latency.  Alert at queue depth > 500 ms wait or P95 > 3 s.

**NIM option:** NVIDIA NIM packages Whisper with a production-ready container, TRT-optimized engine, and standardized REST API.  Reduces operator burden significantly for a customer that wants day-1 reliability.

---

## 5. "What is dynamic batching and when does it help?"

Dynamic batching collects multiple independent inference requests that arrive within a configurable time window and dispatches them to the GPU as a single batched tensor.

**Why it helps:** GPU utilization is highest when tensor dimensions are large.  A single batch-1 inference uses ~0.5% of A100 compute for typical Whisper.  A batch of 32 uses ~16%.  Dynamic batching lets a shared model instance serve multiple clients concurrently without requiring clients to coordinate.

**Triton config knobs:**
- `preferred_batch_size: [1, 4, 8]` — preferred batch sizes (GPU-optimized padding points)
- `max_queue_delay_microseconds: 5000` — wait at most 5 ms to fill a batch

**When it helps most:** High-request-rate scenarios (>10 req/s per GPU instance) with variable inter-arrival times.  At low traffic, the delay hurts latency without throughput benefit — tune the delay based on traffic patterns.

**When it does NOT help:** Streaming/real-time ASR (batch size is forced to 1 by the streaming constraint), or very long audio where one clip saturates the GPU by itself.

---

## 6. "Explain tensor parallelism vs pipeline parallelism."

Both are strategies for distributing a model across multiple GPUs when it exceeds single-GPU memory.

**Tensor Parallelism (TP)**

Splits individual weight matrices across GPUs along the hidden dimension.  For a linear layer `y = xW`, each GPU holds a column shard `W_i` and computes a partial output; an all-reduce across GPUs produces the full result.

- Latency benefit: reduces per-GPU memory footprint AND splits compute, so layers run faster
- Communication: one all-reduce per layer (synchronous)
- Best for: encoder/decoder stacks with large hidden dimensions; requires fast GPU interconnect (NVLink > PCIe)
- Megatron-LM / TensorRT-LLM implement optimized TP

**Pipeline Parallelism (PP)**

Splits model layers sequentially across GPUs.  GPU 0 holds layers 0–11, GPU 1 holds layers 12–23, etc.  Each GPU processes its layers and passes activations to the next GPU via P2P.

- Latency cost: introduces pipeline bubbles (GPU idle time while waiting for the previous stage)
- Communication: one activation tensor per stage boundary (usually cheaper than all-reduce)
- Best for: extremely deep models with many layers; tolerates slower interconnect
- Micro-batching reduces bubble fraction

**In practice:** Large language model serving (TensorRT-LLM, vLLM) typically uses **TP within a node** (fast NVLink) and **PP across nodes** (InfiniBand).  For Whisper, the model is small enough that a single A100 handles it in FP16; TP is only needed for serving at very large batch sizes or with multiple simultaneous model replicas.
