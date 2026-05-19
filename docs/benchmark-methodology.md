# Benchmark Methodology

## Measurement Protocol

All latency measurements follow this protocol to ensure reproducibility and accuracy:

1. **Warmup:** 10 un-timed inference iterations before any measurement begins.  This populates the GPU kernel cache, stabilizes clock frequencies, and eliminates CUDA JIT compilation overhead from results.

2. **CUDA synchronization:** `torch.cuda.synchronize()` is called immediately before `time.perf_counter()` start and immediately after the inference call before the stop measurement.  Without synchronization, CUDA kernel launches are asynchronous — the Python timer returns before the GPU finishes, yielding systematically low latency readings.

3. **Iteration count:** 10 timed iterations per configuration.  Sufficient for stable P95 on a single, consistent audio clip.

4. **Metric reporting:** Mean, P50, and P95.  P95 is the canonical production SLA metric — it captures tail latency without being distorted by rare outliers.

5. **Memory measurement:** `torch.cuda.max_memory_allocated()` after benchmark, reset with `torch.cuda.reset_peak_memory_stats()` before each model load.

## Test Input

- **Dataset:** `hf-internal-testing/librispeech_asr_dummy`, `clean` split, first sample
- **Duration:** ~5.1 seconds of clean read speech
- **Sample rate:** 16 kHz mono
- **Ground truth:** LibriSpeech transcript, lowercased for CER/WER computation

## Precision Comparison Note

FP16 and INT8 CER measurements are computed against the same FP32 Whisper output as reference (not the LibriSpeech gold transcription) when the goal is to measure quantization degradation.  When measuring absolute accuracy, the gold transcript is used.

## Environment

- AWS `p3.8xlarge` or `g5.xlarge` (NVIDIA A10G)
- CUDA 12.x
- Python 3.10
- PyTorch 2.x
- faster-whisper 1.x (CTranslate2 4.x)
