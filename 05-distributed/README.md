# Phase 5 — Distributed Inference

Multi-GPU tensor parallelism for Whisper large-v3 and scaling experiments.

## Scripts

| Script | Purpose |
|--------|---------|
| `tensor_parallel.py` | Shard Whisper attention heads across N GPUs with tensor parallelism |
| `gpu_monitor.py` | Real-time GPU utilization, memory, and NVLink bandwidth monitor |
| `scaling_experiment.py` | Strong/weak scaling curves: latency vs GPU count |

## How to Run

```bash
# Run tensor-parallel inference on 4 GPUs
python 05-distributed/tensor_parallel.py --gpus 4

# Live GPU monitor (runs alongside inference script)
python 05-distributed/gpu_monitor.py --interval 0.5

# Scaling experiment (1, 2, 4 GPUs)
python 05-distributed/scaling_experiment.py --max-gpus 4
```
