"""
Tensor-parallel inference for Whisper large-v3 across multiple GPUs.

Tensor parallelism splits individual weight matrices across GPUs along a chosen
dimension.  For a linear layer W of shape (d_in, d_out), splitting column-wise
gives each GPU a shard W_i of shape (d_in, d_out/N).  Each GPU computes a
partial output; an all-reduce synchronizes the result.  This allows models
that exceed single-GPU HBM to be served, and halves the per-GPU memory footprint
with 2 GPUs (etc.).

This script uses a simplified manual sharding approach for illustration.
For production, use Megatron-LM or NVIDIA TensorRT-LLM which implement
optimized tensor-parallel communication kernels (e.g., fused all-reduce).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset


def setup(rank: int, world_size: int) -> None:
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup() -> None:
    dist.destroy_process_group()


def shard_linear(linear: torch.nn.Linear, dim: int, rank: int, world_size: int) -> torch.nn.Linear:
    """Return a new Linear holding only this rank's slice of the weight."""
    weight = linear.weight.data
    shard_size = weight.shape[dim] // world_size
    start = rank * shard_size
    end = start + shard_size
    if dim == 0:
        shard_weight = weight[start:end, :].clone()
    else:
        shard_weight = weight[:, start:end].clone()
    new_linear = torch.nn.Linear(shard_weight.shape[1], shard_weight.shape[0], bias=linear.bias is not None)
    new_linear.weight.data = shard_weight
    return new_linear


def benchmark_tensor_parallel(rank: int, world_size: int, num_runs: int = 10) -> None:
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    import whisper
    model = whisper.load_model("large-v3", device=device).half()
    model.eval()

    # Shard encoder attention projections column-wise across GPUs
    for layer in model.encoder.blocks:
        layer.attn.query = shard_linear(layer.attn.query, dim=0, rank=rank, world_size=world_size).to(device)
        layer.attn.key = shard_linear(layer.attn.key, dim=0, rank=rank, world_size=world_size).to(device)
        layer.attn.value = shard_linear(layer.attn.value, dim=0, rank=rank, world_size=world_size).to(device)

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:1]")
    audio = np.array(ds[0]["audio"]["array"], dtype=np.float32)

    latencies: list[float] = []
    for _ in range(num_runs):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.transcribe(audio, language="en", fp16=True)
        torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - t0) * 1000)

    if rank == 0:
        print(f"Tensor-parallel ({world_size} GPUs) — mean latency: {np.mean(latencies):.1f} ms")

    cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    torch.multiprocessing.spawn(
        benchmark_tensor_parallel,
        args=(args.gpus, args.runs),
        nprocs=args.gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
