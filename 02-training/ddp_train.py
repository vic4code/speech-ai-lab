"""
Multi-GPU full fine-tuning of Whisper with PyTorch DistributedDataParallel (DDP).

Demonstrates: torchrun launch, process group initialization, DDP model wrapping,
DistributedSampler, and gradient synchronization across GPUs.

Launch with:
    torchrun --nproc_per_node=4 02-training/ddp_train.py --model large-v3 --data data/clean/train
"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def setup_ddp() -> tuple[int, int]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="DDP fine-tune Whisper")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/ddp-whisper"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"DDP training on {world_size} GPUs")

    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{args.model}")
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    ds = load_from_disk(str(args.data))
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_features=input_features, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()

            if rank == 0 and step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {outputs.loss.item():.4f}")

        if rank == 0:
            print(f"Epoch {epoch} complete — avg loss {total_loss/len(loader):.4f}")

    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        model.module.save_pretrained(str(args.output))
        print(f"Model saved → {args.output}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
