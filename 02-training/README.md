# Phase 2 — Fine-Tuning & Training

Baseline evaluation, LoRA fine-tuning, and multi-GPU DDP training for Whisper and Parakeet.

## Scripts

| Script | Purpose |
|--------|---------|
| `baseline_eval.py` | Evaluate pretrained Whisper on a test set, report WER/CER |
| `lora_finetune.py` | Fine-tune Whisper with LoRA adapters (PEFT) |
| `ddp_train.py` | Full fine-tune with PyTorch DDP across multiple GPUs |
| `nemo_config.yaml` | NeMo experiment config for Parakeet CTC training |

## How to Run

```bash
# Baseline WER on LibriSpeech test-clean
python 02-training/baseline_eval.py --model large-v3 --dataset librispeech_asr

# LoRA fine-tune (single GPU, low memory)
python 02-training/lora_finetune.py --model large-v3 --data data/clean/train

# Multi-GPU DDP (4 × A100)
torchrun --nproc_per_node=4 02-training/ddp_train.py --model large-v3 --data data/clean/train
```
