# Phase 1 — Data Collection & Preparation

This phase covers sourcing, cleaning, and augmenting audio data for ASR fine-tuning.

## Scripts

| Script | Purpose |
|--------|---------|
| `collect.py` | Download and cache datasets from HuggingFace Hub |
| `clean.py` | Filter by duration, SNR, and transcript quality |
| `augment.py` | Apply speed perturbation, noise injection, SpecAugment |

## Datasets

- **LibriSpeech** — 960 h clean read speech (baseline)
- **Common Voice** — multilingual, crowd-sourced
- **GigaSpeech** — diverse domains (podcast, YouTube, audiobook)
- **NOAA Weather Radio** — domain-specific for environmental use cases

## How to Run

```bash
# Collect a small split for quick iteration
python 01-data/collect.py --dataset librispeech_asr --split validation

# Clean: drop clips shorter than 1 s or longer than 30 s, SNR < 20 dB
python 01-data/clean.py --input data/raw --output data/clean

# Augment training split
python 01-data/augment.py --input data/clean/train --output data/augmented
```
