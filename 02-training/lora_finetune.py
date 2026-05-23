"""
Fine-tune Whisper with LoRA adapters using HuggingFace PEFT.

Demonstrates: LoRA adapter injection, frozen backbone fine-tuning, and
training a fraction of parameters (~1%) to adapt to a new domain with
minimal compute and no catastrophic forgetting of the pretrained weights.

Key insight: LoRA adds low-rank decomposition matrices (A, B) to attention
weight matrices. Only A and B are trained; the original weights W stay frozen.
At inference, the adapter can be merged: W' = W + α·B·A.
"""

import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def build_lora_model(model_name: str, lora_r: int = 8, lora_alpha: int = 32) -> tuple:
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_name}")
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_name}")

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Whisper")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/lora-whisper"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=8)
    args = parser.parse_args()

    model, processor = build_lora_model(args.model, args.lora_r)
    ds = load_from_disk(str(args.data))

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=500,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
    )
    trainer.train()

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    processor.save_pretrained(str(args.output))
    print(f"LoRA adapter saved → {args.output}")


if __name__ == "__main__":
    main()
