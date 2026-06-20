"""
train_qlora.py — QLoRA fine-tuning of Qwen3.5-4B for NL→SQL on NVIDIA GPU.

DESIGN DECISIONS (T4-specific, compute capability 7.5):
  - fp16=True, NOT bf16 — T4 does not support bfloat16.
  - BitsAndBytesConfig: load_in_4bit, nf4, double_quant, compute_dtype=float16.
  - prepare_model_for_kbit_training + gradient_checkpointing (reduces VRAM).
  - use_cache=False required alongside gradient checkpointing.
  - packing=False — packing breaks completion-only loss masking.
  - DataCollatorForCompletionOnlyLM: guarantees prompt tokens get label=-100,
    loss computed only on completion (gold SQL). This satisfies the requirement
    that the WRONG assistant SQL in correction chains is never a training target.
  - LoRA targets all projection layers for maximum coverage.
  - Resume-from-checkpoint: pass --resume-from-checkpoint path or "latest".

Usage (Colab):
    python train_qlora.py \
        --train-file /drive/data/train.prompt.jsonl \
        --val-file   /drive/data/val.prompt.jsonl \
        --output-dir /drive/adapters/qwen-sql-qlora \
        --base-model Qwen/Qwen3.5-4B

All hyperparams exposed as argparse flags with T4-tuned defaults.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tuning — Qwen3.5-4B, T4 (fp16)")

    # Paths
    p.add_argument("--base-model", default="Qwen/Qwen3.5-4B",
                   help="HF hub id or local path to base model. "
                        "On Colab: 'Qwen/Qwen3.5-4B' (downloads) or Drive path.")
    p.add_argument("--train-file", required=True,
                   help="Path to train.prompt.jsonl (prompt/completion format).")
    p.add_argument("--val-file", required=True,
                   help="Path to val.prompt.jsonl.")
    p.add_argument("--output-dir", required=True,
                   help="Where to save adapter checkpoints. Use a Drive path so "
                        "Colab restarts can resume.")
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Checkpoint dir to resume from, or 'latest' to auto-detect.")

    # LoRA
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")

    # Training
    p.add_argument("--max-seq-length", type=int, default=1024,
                   help="Max sequence length. Up to 2048 fits on T4 with batch=2, "
                        "grad-accum=8, but increases risk of OOM. Default 1024.")
    p.add_argument("--per-device-batch-size", type=int, default=2,
                   help="Per-device train batch size. T4 16GB: 2 at seq=1024.")
    p.add_argument("--gradient-accumulation-steps", type=int, default=8,
                   help="Effective batch = per_device * grad_accum = 16 default.")
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lr-scheduler", default="cosine",
                   choices=["cosine", "linear", "constant", "constant_with_warmup"])
    p.add_argument("--num-epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Override num-epochs with fixed step count (-1 = disabled).")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataloader-num-workers", type=int, default=0,
                   help="0 = main process only (safe in Colab).")

    return p.parse_args()


def load_dataset_from_jsonl(path: str) -> "datasets.Dataset":
    """Load prompt/completion JSONL into a HF Dataset."""
    import datasets as ds
    return ds.Dataset.from_json(path)


def main() -> None:
    args = parse_args()

    import torch
    import transformers
    import peft
    import trl
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    print(f"torch        : {torch.__version__}")
    print(f"transformers : {transformers.__version__}")
    print(f"peft         : {peft.__version__}")
    print(f"trl          : {trl.__version__}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires an NVIDIA GPU.", file=sys.stderr)
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # -------------------------------------------------------------------------
    # 4-bit quantization config — fp16, T4-compatible (CC 7.5, no bf16)
    # -------------------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,   # NOT bfloat16 — T4 lacks bf16 support
    )

    # -------------------------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------------------------
    print(f"\nLoading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="right",   # required for DataCollatorForCompletionOnlyLM
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------------------------------------------
    # Base model
    # -------------------------------------------------------------------------
    print(f"Loading base model (4-bit): {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Required for gradient checkpointing with kbit training
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False   # must disable with gradient checkpointing

    # -------------------------------------------------------------------------
    # LoRA config
    # -------------------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
    print(f"\nLoading datasets...")
    train_dataset = load_dataset_from_jsonl(args.train_file)
    val_dataset   = load_dataset_from_jsonl(args.val_file)
    print(f"  train: {len(train_dataset)} examples")
    print(f"  val:   {len(val_dataset)} examples")

    # -------------------------------------------------------------------------
    # DataCollator — completion-only loss
    # Masks prompt tokens (label=-100). Loss only on completion (gold SQL).
    # This is THE correctness guarantee: wrong SQL in correction chains is prompt,
    # so it gets masked. Only the final gold SQL is trained on.
    #
    # response_template: the token sequence that marks the start of the completion.
    # After enable_thinking=False rendering, prompt ends with:
    #   <|im_start|>assistant\n<think>\n\n</think>\n\n
    # So completion starts immediately after "</think>\n\n".
    # We use the assistant header + think block as response template.
    # -------------------------------------------------------------------------
    response_template = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # -------------------------------------------------------------------------
    # Output dir
    # -------------------------------------------------------------------------
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # TrainingArguments — fp16, T4-tuned
    # -------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        fp16=True,       # T4 requires fp16; bf16=False
        bf16=False,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="none",   # disable wandb/tensorboard by default
        seed=args.seed,
        remove_unused_columns=False,
    )

    # -------------------------------------------------------------------------
    # SFTTrainer
    # packing=False — required with DataCollatorForCompletionOnlyLM
    # -------------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        args=training_args,
        max_seq_length=args.max_seq_length,
        dataset_text_field=None,   # use prompt+completion columns directly
        packing=False,
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    resume = args.resume_from_checkpoint
    if resume == "latest":
        # find latest checkpoint dir
        ckpts = sorted(output_dir.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]))
        resume = str(ckpts[-1]) if ckpts else None
        print(f"Resuming from: {resume}")

    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=resume)

    # -------------------------------------------------------------------------
    # Save final adapter
    # -------------------------------------------------------------------------
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nFinal adapter saved to: {final_dir}")
    print("Training complete.")


if __name__ == "__main__":
    main()
