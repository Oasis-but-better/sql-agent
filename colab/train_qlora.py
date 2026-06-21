"""
train_qlora.py — QLoRA fine-tuning of Qwen3.5-4B for NL→SQL on NVIDIA GPU.

DESIGN DECISIONS (T4-specific, compute capability 7.5):
  - fp16=True, NOT bf16 — T4 does not support bfloat16.
  - BitsAndBytesConfig: load_in_4bit, nf4, double_quant, compute_dtype=float16.
  - prepare_model_for_kbit_training + gradient_checkpointing (reduces VRAM).
  - use_cache=False required alongside gradient checkpointing.
  - packing=False — packing breaks completion-only loss masking.
  - completion_only_loss=True (SFTConfig, trl>=1.0): native prompt/completion
    masking — prompt tokens get label=-100, loss only on completion (gold SQL).
    This satisfies the requirement that wrong assistant SQL in correction chains
    (which lives in the prompt field from prepare_data.py) is never a training target.
  - LoRA targets all projection layers for maximum coverage.
  - Resume-from-checkpoint: pass --resume-from-checkpoint path or "latest".

BF16 UNSCALE FIX (T4 / CC 7.5):
  Root cause: accelerate reads ~/.cache/huggingface/accelerate/default_config.yaml
  at state-singleton init, which happens when from_pretrained(..., device_map="auto")
  dispatches the model. If that file has mixed_precision: bf16, accelerate's
  Accelerator is created with bf16 autocast — BEFORE Trainer ever sets its own env.
  The fp16 GradScaler then calls _amp_foreach_non_finite_check_and_unscale_cuda on
  bf16 grads → NotImplementedError at step 0 grad-clip.
  PRIMARY FIX: force ACCELERATE_MIXED_PRECISION=fp16 at module top (hard =, not
  setdefault) and rename the cached config, both BEFORE any accelerate/transformers
  import. See: pytorch/pytorch#127176, huggingface/trl#4901, huggingface/transformers#29510.
  FALLBACK (--no-scaler): fp16=False, bf16=False, no GradScaler at all, entire
  model cast to fp16 (LoRA included). Crash is structurally impossible. May be
  less numerically stable — if loss goes NaN, lower --learning-rate to 1e-4.

REQUIRES:
  - transformers>=5.2.0  (qwen3_5 model_type added in v5.2.0)
  - trl>=1.0.0           (SFTConfig stable API with completion_only_loss)
  - peft>=0.19.0
  - accelerate>=1.4.0
  - bitsandbytes>=0.49.0

Usage (Colab):
    python train_qlora.py \
        --train-file /drive/data/train.prompt.jsonl \
        --val-file   /drive/data/val.prompt.jsonl \
        --output-dir /drive/adapters/qwen-sql-qlora \
        --base-model Qwen/Qwen3.5-4B

    # If bf16-unscale crash persists after primary fix:
    python train_qlora.py ... --no-scaler

All hyperparams exposed as argparse flags with T4-tuned defaults.
"""

from __future__ import annotations

# =============================================================================
# MODULE-TOP BF16 FIX — must run before ANY accelerate/transformers import.
# device_map="auto" triggers accelerate state-singleton init at model load time,
# freezing mixed_precision. Setting env here (hard =, not setdefault) and
# renaming a cached bf16 default_config.yaml are the only reliable intercepts.
# =============================================================================
import os as _os

_os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"

# Rename cached accelerate config that may have mixed_precision: bf16.
# Rename (not delete) so it is recoverable. Guarded — missing file is fine.
_accel_cfg = _os.path.expanduser(
    "~/.cache/huggingface/accelerate/default_config.yaml"
)
if _os.path.exists(_accel_cfg):
    try:
        _os.rename(_accel_cfg, _accel_cfg + ".bf16_bak")
        print(f"[bf16-fix] Renamed accelerate config → {_accel_cfg}.bf16_bak")
    except OSError as _e:
        print(f"[bf16-fix] WARNING: could not rename accelerate config: {_e}")

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
    p.add_argument("--no-scaler", action="store_true",
                   help="Disable fp16 GradScaler entirely (fp16=False, bf16=False). "
                        "Casts ALL params including LoRA to float16 to avoid dtype "
                        "mismatches. Structurally prevents the bf16-unscale crash. "
                        "May be less numerically stable — lower --learning-rate to "
                        "1e-4 if loss goes NaN. Use when primary fix still crashes.")

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
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    # SFTConfig is the unified API in trl>=1.0 (absorbs TrainingArguments + SFT params).
    # Fallback to separate TrainingArguments + SFTTrainer for older trl, but trl>=1.0
    # is required by requirements-colab.txt so the fallback is a safety net only.
    try:
        from trl import SFTTrainer, SFTConfig
        _HAS_SFTCONFIG = True
    except ImportError:
        from trl import SFTTrainer
        from transformers import TrainingArguments as SFTConfig  # type: ignore[assignment]
        _HAS_SFTCONFIG = False

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
        padding_side="right",
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
        torch_dtype=torch.float16,
    )

    # NUCLEAR STEP 1: Override the config so HF Trainer doesn't get confused
    model.config.torch_dtype = torch.float16

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
    # Parameter dtype consolidation — ONE canonical loop.
    #
    # Normal mode (default, fp16 GradScaler active):
    #   - LoRA trainable params → fp32 master weights (correct AMP pattern).
    #     DO NOT downcast to fp16: triggers "Attempting to unscale FP16 gradients".
    #   - All other floating params → fp16 if bf16, else unchanged.
    #     Uint8 quantized weights: is_floating_point()==False → skipped automatically.
    #   Primary bf16 source: Qwen3.5 config.json sets torch_dtype=bfloat16, so
    #   norms/embeddings/lm_head survive as bf16 even with torch_dtype=float16 in
    #   from_pretrained. The accelerate autocast layer (fixed at module top via
    #   ACCELERATE_MIXED_PRECISION=fp16) is the other bf16 source.
    #   Evidence: pytorch/pytorch#127176, huggingface/trl#4901.
    #
    # --no-scaler fallback mode (GradScaler disabled entirely):
    #   - EVERY floating param → fp16, including LoRA.
    #   - With no scaler there is no unscale call → bf16-unscale crash impossible.
    #   - Also sets ACCELERATE_MIXED_PRECISION=no so Accelerator skips autocast.
    # -------------------------------------------------------------------------
    if args.no_scaler:
        # Fallback: cast everything to fp16, disable accelerate autocast.
        _os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
        for _p in model.parameters():
            if _p.is_floating_point():
                _p.data = _p.data.to(torch.float16)
        print("[--no-scaler] All params cast to fp16. GradScaler disabled.")
    else:
        # Normal: fp32 LoRA master weights + fp16 non-trainable params.
        for _p in model.parameters():
            if _p.requires_grad:
                _p.data = _p.data.to(torch.float32)  # LoRA: fp32 master copy
            elif _p.is_floating_point() and _p.dtype == torch.bfloat16:
                _p.data = _p.data.to(torch.float16)  # bf16→fp16 (norms, embeds, lm_head)

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
    print(f"\nLoading datasets...")
    train_dataset = load_dataset_from_jsonl(args.train_file)
    val_dataset   = load_dataset_from_jsonl(args.val_file)
    print(f"  train: {len(train_dataset)} examples")
    print(f"  val:   {len(val_dataset)} examples")

    # -------------------------------------------------------------------------
    # Output dir
    # -------------------------------------------------------------------------
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # SFTConfig — fp16, T4-tuned.
    #
    # completion_only_loss=True: native prompt/completion masking (trl>=1.0).
    #   Labels are -100 for prompt tokens; loss computed only on completion
    #   (gold SQL). This is the correctness guarantee for correction chains:
    #   the wrong assistant SQL lives in the prompt field → automatically masked.
    #
    # max_length replaces the old max_seq_length (deprecated in trl>=0.16).
    # eval_strategy replaces evaluation_strategy (renamed in transformers>=4.46).
    # processing_class replaces tokenizer (deprecated in trl>=0.16, removed 0.17+).
    #
    # packing=False — required for completion_only_loss to work correctly.
    #
    # --no-scaler: fp16=False, bf16=False → no GradScaler instantiated.
    # -------------------------------------------------------------------------
    _use_fp16 = not args.no_scaler   # False when --no-scaler: no GradScaler at all

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        fp16=_use_fp16,  # False with --no-scaler: disables GradScaler entirely
        bf16=False,
        logging_steps=args.logging_steps,
        eval_strategy="steps",          # renamed from evaluation_strategy in transformers>=4.46
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="none",               # disable wandb/tensorboard by default
        seed=args.seed,
        remove_unused_columns=False,
        # SFT-specific params (trl>=1.0 SFTConfig fields)
        max_length=args.max_seq_length, # replaces deprecated max_seq_length
        packing=False,                  # required with completion_only_loss
        completion_only_loss=True,      # mask prompt tokens, train on completion only
        dataset_text_field=None,        # use prompt+completion columns directly
    ) if _HAS_SFTCONFIG else SFTConfig(
        # Fallback: pure TrainingArguments (should not be reached with trl>=1.0)
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        fp16=_use_fp16,
        bf16=False,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    # -------------------------------------------------------------------------
    # SFTTrainer
    # processing_class= replaces deprecated tokenizer= (trl>=0.16).
    # Native prompt/completion masking via completion_only_loss=True in SFTConfig.
    # No data collator needed — SFTTrainer handles masking internally when
    # dataset has "prompt" + "completion" columns and completion_only_loss=True.
    # -------------------------------------------------------------------------
    trainer_kwargs: dict = dict(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )
    # processing_class replaces tokenizer in trl>=0.16
    trainer_kwargs["processing_class"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

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

    # -------------------------------------------------------------------------
    # Diagnostic — confirm which autocast + scaler is actually active.
    # If mixed_precision shows "bf16" here, the cached config fix did not land;
    # rerun with --no-scaler as the guaranteed fallback.
    # -------------------------------------------------------------------------
    _mp = getattr(trainer.accelerator, "mixed_precision", "unknown")
    _scaler = getattr(trainer.accelerator, "scaler", None)
    print(f"\n[DIAG] accelerator.mixed_precision = {_mp!r}")
    print(f"[DIAG] accelerator.scaler           = {_scaler!r}")
    if _mp == "bf16":
        print("[DIAG] WARNING: mixed_precision is bf16 — bf16-unscale crash likely! "
              "Add --no-scaler to bypass GradScaler entirely.")

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
