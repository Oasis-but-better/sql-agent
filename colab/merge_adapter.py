"""
merge_adapter.py — Merge a LoRA adapter into the Qwen3.5-4B base model and
save the full merged model (for export / inference without PEFT).

Usage:
    python merge_adapter.py \
        --base-model  Qwen/Qwen3.5-4B \
        --adapter-dir /drive/adapters/qwen-sql-qlora/final_adapter \
        --output-dir  /drive/models/qwen-sql-merged

NOTE: Merge happens in fp16 on CPU (no GPU required). Peak RAM ~16 GB.
On Colab: ensure high-RAM runtime if merging in-session. Alternatively,
download the adapter locally and merge on a machine with enough RAM.
"""

from __future__ import annotations

import argparse
import pathlib


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge LoRA adapter into Qwen3.5-4B base.")
    p.add_argument("--base-model", required=True,
                   help="HF hub id or local path to base model (fp16 weights).")
    p.add_argument("--adapter-dir", required=True,
                   help="Directory containing the saved PEFT adapter.")
    p.add_argument("--output-dir", required=True,
                   help="Where to save the merged model + tokenizer.")
    p.add_argument("--device", default="cpu",
                   help="Device for loading base model during merge. 'cpu' is safe "
                        "and avoids VRAM limits. 'cuda' is faster if VRAM allows fp16.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )

    print(f"Loading adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    print("Merging adapter weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(str(output_dir), safe_serialization=True)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_dir, trust_remote_code=True
    )
    tokenizer.save_pretrained(str(output_dir))

    print(f"\nMerged model saved to: {output_dir}")
    print("Load for inference with: AutoModelForCausalLM.from_pretrained(output_dir)")


if __name__ == "__main__":
    main()
