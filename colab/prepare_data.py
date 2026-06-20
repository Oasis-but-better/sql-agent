"""
prepare_data.py — Render train/val/test JSONL into prompt/completion pairs
for Colab QLoRA fine-tuning of Qwen3.5-4B on the sql-agent dataset.

Key correctness decisions:
  - tool role → user role with "Execution feedback:\n<content>" prefix
    (HF chat templates reject raw "tool" role)
  - Prompt = apply_chat_template(messages[:-1], add_generation_prompt=True,
    enable_thinking=False) → ends with <think>\n\n</think>\n\n
  - Completion = final assistant content + EOS token
    (ensures model learns to stop; no duplication since TRL strips it for loss)
  - enable_thinking=False matches project inference convention:
    qwen_model_fn.py prefills "<think></think>" to suppress reasoning prose.
    NOTE: qwen_model_fn uses bare "<think></think>" (no newlines); training uses
    the template's "<think>\n\n</think>\n\n". Reconcile at inference by switching
    qwen_model_fn to also use enable_thinking=False template rendering.
  - Examples where prompt+completion > max_seq_length are filtered (not truncated).
    Truncating from right would destroy the SQL target; left-truncation risks
    corrupting the system/schema turn. Filtering is safer.
  - Wrong assistant SQL in correction chains is INPUT CONTEXT, never a target.
    Loss is computed only on the final (gold) assistant turn.

Usage:
    python prepare_data.py [--data-dir /path/to/dataset] [--out-dir ./data]
                           [--tokenizer Qwen/Qwen3.5-4B] [--max-seq 1024]
                           [--sample N]

    --sample N: only process first N examples per split (for local validation).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Convert 'tool' role turns to user turns with Execution feedback prefix."""
    out = []
    for m in messages:
        if m["role"] == "tool":
            out.append({
                "role": "user",
                "content": "Execution feedback:\n" + m["content"],
            })
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def process_split(
    src_path: pathlib.Path,
    dst_path: pathlib.Path,
    tokenizer,
    max_seq: int,
    sample: int | None,
    eos_token: str,
) -> dict:
    """Process one split. Returns stats dict."""
    stats: dict[str, int] = Counter()
    bucket_1024 = bucket_1536 = bucket_2048 = 0

    with src_path.open() as fin, dst_path.open("w") as fout:
        for lineno, raw_line in enumerate(fin):
            if sample is not None and lineno >= sample:
                break
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                ex = json.loads(raw_line)
            except json.JSONDecodeError as e:
                print(f"  WARN line {lineno+1}: JSON error {e}", file=sys.stderr)
                stats["skipped_json"] += 1
                continue

            messages = ex.get("messages", [])
            if not messages:
                stats["skipped_no_messages"] += 1
                continue

            # Normalize (tool → user)
            messages = normalize_messages(messages)

            # Validate final turn is assistant (gold SQL)
            if messages[-1]["role"] != "assistant":
                print(
                    f"  WARN line {lineno+1}: last role is '{messages[-1]['role']}', expected 'assistant' — skipping",
                    file=sys.stderr,
                )
                stats["skipped_bad_final_role"] += 1
                continue

            # Split into prompt-messages and completion
            prompt_messages = messages[:-1]
            completion_text = messages[-1]["content"]

            # Render prompt with enable_thinking=False so the prompt ends with
            # <think>\n\n</think>\n\n  — suppresses reasoning, matches inference convention
            try:
                prompt = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except Exception as e:
                print(f"  WARN line {lineno+1}: template error {e}", file=sys.stderr)
                stats["skipped_template_error"] += 1
                continue

            # Append EOS so model learns to stop after SQL
            completion = completion_text + eos_token

            # Token-length check (filter, not truncate)
            prompt_ids = tokenizer.encode(prompt)
            completion_ids = tokenizer.encode(completion)
            total_len = len(prompt_ids) + len(completion_ids)

            if total_len > max_seq:
                stats["skipped_too_long"] += 1
                continue

            # Bucket counts (against max thresholds, not max_seq)
            if total_len <= 1024:
                bucket_1024 += 1
            if total_len <= 1536:
                bucket_1536 += 1
            if total_len <= 2048:
                bucket_2048 += 1

            stats["written"] += 1
            stats["total_tokens"] += total_len
            fout.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")

    stats["bucket_le1024"] = bucket_1024
    stats["bucket_le1536"] = bucket_1536
    stats["bucket_le2048"] = bucket_2048
    return dict(stats)


def print_stats(split_name: str, stats: dict, total_read: int) -> None:
    written = stats.get("written", 0)
    total_toks = stats.get("total_tokens", 0)
    avg = total_toks / written if written else 0

    print(f"\n=== {split_name} ===")
    print(f"  read:         {total_read}")
    print(f"  written:      {written} ({100*written/total_read:.1f}% kept)")
    skipped = total_read - written
    print(f"  skipped:      {skipped}")
    for key in ("skipped_json", "skipped_no_messages", "skipped_bad_final_role",
                "skipped_template_error", "skipped_too_long"):
        if stats.get(key, 0):
            print(f"    {key}: {stats[key]}")
    if written:
        print(f"  avg tokens:   {avg:.0f}")
        print(f"  fit ≤1024:    {stats['bucket_le1024']} ({100*stats['bucket_le1024']/written:.1f}%)")
        print(f"  fit ≤1536:    {stats['bucket_le1536']} ({100*stats['bucket_le1536']/written:.1f}%)")
        print(f"  fit ≤2048:    {stats['bucket_le2048']} ({100*stats['bucket_le2048']/written:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare prompt/completion JSONL for QLoRA fine-tuning.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing train.jsonl, val.jsonl, test.jsonl. "
             "Defaults to <project-root>/data/dataset.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <this-script's-dir>/data.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HuggingFace tokenizer path or hub id. "
             "Defaults to <project-root>/Qwen3.5-4B (local copy). "
             "On Colab: set to 'Qwen/Qwen3.5-4B' or a Drive path.",
    )
    parser.add_argument(
        "--max-seq",
        type=int,
        default=1024,
        help="Maximum total sequence length (prompt+completion tokens). "
             "Examples exceeding this are filtered. Default 1024.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process first N examples per split (local validation mode).",
    )
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent
    project_root = script_dir.parent

    data_dir = pathlib.Path(args.data_dir) if args.data_dir else project_root / "data" / "dataset"
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else script_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = args.tokenizer if args.tokenizer else str(project_root / "Qwen3.5-4B")

    print(f"Loading tokenizer from: {tokenizer_path}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    eos_token = tokenizer.eos_token or "<|im_end|>"
    print(f"EOS token: {repr(eos_token)}")

    splits = [
        ("train",  "train.jsonl",  "train.prompt.jsonl"),
        ("val",    "val.jsonl",    "val.prompt.jsonl"),
        ("test",   "test.jsonl",   "test.prompt.jsonl"),
    ]

    for split_name, src_fname, dst_fname in splits:
        src = data_dir / src_fname
        dst = out_dir / dst_fname
        if not src.exists():
            print(f"SKIP {src} — not found")
            continue

        # Count total lines (for stats), then process
        with src.open() as f:
            total_lines = sum(1 for line in f if line.strip())
        if args.sample is not None:
            total_lines = min(total_lines, args.sample)

        stats = process_split(src, dst, tokenizer, args.max_seq, args.sample, eos_token)
        print_stats(split_name, stats, total_lines)
        print(f"  → {dst}")

    print("\nDone.")


if __name__ == "__main__":
    main()
