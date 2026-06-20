"""
build_dataset.py — Orchestrator: combine clean+correction → dedup → split → emit JSONL.

Dedup key: normalize(sql + " ||| " + question) where normalize lowercases,
collapses whitespace, and replaces digit-runs with <NUM> and string literals
with <STR> (same strategy as llm-training/scripts/validate_data.py).

Split: deterministic structural split — sorted normalized-key index mod 10:
  bucket 0 → test (~10%)
  bucket 1 → val  (~10%)
  buckets 2-9 → train (~80%)

Output: data/dataset/train.jsonl, val.jsonl, test.jsonl (one JSON object per line).
Each line preserves the full example dict including messages.
"""

import json
import re
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Normalization (mirrors validate_data._normalize_template)
# ---------------------------------------------------------------------------

def normalize_key(sql: str, question: str) -> str:
    combined = sql + " ||| " + question
    combined = re.sub(r"'[^']*'", "<STR>", combined)
    combined = re.sub(r"\b\d+\b", "<NUM>", combined)
    combined = re.sub(r"\s+", " ", combined.lower().strip())
    return combined


# ---------------------------------------------------------------------------
# dedup_examples
# ---------------------------------------------------------------------------

def dedup_examples(examples: list[dict]) -> tuple[list[dict], int]:
    """Dedup by normalized (sql + question) hash. Returns (kept, dropped_count)."""
    seen: set[str] = set()
    kept = []
    dropped = 0
    for ex in examples:
        # Extract SQL from assistant turn and question from user turn
        sql = ""
        question = ""
        for msg in ex.get("messages", []):
            if msg["role"] == "assistant" and not sql:
                sql = msg["content"]
            if msg["role"] == "user":
                question = msg["content"]
        key = normalize_key(sql, question)
        if key in seen:
            dropped += 1
        else:
            seen.add(key)
            kept.append(ex)
    return kept, dropped


# ---------------------------------------------------------------------------
# split_examples — deterministic structural split
# ---------------------------------------------------------------------------

def split_examples(
    examples: list[dict],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Assign examples to train/val/test by sorted normalized key mod 10.

    bucket 0 → test, bucket 1 → val, buckets 2-9 → train.
    val_frac and test_frac params are informational — the mod-10 split
    approximates 10%/10%/80% deterministically.
    """
    key_to_examples: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        sql = ""
        question = ""
        for msg in ex.get("messages", []):
            if msg["role"] == "assistant" and not sql:
                sql = msg["content"]
            if msg["role"] == "user":
                question = msg["content"]
        key = normalize_key(sql, question)
        key_to_examples[key].append(ex)

    sorted_keys = sorted(key_to_examples.keys())
    train, val, test = [], [], []
    for i, key in enumerate(sorted_keys):
        bucket = i % 10
        if bucket == 0:
            test.extend(key_to_examples[key])
        elif bucket == 1:
            val.extend(key_to_examples[key])
        else:
            train.extend(key_to_examples[key])

    return train, val, test


# ---------------------------------------------------------------------------
# build_dataset — main orchestrator
# ---------------------------------------------------------------------------

def build_dataset(
    clean_examples: list[dict],
    correction_examples: list[dict],
    output_dir: str,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> dict:
    """Combine clean + correction, dedup, split, write JSONL.

    Returns stats dict: {total_input, dropped_duplicates, total_after_dedup,
                         train, val, test, clean_count, correction_count}.
    """
    all_examples = clean_examples + correction_examples
    total_input = len(all_examples)
    clean_count = len(clean_examples)
    correction_count = len(correction_examples)

    deduped, dropped = dedup_examples(all_examples)
    total_after_dedup = len(deduped)

    train, val, test = split_examples(deduped, val_frac, test_frac)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        with open(out / f"{split_name}.jsonl", "w", encoding="utf-8") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")

    return {
        "total_input": total_input,
        "clean_count": clean_count,
        "correction_count": correction_count,
        "dropped_duplicates": dropped,
        "total_after_dedup": total_after_dedup,
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }
