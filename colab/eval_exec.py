"""
eval_exec.py — Execution-accuracy evaluation of the fine-tuned Qwen3.5-4B adapter
against the sql-agent test set.

REQUIREMENTS:
  - SQLite database files (Spider/BIRD) must be present locally.
    Default path: --db-root /drive/spider_data/database
    Each DB lives at: <db-root>/<db_id>/<db_id>.sqlite
  - The model generates SQL; we execute predicted vs gold against the same DB
    and compare result sets.

SELF-CONTAINED: does not import from project src/. Minimal verify logic restated.

Metrics:
  - exec_acc_strict:    result sets equal (same rows, same order after sort)
  - exec_acc_superset:  gold result is a subset of predicted result (permissive)

Usage:
    python eval_exec.py \
        --test-file   /drive/data/test.prompt.jsonl \
        --test-src    /path/to/data/dataset/test.jsonl \
        --db-root     /drive/spider_data/database \
        --base-model  Qwen/Qwen3.5-4B \
        --adapter-dir /drive/adapters/qwen-sql-qlora/final_adapter \
        --max-new-tokens 256 \
        --output-file eval_results.jsonl

Pass --base-only to evaluate base model without adapter (for baseline comparison).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sqlite3
import sys
from typing import Any


# ---------------------------------------------------------------------------
# SQL helpers (restated from project src/verify.py logic)
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SQL_RE = re.compile(
    r"(SELECT|INSERT|UPDATE|DELETE|WITH)\b.*",
    re.IGNORECASE | re.DOTALL,
)


def extract_sql(raw: str) -> str:
    """Strip <think>...</think>, return first SQL statement found."""
    cleaned = _THINK_RE.sub("", raw).strip()
    m = _SQL_RE.search(cleaned)
    if m:
        return m.group(0).strip()
    return cleaned


def execute_sql(db_path: pathlib.Path, sql: str) -> tuple[list[Any] | None, str | None]:
    """Execute SQL against SQLite DB. Returns (rows, error_msg)."""
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql)
        rows = [tuple(r) for r in cur.fetchall()]
        conn.close()
        return rows, None
    except sqlite3.Error as e:
        return None, str(e)


def rows_equal(pred: list, gold: list) -> bool:
    """Strict equality: sorted row sets must match."""
    try:
        return sorted(str(r) for r in pred) == sorted(str(r) for r in gold)
    except Exception:
        return False


def rows_superset(pred: list, gold: list) -> bool:
    """Gold is a subset of pred (permissive match)."""
    try:
        pred_set = set(str(r) for r in pred)
        gold_set = set(str(r) for r in gold)
        return gold_set.issubset(pred_set)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def build_model_fn(base_model: str, adapter_dir: str | None, max_new_tokens: int):
    """Return a callable: messages → predicted SQL string."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_dir:
        from peft import PeftModel
        print(f"Loading adapter: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()

    def model_fn(messages: list[dict]) -> str:
        # Render with enable_thinking=False (matches training format)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only new tokens
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return extract_sql(raw)

    return model_fn


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Execution-accuracy eval for Qwen3.5-4B sql-agent")
    p.add_argument("--test-file", required=True,
                   help="test.prompt.jsonl (prompt/completion pairs).")
    p.add_argument("--test-src", required=True,
                   help="Original test.jsonl (for db_id and gold SQL).")
    p.add_argument("--db-root", required=True,
                   help="Root dir containing <db_id>/<db_id>.sqlite files.")
    p.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--adapter-dir", default=None,
                   help="PEFT adapter dir. Omit for base-only eval.")
    p.add_argument("--base-only", action="store_true",
                   help="Skip adapter, eval base model only.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-examples", type=int, default=None,
                   help="Limit number of eval examples (for quick checks).")
    p.add_argument("--output-file", default="eval_results.jsonl",
                   help="Per-example results JSONL output.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    db_root = pathlib.Path(args.db_root)
    adapter = None if args.base_only else args.adapter_dir

    print("Building model...")
    model_fn = build_model_fn(args.base_model, adapter, args.max_new_tokens)

    # Load test source (db_id, gold SQL, messages)
    src_examples = []
    with open(args.test_src) as f:
        for line in f:
            line = line.strip()
            if line:
                src_examples.append(json.loads(line))

    if args.max_examples:
        src_examples = src_examples[:args.max_examples]

    n_strict = n_superset = n_exec_err = n_gold_err = n_total = 0
    results = []

    for i, ex in enumerate(src_examples):
        db_id = ex["db_id"]
        # Gold SQL: last assistant message in original messages
        orig_msgs = ex["messages"]
        gold_sql = None
        for m in reversed(orig_msgs):
            if m["role"] == "assistant":
                gold_sql = m["content"]
                break
        if gold_sql is None:
            print(f"  WARN example {i}: no gold SQL", file=sys.stderr)
            continue

        # Build inference messages (system + user only, no correction chain for eval)
        # Use first system + first user turn as standard single-shot eval
        infer_msgs = []
        for m in orig_msgs:
            if m["role"] in ("system", "user"):
                infer_msgs.append({"role": m["role"], "content": m["content"]})
            if m["role"] == "user" and len(infer_msgs) >= 2:
                break

        pred_sql = model_fn(infer_msgs)

        db_path = db_root / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            # Some DBs live at db_root/<db_id>.sqlite (flat layout)
            db_path = db_root / f"{db_id}.sqlite"
        if not db_path.exists():
            print(f"  WARN example {i}: DB not found for db_id={db_id}", file=sys.stderr)
            continue

        pred_rows, pred_err = execute_sql(db_path, pred_sql)
        gold_rows, gold_err = execute_sql(db_path, gold_sql)

        strict = superset = False
        if pred_err:
            n_exec_err += 1
        elif gold_err:
            n_gold_err += 1
        else:
            strict = rows_equal(pred_rows, gold_rows)
            superset = rows_superset(pred_rows, gold_rows)
            if strict:
                n_strict += 1
            if superset:
                n_superset += 1

        n_total += 1
        results.append({
            "db_id": db_id,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "exec_strict": strict,
            "exec_superset": superset,
            "pred_error": pred_err,
        })

        if (i + 1) % 50 == 0:
            acc = n_strict / n_total if n_total else 0
            print(f"  [{i+1}/{len(src_examples)}] strict_acc={acc:.3f}")

    # Summary
    if n_total:
        print(f"\n=== Evaluation Results ===")
        print(f"Total evaluated: {n_total}")
        print(f"Exec errors (pred): {n_exec_err} ({100*n_exec_err/n_total:.1f}%)")
        print(f"Gold errors:        {n_gold_err}")
        print(f"Exec acc (strict):  {n_strict}/{n_total} = {100*n_strict/n_total:.2f}%")
        print(f"Exec acc (superset):{n_superset}/{n_total} = {100*n_superset/n_total:.2f}%")

    # Save results
    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nPer-example results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
