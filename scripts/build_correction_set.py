#!/usr/bin/env python3
"""
build_correction_set.py — CP2: Build correction training set.

Two-phase hybrid:
  A) Base-model bootstrap: Qwen 4-bit generates wrong SQL → verify → correction chain.
  B) Programmatic perturbation: Perturb gold SQL → verify → correction chain.

Output: data/dataset/correction.jsonl
Resumable: skips already-written (question+gold_sql) keys on restart.

Usage:
    python scripts/build_correction_set.py [--target 6000] [--bootstrap-budget-min 30]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_JSONL = PROJECT_ROOT / "data" / "dataset" / "clean.jsonl"
OUT_JSONL = PROJECT_ROOT / "data" / "dataset" / "correction.jsonl"
MODEL_PATH = str(PROJECT_ROOT / "models" / "qwen-4bit")

# db_root candidates (Spider and BIRD layouts)
_DB_ROOTS = [
    PROJECT_ROOT / "data" / "spider" / "spider_data" / "database",
    PROJECT_ROOT / "data" / "bird" / "dev_20240627" / "dev_databases",
]

sys.path.insert(0, str(PROJECT_ROOT))
from src.verify import verify_sql, MatchVerdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_db_path(db_id: str) -> Optional[Path]:
    for root in _DB_ROOTS:
        candidate = root / db_id / f"{db_id}.sqlite"
        if candidate.exists():
            return candidate
    return None


def norm_key(question: str, gold_sql: str) -> str:
    """Dedup key: normalized question + gold SQL (case-fold, collapse whitespace)."""
    q = re.sub(r"\s+", " ", question.strip().lower())
    s = re.sub(r"\s+", " ", gold_sql.strip().lower())
    return f"{q}|||{s}"


def load_existing_keys() -> set:
    """Load keys already written to correction.jsonl (resumability)."""
    keys = set()
    if OUT_JSONL.exists():
        with open(OUT_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    # key is in metadata fields; messages[1]=user, messages[4]=gold
                    msgs = rec.get("messages", [])
                    question = ""
                    gold_sql = ""
                    for m in msgs:
                        if m["role"] == "user":
                            question = m["content"]
                        if m["role"] == "assistant" and "gold_sql" not in rec:
                            # last assistant is gold
                            gold_sql = m["content"]
                    # use stored fields if present
                    question = rec.get("question", question)
                    gold_sql = rec.get("gold_sql", gold_sql)
                    if question and gold_sql:
                        keys.add(norm_key(question, gold_sql))
                except Exception:
                    pass
    return keys


def build_correction_chain(
    schema_content: str,
    question: str,
    wrong_sql: str,
    error_msg: str,
    gold_sql: str,
) -> list[dict]:
    """Assemble 5-turn correction chain message list."""
    return [
        {"role": "system", "content": schema_content},
        {"role": "user", "content": question},
        {"role": "assistant", "content": wrong_sql},
        {"role": "tool", "content": error_msg},
        {"role": "assistant", "content": gold_sql},
    ]


def write_chain(
    fout,
    db_id: str,
    source: str,
    difficulty,
    wrong_source: str,
    question: str,
    gold_sql: str,
    chain: list[dict],
):
    rec = {
        "db_id": db_id,
        "source": source,
        "difficulty": difficulty,
        "type": "correction",
        "wrong_source": wrong_source,
        "question": question,
        "gold_sql": gold_sql,
        "messages": chain,
    }
    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fout.flush()


# ---------------------------------------------------------------------------
# Bootstrap (Phase A)
# ---------------------------------------------------------------------------

def build_prefill_prompt(tokenizer, schema_content: str, question: str) -> str:
    sys_msg = (
        "You are a SQL expert. Return ONLY a single valid SQLite query. "
        "No explanation. No markdown. No reasoning."
    )
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"Schema: {schema_content}\n\nQuestion: {question}"},
        {"role": "assistant", "content": "<think>\n\n</think>\n\n"},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    for eos in [tokenizer.eos_token, "<|im_end|>\n", "<|im_end|>"]:
        if eos and prompt.endswith(eos):
            prompt = prompt[: -len(eos)]
    return prompt


def extract_sql(text: str) -> Optional[str]:
    """Parse first SQL statement from model output."""
    # strip chat tokens
    text = re.sub(r"<\|im_start\|>.*?\n", "", text)
    text = re.sub(r"<\|im_end\|>.*", "", text)
    # strip think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # strip markdown fences
    text = re.sub(r"```sql\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)
    # split on ; + newline, take first
    parts = re.split(r";\s*\n", text + "\n")
    candidate = parts[0].strip().rstrip(";").strip()
    if re.match(r"^\s*(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|EXPLAIN)\b", candidate, re.IGNORECASE):
        return candidate
    # scan lines for SQL keyword start
    for i, line in enumerate(candidate.split("\n")):
        stripped = line.strip()
        if re.match(r"^(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE)\b", stripped, re.IGNORECASE):
            remainder = "\n".join(candidate.split("\n")[i:]).strip().rstrip(";")
            return remainder
    return None


def run_bootstrap(
    rows: list[dict],
    existing_keys: set,
    fout,
    budget_seconds: float,
    stats: dict,
) -> int:
    """Generate wrong SQL via model, verify, write corrections. Returns count."""
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    print("Loading model...", flush=True)
    model, tokenizer = load(MODEL_PATH)
    sampler = make_sampler(temp=0.0)
    print("Model loaded.", flush=True)

    # Throughput probe: time 20 gens
    print("Throughput probe (20 gens)...", flush=True)
    probe_times = []
    for row in rows[:20]:
        schema = row["messages"][0]["content"]
        question = row["messages"][1]["content"]
        prompt = build_prefill_prompt(tokenizer, schema, question)
        t0 = time.monotonic()
        generate(model, tokenizer, prompt=prompt, max_tokens=256, sampler=sampler)
        probe_times.append(time.monotonic() - t0)

    sec_per_gen = sum(probe_times) / len(probe_times)
    stats["sec_per_gen"] = round(sec_per_gen, 3)
    print(f"sec/gen = {sec_per_gen:.3f}", flush=True)

    if sec_per_gen > 20.0:
        print(f"SKIP bootstrap: sec/gen={sec_per_gen:.3f} > 20s threshold.", flush=True)
        stats["bootstrap_skipped"] = True
        return 0

    bootstrap_n = min(len(rows), int(budget_seconds / sec_per_gen))
    stats["bootstrap_n"] = bootstrap_n
    print(f"Bootstrap target: {bootstrap_n} gens (budget {budget_seconds/60:.1f}min @ {sec_per_gen:.2f}s/gen)", flush=True)

    count = 0
    tried = 0
    wall_start = time.monotonic()

    for row in rows[:bootstrap_n]:
        if time.monotonic() - wall_start > budget_seconds:
            print(f"Bootstrap wall-time budget exhausted at {tried} tried.", flush=True)
            break

        db_id = row["db_id"]
        source = row.get("source", "")
        difficulty = row.get("difficulty")
        schema = row["messages"][0]["content"]
        question = row["messages"][1]["content"]
        gold_sql = row["messages"][2]["content"]

        key = norm_key(question, gold_sql)
        if key in existing_keys:
            continue

        db_path = find_db_path(db_id)
        if db_path is None:
            tried += 1
            continue

        prompt = build_prefill_prompt(tokenizer, schema, question)
        raw_out = generate(model, tokenizer, prompt=prompt, max_tokens=256, sampler=sampler)
        tried += 1

        model_sql = extract_sql(raw_out)
        if not model_sql:
            stats["bootstrap_parse_fail"] = stats.get("bootstrap_parse_fail", 0) + 1
            continue

        # verify model SQL
        try:
            result = verify_sql(str(db_path), model_sql, gold_sql)
        except ValueError:
            # gold failed — skip (shouldn't happen for clean set)
            continue

        if result.is_valid:
            # Model got it right — not a correction chain
            stats["bootstrap_correct"] = stats.get("bootstrap_correct", 0) + 1
            continue

        # Build error message
        if result.verdict == MatchVerdict.ERROR:
            error_msg = result.traceback or "execution error"
        else:
            error_msg = "incorrect result: query executed but result does not match expected output"

        chain = build_correction_chain(schema, question, model_sql, error_msg, gold_sql)
        write_chain(fout, db_id, source, difficulty, "bootstrap", question, gold_sql, chain)
        existing_keys.add(key)
        count += 1
        stats["db_ids_seen"].add(db_id)

        if count % 50 == 0:
            elapsed = time.monotonic() - wall_start
            print(f"  Bootstrap: {count} corrections, {tried} tried, {elapsed:.0f}s elapsed", flush=True)

    stats["bootstrap_tried"] = tried
    stats["bootstrap_wall_seconds"] = round(time.monotonic() - wall_start, 1)
    stats["bootstrap_failure_rate"] = round(
        (tried - stats.get("bootstrap_correct", 0) - stats.get("bootstrap_parse_fail", 0)) / max(tried, 1),
        4,
    )
    print(f"Bootstrap done: {count} corrections from {tried} attempts.", flush=True)
    return count


# ---------------------------------------------------------------------------
# Perturbation (Phase B)
# ---------------------------------------------------------------------------

_PERTURBATIONS = []


def _perturb_drop_where(sql: str) -> Optional[str]:
    """Remove WHERE clause."""
    m = re.search(r"\bWHERE\b", sql, re.IGNORECASE)
    if not m:
        return None
    # find end of WHERE clause (before GROUP BY / ORDER BY / HAVING / LIMIT or end)
    end_m = re.search(
        r"\b(GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|UNION|EXCEPT|INTERSECT)\b",
        sql[m.start():],
        re.IGNORECASE,
    )
    if end_m:
        result = sql[: m.start()] + sql[m.start() + end_m.start():]
    else:
        result = sql[: m.start()]
    return result.strip() or None


def _perturb_drop_group_by(sql: str) -> Optional[str]:
    """Remove GROUP BY clause (and HAVING if present)."""
    m = re.search(r"\bGROUP\s+BY\b", sql, re.IGNORECASE)
    if not m:
        return None
    end_m = re.search(
        r"\b(ORDER\s+BY|LIMIT|UNION|EXCEPT|INTERSECT)\b",
        sql[m.start():],
        re.IGNORECASE,
    )
    if end_m:
        result = sql[: m.start()] + sql[m.start() + end_m.start():]
    else:
        result = sql[: m.start()]
    return result.strip() or None


def _perturb_drop_join(sql: str) -> Optional[str]:
    """Remove first JOIN clause."""
    m = re.search(r"\b(INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|JOIN)\b", sql, re.IGNORECASE)
    if not m:
        return None
    # find next JOIN or WHERE/GROUP BY/ORDER BY
    after = sql[m.end():]
    end_m = re.search(
        r"\b(INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|JOIN|WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b",
        after,
        re.IGNORECASE,
    )
    if end_m:
        result = sql[: m.start()] + sql[m.end() + end_m.start():]
    else:
        result = sql[: m.start()]
    return result.strip() or None


def _perturb_wrong_column(sql: str) -> Optional[str]:
    """Rename first column reference to invalid name."""
    m = re.search(r"\bSELECT\s+([\w.]+)", sql, re.IGNORECASE)
    if not m:
        return None
    orig = m.group(1)
    # avoid replacing * or numeric literals
    if orig in ("*", "1") or orig.isdigit():
        return None
    replacement = orig + "_WRONG_COL_XYZ"
    return sql[: m.start(1)] + replacement + sql[m.end(1):]


def _perturb_drop_having(sql: str) -> Optional[str]:
    """Remove HAVING clause."""
    m = re.search(r"\bHAVING\b", sql, re.IGNORECASE)
    if not m:
        return None
    end_m = re.search(
        r"\b(ORDER\s+BY|LIMIT|UNION|EXCEPT|INTERSECT)\b",
        sql[m.start():],
        re.IGNORECASE,
    )
    if end_m:
        result = sql[: m.start()] + sql[m.start() + end_m.start():]
    else:
        result = sql[: m.start()]
    return result.strip() or None


def _perturb_drop_limit(sql: str) -> Optional[str]:
    """Remove LIMIT clause."""
    m = re.search(r"\bLIMIT\s+\d+", sql, re.IGNORECASE)
    if not m:
        return None
    # removing LIMIT may make results match MORE not less — only use if result is wrong
    return (sql[: m.start()] + sql[m.end():]).strip() or None


PERTURBATION_FNS = [
    _perturb_drop_where,
    _perturb_wrong_column,
    _perturb_drop_join,
    _perturb_drop_group_by,
    _perturb_drop_having,
    _perturb_drop_limit,
]


def perturb_sql(gold_sql: str) -> list[str]:
    """Return list of distinct perturbed variants (may be empty)."""
    results = []
    seen = set()
    for fn in PERTURBATION_FNS:
        try:
            p = fn(gold_sql)
            if p and p != gold_sql and p not in seen:
                seen.add(p)
                results.append(p)
        except Exception:
            pass
    return results


def run_perturbation(
    rows: list[dict],
    existing_keys: set,
    fout,
    target: int,
    stats: dict,
) -> int:
    """Apply perturbations to gold SQL until target total corrections reached."""
    count = 0
    tried_pairs = 0

    for row in rows:
        if count >= target:
            break

        db_id = row["db_id"]
        source = row.get("source", "")
        difficulty = row.get("difficulty")
        schema = row["messages"][0]["content"]
        question = row["messages"][1]["content"]
        gold_sql = row["messages"][2]["content"]

        key = norm_key(question, gold_sql)
        if key in existing_keys:
            continue  # bootstrap already handled this pair

        db_path = find_db_path(db_id)
        if db_path is None:
            continue

        # verify gold is still good (sanity — should always pass from clean set)
        try:
            gold_result = verify_sql(str(db_path), gold_sql, gold_sql)
        except ValueError:
            continue
        if not gold_result.is_valid:
            continue

        tried_pairs += 1
        candidates = perturb_sql(gold_sql)
        accepted = False

        for perturbed in candidates:
            if count >= target:
                break
            try:
                result = verify_sql(str(db_path), perturbed, gold_sql)
            except ValueError:
                continue

            if result.is_valid:
                # perturbed still matches — not useful
                continue

            # genuine failure
            if result.verdict == MatchVerdict.ERROR:
                error_msg = result.traceback or "execution error"
            else:
                error_msg = "incorrect result: query executed but result does not match expected output"

            chain = build_correction_chain(schema, question, perturbed, error_msg, gold_sql)
            write_chain(fout, db_id, source, difficulty, "perturbation", question, gold_sql, chain)
            existing_keys.add(key)
            count += 1
            accepted = True
            stats["db_ids_seen"].add(db_id)
            break  # one correction per pair (dedup by key)

        if count % 500 == 0 and count > 0:
            print(f"  Perturbation: {count} corrections from {tried_pairs} pairs tried", flush=True)

    stats["perturbation_tried_pairs"] = tried_pairs
    print(f"Perturbation done: {count} corrections.", flush=True)
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=6000)
    parser.add_argument("--bootstrap-budget-min", type=float, default=30.0)
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()

    target = args.target
    bootstrap_budget_sec = args.bootstrap_budget_min * 60.0

    # Load clean rows
    print(f"Loading {CLEAN_JSONL}...", flush=True)
    rows = []
    with open(CLEAN_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} clean rows.", flush=True)

    # Shuffle deterministically for bootstrap variety
    import random
    rng = random.Random(42)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    # Load existing keys for resumability
    existing_keys = load_existing_keys()
    print(f"Resuming: {len(existing_keys)} already-written keys found.", flush=True)

    stats: dict = {
        "db_ids_seen": set(),
        "bootstrap_corrections": 0,
        "perturbation_corrections": 0,
        "sec_per_gen": None,
        "bootstrap_wall_seconds": 0,
        "bootstrap_skipped": False,
    }

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    fout = open(OUT_JSONL, "a", encoding="utf-8")

    try:
        # Count already written
        already_written = sum(1 for _ in existing_keys)

        # Phase A: Bootstrap
        bootstrap_count = 0
        if not args.skip_bootstrap:
            bootstrap_count = run_bootstrap(shuffled, existing_keys, fout, bootstrap_budget_sec, stats)
        else:
            print("Bootstrap skipped via --skip-bootstrap flag.", flush=True)
            stats["bootstrap_skipped"] = True
        stats["bootstrap_corrections"] = bootstrap_count

        # Phase B: Perturbation (fill remainder)
        current_total = already_written + bootstrap_count
        # Count actual lines in file
        fout.flush()
        with open(OUT_JSONL, encoding="utf-8") as tmp:
            current_total = sum(1 for ln in tmp if ln.strip())

        perturb_needed = max(0, target - current_total)
        print(f"Current total: {current_total}. Need {perturb_needed} more via perturbation.", flush=True)

        perturb_count = 0
        if perturb_needed > 0:
            # Shuffle rows differently for perturbation (different seed)
            perturb_rows = list(rows)
            rng2 = random.Random(99)
            rng2.shuffle(perturb_rows)
            perturb_count = run_perturbation(perturb_rows, existing_keys, fout, perturb_needed, stats)
        stats["perturbation_corrections"] = perturb_count

    finally:
        fout.close()

    # Final count
    with open(OUT_JSONL, encoding="utf-8") as f:
        final_count = sum(1 for ln in f if ln.strip())

    # Model failure rate
    bootstrap_tried = stats.get("bootstrap_tried", 0)
    bootstrap_correct = stats.get("bootstrap_correct", 0)
    bootstrap_parse_fail = stats.get("bootstrap_parse_fail", 0)
    if bootstrap_tried > 0:
        failure_pct = 100.0 * (bootstrap_tried - bootstrap_correct - bootstrap_parse_fail) / bootstrap_tried
    else:
        failure_pct = 0.0

    print("\n" + "=" * 60)
    print("CP2 RESULTS")
    print("=" * 60)
    print(f"bootstrap_corrections : {stats['bootstrap_corrections']}")
    print(f"perturbation_corrections: {stats['perturbation_corrections']}")
    print(f"total_lines           : {final_count}")
    print(f"bootstrap_skipped     : {stats['bootstrap_skipped']}")
    if stats.get("sec_per_gen") is not None:
        print(f"sec_per_gen           : {stats['sec_per_gen']:.3f}")
    print(f"bootstrap_wall_sec    : {stats['bootstrap_wall_seconds']:.1f}")
    print(f"model_first_try_fail% : {failure_pct:.1f}%")
    print(f"bootstrap_tried       : {bootstrap_tried}")
    print(f"bootstrap_correct     : {bootstrap_correct}")
    print(f"distinct_db_ids       : {len(stats['db_ids_seen'])}")
    print(f"output_file           : {OUT_JSONL}")
    print("=" * 60)

    # Return stats dict for caller
    return {
        "bootstrap_corrections": stats["bootstrap_corrections"],
        "perturbation_corrections": stats["perturbation_corrections"],
        "total": final_count,
        "sec_per_gen": stats.get("sec_per_gen"),
        "bootstrap_wall_seconds": stats["bootstrap_wall_seconds"],
        "bootstrap_skipped": stats["bootstrap_skipped"],
        "model_first_try_fail_pct": round(failure_pct, 1),
        "distinct_db_ids": len(stats["db_ids_seen"]),
    }


if __name__ == "__main__":
    main()
