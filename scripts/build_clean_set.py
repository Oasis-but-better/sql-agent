"""
build_clean_set.py — CP1 runner: load Spider+BIRD gold, verify, write clean.jsonl.

Usage:
    .venv/bin/python scripts/build_clean_set.py

Outputs:
    data/dataset/clean.jsonl   — one JSON dict per verified clean example
    (stats printed to stdout)

DB path resolution:
    Spider: data/spider/spider_data/database/<db_id>/<db_id>.sqlite
    BIRD:   data/bird/dev_20240627/dev_databases/<db_id>/<db_id>.sqlite
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure repo root on sys.path so "from src.X import ..." works
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.gold_ingest import load_spider_gold, load_bird_gold, build_message_list
from src.schema_cache import compile_schema_cache
from src.verify import run_query_with_timeout

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SPIDER_ROOT = _REPO_ROOT / "data" / "spider" / "spider_data"
SPIDER_DB_ROOT = SPIDER_ROOT / "database"
SPIDER_FILES = [
    SPIDER_ROOT / "train_spider.json",
    SPIDER_ROOT / "train_others.json",
]

BIRD_ROOT = _REPO_ROOT / "data" / "bird" / "dev_20240627"
BIRD_DB_ROOT = BIRD_ROOT / "dev_databases"
BIRD_FILE = BIRD_ROOT / "dev.json"

OUTPUT_PATH = _REPO_ROOT / "data" / "dataset" / "clean.jsonl"

# Per-query timeout (seconds) — same as verify.py default
WALL_SECONDS = 3.0
MAX_OPS = 20_000_000


def _categorize_error(err: str) -> str:
    """Bucket raw SQLite error string into a category label."""
    e = err.lower()
    if "no such table" in e:
        return "no_such_table"
    if "no such column" in e:
        return "no_such_column"
    if "syntax error" in e or "parse error" in e:
        return "syntax_error"
    if "timeout" in e or "interrupted" in e:
        return "timeout"
    if "ambiguous" in e:
        return "ambiguous_column"
    return "other_error"


def _db_path(rec, spider_db_root: Path, bird_db_root: Path) -> Path:
    if rec.source == "spider":
        return spider_db_root / rec.db_id / f"{rec.db_id}.sqlite"
    else:  # bird
        return bird_db_root / rec.db_id / f"{rec.db_id}.sqlite"


def main():
    t0 = time.monotonic()

    # --- Load gold records ---
    all_records = []
    for fpath in SPIDER_FILES:
        if fpath.exists():
            recs = load_spider_gold(str(fpath))
            print(f"Loaded {len(recs):,} Spider records from {fpath.name}")
            all_records.extend(recs)
        else:
            print(f"WARNING: Spider file not found: {fpath}")

    if BIRD_FILE.exists():
        bird_recs = load_bird_gold(str(BIRD_FILE))
        print(f"Loaded {len(bird_recs):,} BIRD records from {BIRD_FILE.name}")
        all_records.extend(bird_recs)
    else:
        print(f"WARNING: BIRD file not found: {BIRD_FILE}")

    total_in = len(all_records)
    print(f"\nTotal pairs loaded: {total_in:,}")

    # --- Caches ---
    # Schema cache: keyed by (source, db_id) → dict
    schema_cache_map: dict[tuple, dict] = {}

    # --- Stats ---
    stats = {
        "total": total_in,
        "valid": 0,
        "discarded_db_not_found": 0,
        "error_categories": defaultdict(int),
    }
    per_source = defaultdict(lambda: {"total": 0, "valid": 0})
    per_db = defaultdict(int)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    examples = []

    print("\nProcessing...")
    tick = max(1, total_in // 50)  # progress every ~2%

    for i, rec in enumerate(all_records):
        per_source[rec.source]["total"] += 1

        if i % tick == 0:
            elapsed = time.monotonic() - t0
            pct = 100 * i / total_in if total_in else 0
            print(f"  [{pct:5.1f}%] {i:,}/{total_in:,}  elapsed={elapsed:.1f}s", flush=True)

        db_path = _db_path(rec, SPIDER_DB_ROOT, BIRD_DB_ROOT)
        if not db_path.exists():
            stats["discarded_db_not_found"] += 1
            stats["error_categories"]["db_not_found"] += 1
            continue

        rows, err = run_query_with_timeout(
            str(db_path), rec.gold_sql,
            wall_seconds=WALL_SECONDS, max_ops=MAX_OPS
        )
        if err is not None:
            cat = _categorize_error(err)
            stats["error_categories"][cat] += 1
            continue

        # Valid — compile schema (reuse per db_id+source)
        cache_key = (rec.source, rec.db_id)
        if cache_key not in schema_cache_map:
            try:
                schema_cache_map[cache_key] = compile_schema_cache(str(db_path))
            except Exception:
                # Reserved keyword table name or other PRAGMA error
                schema_cache_map[cache_key] = None  # sentinel
        cache = schema_cache_map[cache_key]
        if cache is None:
            stats["error_categories"]["schema_compile_error"] += 1
            continue

        messages = build_message_list(rec, cache)
        example = {
            "db_id": rec.db_id,
            "source": rec.source,
            "difficulty": rec.difficulty,
            "type": "clean",
            "query_types": [],
            "messages": messages,
        }
        examples.append(example)
        stats["valid"] += 1
        per_source[rec.source]["valid"] += 1
        per_db[rec.db_id] += 1

    elapsed_total = time.monotonic() - t0

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    # --- Report ---
    discarded = total_in - stats["valid"]
    discard_pct = 100 * discarded / total_in if total_in else 0
    valid_pct = 100 * stats["valid"] / total_in if total_in else 0
    distinct_dbs = len(per_db)

    print("\n" + "=" * 60)
    print("CP1 RESULTS")
    print("=" * 60)
    print(f"Total in:       {total_in:,}")
    print(f"Valid out:       {stats['valid']:,}  ({valid_pct:.1f}%)")
    print(f"Discarded:       {discarded:,}  ({discard_pct:.1f}%)")
    print(f"  db_not_found:  {stats['error_categories'].get('db_not_found', 0):,}")
    for cat, count in sorted(stats["error_categories"].items(), key=lambda x: -x[1]):
        if cat != "db_not_found":
            print(f"  {cat}: {count:,}")
    print(f"Distinct DBs:   {distinct_dbs:,}")
    print(f"Output:          {OUTPUT_PATH}")
    print(f"Elapsed:         {elapsed_total:.1f}s")
    print()
    print("Per-source breakdown:")
    for src, s in sorted(per_source.items()):
        src_pct = 100 * s["valid"] / s["total"] if s["total"] else 0
        print(f"  {src}: {s['valid']:,}/{s['total']:,}  ({src_pct:.1f}%)")
    print()
    print("Top 10 DBs by count:")
    for db_id, count in sorted(per_db.items(), key=lambda x: -x[1])[:10]:
        print(f"  {db_id}: {count}")


if __name__ == "__main__":
    main()
