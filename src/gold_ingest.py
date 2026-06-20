"""
gold_ingest.py — Normalize Spider and BIRD gold JSON into GoldRecord + message lists.

Spider gold field names: db_id, question, query (SQL string), query_toks,
    query_toks_no_value, question_toks, sql (parsed dict — NOT the SQL string).
BIRD gold field names: question_id, db_id, question, evidence, SQL (uppercase),
    difficulty.

Output message list format (model-agnostic, per spec §4.1):
    [
      {"role": "system",    "content": "<json-serialized db_metadata_cache>"},
      {"role": "user",      "content": "<question>"},
      {"role": "assistant", "content": "<gold SQL>"},
    ]
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GoldRecord:
    db_id: str
    question: str
    gold_sql: str
    source: str          # "spider" or "bird"
    difficulty: str | None  # None for Spider train (no difficulty field)


def load_spider_gold(json_path: str) -> list[GoldRecord]:
    """Load train_spider.json or train_others.json.

    Reads 'db_id', 'question', 'query' (SQL string).
    Ignores 'sql' (parsed dict), 'query_toks', 'question_toks'.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        GoldRecord(
            db_id=row["db_id"],
            question=row["question"],
            gold_sql=row["query"],
            source="spider",
            difficulty=row.get("difficulty"),  # absent in train_spider.json
        )
        for row in data
    ]


def load_bird_gold(json_path: str) -> list[GoldRecord]:
    """Load BIRD dev.json.

    Reads 'db_id', 'question', 'SQL' (uppercase — the SQL string), 'difficulty'.
    Ignores 'question_id', 'evidence'.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        GoldRecord(
            db_id=row["db_id"],
            question=row["question"],
            gold_sql=row["SQL"],
            source="bird",
            difficulty=row.get("difficulty"),
        )
        for row in data
    ]


def build_message_list(record: GoldRecord, schema_cache: dict) -> list[dict]:
    """Assemble a 3-turn message list from a GoldRecord and its compiled schema cache.

    System content = JSON-serialized schema cache (compact).
    User content   = natural language question.
    Assistant      = gold SQL string.
    """
    return [
        {"role": "system",    "content": json.dumps(schema_cache, separators=(",", ":"))},
        {"role": "user",      "content": record.question},
        {"role": "assistant", "content": record.gold_sql},
    ]


from src.schema_cache import compile_schema_cache
from src.verify import verify_sql


def build_clean_examples(
    records: list[GoldRecord],
    db_root: str,
) -> tuple[list[dict], dict]:
    """Verify each GoldRecord against its .sqlite and emit clean examples.

    DB path resolution: <db_root>/<db_id>/<db_id>.sqlite (Spider layout).
    BIRD path is identical: <db_root>/<db_id>/<db_id>.sqlite.

    Discards records whose gold SQL fails to execute on the real DB.
    Returns (examples, stats) where stats = {total, accepted,
    discarded_gold_error, discarded_db_not_found}.
    """
    stats = {
        "total": len(records),
        "accepted": 0,
        "discarded_gold_error": 0,
        "discarded_db_not_found": 0,
    }
    examples = []

    for rec in records:
        db_path = str(Path(db_root) / rec.db_id / f"{rec.db_id}.sqlite")
        if not Path(db_path).exists():
            stats["discarded_db_not_found"] += 1
            continue

        try:
            verify_result = verify_sql(db_path, rec.gold_sql, rec.gold_sql)
        except ValueError:
            # Gold SQL itself failed
            stats["discarded_gold_error"] += 1
            continue

        if not verify_result.is_valid:
            stats["discarded_gold_error"] += 1
            continue

        cache = compile_schema_cache(db_path)
        messages = build_message_list(rec, cache)
        examples.append({
            "db_id": rec.db_id,
            "source": rec.source,
            "difficulty": rec.difficulty,
            "type": "clean",
            "query_types": [],
            "messages": messages,
        })
        stats["accepted"] += 1

    return examples, stats
