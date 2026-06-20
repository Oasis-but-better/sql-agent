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
