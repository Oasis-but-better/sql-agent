import json
import re
from pathlib import Path
import pytest
from src.build_dataset import build_dataset, dedup_examples, split_examples, normalize_key


def _make_example(db_id, question, sql, type_="clean", difficulty=None):
    return {
        "db_id": db_id,
        "source": "spider",
        "difficulty": difficulty,
        "type": type_,
        "query_types": [],
        "messages": [
            {"role": "system", "content": "{}"},
            {"role": "user", "content": question},
            {"role": "assistant", "content": sql},
        ],
    }


# --- normalize_key ---

def test_normalize_key_lowercases_and_collapses_whitespace():
    key = normalize_key("SELECT  count(*) FROM t", "How  many?")
    assert key == normalize_key("select count(*) from t", "how many?")


def test_normalize_key_replaces_number_literals():
    k1 = normalize_key("SELECT * FROM t WHERE age > 56", "singers older than 56")
    k2 = normalize_key("SELECT * FROM t WHERE age > 99", "singers older than 99")
    assert k1 == k2  # both normalize to <NUM>


# --- dedup_examples ---

def test_dedup_removes_exact_duplicates():
    examples = [
        _make_example("db1", "How many?", "SELECT count(*) FROM t"),
        _make_example("db1", "How many?", "SELECT count(*) FROM t"),  # duplicate
    ]
    deduped, dropped = dedup_examples(examples)
    assert len(deduped) == 1
    assert dropped == 1


def test_dedup_keeps_distinct_questions():
    examples = [
        _make_example("db1", "How many singers?", "SELECT count(*) FROM singer"),
        _make_example("db1", "How many concerts?", "SELECT count(*) FROM concert"),
    ]
    deduped, dropped = dedup_examples(examples)
    assert len(deduped) == 2
    assert dropped == 0


def test_dedup_removes_normalized_duplicates():
    """Different number literals in same template → same normalized key → dedup."""
    examples = [
        _make_example("db1", "Singers older than 30", "SELECT name FROM t WHERE age > 30"),
        _make_example("db1", "Singers older than 50", "SELECT name FROM t WHERE age > 50"),
    ]
    deduped, dropped = dedup_examples(examples)
    assert len(deduped) == 1
    assert dropped == 1


# --- split_examples ---

def test_split_returns_three_lists():
    examples = [_make_example("db1", f"Q{i}", f"SELECT {i} FROM t") for i in range(20)]
    train, val, test = split_examples(examples, val_frac=0.1, test_frac=0.1)
    assert len(train) + len(val) + len(test) == 20


def test_split_train_is_largest():
    examples = [_make_example("db1", f"Q{i}", f"SELECT {i} FROM t") for i in range(100)]
    train, val, test = split_examples(examples, val_frac=0.1, test_frac=0.1)
    assert len(train) > len(val)
    assert len(train) > len(test)


# --- build_dataset (integration) ---

def test_build_dataset_emits_jsonl_files(tmp_path):
    examples = [_make_example("db1", f"Q{i}", f"SELECT {i} FROM t") for i in range(30)]
    stats = build_dataset(examples, [], output_dir=str(tmp_path))
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "val.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()


def test_build_dataset_jsonl_each_line_valid_json(tmp_path):
    examples = [_make_example("db1", f"Q{i}", f"SELECT {i} FROM t") for i in range(30)]
    build_dataset(examples, [], output_dir=str(tmp_path))
    with open(tmp_path / "train.jsonl") as f:
        for line in f:
            obj = json.loads(line.strip())
            assert "messages" in obj
            assert "db_id" in obj


def test_build_dataset_stats_returned(tmp_path):
    examples = [_make_example("db1", f"Q{i}", f"SELECT {i} FROM t") for i in range(30)]
    stats = build_dataset(examples, [], output_dir=str(tmp_path))
    assert "total_after_dedup" in stats
    assert "train" in stats
    assert "val" in stats
    assert "test" in stats
    assert "dropped_duplicates" in stats


def test_build_dataset_combines_clean_and_correction(tmp_path):
    clean = [_make_example("db1", f"clean Q{i}", f"SELECT {i} FROM t", type_="clean") for i in range(20)]
    correction = [_make_example("db1", f"corr Q{i}", f"SELECT {i}+100 FROM t", type_="correction") for i in range(10)]
    stats = build_dataset(clean, correction, output_dir=str(tmp_path))
    assert stats["total_after_dedup"] <= 30
