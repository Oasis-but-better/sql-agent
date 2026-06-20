import json
import sqlite3
import pytest
from src.gold_ingest import GoldRecord, build_clean_examples


@pytest.fixture
def two_db_setup(tmp_path):
    """Two sqlite DBs, a valid and an invalid gold record."""
    # DB 1: concert_singer — valid gold
    db1 = tmp_path / "concert_singer" / "concert_singer.sqlite"
    db1.parent.mkdir()
    conn = sqlite3.connect(str(db1))
    conn.executescript("""
        CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        INSERT INTO singer VALUES (1, 'Alice', 30), (2, 'Bob', 22);
    """)
    conn.commit()
    conn.close()

    # DB 2: broken_db — invalid gold (column doesn't exist)
    db2 = tmp_path / "broken_db" / "broken_db.sqlite"
    db2.parent.mkdir()
    conn = sqlite3.connect(str(db2))
    conn.executescript("CREATE TABLE t (id INTEGER);")
    conn.commit()
    conn.close()

    records = [
        GoldRecord(
            db_id="concert_singer",
            question="How many singers are older than 20?",
            gold_sql="SELECT count(*) FROM singer WHERE age > 20",
            source="spider",
            difficulty=None,
        ),
        GoldRecord(
            db_id="broken_db",
            question="What are the names?",
            gold_sql="SELECT nonexistent FROM t",  # will fail
            source="spider",
            difficulty=None,
        ),
    ]
    return records, str(tmp_path)


def test_build_clean_examples_returns_valid_only(two_db_setup):
    records, db_root = two_db_setup
    examples, stats = build_clean_examples(records, db_root)
    assert len(examples) == 1
    assert examples[0]["db_id"] == "concert_singer"


def test_build_clean_examples_stats_track_yield(two_db_setup):
    records, db_root = two_db_setup
    examples, stats = build_clean_examples(records, db_root)
    assert stats["total"] == 2
    assert stats["accepted"] == 1
    assert stats["discarded_gold_error"] == 1


def test_example_has_required_fields(two_db_setup):
    records, db_root = two_db_setup
    examples, _ = build_clean_examples(records, db_root)
    ex = examples[0]
    assert "db_id" in ex
    assert "source" in ex
    assert "difficulty" in ex
    assert "type" in ex
    assert ex["type"] == "clean"
    assert "query_types" in ex
    assert "messages" in ex
    assert len(ex["messages"]) == 3


def test_example_messages_well_formed(two_db_setup):
    records, db_root = two_db_setup
    examples, _ = build_clean_examples(records, db_root)
    msgs = examples[0]["messages"]
    assert msgs[0]["role"] == "system"
    system_obj = json.loads(msgs[0]["content"])
    assert system_obj["db_id"] == "concert_singer"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    assert "singer" in msgs[2]["content"].lower()


def test_db_root_path_resolution(two_db_setup):
    """DB is found at db_root/<db_id>/<db_id>.sqlite (Spider layout)."""
    records, db_root = two_db_setup
    examples, stats = build_clean_examples(records, db_root)
    # Only the resolvable DB produces an example
    assert stats["accepted"] == 1
