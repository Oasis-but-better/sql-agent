import json
import sqlite3
import pytest
from unittest.mock import MagicMock
from src.generate_examples import (
    ModelCallable,
    generate_correction_example,
    CorrectionResult,
)
from src.gold_ingest import GoldRecord
from src.schema_cache import compile_schema_cache


@pytest.fixture
def singer_db(tmp_path):
    db_path = tmp_path / "concert_singer.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        INSERT INTO singer VALUES (1, 'Alice', 30), (2, 'Bob', 22);
    """)
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture
def gold_record():
    return GoldRecord(
        db_id="concert_singer",
        question="How many singers are older than 25?",
        gold_sql="SELECT count(*) FROM singer WHERE age > 25",
        source="spider",
        difficulty=None,
    )


def test_generate_correction_example_wrong_then_fixed(singer_db, gold_record):
    """Mock produces wrong SQL first, then correct SQL on retry."""
    call_count = [0]

    def mock_model(messages: list[dict]) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            return "SELECT name FROM singer WHERE age > 25"  # wrong — returns names, not count
        return gold_record.gold_sql  # correct on retry

    cache = compile_schema_cache(singer_db)
    result = generate_correction_example(
        record=gold_record,
        db_path=singer_db,
        schema_cache=cache,
        model_fn=mock_model,
        max_attempts=3,
    )

    assert result is not None
    assert isinstance(result, CorrectionResult)
    assert result.accepted is True
    assert len(result.messages) == 5  # system + user + wrong_sql + tool(traceback) + fixed_sql


def test_generate_correction_skipped_when_first_attempt_correct(singer_db, gold_record):
    """If base model gets it right first try, no correction chain — skip."""
    def mock_model(messages: list[dict]) -> str:
        return gold_record.gold_sql  # immediately correct

    cache = compile_schema_cache(singer_db)
    result = generate_correction_example(
        record=gold_record,
        db_path=singer_db,
        schema_cache=cache,
        model_fn=mock_model,
        max_attempts=3,
    )
    assert result is None  # nothing to correct


def test_generate_correction_rejected_when_fix_never_works(singer_db, gold_record):
    """Model keeps returning wrong SQL — correction rejected after max_attempts."""
    def mock_model(messages: list[dict]) -> str:
        return "SELECT bad_col FROM singer"  # always wrong

    cache = compile_schema_cache(singer_db)
    result = generate_correction_example(
        record=gold_record,
        db_path=singer_db,
        schema_cache=cache,
        model_fn=mock_model,
        max_attempts=2,
    )
    assert result is not None
    assert result.accepted is False


def test_correction_messages_have_tool_role_for_traceback(singer_db, gold_record):
    """Turn-2 must inject real sqlite3 traceback via a 'tool' role message."""
    call_count = [0]

    def mock_model(messages: list[dict]) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            return "SELECT nonexistent FROM singer"  # real sqlite3 error
        return gold_record.gold_sql

    cache = compile_schema_cache(singer_db)
    result = generate_correction_example(
        record=gold_record,
        db_path=singer_db,
        schema_cache=cache,
        model_fn=mock_model,
        max_attempts=3,
    )
    assert result is not None
    assert result.accepted is True
    roles = [m["role"] for m in result.messages]
    assert "tool" in roles
    tool_msg = next(m for m in result.messages if m["role"] == "tool")
    # Must be a real sqlite3 error, not invented
    assert "no such column" in tool_msg["content"].lower()


def test_correction_result_to_example_dict(singer_db, gold_record):
    call_count = [0]

    def mock_model(messages: list[dict]) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            return "SELECT name FROM singer WHERE age > 25"
        return gold_record.gold_sql

    cache = compile_schema_cache(singer_db)
    result = generate_correction_example(
        record=gold_record,
        db_path=singer_db,
        schema_cache=cache,
        model_fn=mock_model,
        max_attempts=3,
    )
    example = result.to_example_dict()
    assert example["type"] == "correction"
    assert example["db_id"] == "concert_singer"
    assert example["source"] == "spider"
    assert len(example["messages"]) == 5
    serialized = json.dumps(example)
    assert "correction" in serialized
