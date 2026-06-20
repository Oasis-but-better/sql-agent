"""
test_schema_cache_reserved.py — regression for PRAGMA reserved-keyword table name bug.

Creates an in-memory-backed temp .sqlite with a table named `order` (SQLite reserved word),
then asserts compile_schema_cache succeeds and returns the expected columns.
"""

import sqlite3
import tempfile
import os
import pytest

from src.schema_cache import compile_schema_cache


def _make_reserved_word_db() -> str:
    """Create a temp .sqlite with a table named `order` (reserved keyword)."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute(
        'CREATE TABLE "order" (id INTEGER PRIMARY KEY, customer TEXT, amount REAL)'
    )
    conn.execute('INSERT INTO "order" VALUES (1, "Alice", 99.99)')
    conn.commit()
    conn.close()
    return path


def test_reserved_keyword_table_compiles():
    """compile_schema_cache must not crash on a reserved-keyword table name."""
    path = _make_reserved_word_db()
    try:
        result = compile_schema_cache(path)
    finally:
        os.unlink(path)

    assert result["db_id"] is not None
    tables = result["tables"]
    assert len(tables) == 1
    assert tables[0]["name"] == "order"
    col_names = [c["name"] for c in tables[0]["columns"]]
    assert "id" in col_names
    assert "customer" in col_names
    assert "amount" in col_names


def test_reserved_keyword_sample_rows():
    """Sample rows must be returned without error for reserved-keyword table."""
    path = _make_reserved_word_db()
    try:
        result = compile_schema_cache(path)
    finally:
        os.unlink(path)

    table = result["tables"][0]
    assert len(table["sample_rows"]) == 1
    assert table["sample_rows"][0][0] == 1  # id=1
