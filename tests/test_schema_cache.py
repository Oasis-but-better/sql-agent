import json
import sqlite3
import tempfile
from pathlib import Path
import pytest
from src.schema_cache import compile_schema_cache


@pytest.fixture
def tmp_db(tmp_path):
    """Minimal two-table sqlite with a FK, matching PRAGMA returns."""
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE singer (
            singer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            birth_year INTEGER
        );
        CREATE TABLE concert (
            concert_id INTEGER PRIMARY KEY,
            theme TEXT,
            singer_id INTEGER,
            FOREIGN KEY (singer_id) REFERENCES singer(singer_id)
        );
        INSERT INTO singer VALUES (1, 'Alice', 1990), (2, 'Bob', 1985), (3, 'Carol', 1978);
        INSERT INTO concert VALUES (101, 'Pop Night', 1), (102, 'Jazz', 2);
    """)
    conn.commit()
    conn.close()
    return str(db_path)


def test_compile_returns_dict_with_required_keys(tmp_db):
    cache = compile_schema_cache(tmp_db)
    assert isinstance(cache, dict)
    assert "db_id" in cache
    assert "tables" in cache


def test_tables_have_columns_and_types(tmp_db):
    cache = compile_schema_cache(tmp_db)
    tables = {t["name"]: t for t in cache["tables"]}
    assert "singer" in tables
    assert "concert" in tables
    singer_cols = {c["name"]: c["type"] for c in tables["singer"]["columns"]}
    assert singer_cols["name"] == "TEXT"
    assert singer_cols["birth_year"] == "INTEGER"


def test_fk_edges_captured(tmp_db):
    cache = compile_schema_cache(tmp_db)
    tables = {t["name"]: t for t in cache["tables"]}
    fks = tables["concert"]["foreign_keys"]
    assert len(fks) == 1
    assert fks[0]["from_col"] == "singer_id"
    assert fks[0]["to_table"] == "singer"
    assert fks[0]["to_col"] == "singer_id"


def test_sample_rows_max_three(tmp_db):
    cache = compile_schema_cache(tmp_db)
    tables = {t["name"]: t for t in cache["tables"]}
    assert len(tables["singer"]["sample_rows"]) == 3
    assert len(tables["concert"]["sample_rows"]) == 2  # only 2 rows inserted


def test_tables_with_no_fk_have_empty_list(tmp_db):
    cache = compile_schema_cache(tmp_db)
    tables = {t["name"]: t for t in cache["tables"]}
    assert tables["singer"]["foreign_keys"] == []


def test_db_id_derived_from_filename(tmp_db):
    cache = compile_schema_cache(tmp_db)
    assert cache["db_id"] == "test"


def test_compile_real_spider_db():
    """Integration: compile against the real department_management.sqlite."""
    db_path = (
        "/Users/hiten/IU Coursework/AI Projects/sql-agent/data/spider/spider_data"
        "/database/department_management/department_management.sqlite"
    )
    cache = compile_schema_cache(db_path)
    assert cache["db_id"] == "department_management"
    table_names = [t["name"] for t in cache["tables"]]
    assert "head" in table_names
    head = next(t for t in cache["tables"] if t["name"] == "head")
    col_names = [c["name"] for c in head["columns"]]
    assert "name" in col_names
    assert "age" in col_names
    assert isinstance(cache["tables"], list)
    assert len(cache["tables"]) == 3  # department, head, management


def test_to_json_is_serializable(tmp_db):
    cache = compile_schema_cache(tmp_db)
    serialized = json.dumps(cache)
    assert isinstance(serialized, str)
    assert len(serialized) > 10
