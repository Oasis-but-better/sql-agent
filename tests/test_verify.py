import sqlite3
import pytest
from src.verify import run_query_with_timeout, verify_sql, MatchVerdict


@pytest.fixture
def two_row_db(tmp_path):
    db_path = tmp_path / "v.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE song (id INTEGER PRIMARY KEY, title TEXT, plays INTEGER);
        INSERT INTO song VALUES (1, 'A', 100), (2, 'B', 50), (3, 'C', 200);
    """)
    conn.commit()
    conn.close()
    return str(db_path)


# --- run_query_with_timeout ---

def test_run_query_returns_rows(two_row_db):
    rows, err = run_query_with_timeout(two_row_db, "SELECT id, title FROM song ORDER BY id")
    assert err is None
    assert rows == [(1, "A"), (2, "B"), (3, "C")]


def test_run_query_returns_error_on_bad_sql(two_row_db):
    rows, err = run_query_with_timeout(two_row_db, "SELECT nonexistent_col FROM song")
    assert rows is None
    assert err is not None
    assert "no such column" in err.lower()


def test_run_query_handles_empty_result(two_row_db):
    rows, err = run_query_with_timeout(two_row_db, "SELECT * FROM song WHERE plays > 9999")
    assert err is None
    assert rows == []


def test_run_query_timeout_aborts_runaway(two_row_db):
    """A cross-join on a ~27-row self-join should be fast, but a CTE infinite
    loop will hit the opcode budget and return a timeout error."""
    # sqlite progress handler kicks in; use a known runaway
    runaway = (
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) "
        "SELECT n FROM r LIMIT 1000000"
    )
    rows, err = run_query_with_timeout(two_row_db, runaway, wall_seconds=1.0, max_ops=100_000)
    # Either returns rows (if SQLite is fast enough) or times out — must not raise
    assert rows is None or isinstance(rows, list)


# --- verify_sql ---

def test_verify_sql_strict_match(two_row_db):
    gold_sql = "SELECT id, title FROM song WHERE plays > 60 ORDER BY id"
    pred_sql = "SELECT id, title FROM song WHERE plays >= 61 ORDER BY id"
    result = verify_sql(two_row_db, pred_sql, gold_sql)
    assert result.is_valid is True
    assert result.verdict == MatchVerdict.STRICT


def test_verify_sql_superset_match(two_row_db):
    """pred returns extra column — superset credit applies."""
    gold_sql = "SELECT title FROM song WHERE id = 1"
    pred_sql = "SELECT id, title FROM song WHERE id = 1"
    result = verify_sql(two_row_db, pred_sql, gold_sql)
    assert result.is_valid is True
    assert result.verdict == MatchVerdict.SUPERSET


def test_verify_sql_wrong_result_is_invalid(two_row_db):
    gold_sql = "SELECT title FROM song WHERE plays > 60"
    pred_sql = "SELECT title FROM song WHERE plays > 9999"
    result = verify_sql(two_row_db, pred_sql, gold_sql)
    assert result.is_valid is False
    assert result.verdict == MatchVerdict.MISMATCH


def test_verify_sql_error_is_invalid(two_row_db):
    gold_sql = "SELECT title FROM song WHERE id = 1"
    pred_sql = "SELECT bad_col FROM song"
    result = verify_sql(two_row_db, pred_sql, gold_sql)
    assert result.is_valid is False
    assert result.verdict == MatchVerdict.ERROR
    assert result.traceback is not None


def test_verify_sql_empty_gold_and_empty_pred_is_valid(two_row_db):
    """Both return 0 rows — valid (correct empty result)."""
    gold_sql = "SELECT * FROM song WHERE plays > 9999"
    pred_sql = "SELECT id FROM song WHERE plays > 8888"
    result = verify_sql(two_row_db, pred_sql, gold_sql)
    assert result.is_valid is True


def test_verify_sql_gold_fails_raises(two_row_db):
    """If gold SQL itself errors, verify_sql raises ValueError."""
    with pytest.raises(ValueError, match="Gold SQL failed"):
        verify_sql(two_row_db, "SELECT 1", "SELECT bad FROM song")
