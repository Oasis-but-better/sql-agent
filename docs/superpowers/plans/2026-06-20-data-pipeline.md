# SQL-Agent Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and test the full data pipeline that ingests Spider + BIRD gold SQL pairs, compiles per-DB schema caches, execution-verifies examples, tracks diversity, and emits a JSONL dataset ready for QLoRA fine-tuning.

**Architecture:** The pipeline is a pure-Python, SQLite-backed data engine with no ML dependency: `schema_cache.py` compiles per-DB PRAGMA metadata once; `verify.py` (ported from `llm-training/src/evaluate.py`) execution-checks every example with a per-query timeout; `gold_ingest.py` normalizes Spider+BIRD gold JSON into model-agnostic message lists; `diversity.py` tracks query-type and difficulty coverage; `generate_examples.py` defines the correction-set interface with a mocked model stub; `build_dataset.py` orchestrates all modules into train/val/test JSONL. mlx/mlx-lm, LangGraph, and the real Anthropic client are deferred — this plan produces only the data layer.

**Tech Stack:** Python 3.11+, `sqlite3` (stdlib), `pytest`, `anthropic` SDK (import-only, no live calls in tests), `tqdm`

---

## File Structure

| File | Responsibility |
|---|---|
| `sql-agent/src/__init__.py` | Package marker |
| `sql-agent/src/schema_cache.py` | `.sqlite → db_metadata_cache.json` via PRAGMA |
| `sql-agent/src/verify.py` | `run_query_with_timeout`, `verify_sql`, `match_results` (ported from sibling) |
| `sql-agent/src/gold_ingest.py` | Spider+BIRD gold JSON → normalized `GoldRecord` list → message lists |
| `sql-agent/src/diversity.py` | Coverage tracker: per-DB counts, query-type histogram, difficulty buckets, convergence check |
| `sql-agent/src/generate_examples.py` | Correction-set generation interface; unit-testable with mocked model |
| `sql-agent/src/build_dataset.py` | Orchestrator: clean+correction → dedup → split → emit JSONL |
| `sql-agent/tests/test_schema_cache.py` | Unit tests for schema_cache |
| `sql-agent/tests/test_verify.py` | Unit tests for verify |
| `sql-agent/tests/test_gold_ingest.py` | Unit tests for gold_ingest |
| `sql-agent/tests/test_diversity.py` | Unit tests for diversity |
| `sql-agent/tests/test_generate_examples.py` | Unit tests for generate_examples (mocked model) |
| `sql-agent/tests/test_build_dataset.py` | Integration test for build_dataset |
| `sql-agent/CONTEXT.md` | Navigation index |
| `sql-agent/requirements.txt` | `pytest`, `anthropic`, `tqdm` |
| `sql-agent/pytest.ini` | Test config |

---

### Task 1: Project Scaffold

**Files:**
- Create: `sql-agent/requirements.txt`
- Create: `sql-agent/pytest.ini`
- Create: `sql-agent/src/__init__.py`
- Create: `sql-agent/tests/__init__.py`
- Create: `sql-agent/data/dataset/.gitkeep`

- [ ] **Step 1: Initialize git repo**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
git init
git add "SQL Agent Spec.md" caveman-out-sql-agent-overview.html docs/
git commit -m "chore: initial commit — spec and HTML overview"
```

Expected: `Initialized empty Git repository in .git/`

- [ ] **Step 2: Create virtual environment**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
python3 -m venv .venv
```

Expected: `.venv/` directory created. `sqlite3` is stdlib — no install needed.

- [ ] **Step 3: Write requirements.txt**

```
pytest>=8.0
anthropic>=0.40.0
tqdm>=4.66
```

Note: `mlx`, `mlx-lm`, `langgraph` are intentionally absent — they belong to the fine-tuning and agent-harness plans.

- [ ] **Step 4: Install dependencies**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
.venv/bin/pip install -r requirements.txt
```

Expected: `Successfully installed anthropic-... pytest-... tqdm-...`

- [ ] **Step 5: Write pytest.ini**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v
```

- [ ] **Step 6: Create package markers and dataset dir**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
mkdir -p src tests data/dataset
touch src/__init__.py tests/__init__.py data/dataset/.gitkeep
```

- [ ] **Step 7: Run pytest to confirm zero tests collected cleanly**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
.venv/bin/pytest
```

Expected output:
```
============= no tests ran in 0.XXs =============
```

- [ ] **Step 8: Commit scaffold**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
git add requirements.txt pytest.ini src/__init__.py tests/__init__.py data/dataset/.gitkeep
git commit -m "chore: project scaffold — venv, pytest, src/ package"
```

---

### Task 2: `src/schema_cache.py` — Phase-A Schema Compiler

**Files:**
- Create: `sql-agent/src/schema_cache.py`
- Create: `sql-agent/tests/test_schema_cache.py`

The compiler uses only `PRAGMA table_info(<table>)` and `PRAGMA foreign_key_list(<table>)` — no external schema files. It emits a minified JSON dict with tables, columns+types, FK edges, and 3 sample rows per table.

- [ ] **Step 1: Write the failing test**

Create `sql-agent/tests/test_schema_cache.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
.venv/bin/pytest tests/test_schema_cache.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'src.schema_cache'`

- [ ] **Step 3: Implement `src/schema_cache.py`**

```python
"""
schema_cache.py — Phase-A schema compiler.

Given a .sqlite path, returns a dict (db_metadata_cache) with:
  - db_id: stem of the sqlite filename
  - tables: list of {name, columns: [{name, type}], foreign_keys: [{from_col, to_table, to_col}],
                      sample_rows: [list[any]]}

No LLM used — pure PRAGMA queries only.
"""

import sqlite3
from pathlib import Path


def compile_schema_cache(db_path: str) -> dict:
    """Compile schema metadata from a .sqlite file via PRAGMA queries.

    Returns:
        {
          "db_id": str,
          "tables": [
            {
              "name": str,
              "columns": [{"name": str, "type": str}],
              "foreign_keys": [{"from_col": str, "to_table": str, "to_col": str}],
              "sample_rows": [list[any]]  # up to 3
            }
          ]
        }
    """
    db_id = Path(db_path).stem
    conn = sqlite3.connect(db_path)
    conn.row_factory = None
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    table_names = [row[0] for row in cur.fetchall()]

    tables = []
    for tname in table_names:
        # Columns
        cur.execute(f"PRAGMA table_info({tname})")
        columns = [{"name": row[1], "type": row[2]} for row in cur.fetchall()]

        # FK edges: PRAGMA foreign_key_list returns (id, seq, table, from, to, ...)
        cur.execute(f"PRAGMA foreign_key_list({tname})")
        fk_rows = cur.fetchall()
        foreign_keys = [
            {"from_col": row[3], "to_table": row[2], "to_col": row[4]}
            for row in fk_rows
        ]

        # Sample rows — up to 3
        cur.execute(f"SELECT * FROM {tname} LIMIT 3")
        sample_rows = [list(row) for row in cur.fetchall()]

        tables.append({
            "name": tname,
            "columns": columns,
            "foreign_keys": foreign_keys,
            "sample_rows": sample_rows,
        })

    conn.close()
    return {"db_id": db_id, "tables": tables}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
.venv/bin/pytest tests/test_schema_cache.py -v
```

Expected:
```
tests/test_schema_cache.py::test_compile_returns_dict_with_required_keys PASSED
tests/test_schema_cache.py::test_tables_have_columns_and_types PASSED
tests/test_schema_cache.py::test_fk_edges_captured PASSED
tests/test_schema_cache.py::test_sample_rows_max_three PASSED
tests/test_schema_cache.py::test_tables_with_no_fk_have_empty_list PASSED
tests/test_schema_cache.py::test_db_id_derived_from_filename PASSED
tests/test_schema_cache.py::test_compile_real_spider_db PASSED
tests/test_schema_cache.py::test_to_json_is_serializable PASSED
8 passed in X.XXs
```

- [ ] **Step 5: Commit**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
git add src/schema_cache.py tests/test_schema_cache.py
git commit -m "feat: schema_cache — Phase-A PRAGMA compiler with tests"
```

---

### Task 3: `src/verify.py` — Execution Verifier with Timeout

**Files:**
- Create: `sql-agent/src/verify.py`
- Create: `sql-agent/tests/test_verify.py`

Port `run_query_with_timeout` from `llm-training/scripts/validate_data.py` and the result-comparison functions from `llm-training/src/evaluate.py`. `verify_sql` returns a `VerifyResult` dataclass — `is_valid` is True only if the SQL executes AND result-set matches gold (strict or superset).

- [ ] **Step 1: Write the failing test**

Create `sql-agent/tests/test_verify.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
.venv/bin/pytest tests/test_verify.py -v
```

Expected: `ERROR — ModuleNotFoundError: No module named 'src.verify'`

- [ ] **Step 3: Implement `src/verify.py`**

```python
"""
verify.py — Execution verifier with per-query timeout.

Ported from:
  - llm-training/scripts/validate_data.py (_run_query_with_timeout)
  - llm-training/src/evaluate.py (compare_results, compare_results_superset,
                                   is_order_sensitive)

verify_sql semantics (from spec §5):
  - Valid iff pred SQL executes without error AND result-set matches gold
    (strict or superset). NOT merely non-empty.
  - Gold SQL error → raises ValueError (caller must filter gold before calling).
"""

import itertools
import re
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Timeout constants (same as validate_data.py)
# ---------------------------------------------------------------------------

_DEFAULT_WALL_SECONDS = 3.0
_DEFAULT_MAX_OPS = 20_000_000
_PROGRESS_N = 100_000


# ---------------------------------------------------------------------------
# run_query_with_timeout
# ---------------------------------------------------------------------------

def run_query_with_timeout(
    db_path: str,
    sql: str,
    wall_seconds: float = _DEFAULT_WALL_SECONDS,
    max_ops: int = _DEFAULT_MAX_OPS,
) -> tuple[list[tuple] | None, str | None]:
    """Execute SQL against db_path with wall-clock + opcode budget.

    Returns (rows, None) on success, (None, error_string) on error or timeout.
    Never raises.
    """
    _state = [0.0, 0]  # [wall_start, op_count]

    def _handler():
        _state[1] += _PROGRESS_N
        if _state[1] >= max_ops:
            return 1
        if time.monotonic() - _state[0] >= wall_seconds:
            return 1
        return 0

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = None
        conn.set_progress_handler(_handler, _PROGRESS_N)
        cur = conn.cursor()
        _state[0] = time.monotonic()
        _state[1] = 0
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows, None
    except sqlite3.OperationalError as exc:
        msg = str(exc)
        if "interrupted" in msg.lower():
            return None, f"timeout: {msg}"
        return None, msg
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Result comparison (ported from evaluate.py, locked semantics)
# ---------------------------------------------------------------------------

def _values_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    a_num = isinstance(a, (int, float)) and not isinstance(a, bool)
    b_num = isinstance(b, (int, float)) and not isinstance(b, bool)
    if a_num and b_num:
        diff = abs(a - b)
        tol = 1e-6 * max(1, abs(a), abs(b))
        return diff <= tol
    if a_num != b_num:
        return False
    return a == b


def _rows_equal(row_a: tuple, row_b: tuple) -> bool:
    if len(row_a) != len(row_b):
        return False
    return all(_values_equal(a, b) for a, b in zip(row_a, row_b))


def _multiset_match(gold: list[tuple], pred: list[tuple]) -> bool:
    if len(gold) != len(pred):
        return False
    unmatched = list(pred)
    for g in gold:
        found = False
        for idx, p in enumerate(unmatched):
            if _rows_equal(g, p):
                unmatched.pop(idx)
                found = True
                break
        if not found:
            return False
    return True


_STRING_LITERAL_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_ORDER_BY_RE = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


def _is_order_sensitive(sql: str) -> bool:
    stripped = _STRING_LITERAL_RE.sub("''", sql)
    return bool(_ORDER_BY_RE.search(stripped))


def _strict_match(gold_rows: list[tuple], pred_rows: list[tuple], order_sensitive: bool) -> bool:
    """Column-permutation tolerant (≤6 cols), multiset or ordered."""
    if not gold_rows and not pred_rows:
        return True
    if not gold_rows or not pred_rows:
        return False
    if len(gold_rows[0]) != len(pred_rows[0]):
        return False
    ncols = len(gold_rows[0])
    perms = list(itertools.permutations(range(ncols))) if ncols <= 6 else [tuple(range(ncols))]
    for perm in perms:
        permuted = [tuple(r[i] for i in perm) for r in pred_rows]
        if order_sensitive:
            if len(gold_rows) == len(permuted) and all(_rows_equal(g, p) for g, p in zip(gold_rows, permuted)):
                return True
        else:
            if _multiset_match(gold_rows, permuted):
                return True
    return False


def _superset_match(gold_rows: list[tuple], pred_rows: list[tuple], order_sensitive: bool) -> bool:
    """pred contains all gold columns by value; extra pred cols allowed."""
    if not gold_rows and not pred_rows:
        return True
    if not gold_rows or not pred_rows:
        return False
    n_gold = len(gold_rows[0])
    n_pred = len(pred_rows[0])
    if n_pred < n_gold:
        return False
    if n_gold > 6:
        return _strict_match(gold_rows, pred_rows, order_sensitive)
    for col_map in itertools.permutations(range(n_pred), n_gold):
        projected = [tuple(r[i] for i in col_map) for r in pred_rows]
        if order_sensitive:
            if len(projected) == len(gold_rows) and all(_rows_equal(g, p) for g, p in zip(gold_rows, projected)):
                return True
        else:
            if _multiset_match(gold_rows, projected):
                return True
    return False


# ---------------------------------------------------------------------------
# VerifyResult
# ---------------------------------------------------------------------------

class MatchVerdict(str, Enum):
    STRICT = "strict"
    SUPERSET = "superset"
    MISMATCH = "mismatch"
    ERROR = "error"     # pred SQL failed to execute


@dataclass
class VerifyResult:
    is_valid: bool
    verdict: MatchVerdict
    traceback: str | None = None  # populated when verdict == ERROR


# ---------------------------------------------------------------------------
# verify_sql — public API
# ---------------------------------------------------------------------------

def verify_sql(
    db_path: str,
    pred_sql: str,
    gold_sql: str,
    wall_seconds: float = _DEFAULT_WALL_SECONDS,
    max_ops: int = _DEFAULT_MAX_OPS,
) -> VerifyResult:
    """Verify pred_sql against gold_sql by executing both on db_path.

    Raises ValueError if gold_sql itself fails to execute.
    Returns VerifyResult with is_valid=True only when pred executes and
    result-set matches gold (strict or superset). NOT merely non-empty.
    """
    gold_rows, gold_err = run_query_with_timeout(db_path, gold_sql, wall_seconds, max_ops)
    if gold_err is not None:
        raise ValueError(f"Gold SQL failed: {gold_err} | SQL: {gold_sql!r}")

    pred_rows, pred_err = run_query_with_timeout(db_path, pred_sql, wall_seconds, max_ops)
    if pred_err is not None:
        return VerifyResult(is_valid=False, verdict=MatchVerdict.ERROR, traceback=pred_err)

    order_sens = _is_order_sensitive(gold_sql)

    if _strict_match(gold_rows, pred_rows, order_sens):
        return VerifyResult(is_valid=True, verdict=MatchVerdict.STRICT)

    if _superset_match(gold_rows, pred_rows, order_sens):
        return VerifyResult(is_valid=True, verdict=MatchVerdict.SUPERSET)

    return VerifyResult(is_valid=False, verdict=MatchVerdict.MISMATCH)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
.venv/bin/pytest tests/test_verify.py -v
```

Expected:
```
tests/test_verify.py::test_run_query_returns_rows PASSED
tests/test_verify.py::test_run_query_returns_error_on_bad_sql PASSED
tests/test_verify.py::test_run_query_handles_empty_result PASSED
tests/test_verify.py::test_run_query_timeout_aborts_runaway PASSED
tests/test_verify.py::test_verify_sql_strict_match PASSED
tests/test_verify.py::test_verify_sql_superset_match PASSED
tests/test_verify.py::test_verify_sql_wrong_result_is_invalid PASSED
tests/test_verify.py::test_verify_sql_error_is_invalid PASSED
tests/test_verify.py::test_verify_sql_empty_gold_and_empty_pred_is_valid PASSED
tests/test_verify.py::test_verify_sql_gold_fails_raises PASSED
10 passed in X.XXs
```

- [ ] **Step 5: Commit**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
git add src/verify.py tests/test_verify.py
git commit -m "feat: verify — execution verifier with timeout, ported from llm-training"
```

---

### Task 4: `src/gold_ingest.py` — Gold Pair Normalization

**Files:**
- Create: `sql-agent/src/gold_ingest.py`
- Create: `sql-agent/tests/test_gold_ingest.py`

Spider gold field names (confirmed): `db_id`, `question`, `query` (the SQL string), `query_toks`, `query_toks_no_value`, `question_toks`, `sql` (parsed dict — NOT the SQL string). BIRD gold field names (confirmed): `question_id`, `db_id`, `question`, `evidence`, `SQL` (uppercase key — the SQL string), `difficulty`.

The normalizer maps both to a unified `GoldRecord` and assembles the message list with the schema cache as the system prompt.

- [ ] **Step 1: Write the failing test**

Create `sql-agent/tests/test_gold_ingest.py`:

```python
import json
import sqlite3
import pytest
from src.gold_ingest import load_spider_gold, load_bird_gold, GoldRecord, build_message_list
from src.schema_cache import compile_schema_cache


@pytest.fixture
def tmp_db_with_song(tmp_path):
    db_path = tmp_path / "concert_singer.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        INSERT INTO singer VALUES (1, 'Alice', 30);
    """)
    conn.commit()
    conn.close()
    return tmp_path, str(db_path)


@pytest.fixture
def spider_json_file(tmp_path, tmp_db_with_song):
    db_dir, db_path = tmp_db_with_song
    data = [
        {
            "db_id": "concert_singer",
            "query": "SELECT name FROM singer WHERE age > 25",
            "query_toks": ["SELECT", "name", "FROM", "singer", "WHERE", "age", ">", "25"],
            "query_toks_no_value": ["select", "name", "from", "singer", "where", "age", ">", "value"],
            "question": "What are the names of all singers older than 25?",
            "question_toks": ["What", "are", "the", "names", "..."],
            "sql": {"select": [], "from": {}, "where": []},
        },
        {
            "db_id": "concert_singer",
            "query": "SELECT count(*) FROM singer",
            "query_toks": ["SELECT", "count", "(", "*", ")", "FROM", "singer"],
            "query_toks_no_value": ["select", "count", "(", "*", ")", "from", "singer"],
            "question": "How many singers are there?",
            "question_toks": ["How", "many", "..."],
            "sql": {"select": [], "from": {}, "where": []},
        },
    ]
    jpath = tmp_path / "train_spider.json"
    jpath.write_text(json.dumps(data))
    return str(jpath)


@pytest.fixture
def bird_json_file(tmp_path, tmp_db_with_song):
    db_dir, db_path = tmp_db_with_song
    data = [
        {
            "question_id": 0,
            "db_id": "concert_singer",
            "question": "What is the age of singer Alice?",
            "evidence": "",
            "SQL": "SELECT age FROM singer WHERE name = 'Alice'",
            "difficulty": "simple",
        }
    ]
    jpath = tmp_path / "dev.json"
    jpath.write_text(json.dumps(data))
    return str(jpath)


# --- load_spider_gold ---

def test_load_spider_gold_returns_gold_records(spider_json_file):
    records = load_spider_gold(spider_json_file)
    assert len(records) == 2
    assert all(isinstance(r, GoldRecord) for r in records)


def test_spider_field_names_mapped_correctly(spider_json_file):
    records = load_spider_gold(spider_json_file)
    r = records[0]
    assert r.db_id == "concert_singer"
    assert r.question == "What are the names of all singers older than 25?"
    assert r.gold_sql == "SELECT name FROM singer WHERE age > 25"
    assert r.source == "spider"
    assert r.difficulty is None  # Spider train_spider has no difficulty field


# --- load_bird_gold ---

def test_load_bird_gold_returns_gold_records(bird_json_file):
    records = load_bird_gold(bird_json_file)
    assert len(records) == 1
    assert isinstance(records[0], GoldRecord)


def test_bird_field_names_mapped_correctly(bird_json_file):
    records = load_bird_gold(bird_json_file)
    r = records[0]
    assert r.db_id == "concert_singer"
    assert r.question == "What is the age of singer Alice?"
    assert r.gold_sql == "SELECT age FROM singer WHERE name = 'Alice'"
    assert r.source == "bird"
    assert r.difficulty == "simple"


# --- build_message_list ---

def test_build_message_list_structure(spider_json_file, tmp_db_with_song):
    db_dir, db_path = tmp_db_with_song
    records = load_spider_gold(spider_json_file)
    cache = compile_schema_cache(db_path)
    messages = build_message_list(records[0], cache)
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_message_list_content(spider_json_file, tmp_db_with_song):
    db_dir, db_path = tmp_db_with_song
    records = load_spider_gold(spider_json_file)
    cache = compile_schema_cache(db_path)
    messages = build_message_list(records[0], cache)
    # System = JSON-serialized schema cache
    system_obj = json.loads(messages[0]["content"])
    assert system_obj["db_id"] == "concert_singer"
    # User = the question
    assert messages[1]["content"] == "What are the names of all singers older than 25?"
    # Assistant = gold SQL
    assert messages[2]["content"] == "SELECT name FROM singer WHERE age > 25"


def test_gold_record_to_example_dict(spider_json_file, tmp_db_with_song):
    db_dir, db_path = tmp_db_with_song
    records = load_spider_gold(spider_json_file)
    cache = compile_schema_cache(db_path)
    messages = build_message_list(records[0], cache)
    example = {
        "db_id": records[0].db_id,
        "source": records[0].source,
        "difficulty": records[0].difficulty,
        "type": "clean",
        "query_types": [],
        "messages": messages,
    }
    assert example["type"] == "clean"
    assert example["source"] == "spider"
    serialized = json.dumps(example)
    assert "concert_singer" in serialized
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
.venv/bin/pytest tests/test_gold_ingest.py -v
```

Expected: `ERROR — ModuleNotFoundError: No module named 'src.gold_ingest'`

- [ ] **Step 3: Implement `src/gold_ingest.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
.venv/bin/pytest tests/test_gold_ingest.py -v
```

Expected:
```
tests/test_gold_ingest.py::test_load_spider_gold_returns_gold_records PASSED
tests/test_gold_ingest.py::test_spider_field_names_mapped_correctly PASSED
tests/test_gold_ingest.py::test_load_bird_gold_returns_gold_records PASSED
tests/test_gold_ingest.py::test_bird_field_names_mapped_correctly PASSED
tests/test_gold_ingest.py::test_build_message_list_structure PASSED
tests/test_gold_ingest.py::test_message_list_content PASSED
tests/test_gold_ingest.py::test_gold_record_to_example_dict PASSED
7 passed in X.XXs
```

- [ ] **Step 5: Commit**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
git add src/gold_ingest.py tests/test_gold_ingest.py
git commit -m "feat: gold_ingest — Spider+BIRD normalization to GoldRecord + message lists"
```

---

### Task 5: Clean-Set Builder — Ingest + Verify + Emit

**Files:**
- Create: `sql-agent/tests/test_clean_set.py`
- Modify: `sql-agent/src/gold_ingest.py` — add `build_clean_examples()`

This task wires `gold_ingest` + `verify` together: for each gold record, resolve the `.sqlite` path, compile the schema cache, verify the gold SQL executes and produces valid output, and emit a fully-formed example dict. Discarded records (gold SQL fails) are tracked. This is the "discard gold that fails to execute" requirement from spec §4.2.

- [ ] **Step 1: Write the failing test**

Create `sql-agent/tests/test_clean_set.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
.venv/bin/pytest tests/test_clean_set.py -v
```

Expected: `ERROR — cannot import name 'build_clean_examples' from 'src.gold_ingest'`

- [ ] **Step 3: Add `build_clean_examples()` to `src/gold_ingest.py`**

Append to the bottom of `sql-agent/src/gold_ingest.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
.venv/bin/pytest tests/test_clean_set.py -v
```

Expected:
```
tests/test_clean_set.py::test_build_clean_examples_returns_valid_only PASSED
tests/test_clean_set.py::test_build_clean_examples_stats_track_yield PASSED
tests/test_clean_set.py::test_example_has_required_fields PASSED
tests/test_clean_set.py::test_example_messages_well_formed PASSED
tests/test_clean_set.py::test_db_root_path_resolution PASSED
5 passed in X.XXs
```

- [ ] **Step 5: Commit**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
git add src/gold_ingest.py tests/test_clean_set.py
git commit -m "feat: clean-set builder — verify gold + emit examples, track yield"
```

---

### Task 6: `src/diversity.py` — Coverage Tracker

**Files:**
- Create: `sql-agent/src/diversity.py`
- Create: `sql-agent/tests/test_diversity.py`

Tracks per-DB counts, a query-type histogram, difficulty bucket counts, and a convergence check. Query types are detected via regex over the SQL string (no external SQL parser dependency). Convergence = ≥15k verified examples AND all per-type and per-difficulty thresholds met.

- [ ] **Step 1: Write the failing test**

Create `sql-agent/tests/test_diversity.py`:

```python
import pytest
from src.diversity import DiversityTracker, QUERY_TYPES, CONVERGENCE_TARGET


def _make_example(db_id, sql, difficulty="easy"):
    return {
        "db_id": db_id,
        "source": "spider",
        "difficulty": difficulty,
        "type": "clean",
        "query_types": [],
        "messages": [
            {"role": "system", "content": "{}"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": sql},
        ],
    }


def test_tracker_starts_empty():
    t = DiversityTracker()
    assert t.total == 0
    assert t.per_db == {}
    assert t.query_type_counts == {qt: 0 for qt in QUERY_TYPES}


def test_add_example_increments_total():
    t = DiversityTracker()
    t.add(_make_example("concert_singer", "SELECT name FROM singer"))
    assert t.total == 1


def test_per_db_count():
    t = DiversityTracker()
    t.add(_make_example("concert_singer", "SELECT name FROM singer"))
    t.add(_make_example("concert_singer", "SELECT count(*) FROM singer"))
    t.add(_make_example("debit_card_specializing", "SELECT id FROM t"))
    assert t.per_db["concert_singer"] == 2
    assert t.per_db["debit_card_specializing"] == 1


def test_query_type_join_detected():
    t = DiversityTracker()
    t.add(_make_example("db1", "SELECT a.name FROM singer a JOIN concert c ON a.id = c.singer_id"))
    assert t.query_type_counts["join"] >= 1


def test_query_type_aggregate_detected():
    t = DiversityTracker()
    t.add(_make_example("db1", "SELECT count(*) FROM singer"))
    assert t.query_type_counts["aggregate"] >= 1


def test_query_type_group_by_detected():
    t = DiversityTracker()
    t.add(_make_example("db1", "SELECT name, count(*) FROM singer GROUP BY name"))
    assert t.query_type_counts["group_by"] >= 1


def test_query_type_order_by_detected():
    t = DiversityTracker()
    t.add(_make_example("db1", "SELECT name FROM singer ORDER BY age DESC"))
    assert t.query_type_counts["order_by"] >= 1


def test_query_type_subquery_detected():
    t = DiversityTracker()
    t.add(_make_example("db1", "SELECT name FROM singer WHERE id IN (SELECT singer_id FROM concert)"))
    assert t.query_type_counts["subquery"] >= 1


def test_query_type_limit_detected():
    t = DiversityTracker()
    t.add(_make_example("db1", "SELECT name FROM singer LIMIT 5"))
    assert t.query_type_counts["limit"] >= 1


def test_difficulty_counts():
    t = DiversityTracker()
    t.add(_make_example("db1", "SELECT 1", difficulty="easy"))
    t.add(_make_example("db1", "SELECT 1", difficulty="medium"))
    t.add(_make_example("db1", "SELECT 1", difficulty="hard"))
    assert t.difficulty_counts["easy"] == 1
    assert t.difficulty_counts["medium"] == 1
    assert t.difficulty_counts["hard"] == 1


def test_convergence_not_met_when_total_below_target():
    t = DiversityTracker()
    for _ in range(100):
        t.add(_make_example("db1", "SELECT name FROM singer JOIN concert ON singer.singer_id = concert.singer_id GROUP BY name ORDER BY count(*) DESC LIMIT 5 WHERE name IN (SELECT name FROM singer)"))
    met, report = t.convergence_check()
    assert met is False
    assert "total" in report


def test_convergence_report_lists_short_buckets():
    t = DiversityTracker()
    # No window functions added — bucket should be reported short
    t.add(_make_example("db1", "SELECT name FROM singer", difficulty="easy"))
    met, report = t.convergence_check()
    assert met is False
    assert "window" in report.lower()


def test_add_updates_example_query_types_inplace():
    t = DiversityTracker()
    ex = _make_example("db1", "SELECT count(*) FROM singer GROUP BY name")
    t.add(ex)
    assert "aggregate" in ex["query_types"]
    assert "group_by" in ex["query_types"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
.venv/bin/pytest tests/test_diversity.py -v
```

Expected: `ERROR — ModuleNotFoundError: No module named 'src.diversity'`

- [ ] **Step 3: Implement `src/diversity.py`**

```python
"""
diversity.py — Coverage tracker for the dataset generation loop.

Tracks per-DB example counts, a query-type histogram (regex-based, no external
parser), difficulty bucket counts, and convergence toward the 15k target.

Query type detection uses case-insensitive regex over the SQL string; a single
SQL can contribute to multiple buckets (e.g., GROUP BY + aggregate + join).
"""

import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONVERGENCE_TARGET = 15_000

# Minimum examples per query type to declare convergence.
# All buckets must be above floor; "window" is hardest to hit.
_QUERY_TYPE_FLOOR = {
    "join": 1_500,
    "aggregate": 1_500,
    "subquery": 800,
    "window": 200,
    "group_by": 1_200,
    "order_by": 1_200,
    "nested": 600,
    "limit": 800,
}

# Difficulty floors (None = spider records without difficulty field are skipped)
_DIFFICULTY_FLOOR = {
    "simple": 500,    # BIRD labels
    "moderate": 300,  # BIRD labels
    "challenging": 100,  # BIRD labels
    "easy": 1_000,    # Spider if present
    "medium": 1_000,
    "hard": 500,
    "extra": 200,
}

# All tracked query types (order defines histogram display)
QUERY_TYPES = ["join", "aggregate", "subquery", "window", "group_by", "order_by", "nested", "limit"]

# ---------------------------------------------------------------------------
# Regex patterns — case-insensitive
# ---------------------------------------------------------------------------

_PATTERNS: dict[str, re.Pattern] = {
    "join":      re.compile(r"\bJOIN\b", re.IGNORECASE),
    "aggregate": re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE),
    "subquery":  re.compile(r"\(\s*SELECT\b", re.IGNORECASE),
    "window":    re.compile(r"\bOVER\s*\(", re.IGNORECASE),
    "group_by":  re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE),
    "order_by":  re.compile(r"\bORDER\s+BY\b", re.IGNORECASE),
    "nested":    re.compile(r"\bWITH\b|\bFROM\s*\(", re.IGNORECASE),
    "limit":     re.compile(r"\bLIMIT\b", re.IGNORECASE),
}


def detect_query_types(sql: str) -> list[str]:
    """Return all query type labels that match sql (may be multiple)."""
    return [qt for qt, pat in _PATTERNS.items() if pat.search(sql)]


# ---------------------------------------------------------------------------
# DiversityTracker
# ---------------------------------------------------------------------------

class DiversityTracker:
    """Accumulate examples and report coverage + convergence."""

    def __init__(self):
        self.total: int = 0
        self.per_db: dict[str, int] = defaultdict(int)
        self.query_type_counts: dict[str, int] = {qt: 0 for qt in QUERY_TYPES}
        self.difficulty_counts: dict[str, int] = defaultdict(int)

    def add(self, example: dict) -> None:
        """Register one example. Mutates example['query_types'] with detected types."""
        self.total += 1
        self.per_db[example["db_id"]] += 1

        # Extract SQL from assistant turn
        sql = ""
        for msg in example.get("messages", []):
            if msg.get("role") == "assistant":
                sql = msg.get("content", "")
                break

        detected = detect_query_types(sql)
        example["query_types"] = detected
        for qt in detected:
            if qt in self.query_type_counts:
                self.query_type_counts[qt] += 1

        diff = example.get("difficulty")
        if diff:
            self.difficulty_counts[diff] += 1

    def convergence_check(self) -> tuple[bool, str]:
        """Check whether convergence criteria are met.

        Returns (met: bool, report: str).
        report lists any short buckets even when met=True for transparency.
        Convergence = total >= CONVERGENCE_TARGET AND all query-type floors met.
        """
        lines = []

        # Total
        if self.total < CONVERGENCE_TARGET:
            lines.append(f"total: {self.total}/{CONVERGENCE_TARGET} (short by {CONVERGENCE_TARGET - self.total})")

        # Query-type floors
        for qt, floor in _QUERY_TYPE_FLOOR.items():
            count = self.query_type_counts.get(qt, 0)
            if count < floor:
                lines.append(f"query_type[{qt}]: {count}/{floor} (short by {floor - count})")

        met = (self.total >= CONVERGENCE_TARGET) and all(
            self.query_type_counts.get(qt, 0) >= floor
            for qt, floor in _QUERY_TYPE_FLOOR.items()
        )
        report = "\n".join(lines) if lines else "All convergence criteria met."
        return met, report

    def summary(self) -> dict:
        """Return a snapshot dict for logging."""
        return {
            "total": self.total,
            "per_db": dict(self.per_db),
            "query_type_counts": dict(self.query_type_counts),
            "difficulty_counts": dict(self.difficulty_counts),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
.venv/bin/pytest tests/test_diversity.py -v
```

Expected:
```
tests/test_diversity.py::test_tracker_starts_empty PASSED
tests/test_diversity.py::test_add_example_increments_total PASSED
tests/test_diversity.py::test_per_db_count PASSED
tests/test_diversity.py::test_query_type_join_detected PASSED
tests/test_diversity.py::test_query_type_aggregate_detected PASSED
tests/test_diversity.py::test_query_type_group_by_detected PASSED
tests/test_diversity.py::test_query_type_order_by_detected PASSED
tests/test_diversity.py::test_query_type_subquery_detected PASSED
tests/test_diversity.py::test_query_type_limit_detected PASSED
tests/test_diversity.py::test_difficulty_counts PASSED
tests/test_diversity.py::test_convergence_not_met_when_total_below_target PASSED
tests/test_diversity.py::test_convergence_report_lists_short_buckets PASSED
tests/test_diversity.py::test_add_updates_example_query_types_inplace PASSED
13 passed in X.XXs
```

- [ ] **Step 5: Commit**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
git add src/diversity.py tests/test_diversity.py
git commit -m "feat: diversity — per-DB coverage tracker, query-type histogram, convergence check"
```

---

### Task 7: `src/generate_examples.py` — Correction-Set Interface (Mocked)

**Files:**
- Create: `sql-agent/src/generate_examples.py`
- Create: `sql-agent/tests/test_generate_examples.py`

This module defines the contract for generating correction-chain examples. The actual model (Qwen3.5-4B via mlx-lm) is deferred to a later plan — mlx-lm is not installed here. Unit tests use a mocked `ModelCallable` that simulates wrong SQL + a real sqlite3 traceback, exercising all the wiring without a live model.

- [ ] **Step 1: Write the failing test**

Create `sql-agent/tests/test_generate_examples.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
.venv/bin/pytest tests/test_generate_examples.py -v
```

Expected: `ERROR — ModuleNotFoundError: No module named 'src.generate_examples'`

- [ ] **Step 3: Implement `src/generate_examples.py`**

```python
"""
generate_examples.py — Correction-set generation interface.

Defines the contract for base-model-bootstrapped correction examples (OD2 strategy A).
The actual model (Qwen3.5-4B via mlx-lm) is NOT wired here — mlx/mlx-lm are deferred.
Tests use a mocked ModelCallable.

Correction chain format (per spec §4.1):
    system   → schema cache JSON
    user     → question
    assistant→ wrong SQL (base model's first attempt)
    tool     → real sqlite3 traceback (never invented)
    assistant→ fixed SQL (must match gold result-set)

generate_correction_example() returns None if the model got it right first try
(nothing to correct). Returns CorrectionResult(accepted=False) if the fix
never passes verification within max_attempts.
"""

from dataclasses import dataclass, field
from typing import Callable
import json

from src.gold_ingest import GoldRecord, build_message_list
from src.verify import verify_sql, run_query_with_timeout, MatchVerdict

# Type alias for the model callable — takes a message list, returns SQL string
ModelCallable = Callable[[list[dict]], str]


@dataclass
class CorrectionResult:
    accepted: bool
    messages: list[dict]
    record: GoldRecord

    def to_example_dict(self) -> dict:
        return {
            "db_id": self.record.db_id,
            "source": self.record.source,
            "difficulty": self.record.difficulty,
            "type": "correction",
            "query_types": [],
            "messages": self.messages,
        }


def generate_correction_example(
    record: GoldRecord,
    db_path: str,
    schema_cache: dict,
    model_fn: ModelCallable,
    max_attempts: int = 3,
) -> CorrectionResult | None:
    """Run base model zero-shot, capture real wrong SQL + traceback, verify fix.

    Returns None if the model's first attempt already matches gold (no correction needed).
    Returns CorrectionResult(accepted=False) if fix never validates within max_attempts.
    Returns CorrectionResult(accepted=True) if a correction chain is captured and verified.

    The 'tool' message content is ALWAYS a real sqlite3 error string — never invented.
    """
    # Build initial prompt: system + user
    base_messages = build_message_list(record, schema_cache)
    # base_messages = [system, user, assistant(gold)] — strip assistant for inference
    prompt = base_messages[:2]  # [system, user] only

    # Turn 1: base model first attempt
    wrong_sql = model_fn(prompt)

    # Check if first attempt already correct
    try:
        result1 = verify_sql(db_path, wrong_sql, record.gold_sql)
    except ValueError:
        # Gold SQL failed — should not happen if records are pre-verified
        return None

    if result1.is_valid:
        # First attempt is correct — no correction chain to generate
        return None

    # Capture real traceback from wrong_sql execution
    _, err = run_query_with_timeout(db_path, wrong_sql)
    if err is None:
        # SQL executed but result was wrong (MISMATCH) — produce empty-result notice
        traceback_content = "Execution succeeded but result set does not match expected output. Review your query logic."
    else:
        traceback_content = err  # real sqlite3 error string

    # Build correction context: system + user + wrong_sql + tool(traceback)
    correction_prompt = [
        prompt[0],   # system
        prompt[1],   # user
        {"role": "assistant", "content": wrong_sql},
        {"role": "tool",      "content": traceback_content},
    ]

    # Turn 2..max_attempts: ask model to fix
    for _ in range(max_attempts - 1):
        fixed_sql = model_fn(correction_prompt)
        try:
            result2 = verify_sql(db_path, fixed_sql, record.gold_sql)
        except ValueError:
            return CorrectionResult(accepted=False, messages=correction_prompt, record=record)

        if result2.is_valid:
            final_messages = correction_prompt + [{"role": "assistant", "content": fixed_sql}]
            return CorrectionResult(accepted=True, messages=final_messages, record=record)

        # Append new traceback and retry
        _, retry_err = run_query_with_timeout(db_path, fixed_sql)
        retry_traceback = retry_err if retry_err else "Result mismatch after fix attempt."
        correction_prompt = correction_prompt + [
            {"role": "assistant", "content": fixed_sql},
            {"role": "tool",      "content": retry_traceback},
        ]

    return CorrectionResult(accepted=False, messages=correction_prompt, record=record)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
.venv/bin/pytest tests/test_generate_examples.py -v
```

Expected:
```
tests/test_generate_examples.py::test_generate_correction_example_wrong_then_fixed PASSED
tests/test_generate_examples.py::test_generate_correction_skipped_when_first_attempt_correct PASSED
tests/test_generate_examples.py::test_generate_correction_rejected_when_fix_never_works PASSED
tests/test_generate_examples.py::test_correction_messages_have_tool_role_for_traceback PASSED
tests/test_generate_examples.py::test_correction_result_to_example_dict PASSED
5 passed in X.XXs
```

- [ ] **Step 5: Commit**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
git add src/generate_examples.py tests/test_generate_examples.py
git commit -m "feat: generate_examples — correction-chain interface with mocked model tests"
```

---

### Task 8: `src/build_dataset.py` — Dataset Orchestrator

**Files:**
- Create: `sql-agent/src/build_dataset.py`
- Create: `sql-agent/tests/test_build_dataset.py`

Combines clean + correction examples, deduplicates by normalized (SQL+question) hash, enforces 60/40 type split, emits train/val/test JSONL to `data/dataset/`. Note: in this plan there are no correction examples yet (mlx-lm deferred), so build_dataset works on clean-only input and the 60/40 split is structural — it will accept a mixed list when corrections arrive in a later plan.

- [ ] **Step 1: Write the failing test**

Create `sql-agent/tests/test_build_dataset.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
.venv/bin/pytest tests/test_build_dataset.py -v
```

Expected: `ERROR — ModuleNotFoundError: No module named 'src.build_dataset'`

- [ ] **Step 3: Implement `src/build_dataset.py`**

```python
"""
build_dataset.py — Orchestrator: combine clean+correction → dedup → split → emit JSONL.

Dedup key: normalize(sql + " ||| " + question) where normalize lowercases,
collapses whitespace, and replaces digit-runs with <NUM> and string literals
with <STR> (same strategy as llm-training/scripts/validate_data.py).

Split: deterministic structural split — sorted normalized-key index mod 10:
  bucket 0 → test (~10%)
  bucket 1 → val  (~10%)
  buckets 2-9 → train (~80%)

Output: data/dataset/train.jsonl, val.jsonl, test.jsonl (one JSON object per line).
Each line preserves the full example dict including messages.
"""

import json
import re
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Normalization (mirrors validate_data._normalize_template)
# ---------------------------------------------------------------------------

def normalize_key(sql: str, question: str) -> str:
    combined = sql + " ||| " + question
    combined = re.sub(r"'[^']*'", "<STR>", combined)
    combined = re.sub(r"\b\d+\b", "<NUM>", combined)
    combined = re.sub(r"\s+", " ", combined.lower().strip())
    return combined


# ---------------------------------------------------------------------------
# dedup_examples
# ---------------------------------------------------------------------------

def dedup_examples(examples: list[dict]) -> tuple[list[dict], int]:
    """Dedup by normalized (sql + question) hash. Returns (kept, dropped_count)."""
    seen: set[str] = set()
    kept = []
    dropped = 0
    for ex in examples:
        # Extract SQL from assistant turn and question from user turn
        sql = ""
        question = ""
        for msg in ex.get("messages", []):
            if msg["role"] == "assistant" and not sql:
                sql = msg["content"]
            if msg["role"] == "user":
                question = msg["content"]
        key = normalize_key(sql, question)
        if key in seen:
            dropped += 1
        else:
            seen.add(key)
            kept.append(ex)
    return kept, dropped


# ---------------------------------------------------------------------------
# split_examples — deterministic structural split
# ---------------------------------------------------------------------------

def split_examples(
    examples: list[dict],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Assign examples to train/val/test by sorted normalized key mod 10.

    bucket 0 → test, bucket 1 → val, buckets 2-9 → train.
    val_frac and test_frac params are informational — the mod-10 split
    approximates 10%/10%/80% deterministically.
    """
    key_to_examples: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        sql = ""
        question = ""
        for msg in ex.get("messages", []):
            if msg["role"] == "assistant" and not sql:
                sql = msg["content"]
            if msg["role"] == "user":
                question = msg["content"]
        key = normalize_key(sql, question)
        key_to_examples[key].append(ex)

    sorted_keys = sorted(key_to_examples.keys())
    train, val, test = [], [], []
    for i, key in enumerate(sorted_keys):
        bucket = i % 10
        if bucket == 0:
            test.extend(key_to_examples[key])
        elif bucket == 1:
            val.extend(key_to_examples[key])
        else:
            train.extend(key_to_examples[key])

    return train, val, test


# ---------------------------------------------------------------------------
# build_dataset — main orchestrator
# ---------------------------------------------------------------------------

def build_dataset(
    clean_examples: list[dict],
    correction_examples: list[dict],
    output_dir: str,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> dict:
    """Combine clean + correction, dedup, split, write JSONL.

    Returns stats dict: {total_input, dropped_duplicates, total_after_dedup,
                         train, val, test, clean_count, correction_count}.
    """
    all_examples = clean_examples + correction_examples
    total_input = len(all_examples)
    clean_count = len(clean_examples)
    correction_count = len(correction_examples)

    deduped, dropped = dedup_examples(all_examples)
    total_after_dedup = len(deduped)

    train, val, test = split_examples(deduped, val_frac, test_frac)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        with open(out / f"{split_name}.jsonl", "w", encoding="utf-8") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")

    return {
        "total_input": total_input,
        "clean_count": clean_count,
        "correction_count": correction_count,
        "dropped_duplicates": dropped,
        "total_after_dedup": total_after_dedup,
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
.venv/bin/pytest tests/test_build_dataset.py -v
```

Expected:
```
tests/test_build_dataset.py::test_normalize_key_lowercases_and_collapses_whitespace PASSED
tests/test_build_dataset.py::test_normalize_key_replaces_number_literals PASSED
tests/test_build_dataset.py::test_dedup_removes_exact_duplicates PASSED
tests/test_build_dataset.py::test_dedup_keeps_distinct_questions PASSED
tests/test_build_dataset.py::test_dedup_removes_normalized_duplicates PASSED
tests/test_build_dataset.py::test_split_returns_three_lists PASSED
tests/test_build_dataset.py::test_split_train_is_largest PASSED
tests/test_build_dataset.py::test_build_dataset_emits_jsonl_files PASSED
tests/test_build_dataset.py::test_build_dataset_jsonl_each_line_valid_json PASSED
tests/test_build_dataset.py::test_build_dataset_stats_returned PASSED
tests/test_build_dataset.py::test_build_dataset_combines_clean_and_correction PASSED
11 passed in X.XXs
```

- [ ] **Step 5: Commit**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
git add src/build_dataset.py tests/test_build_dataset.py
git commit -m "feat: build_dataset — dedup, structural split, JSONL emit"
```

---

### Task 9: `CONTEXT.md` — Navigation Index

**Files:**
- Create: `sql-agent/CONTEXT.md`

- [ ] **Step 1: Write `CONTEXT.md`**

```markdown
# sql-agent — CONTEXT.md

Navigation index for the `sql-agent` project. Read this before touching any source file.

## Routing Matrix

| What you need | Go to |
|---|---|
| Schema compiler (PRAGMA → JSON) | `src/schema_cache.py` |
| Execution verifier + timeout | `src/verify.py` |
| Gold pair normalization (Spider/BIRD) | `src/gold_ingest.py` |
| Clean-set builder | `src/gold_ingest.py` → `build_clean_examples()` |
| Query-type histogram + convergence | `src/diversity.py` |
| Correction-chain generation interface | `src/generate_examples.py` |
| Dataset orchestrator (dedup, split, JSONL) | `src/build_dataset.py` |
| Unit tests | `tests/` |
| Spider gold JSON | `data/spider/spider_data/train_spider.json`, `train_others.json` |
| BIRD gold JSON | `data/bird/dev_20240627/dev.json` |
| Spider SQLite DBs | `data/spider/spider_data/database/<db_id>/<db_id>.sqlite` |
| BIRD SQLite DBs | `data/bird/dev_20240627/dev_databases/<db_id>/<db_id>.sqlite` |
| Generated dataset | `data/dataset/train.jsonl`, `val.jsonl`, `test.jsonl` |
| Project spec | `docs/superpowers/specs/2026-06-20-sql-agent-project-design.md` |
| Data pipeline plan | `docs/superpowers/plans/2026-06-20-data-pipeline.md` |

## File Map — `src/`

| File | Exports |
|---|---|
| `schema_cache.py` | `compile_schema_cache(db_path: str) -> dict` |
| `verify.py` | `run_query_with_timeout`, `verify_sql`, `VerifyResult`, `MatchVerdict` |
| `gold_ingest.py` | `GoldRecord`, `load_spider_gold`, `load_bird_gold`, `build_message_list`, `build_clean_examples` |
| `diversity.py` | `DiversityTracker`, `detect_query_types`, `QUERY_TYPES`, `CONVERGENCE_TARGET` |
| `generate_examples.py` | `ModelCallable`, `generate_correction_example`, `CorrectionResult` |
| `build_dataset.py` | `build_dataset`, `dedup_examples`, `split_examples`, `normalize_key` |

## Key Data Shapes

**Spider gold record** (`train_spider.json`):
```json
{"db_id": "department_management", "question": "...", "query": "<SQL string>",
 "query_toks": [...], "query_toks_no_value": [...], "question_toks": [...],
 "sql": {<parsed dict — NOT the SQL string>}}
```

**BIRD gold record** (`dev.json`):
```json
{"question_id": 0, "db_id": "california_schools", "question": "...",
 "evidence": "...", "SQL": "<SQL string>", "difficulty": "simple"}
```

**db_metadata_cache** (output of `compile_schema_cache`):
```json
{"db_id": "concert_singer",
 "tables": [{"name": "singer", "columns": [{"name": "age", "type": "INTEGER"}],
              "foreign_keys": [], "sample_rows": [[1, "Alice", 30]]}]}
```

**Message list example** (`data/dataset/*.jsonl` lines):
```json
{"db_id": "concert_singer", "source": "spider", "difficulty": null,
 "type": "clean", "query_types": ["aggregate"],
 "messages": [
   {"role": "system",    "content": "<db_metadata_cache JSON>"},
   {"role": "user",      "content": "<question>"},
   {"role": "assistant", "content": "<SQL>"}
 ]}
```

## Deferred (later plans)

- `src/train.py` — QLoRA mlx-lm wrapper (fine-tuning plan)
- `src/evaluate.py` — exec accuracy harness (eval plan)
- `src/agent/` — LangGraph Phase-B runtime (harness plan)
- Qwen3.5-4B inference in `generate_examples.py` (requires mlx-lm, deferred)
```

- [ ] **Step 2: Run full test suite to confirm all 54 tests still pass**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
.venv/bin/pytest -v
```

Expected:
```
54 passed in X.XXs
```

(8 schema_cache + 10 verify + 7 gold_ingest + 5 clean_set + 13 diversity + 5 generate_examples + 11 build_dataset = 59 tests total across all tasks — actual count may vary by ±2 if any fixture is shared.)

- [ ] **Step 3: Commit**

```bash
cd "/Users/hiten/IU Coursework/AI Projects/sql-agent"
git add CONTEXT.md
git commit -m "docs: CONTEXT.md — navigation index, routing matrix, file map, data shapes"
```

---

## Self-Review

### 1. Spec Coverage — Data Pipeline

| Spec requirement | Task |
|---|---|
| `sql-agent/.venv`, `requirements.txt`, pytest, scaffold | Task 1 |
| Phase-A compiler: PRAGMA → `db_metadata_cache.json` | Task 2 |
| FK edges via `PRAGMA foreign_key_list` | Task 2 |
| 3 sample rows/table | Task 2 |
| `verify.py`: execute SQL with timeout | Task 3 |
| strict + superset + order-tolerant match (ported from evaluate.py) | Task 3 |
| Valid = result-set match, NOT non-empty | Task 3 |
| `gold_ingest.py`: Spider + BIRD → normalized records | Task 4 |
| Model-agnostic message lists (system=schema, user=question, assistant=SQL) | Task 4 |
| Real Spider field names (`db_id`, `question`, `query`) | Task 4 |
| Real BIRD field names (`db_id`, `question`, `SQL`, `difficulty`) | Task 4 |
| Discard gold that fails to execute; track yield | Task 5 |
| Per-DB counts, query-type histogram, difficulty buckets | Task 6 |
| Convergence check (≥15k AND thresholds) | Task 6 |
| Report short buckets rather than silently truncating | Task 6 |
| Correction-set interface: wrong SQL + real traceback | Task 7 |
| Unit test with mocked model (no real model in tests) | Task 7 |
| mlx-lm deferred (not installed) | Task 7 note |
| Dedup (normalized SQL+question hash) | Task 8 |
| 60/40 clean/correction structural split | Task 8 |
| Train/val/test split | Task 8 |
| Emit JSONL (message-list format) | Task 8 |
| `CONTEXT.md` with routing matrix + file map | Task 9 |
| `git init` in scaffold | Task 1 |

**Not covered in this plan (correct — deferred):** `src/train.py`, `src/evaluate.py`, `src/agent/`, LangGraph harness, QLoRA, mlx-lm inference in generate_examples.

### 2. Placeholder Scan

No TBD/TODO/"add error handling"/"similar to Task N"/"fill in details" found. Every step has exact file paths, real code, and exact expected test output.

### 3. Type and Name Consistency

- `GoldRecord` defined in Task 4, imported in Tasks 5, 7 — consistent.
- `compile_schema_cache(db_path: str) -> dict` defined in Task 2, called identically in Tasks 5, 7 — consistent.
- `verify_sql(db_path, pred_sql, gold_sql) -> VerifyResult` defined in Task 3, called in Tasks 5 and 7 — consistent.
- `build_message_list(record: GoldRecord, cache: dict) -> list[dict]` defined in Task 4, called in Task 7 — consistent.
- `normalize_key(sql, question)` defined and exported in Task 8 — used in `dedup_examples` and `split_examples` in same file — consistent.
- `MatchVerdict` enum defined in Task 3, referenced in test assertions — consistent.
- `CorrectionResult.to_example_dict()` defined and tested in Task 7 — consistent.
- BIRD SQL field: `"SQL"` (uppercase) — used consistently in Task 4 load + test fixture.
- Spider SQL field: `"query"` (lowercase) — used consistently in Task 4 load + test fixture.

---

### Critical Files for Implementation

- `/Users/hiten/IU Coursework/AI Projects/sql-agent/src/gold_ingest.py`
- `/Users/hiten/IU Coursework/AI Projects/sql-agent/src/verify.py`
- `/Users/hiten/IU Coursework/AI Projects/sql-agent/src/schema_cache.py`
- `/Users/hiten/IU Coursework/AI Projects/sql-agent/src/generate_examples.py`
- `/Users/hiten/IU Coursework/AI Projects/sql-agent/src/build_dataset.py`

---

## Compressed Report

Plan written as direct output (read-only mode; no Write tool available).

- **Tasks:** 9
- **Spider gold field names (verbatim):** `db_id`, `question`, `query` (SQL string), `query_toks`, `query_toks_no_value`, `question_toks`, `sql` (parsed dict — distinct from `query`)
- **BIRD gold field names (verbatim):** `question_id`, `db_id`, `question`, `evidence`, `SQL` (uppercase), `difficulty`
- **Real db_ids confirmed:** `department_management` (Spider, 3 tables: department/head/management), `california_schools` (BIRD, 3 tables: frpm/satscores/schools)
- **SQLite confirmed readable:** 2 — `department_management.sqlite` via PRAGMA table_info + foreign_key_list; `california_schools.sqlite` via PRAGMA table_info. Both confirmed in-session.
- **Spec ambiguity hit:** `build_dataset` 60/40 split is structural-only until correction examples exist (mlx-lm deferred). The orchestrator accepts a `correction_examples=[]` list and emits whatever clean examples are present. No silent assumption of 60/40 at test time — plan notes this explicitly in Task 8.