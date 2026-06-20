"""
test_eval_harness.py — TDD for CP7 eval harness.

Uses MOCK model_fn + tiny temp sqlite + synthetic test JSONL.
No real model inference. No mlx imports.

Covers:
  1. All-correct mock → exec_acc_strict=1.0, exec_acc_superset=1.0,
     loop_resolution_rate=None (all succeeded first try), n=4.
  2. Mock that fails attempt-1 (SQL error) then fixes → loop_resolution_rate=1.0.
  3. Always-wrong mock → exec_acc=0.0, loop_resolution_rate=0.0.
  4. Mixed: 2 correct, 1 wrong → exec_acc_strict=0.5 (out of 2 valid examples).
"""

from __future__ import annotations

import json
import sqlite3
import pathlib

import pytest

from src.eval_harness import evaluate, record_run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_db(tmp_path) -> tuple[str, str]:
    """Create a tiny sqlite with `users` table; return (db_path, db_id)."""
    db_id = "testdb"
    db_dir = tmp_path / db_id
    db_dir.mkdir()
    db_path = str(db_dir / f"{db_id}.sqlite")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice')")
    conn.execute("INSERT INTO users VALUES (2, 'Bob')")
    conn.commit()
    conn.close()
    return db_path, db_id, str(tmp_path)


def _make_record(db_id: str, question: str, gold_sql: str) -> str:
    """Build a JSONL line in CP3 mlx test format."""
    schema_stub = {"db_id": db_id, "tables": []}
    record = {
        "messages": [
            {"role": "system",    "content": json.dumps(schema_stub)},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": gold_sql},
        ]
    }
    return json.dumps(record)


@pytest.fixture()
def test_jsonl(tiny_db, tmp_path) -> str:
    """4-example test JSONL; all use same db_id + gold SQL."""
    _, db_id, _ = tiny_db
    gold = "SELECT * FROM users"
    lines = [_make_record(db_id, f"Q{i}", gold) for i in range(4)]
    path = tmp_path / "test.jsonl"
    path.write_text("\n".join(lines))
    return str(path)


def _mock_fn(sql_responses: list[str]):
    """Return model_fn that pops responses in order (restarted per call to factory)."""
    responses = list(sql_responses)

    def fn(messages):
        return responses.pop(0)

    return fn


# ---------------------------------------------------------------------------
# Test 1: all-correct mock → acc=1.0, loop_resolution_rate=None
# ---------------------------------------------------------------------------

def test_all_correct(tiny_db, test_jsonl):
    _, _, dbs_root = tiny_db
    # Always return perfect SQL
    model_fn = _mock_fn(["SELECT * FROM users"] * 20)
    metrics = evaluate(model_fn, test_jsonl, dbs_root, max_attempts=3)

    assert metrics["n"] == 4
    assert metrics["exec_acc_strict"] == pytest.approx(1.0)
    assert metrics["exec_acc_superset"] == pytest.approx(1.0)
    # All correct on first attempt → loop_denom=0 → None
    assert metrics["loop_resolution_rate"] is None
    assert metrics["ttft_ms"] is not None and metrics["ttft_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Test 2: fails attempt-1 (error), fixes attempt-2 → loop_resolution_rate=1.0
# ---------------------------------------------------------------------------

def test_fails_then_fixes(tiny_db, tmp_path):
    _, db_id, dbs_root = tiny_db
    gold = "SELECT * FROM users"
    # 1 example: first SQL errors (bad table), second SQL = gold
    lines = [_make_record(db_id, "List users", gold)]
    path = tmp_path / "test_loop.jsonl"
    path.write_text(lines[0])

    # Provide enough responses: attempt-1 errors, attempt-2 is gold
    bad_sql = "SELECT * FROM nonexistent_table"
    model_fn = _mock_fn([bad_sql, gold])

    metrics = evaluate(model_fn, str(path), dbs_root, max_attempts=3)

    assert metrics["n"] == 1
    assert metrics["exec_acc_superset"] == pytest.approx(1.0)
    # attempt-1 was wrong → loop_denom=1; ultimately correct → loop_numer=1
    assert metrics["loop_resolution_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 3: always-wrong → exec_acc=0.0, loop_resolution_rate=0.0
# ---------------------------------------------------------------------------

def test_always_wrong(tiny_db, tmp_path):
    _, db_id, dbs_root = tiny_db
    gold = "SELECT * FROM users"
    lines = [_make_record(db_id, "Q", gold)]
    path = tmp_path / "test_wrong.jsonl"
    path.write_text(lines[0])

    model_fn = _mock_fn(["SELECT * FROM nonexistent"] * 10)

    metrics = evaluate(model_fn, str(path), dbs_root, max_attempts=3)

    assert metrics["n"] == 1
    assert metrics["exec_acc_strict"] == pytest.approx(0.0)
    assert metrics["exec_acc_superset"] == pytest.approx(0.0)
    # Failed all attempts → loop_denom=1 (attempt-1 wrong), loop_numer=0
    assert metrics["loop_resolution_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 4: record_run writes to metrics.json correctly
# ---------------------------------------------------------------------------

def test_record_run(tmp_path, monkeypatch):
    metrics_path = tmp_path / "docs" / "metrics.json"
    import src.eval_harness as harness
    monkeypatch.setattr(harness, "METRICS_PATH", metrics_path)

    fake_metrics = {
        "exec_acc_strict": 0.75,
        "exec_acc_superset": 0.80,
        "loop_resolution_rate": 0.5,
        "ttft_ms": 123.4,
        "n": 100,
    }
    record_run("test-run", "qwen-4bit", fake_metrics)

    data = json.loads(metrics_path.read_text())
    assert "runs" in data
    assert len(data["runs"]) == 1
    run = data["runs"][0]
    assert run["label"] == "test-run"
    assert run["model"] == "qwen-4bit"
    assert run["exec_acc_strict"] == pytest.approx(0.75)
    assert run["n"] == 100
    assert "ts" in run

    # Second call appends
    record_run("run-2", "qwen-sql", fake_metrics)
    data2 = json.loads(metrics_path.read_text())
    assert len(data2["runs"]) == 2
