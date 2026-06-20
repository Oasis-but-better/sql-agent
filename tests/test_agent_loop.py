"""
test_agent_loop.py — TDD for Phase-B LangGraph self-correction agent.

Uses MOCK model_fn (no real inference) + real tiny temp .sqlite.
Avoids resource contention with concurrent CP5 training.
"""

import sqlite3
import tempfile
import os
import pytest

from src.agent.loop import run_agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_db(tmp_path):
    """Tiny .sqlite with one table + one row for real execution."""
    db_path = str(tmp_path / "tiny.sqlite")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice')")
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_fn(sql_responses: list[str]):
    """Return a model_fn that pops responses from a list in order."""
    responses = list(sql_responses)
    def fn(messages):
        return responses.pop(0)
    return fn


# ---------------------------------------------------------------------------
# Test 1: valid SQL first try → success, 1 attempt
# ---------------------------------------------------------------------------

def test_success_first_try(tiny_db):
    model_fn = _mock_fn(["SELECT * FROM users"])
    state = run_agent(tiny_db, "List all users.", model_fn, max_attempts=3)

    assert state["status"] == "success"
    assert state["attempts"] == 1
    assert state["final_result"] is not None
    assert len(state["final_result"]) == 1
    assert state["history"][-1][0] == "SELECT * FROM users"


# ---------------------------------------------------------------------------
# Test 2: broken SQL then valid → success, 2 attempts, history shows error
# ---------------------------------------------------------------------------

def test_success_second_try(tiny_db):
    bad_sql  = "SELECT * FROM nonexistent_table"
    good_sql = "SELECT * FROM users"
    model_fn = _mock_fn([bad_sql, good_sql])

    state = run_agent(tiny_db, "List all users.", model_fn, max_attempts=3)

    assert state["status"] == "success"
    assert state["attempts"] == 2

    # First history entry: bad SQL + error string
    assert state["history"][0][0] == bad_sql
    assert isinstance(state["history"][0][1], str)  # error message
    assert "no such table" in state["history"][0][1].lower()

    # Second history entry: good SQL + rows
    assert state["history"][1][0] == good_sql
    assert state["final_result"] is not None

    # Messages should contain tool feedback (error injection)
    tool_msgs = [m for m in state["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) >= 1
    assert "error" in tool_msgs[0]["content"].lower()


# ---------------------------------------------------------------------------
# Test 3: always broken → exhausted at 3 attempts
# ---------------------------------------------------------------------------

def test_exhausted_after_max_attempts(tiny_db):
    always_bad = "SELECT * FROM definitely_missing"
    # Provide more than max_attempts to ensure it caps at 3
    model_fn = _mock_fn([always_bad] * 10)

    state = run_agent(tiny_db, "List all users.", model_fn, max_attempts=3)

    assert state["status"] == "exhausted"
    assert state["attempts"] == 3
    assert state["final_result"] is None
    # All 3 history entries should be the bad SQL
    assert len(state["history"]) == 3
    for sql, err in state["history"]:
        assert sql == always_bad
        assert isinstance(err, str)


# ---------------------------------------------------------------------------
# Test 4: empty-result SQL → retry, then success
# ---------------------------------------------------------------------------

def test_empty_result_triggers_retry(tiny_db):
    empty_sql = "SELECT * FROM users WHERE id = 999"  # no matching row
    good_sql  = "SELECT * FROM users"
    model_fn  = _mock_fn([empty_sql, good_sql])

    state = run_agent(tiny_db, "List all users.", model_fn, max_attempts=3)

    assert state["status"] == "success"
    assert state["attempts"] == 2

    # Tool message should mention empty result
    tool_msgs = [m for m in state["messages"] if m["role"] == "tool"]
    assert any("empty" in m["content"].lower() for m in tool_msgs)
