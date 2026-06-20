"""
eval_harness.py — CP7 evaluation harness for NL→SQL self-correction agent.

evaluate(model_fn, test_path, dbs_root, max_attempts=3) -> dict
  Runs Phase-B agent on every test example; returns aggregate metrics.

Metrics returned:
  exec_acc_strict        — fraction with STRICT result-set match
  exec_acc_superset      — fraction with STRICT or SUPERSET match
  loop_resolution_rate   — of examples where attempt-1 wrong, fraction
                           ultimately correct (None if denom == 0)
  ttft_ms                — mean time-to-first-token proxy (ms) per example
  n                      — number of examples evaluated

Record helper:
  record_run(label, model, metrics) — appends to docs/metrics.json
"""

from __future__ import annotations

import json
import pathlib
import time
from datetime import datetime, timezone
from typing import Any, Callable

from src.agent.loop import run_agent
from src.verify import verify_sql, MatchVerdict

ROOT = pathlib.Path(__file__).parent.parent
METRICS_PATH = ROOT / "docs" / "metrics.json"


# ---------------------------------------------------------------------------
# Internal: parse a single test record
# ---------------------------------------------------------------------------

def _parse_record(record: dict) -> tuple[str, str, str]:
    """Return (db_id, question, gold_sql) from a test JSONL record.

    Record shape (CP3 mlx format):
        messages[0]: system  — JSON string containing "db_id"
        messages[1]: user    — NL question
        messages[2]: assistant — gold SQL
    """
    messages = record["messages"]
    system_payload = json.loads(messages[0]["content"])
    db_id = system_payload["db_id"]
    question = messages[1]["content"]
    gold_sql = messages[2]["content"]
    return db_id, question, gold_sql


# ---------------------------------------------------------------------------
# Public: evaluate
# ---------------------------------------------------------------------------

def evaluate(
    model_fn: Callable[[list[dict]], str],
    test_path: str,
    dbs_root: str,
    max_attempts: int = 3,
) -> dict:
    """Run agent on every example in test_path; return aggregate metrics.

    Args:
        model_fn:     Callable(messages) -> sql_text. Real or mock.
        test_path:    Path to JSONL test file (CP3 mlx format).
        dbs_root:     Root dir; DB at <dbs_root>/<db_id>/<db_id>.sqlite.
        max_attempts: Passed to run_agent.

    Returns dict with keys:
        exec_acc_strict, exec_acc_superset, loop_resolution_rate,
        ttft_ms, n
    """
    test_path_obj = pathlib.Path(test_path)
    dbs_root_obj = pathlib.Path(dbs_root)

    records = [json.loads(line) for line in test_path_obj.read_text().splitlines() if line.strip()]

    strict_correct = 0
    superset_correct = 0
    loop_denom = 0   # examples where attempt-1 was wrong
    loop_numer = 0   # of those, ultimately correct
    ttft_ms_list: list[float] = []
    n = 0

    for record in records:
        try:
            db_id, question, gold_sql = _parse_record(record)
        except (KeyError, IndexError, json.JSONDecodeError):
            continue

        db_path = str(dbs_root_obj / db_id / f"{db_id}.sqlite")
        if not pathlib.Path(db_path).exists():
            continue

        # Wrap model_fn to capture first-draft timing (ttft proxy)
        _timing: list[float] = []

        def timed_model_fn(messages, _inner=model_fn, _t=_timing):
            t0 = time.monotonic()
            result = _inner(messages)
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            if not _t:  # only first call per example
                _t.append(elapsed_ms)
            return result

        state = run_agent(db_path, question, timed_model_fn, max_attempts=max_attempts)

        if _timing:
            ttft_ms_list.append(_timing[0])

        n += 1

        # Score final SQL
        pred_sql = state["current_sql"]
        try:
            vr = verify_sql(db_path, pred_sql, gold_sql)
        except ValueError:
            # Gold SQL failed — skip this example from accuracy counts
            n -= 1
            continue

        is_strict = vr.verdict == MatchVerdict.STRICT
        is_valid = vr.is_valid  # STRICT or SUPERSET

        if is_strict:
            strict_correct += 1
        if is_valid:
            superset_correct += 1

        # Loop resolution: was attempt-1 already correct?
        if state["attempts"] > 1 and len(state["history"]) >= 1:
            # Check attempt-1 correctness
            first_sql = state["history"][0][0]
            try:
                first_vr = verify_sql(db_path, first_sql, gold_sql)
                attempt1_correct = first_vr.is_valid
            except ValueError:
                attempt1_correct = False

            if not attempt1_correct:
                loop_denom += 1
                if is_valid:
                    loop_numer += 1
        elif state["attempts"] == 1:
            # Only one attempt — check if it was wrong for loop_denom
            try:
                first_vr = verify_sql(db_path, pred_sql, gold_sql)
                if not first_vr.is_valid:
                    loop_denom += 1  # wrong, no retry → didn't resolve
            except ValueError:
                pass

    if n == 0:
        return {
            "exec_acc_strict": 0.0,
            "exec_acc_superset": 0.0,
            "loop_resolution_rate": None,
            "ttft_ms": None,
            "n": 0,
        }

    return {
        "exec_acc_strict": strict_correct / n,
        "exec_acc_superset": superset_correct / n,
        "loop_resolution_rate": (loop_numer / loop_denom) if loop_denom > 0 else None,
        "ttft_ms": (sum(ttft_ms_list) / len(ttft_ms_list)) if ttft_ms_list else None,
        "n": n,
    }


# ---------------------------------------------------------------------------
# record_run — append to docs/metrics.json
# ---------------------------------------------------------------------------

def record_run(label: str, model: str, metrics: dict) -> None:
    """Append a run entry to docs/metrics.json.

    Schema: {"runs": [{"label", "model", "exec_acc_strict",
                        "exec_acc_superset", "loop_resolution_rate",
                        "ttft_ms", "n", "ts"}, ...]}
    """
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if METRICS_PATH.exists():
        try:
            data = json.loads(METRICS_PATH.read_text())
        except json.JSONDecodeError:
            data = {"runs": []}
    else:
        data = {"runs": []}

    entry = {
        "label": label,
        "model": model,
        "exec_acc_strict": metrics.get("exec_acc_strict"),
        "exec_acc_superset": metrics.get("exec_acc_superset"),
        "loop_resolution_rate": metrics.get("loop_resolution_rate"),
        "ttft_ms": metrics.get("ttft_ms"),
        "n": metrics.get("n"),
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    data["runs"].append(entry)
    METRICS_PATH.write_text(json.dumps(data, indent=2))
