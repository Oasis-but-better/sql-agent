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
