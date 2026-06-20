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
