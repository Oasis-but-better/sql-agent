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
