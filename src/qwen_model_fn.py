"""
qwen_model_fn.py — Real model_fn for Qwen3.5-4B-4bit (MLX).

Key conventions (from project Key-learnings):
  - Must render Qwen chat template AND prefill empty <think></think>
    before SQL generation, else model emits reasoning prose instead of SQL.
  - SQL extracted from output by stripping <think>...</think> block first,
    then taking the first SELECT/INSERT/UPDATE/DELETE/WITH statement found.

Usage:
    from src.qwen_model_fn import make_qwen_model_fn

    model_fn = make_qwen_model_fn()                      # base quantized
    model_fn = make_qwen_model_fn(adapter_path="adapters/qwen-sql")  # finetuned

    sql = model_fn(messages)   # messages: list[{"role": ..., "content": ...}]
"""

from __future__ import annotations

import re
from typing import Callable


# ---------------------------------------------------------------------------
# SQL extraction helper
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SQL_RE = re.compile(
    r"(SELECT|INSERT|UPDATE|DELETE|WITH)\b.*",
    re.IGNORECASE | re.DOTALL,
)


def _extract_sql(raw: str) -> str:
    """Strip <think>...</think>, return first SQL statement found."""
    cleaned = _THINK_RE.sub("", raw).strip()
    m = _SQL_RE.search(cleaned)
    if m:
        return m.group(0).strip()
    # Fallback: return cleaned string as-is (let agent execute + handle error)
    return cleaned


# ---------------------------------------------------------------------------
# make_qwen_model_fn — factory
# ---------------------------------------------------------------------------

def make_qwen_model_fn(adapter_path: str | None = None) -> Callable[[list[dict]], str]:
    """Load Qwen3.5-4B-4bit + optional LoRA adapter; return model_fn.

    Imports mlx_lm lazily to avoid loading the model at module import time
    (prevents memory contention with a concurrently running fine-tune).

    Args:
        adapter_path: Path to LoRA adapter dir (e.g. "adapters/qwen-sql").
                      None = base quantized model only.

    Returns:
        model_fn(messages: list[dict]) -> str
            Renders Qwen chat template with empty <think></think> prefill,
            runs generation, extracts SQL from output.
    """
    import pathlib

    # Lazy import — do NOT move to module top level
    import mlx_lm  # type: ignore

    model_path = str(pathlib.Path(__file__).parent.parent / "models" / "qwen-4bit")

    model, tokenizer = mlx_lm.load(model_path, adapter_path=adapter_path)

    def model_fn(messages: list[dict]) -> str:
        # Render chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Prefill empty <think></think> to suppress reasoning prose
        prompt = prompt + "<think></think>"

        # Generate
        response = mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=512,
            verbose=False,
        )

        return _extract_sql(response)

    return model_fn
