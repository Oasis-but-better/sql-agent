"""
loop.py — Phase-B LangGraph self-correction agent.

State machine: Compile → Draft → Execute → Loop (router).

model_fn is an INJECTED dependency:
  model_fn(messages: list[dict]) -> str   (returns raw SQL text)

Real impl would render Qwen chat template + prefill empty <think></think>.
Tests pass a mock.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from src.schema_cache import compile_schema_cache
from src.verify import run_query_with_timeout


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # Inputs (set before run)
    db_path: str
    question: str
    max_attempts: int
    model_fn: Any  # Callable[[list[dict]], str] — not type-checked by LangGraph

    # Working state
    schema_cache: dict           # populated by Compile node
    messages: list[dict]         # grows with each draft + feedback
    current_sql: str             # most-recently drafted SQL
    attempts: int                # incremented in Draft
    history: list[tuple]         # (sql, result_or_error) tuples
    final_result: Any            # rows on success, None otherwise
    status: str                  # "running" | "success" | "exhausted"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _node_compile(state: AgentState) -> dict:
    """Compile schema once from db_path; build initial messages."""
    cache = compile_schema_cache(state["db_path"])

    messages = [
        {"role": "system", "content": json.dumps(cache, separators=(",", ":"))},
        {"role": "user",   "content": state["question"]},
    ]

    return {
        "schema_cache": cache,
        "messages": messages,
        "attempts": 0,
        "history": [],
        "status": "running",
        "final_result": None,
        "current_sql": "",
    }


def _node_draft(state: AgentState) -> dict:
    """Call model_fn with current messages; increment attempt counter."""
    sql = state["model_fn"](state["messages"])
    attempts = state["attempts"] + 1

    return {
        "current_sql": sql,
        "attempts": attempts,
    }


def _node_execute(state: AgentState) -> dict:
    """Run current_sql; classify outcome; append feedback to messages + history."""
    sql = state["current_sql"]
    rows, err = run_query_with_timeout(state["db_path"], sql)

    messages = list(state["messages"])
    history = list(state["history"])

    if err is not None:
        # SQL error — inject assistant(wrong SQL) + tool(error traceback)
        messages.append({"role": "assistant", "content": sql})
        messages.append({"role": "tool",      "content": f"Error: {err}"})
        history.append((sql, err))
        return {
            "messages": messages,
            "history": history,
            "status": "running",
        }

    if not rows:
        # Empty result — inject assistant(SQL) + tool(empty note)
        messages.append({"role": "assistant", "content": sql})
        messages.append({"role": "tool",      "content": "Empty result: query returned no rows."})
        history.append((sql, "empty_result"))
        return {
            "messages": messages,
            "history": history,
            "status": "running",
        }

    # Success
    history.append((sql, rows))
    return {
        "history": history,
        "final_result": rows,
        "status": "success",
    }


def _router(state: AgentState) -> str:
    """Route after execute: success → end; exhausted → end; else → draft."""
    if state["status"] == "success":
        return "end"
    if state["attempts"] >= state["max_attempts"]:
        return "exhausted"
    return "draft"


def _node_exhausted(state: AgentState) -> dict:
    """Terminal node — mark status exhausted."""
    return {"status": "exhausted"}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("compile",   _node_compile)
    graph.add_node("draft",     _node_draft)
    graph.add_node("execute",   _node_execute)
    graph.add_node("exhausted", _node_exhausted)

    graph.set_entry_point("compile")
    graph.add_edge("compile",   "draft")
    graph.add_edge("draft",     "execute")
    graph.add_conditional_edges(
        "execute",
        _router,
        {
            "end":       END,
            "exhausted": "exhausted",
            "draft":     "draft",
        },
    )
    graph.add_edge("exhausted", END)

    return graph.compile()


_COMPILED_GRAPH = _build_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_agent(
    db_path: str,
    question: str,
    model_fn: Callable[[list[dict]], str],
    max_attempts: int = 3,
) -> AgentState:
    """Run the self-correction agent.

    Args:
        db_path:      Path to .sqlite file.
        question:     Natural language question.
        model_fn:     Callable(messages) -> sql_text. Injected; tests use a mock.
        max_attempts: Hard cap on draft/execute cycles (default 3).

    Returns:
        Final AgentState dict.
          state["status"]       — "success" | "exhausted"
          state["final_result"] — list of row tuples on success, else None
          state["attempts"]     — number of attempts made
          state["history"]      — [(sql, result_or_error), ...]
    """
    initial: AgentState = {
        "db_path":      db_path,
        "question":     question,
        "model_fn":     model_fn,
        "max_attempts": max_attempts,
        # Fields below are populated by nodes; set defaults to satisfy TypedDict
        "schema_cache":  {},
        "messages":      [],
        "current_sql":   "",
        "attempts":      0,
        "history":       [],
        "final_result":  None,
        "status":        "running",
    }
    result = _COMPILED_GRAPH.invoke(initial)
    return result
