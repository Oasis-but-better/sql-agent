# SQL-Agent — End-to-End Project Design

**Date:** 2026-06-20
**Status:** APPROVED 2026-06-20 — all open decisions resolved to recommended options. Implementation authorized (generation run, fine-tuning, harness).

**Locked decisions:** OD1 gold-only clean set · OD2 base-model bootstrapped corrections · OD3 **Qwen3.5-4B runs the entire pipeline; gemma-4-E4B is a fallback, attempted only if Qwen performance stagnates and can no longer improve** · OD4 BIRD dev-only · OD5 15k target.
**Sibling project:** `llm-training` (nanoSQL) — single-schema from-scratch text-to-SQL, build complete. This project (`sql-agent`) is its generalization.

---

## 1. Purpose & Premise

Build a **dynamic, schema-agnostic natural-language-to-SQL system**: no database schema is memorized in model weights. At runtime an arbitrary `.sqlite` is dropped in, its schema is compiled into a compact in-prompt cache, and a fine-tuned small language model (SLM) drafts SQL inside an **execution-driven self-correction loop** — it sees real SQLite errors/empty results and repairs its own query.

Contrast with sibling nanoSQL: that project proved a tiny model can *memorize* one schema (Chinook). This project drops memorization entirely and teaches **(a)** generalize to unseen schemas given in-context, and **(b)** self-correct from execution feedback.

Runs locally on Apple Silicon M1 Pro, 16 GB RAM.

---

## 2. System Architecture

Two phases, matching the original spec.

### Phase A — Cold-Start Schema Compiler
1. User drops a `.sqlite` file.
2. PRAGMA queries extract: DDL per table, foreign-key graph, 3 sample rows/table.
3. An SLM compresses the raw DDL into a compact, semantically-meaningful `db_metadata_cache.json` (minified JSON: table/column names + types, FK edges, sample rows). This is a **semantic cache** computed once per DB, reused across every question.

### Phase B — Runtime Self-Correction Loop (LangGraph)
State-machine nodes: **Compile → Draft → Execute → Loop**.
- Prompt = frozen `db_metadata_cache.json` + user question → model generates candidate SQL.
- **Execution guardrail** runs the query against the live SQLite:
  - **Success (rows returned):** return rows, clear state.
  - **Syntax/runtime error:** inject the *exact* `sqlite3` traceback into context, retry.
  - **Empty result:** inject an "empty result" warning, retry.
- **Hard stop at 3 iterations.**

The dataset (§4–§6) exists to teach the model to behave well inside this exact loop.

---

## 3. Models

| Role | Model | Notes |
|---|---|---|
| **Full pipeline** | `Qwen3.5-4B` | 8.7 GB bf16 downloaded. ~4B params → QLoRA fits 16 GB M1 comfortably. Runs entire pipeline: data-gen bootstrap, fine-tune, agent loop, eval. |
| **Fallback** | `gemma-4-E4B-it` | 15 GB bf16 (~7.5B real params via MatFormer). Attempted **only if** Qwen performance plateaus and cannot improve further. QLoRA on 16 GB M1 is tight — may OOM. |

- **Quantization:** bf16 → **4-bit MLX** (`mlx_lm.convert -q`) before both inference and QLoRA training.
- **Inference backend:** MLX (Metal). Ollama removed — not used in the training/inference path (GGUF incompatible with `mlx-lm.lora`).

---

## 4. Dataset Design

**Target:** ~15k verified examples, **60% clean / 40% self-correction**, spanning Spider (~200 DBs) + BIRD dev (~11 DBs) for schema diversity.

### 4.1 Storage format — model-agnostic
Examples are stored as **message lists** (role + content), NOT pre-rendered ChatML. Qwen and gemma use different chat templates (`<|im_start|>` vs `<start_of_turn>`); the correct template is rendered **per-model at train time**. This avoids a costly retrofit.

```jsonc
{
  "db_id": "concert_singer",
  "source": "spider",
  "difficulty": "medium",
  "type": "clean",                 // or "correction"
  "query_types": ["join", "group_by"],
  "messages": [
    {"role": "system",  "content": "<db_metadata_cache.json for this DB>"},
    {"role": "user",    "content": "<natural language question>"},
    {"role": "assistant","content": "<SQL>"}
    // correction type adds: assistant(wrong SQL) → tool(real traceback) → assistant(fixed SQL)
  ]
}
```

### 4.2 Clean examples (60%, ~9k) — from GOLD, not generated
Spider (~7k train, `train_spider.json` + `train_others.json`) and BIRD dev (`dev.json`) ship **human-annotated, verified `question → gold_SQL → db_id` triples** — all gold files confirmed on disk. These are reused directly:
1. Take each gold pair.
2. Build the `db_metadata_cache.json` for its DB (Phase-A compiler).
3. Assemble the message list (schema → question → gold SQL).
4. **Execution-verify** (§5) — discard pairs whose gold SQL fails to execute on the shipped `.sqlite` (a known Spider data-quality issue).

No LLM generation for the clean set → far lower token cost, higher quality than synthesizing. Optional light **NL paraphrase augmentation** (haiku) may expand question diversity over the same gold SQL — listed as open decision.

### 4.3 Correction examples (40%, ~6k) — the real generation work
Gold gives correct SQL, not `wrong → traceback → fix` chains. Those are generated. **Source of the "wrong attempt"** is the key design choice (open decision OD2), with three strategies (recommend hybrid):

- **(A) Base-model bootstrapping (recommended core):** run the *actual model being fine-tuned* (quantized Qwen/gemma) zero-shot over gold questions; capture its **real** wrong SQL + the **real** `sqlite3` traceback; gold SQL = the fix. Teaches the model to repair *its own* error distribution — proper self-correction. Costs inference infra.
- **(B) Programmatic perturbation:** mutate gold SQL (drop a join, wrong column, bad GROUP BY) → execute → capture real traceback. Cheap, broad coverage, but off-distribution errors.
- **(C) Sonnet-authored:** sonnet writes a plausible wrong attempt. Most expensive, least faithful to model's real failures.

**Two-tier wave roles (refined):**
- **Haiku waves:** high-volume cheap work — programmatic-perturbation candidates, NL paraphrase augmentation, bulk first-pass correction candidates over easy/medium.
- **Sonnet waves:** hard/extra-difficulty corrections, cases where the fix needs genuine reasoning, and synthesis of natural correction phrasing.
- Every wave output is **execution-verified** before acceptance.

---

## 5. Verification Harness

Every example (clean and correction target) is validated by **executing against the real `.sqlite`** — reusing logic from sibling `llm-training/src/evaluate.py` (strict / superset / order-tolerant / id-tolerant match).

- **Clean valid iff:** SQL executes without error **AND result-set matches the gold execution** (strict or superset). *Not* "non-empty" — that wrongly rejects correct queries returning zero rows.
- **Correction valid iff:** turn-1 SQL **genuinely** errors or returns wrong/empty results (real captured traceback — never invented) **AND** the final fixed SQL's result-set matches gold.
- **Dedup:** by normalized (SQL + question) hash.
- **Per-query timeout** (reuse `validate_data.py` patterns) to guard runaway queries.

---

## 6. Diversity & Convergence

The generation loop runs waves, verifies, accumulates — and tracks coverage to guarantee a **dense + diverse** dataset:

- **Per-DB coverage** — minimum N examples per database; backfill under-represented DBs.
- **Query-type histogram** — joins, aggregates, subqueries, window functions, GROUP BY, ORDER BY, nested, LIMIT. Generate into under-filled buckets.
- **Difficulty balance** — easy/medium/hard/extra distribution targets.

**Convergence criterion:** stop when **≥15k verified examples AND** all diversity thresholds met (min/DB, balanced difficulty, every query-type bucket populated). Report any bucket left short rather than silently truncating.

---

## 7. Fine-Tuning

- **Framework:** `mlx-lm.lora` (QLoRA) on 4-bit quantized base.
- **Templating:** render stored message lists into each model's native chat template at load.
- **Loss masking:** `--mask-prompt` — loss computed only on **assistant SQL spans**, not schema/question/traceback.
- **Split:** train / val / **frozen test** (held out, never seen in generation tuning).
- Train Qwen first (primary); attempt gemma only if it fits memory.

---

## 8. Evaluation

Reuse sibling nanoSQL evaluation methodology:

- **Execution accuracy** — strict-match + superset-match rates on the frozen test set.
- **Loop resolution rate** — % of cases failing turn-1 that are fixed on turn-2/turn-3 (measures the self-correction skill, this project's differentiator).
- **TTFT** — time-to-first-token / latency: compiled `db_metadata_cache.json` injection vs raw DDL injection (quantifies the Phase-A cache payoff).
- Baseline: base model zero-shot vs fine-tuned, on unseen-schema test DBs.

---

## 9. Components & Layout

```
sql-agent/
├── data/
│   ├── spider/spider_data/      # 744 .sqlite DBs (database/ + test_database/) + gold JSON (train_spider/train_others/dev/tables.json)
│   ├── bird/dev_20240627/       # 11 dev .sqlite DBs + dev.json gold
│   └── dataset/                 # train.jsonl / val.jsonl / test.jsonl (message-list format)
├── src/
│   ├── schema_cache.py          # Phase-A compiler: .sqlite → db_metadata_cache.json
│   ├── gold_ingest.py           # gold pairs → message lists (clean set)
│   ├── generate_examples.py     # haiku/sonnet wave generation (correction set)
│   ├── verify.py                # execution verification + traceback capture (from evaluate.py)
│   ├── diversity.py             # coverage tracking + convergence check
│   ├── build_dataset.py         # orchestrator: waves → verify → dedup → split → emit
│   ├── train.py                 # mlx-lm.lora QLoRA wrapper (per-model templating, mask-prompt)
│   ├── evaluate.py              # exec accuracy, loop-resolution, TTFT
│   └── agent/                   # Phase-B LangGraph runtime loop (Compile→Draft→Execute→Loop)
├── Qwen3.5-4B/                  # bf16 weights (downloaded)
├── gemma-4-E4B-it/             # bf16 weights (downloaded)
└── docs/                        # this spec + CONTEXT.md (to scaffold at impl start)
```

Generation orchestration (waves + verification loop) implemented via the multi-agent **Workflow** tool.

---

## 10. Open Decisions (surface in HTML for user)

- **OD1 — Clean-set generation:** gold-reformat only (recommended) vs gold + haiku NL-paraphrase augmentation for extra diversity.
- **OD2 — Correction "wrong attempt" source:** base-model bootstrapping (A, recommended) vs programmatic perturbation (B) vs sonnet-authored (C) vs hybrid A+B.
- **OD3 — Primary model:** Qwen3.5-4B primary + gemma stretch (recommended) vs attempt both equally.
- **OD4 — BIRD scope:** dev-only ~11 DBs (recommended, disk) vs add BIRD train (~69 DBs, multi-GB).
- **OD5 — Dataset size:** hold 15k vs scale to gold availability (~16k clean gold reachable; 40% correction = ~6k to generate).

---

## 11. Risks

- **Disk:** BIRD + 15k dataset on a tight disk — mitigated by ollama wipe (~27 GB reclaimed) + BIRD-dev-only scope.
- **gemma OOM:** ~7.5B params QLoRA on 16 GB M1 may not fit — Qwen is the fallback primary.
- **Generation token cost:** correction-set generation (~6k via haiku/sonnet waves) is the main spend — gated behind HTML approval.
- **Spider gold quality:** some gold SQL doesn't execute cleanly — verification filters these (expect some yield loss).
- **Base-model bootstrap infra:** strategy (A) needs the quantized models running inference at scale during generation.

---

## 12. Milestones

1. **Setup** — ✅ ollama wiped (45 GB free), ✅ Spider (744 DBs) + BIRD dev (11 DBs) downloaded with gold JSON. Remaining: create `sql-agent/.venv` (currently borrowing `llm-training/.venv`), scaffold `CONTEXT.md`.
2. **Phase-A compiler** — `schema_cache.py` + `db_metadata_cache.json` per DB.
3. **Clean set** — `gold_ingest.py` + verification → ~9k verified clean examples.
4. **Correction set** — `generate_examples.py` waves + verification → ~6k correction chains. *(largest token spend — post-approval)*
5. **Dataset assembly** — dedup, diversity-convergence, train/val/test split.
6. **Fine-tune** — QLoRA Qwen (primary), gemma (stretch). *(post-approval)*
7. **Phase-B agent** — LangGraph runtime loop. *(post-approval / "harness build")*
8. **Evaluation** — exec accuracy, loop-resolution, TTFT vs baseline.

---

## 13. Approval Gate

Per user: converge this spec → render a **high-level HTML overview** (broad architecture + work plan + open decisions, not low-level detail) → **wait for explicit approval** before implementing the data-generation run, fine-tuning, and Phase-B harness.
