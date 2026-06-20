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
