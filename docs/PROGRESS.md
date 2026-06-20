# SQL-Agent — Execution Progress Ledger

Durable checkpoint log. Each agent appends its outcome here.

## CP0 — Data pipeline code ✅ 2026-06-20
- 9-task TDD build complete. 59 tests pass, 0 fail. 6 src modules + 7 test modules. 10 commits. CONTEXT.md present.
- Modules: schema_cache, verify, gold_ingest, diversity, generate_examples, build_dataset.

## CP1 — Clean set ✅ 2026-06-20
- total in: 10,193 (Spider train_spider 7,000 + train_others 1,659 + BIRD dev 1,534)
- valid out: 10,071 (98.8%)
- discarded: 122 (1.2%)
- distinct DBs: 155
- discard categories: schema_compile_error 106 (reserved-keyword table name in PRAGMA), timeout 13, other_error 2, no_such_table 1
- per-source: spider 8,652/8,659 (99.9%), bird 1,419/1,534 (92.5%)
- clean.jsonl: data/dataset/clean.jsonl — 10,071 lines
- runner: scripts/build_clean_set.py

## CP2 — Correction set ✅ 2026-06-20
- total corrections: 6,000 (all 5-turn chains: system→user→assistant[wrong]→tool[real error/note]→assistant[gold])
- bootstrap corrections: 105 (base-model Qwen 4-bit)
- perturbation corrections: 5,895 (programmatic gold-SQL perturbation)
- dedup: 6,000 unique (question+gold_sql) keys — zero collisions; bootstrap/perturbation cover disjoint pairs
- distinct db_ids: 155
- tool-message kinds: real sqlite3 errors 3,409 / "incorrect result" mismatch notes 2,591 (all captured from execution, none invented)
- HYBRID decision: bootstrap NOT skipped (sec/gen=5.461 < 20s threshold)
- model first-try failure rate: 31.9% (329 attempts, 224 correct, 105 wrong → corrections)
- sec/gen (full-run, post chat-template prefill): 5.461
- bootstrap wall-time: 1,648.1s (~27.5min, under 30min budget; target 329 gens)
- prompt: chat-template + prefilled empty `<think></think>` to skip reasoning tokens (raw prompt produced reasoning prose, not SQL)
- perturbations: drop WHERE / wrong column / drop JOIN / drop GROUP BY / drop HAVING / drop LIMIT — kept only when genuinely error or mismatch vs gold
- correction.jsonl: data/dataset/correction.jsonl — 6,000 lines
- runner: scripts/build_correction_set.py (resumable: skips already-written keys)

## CP3 — Dataset assembly ✅ 2026-06-20
- total_input: 15,000 (clean_sub=9,000 + correction=6,000)
- dropped_duplicates (within-type dedup): 41 clean + 0 correction = 41 total
- total_after_dedup: 14,959
- train: 13,015 | val: 915 | test: 1,029
- train clean/correction: 7,776/5,239 = 59.7%/40.3% (target 60/40)
- split strategy: db-disjoint (val + test each hold out 8 distinct db_ids unseen in train)
- test db_ids (8): academic, club_1, customers_and_invoices, european_football_2, journal_committee, performance_attendance, school_player, thrombosis_prediction
- val db_ids (8): activity_1, codebase_community, customers_and_products_contacts, farm, loan_1, perpetrator, scientist_1, toxicology
- disjoint verified: train∩val=0, train∩test=0, val∩test=0
- all lines valid JSON, all lines have 'messages' key
- outputs: data/dataset/train.jsonl, val.jsonl, test.jsonl
- seed=42 for clean subsample (deterministic/idempotent)

## CP4 — Quantize Qwen 4-bit MLX ✅ 2026-06-20
- mlx-lm version: 0.31.3 (mlx 0.31.2)
- quantized path: models/qwen-4bit (gitignored — not committed)
- effective bits: 4.503 bits/weight
- quantized model size: 2.2G
- load OK: yes
- sample generation: prompt=`SELECT 1;` → `-- 1. 创建表` (8 tokens, model responded in Chinese SQL comment style)
- .gitignore created; models/ and Qwen3.5-4B/ excluded from git

## CP5 — QLoRA fine-tune
- status: pending

## CP6 — Phase-B agent harness ✅ 2026-06-20
- PRAGMA reserved-keyword fix: double-quoted identifiers in table_info, foreign_key_list, SELECT * — all 3 sites fixed in src/schema_cache.py
- PRAGMA regression test: tests/test_schema_cache_reserved.py — 2 tests, cover table named `order`
- LangGraph 1.2.6 agent: src/agent/__init__.py + src/agent/loop.py
- State machine: Compile → Draft → Execute → Loop router (success→END, exhausted→END, retry→Draft)
- model_fn injected dependency; Execute uses run_query_with_timeout (no gold required)
- Feedback injection: assistant(wrong SQL) + tool(error/empty note) on retry — mirrors training chains
- Hard cap: 3 attempts; status ∈ {"success", "exhausted"}
- Tests: tests/test_agent_loop.py — 4 tests (success@1, success@2 w/ error injection, exhausted@3, empty-result retry)
- Full suite: 65 tests pass, 0 fail

## CP7 — Evaluation
- status: pending
