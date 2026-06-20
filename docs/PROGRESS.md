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

## CP3 — Dataset assembly
- status: pending

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

## CP6 — Phase-B agent harness
- status: pending

## CP7 — Evaluation
- status: pending
