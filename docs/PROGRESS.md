# SQL-Agent — Execution Progress Ledger

Durable checkpoint log. Each agent appends its outcome here.

## CP0 — Data pipeline code ✅ 2026-06-20
- 9-task TDD build complete. 59 tests pass, 0 fail. 6 src modules + 7 test modules. 10 commits. CONTEXT.md present.
- Modules: schema_cache, verify, gold_ingest, diversity, generate_examples, build_dataset.

## CP1 — Clean set
- status: IN PROGRESS

## CP2 — Correction set
- status: pending

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
