# Colab T4 QLoRA Fine-tuning Bundle

Fine-tune Qwen3.5-4B for NL→SQL on NVIDIA T4 (16 GB VRAM) using the **full** sql-agent dataset (13,015 train / 915 val / 1,029 test). This bundle exists because the M1 MacBook (16 GB unified memory) cannot run full-dataset training at useful sequence lengths — the T4 on Colab provides a dedicated CUDA GPU with bfloat16-free fp16 support.

---

## Prerequisites

- Google Colab account with **T4 GPU runtime** (Runtime → Change runtime type → T4 GPU)
- Google Drive mounted in Colab (`/content/drive/MyDrive/`)
- ~3–4 GB Drive space for data + ~500 MB for adapter checkpoints

---

## Step 1 — Prepare data locally (run once, upload to Drive)

Run on your M1 (uses only the tokenizer, no GPU needed):

```bash
cd /path/to/sql-agent
source .venv/bin/activate
python colab/prepare_data.py \
    --data-dir  data/dataset \
    --out-dir   colab/data \
    --tokenizer ./Qwen3.5-4B \
    --max-seq   1024
```

This writes three files to `colab/data/`:
- `train.prompt.jsonl` — 13,015 examples (minus those > 1024 tokens)
- `val.prompt.jsonl`   — 915 examples
- `test.prompt.jsonl`  — 1,029 examples

**Upload to Drive:**

```
MyDrive/sql-agent-data/
  train.prompt.jsonl
  val.prompt.jsonl
  test.prompt.jsonl
```

**Note on dataset file names:** The task spec referenced `train.full.jsonl`; the actual files are `train.jsonl` / `val.jsonl` / `test.jsonl` with the same counts (13,015 / 915 / 1,029). These ARE the full sets.

---

## Step 2 — Upload scripts to Colab

Upload (or clone from Drive) the `colab/` folder:
- `train_qlora.py`
- `merge_adapter.py`
- `eval_exec.py` (optional)
- `requirements-colab.txt`

Or mount Drive and access them there.

---

## Step 3 — Set T4 runtime

Runtime → Change runtime type → **T4 GPU**. Verify with `!nvidia-smi`.

---

## Step 4 — Install dependencies

> **Qwen3.5 requires `transformers>=5.2.0`** — the `qwen3_5` model type was added in v5.2.0.
> Older transformers (e.g., 4.45.x) fail with: `ValueError: model type 'qwen3_5' not recognized`.

```bash
pip install "transformers>=5.2.0" "peft>=0.19.0" "trl>=1.0.0" \
    "bitsandbytes>=0.49.0" "accelerate>=1.4.0" \
    "datasets>=4.7.0" "tokenizers>=0.21.0" "sentencepiece>=0.2.0" "safetensors>=0.4.5"
```

Or use the requirements file:
```python
!pip install -r requirements-colab.txt
```

**Restart the Colab runtime after installing** (Runtime → Restart session).
Do NOT reinstall torch unless the Colab default causes version conflicts.

---

## Step 5 — Run training

```python
!python train_qlora.py \
    --base-model      Qwen/Qwen3.5-4B \
    --train-file      /content/drive/MyDrive/sql-agent-data/train.prompt.jsonl \
    --val-file        /content/drive/MyDrive/sql-agent-data/val.prompt.jsonl \
    --output-dir      /content/drive/MyDrive/sql-agent-adapters/qwen-sql-qlora \
    --max-seq-length  1024 \
    --num-epochs      2
```

**Key T4 flags (pre-set as defaults):**
- `fp16=True, bf16=False` — T4 (CC 7.5) lacks bfloat16
- `per_device_train_batch_size=2, gradient_accumulation_steps=8` → effective batch 16
- `packing=False` — required for completion-only loss masking

**Estimated wall time (T4, seq=1024, ~12k train examples, 2 epochs):** ~3–4 hours.

**Colab disconnect caveat:** Colab sessions disconnect after ~90 min of inactivity (Pro: ~24h). The adapter checkpoints are saved every 200 steps to Drive. On reconnect:

```python
!python train_qlora.py ... --resume-from-checkpoint latest
```

The `--output-dir` on Drive ensures checkpoints survive the VM reset.

---

## Step 6 — Merge adapter (optional, for full-model export)

```python
!python merge_adapter.py \
    --base-model  Qwen/Qwen3.5-4B \
    --adapter-dir /content/drive/MyDrive/sql-agent-adapters/qwen-sql-qlora/final_adapter \
    --output-dir  /content/drive/MyDrive/sql-agent-models/qwen-sql-merged
```

Peak RAM during merge: ~16 GB (runs on CPU by default). Use Colab High-RAM runtime if needed, or merge locally.

---

## Step 7 — Evaluate (optional)

Requires Spider/BIRD SQLite DB files on Drive.

```
MyDrive/spider_data/database/<db_id>/<db_id>.sqlite
```

```python
!python eval_exec.py \
    --test-file   /content/drive/MyDrive/sql-agent-data/test.prompt.jsonl \
    --test-src    /content/drive/MyDrive/sql-agent-data-src/test.jsonl \
    --db-root     /content/drive/MyDrive/spider_data/database \
    --base-model  Qwen/Qwen3.5-4B \
    --adapter-dir /content/drive/MyDrive/sql-agent-adapters/qwen-sql-qlora/final_adapter
```

---

## Step 8 — Use adapter in project inference

Bring the adapter back into the M1 project by downloading it from Drive to:

```
sql-agent/adapters/qwen-sql-qlora/final_adapter/
```

Then in `src/qwen_model_fn.py`:

```python
model_fn = make_qwen_model_fn(adapter_path="adapters/qwen-sql-qlora/final_adapter")
```

**Inference format note:** `qwen_model_fn.py` currently appends `"<think></think>"` as a bare string prefill after the template. Training uses `enable_thinking=False` which renders `<think>\n\n</think>\n\n`. For best adapter performance, reconcile by updating `qwen_model_fn.py` to use `enable_thinking=False` in its `apply_chat_template` call (removing the manual prefill) — this exactly matches the training distribution.

---

## Sequence Length Guide

| max_seq | % train kept | Effective train set | VRAM usage (T4) |
|---------|--------------|---------------------|-----------------|
| 1024    | see prepare_data stats | ~12k+ | safe |
| 1536    | higher keep rate | ~13k+ | moderate |
| 2048    | near 100% | ~13k | may OOM at batch=2 |

If using seq=2048: reduce `--per-device-batch-size 1` and increase `--gradient-accumulation-steps 16`.

---

## File Reference

| File | Purpose |
|------|---------|
| `prepare_data.py` | Local data prep — renders prompt/completion JSONL |
| `train_qlora.py` | QLoRA training — CUDA, fp16, T4-tuned defaults |
| `merge_adapter.py` | Merge LoRA adapter into base model |
| `eval_exec.py` | Execution-accuracy eval against SQLite DBs |
| `requirements-colab.txt` | Pinned dependencies |
| `Qwen_SQL_QLoRA_T4.ipynb` | All-in-one Colab notebook |
| `data/` | Output of prepare_data (not committed — upload to Drive) |
