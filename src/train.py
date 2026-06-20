"""
CP5 — QLoRA fine-tune launcher for Qwen 4-bit via mlx-lm.

Usage (smoke test):
    python src/train.py --smoke

Usage (full run):
    python src/train.py

Wraps mlx_lm lora with project-specific hyperparameters.
Designed for Apple Silicon 16GB M1 with models/qwen-4bit (4-bit, ~4B params).

Hyperparameter decisions:
  --max-seq-length 2048   covers 85% of train examples (38% would be cut at 1024)
  --batch-size 1          required for 16GB with 2048 seq-len + grad-checkpoint
  --num-layers 8          tune 8/28 transformer layers (conservative for memory)
  --learning-rate 1e-4    standard LoRA rate
  --iters 1200            ~1 epoch equivalent at batch=1 over 13k examples
  --grad-checkpoint       halves activation memory
  --mask-prompt           loss only on final assistant turn (target SQL)
"""

import subprocess
import sys
import pathlib
import argparse

ROOT = pathlib.Path(__file__).parent.parent


def build_cmd(
    smoke: bool = False,
    data_dir: str | None = None,
    iters: int | None = None,
) -> list[str]:
    base = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", str(ROOT / "models" / "qwen-4bit"),
        "--train",
        "--data", data_dir or str(ROOT / "data" / "mlx"),
        "--batch-size", "1",
        "--num-layers", "8",
        "--learning-rate", "1e-4",
        "--max-seq-length", "2048",
        "--mask-prompt",
        "--grad-checkpoint",
        "--adapter-path", str(ROOT / "adapters" / "qwen-sql"),
        "--steps-per-report", "10",
        "--steps-per-eval", "200",
        "--save-every", "200",
        "--val-batches", "-1" if not smoke else "5",
    ]
    if smoke:
        base += ["--iters", "20"]
    else:
        base += ["--iters", str(iters or 1200)]
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="20-iter smoke test on smoke.jsonl")
    parser.add_argument("--iters", type=int, default=1200)
    args = parser.parse_args()

    if args.smoke:
        # Use smoke subset
        smoke_dir = ROOT / "data" / "mlx" / "smoke_dir"
        smoke_dir.mkdir(exist_ok=True)
        # Symlink or copy smoke.jsonl as train.jsonl + valid.jsonl
        import shutil
        src = ROOT / "data" / "mlx" / "smoke.jsonl"
        for name in ("train.jsonl", "valid.jsonl"):
            dst = smoke_dir / name
            if dst.exists():
                dst.unlink()
            shutil.copy(src, dst)
        cmd = build_cmd(smoke=True, data_dir=str(smoke_dir))
        print("SMOKE TEST — 20 iters on 200-line subset")
    else:
        cmd = build_cmd(smoke=False, iters=args.iters)
        print(f"FULL TRAIN — {args.iters} iters on data/mlx/")

    print("CMD:", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
