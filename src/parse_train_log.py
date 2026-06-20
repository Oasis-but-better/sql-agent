"""
CP5 — Parse train.log and emit live metrics to docs/train_live.json.

Runs as a background loop while training is active.
Reads train.log every 30s, parses mlx-lm loss lines, writes latest metrics.

mlx-lm 0.31.3 log format (observed):
  Iter N: Train loss X.XXX, Learning Rate Y.Ye-0Z, It/sec W.WWW
  Val loss X.XXX, Val took M.M sec

Usage:
    python src/parse_train_log.py [--log train.log] [--out docs/train_live.json] [--interval 30]
"""

import re
import json
import time
import pathlib
import argparse
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).parent.parent

TRAIN_RE = re.compile(
    r"Iter\s+(\d+):\s+Train loss\s+([\d.]+)"
)
VAL_RE = re.compile(
    r"Val loss\s+([\d.]+)"
)


def parse_log(log_path: pathlib.Path) -> dict:
    """Return latest metrics from log file."""
    result = {"iter": None, "train_loss": None, "val_loss": None, "ts": None}
    last_val_loss = None
    last_iter = None
    last_train_loss = None

    try:
        text = log_path.read_text(errors="replace")
    except FileNotFoundError:
        return result

    lines = text.splitlines()
    for line in lines:
        m = TRAIN_RE.search(line)
        if m:
            last_iter = int(m.group(1))
            last_train_loss = float(m.group(2))
        m = VAL_RE.search(line)
        if m:
            last_val_loss = float(m.group(1))

    result["iter"] = last_iter
    result["train_loss"] = last_train_loss
    result["val_loss"] = last_val_loss
    result["ts"] = datetime.now(timezone.utc).isoformat()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=str(ROOT / "train.log"))
    parser.add_argument("--out", default=str(ROOT / "docs" / "train_live.json"))
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--once", action="store_true", help="Parse once and exit")
    args = parser.parse_args()

    log_path = pathlib.Path(args.log)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Watching {log_path} → {out_path} every {args.interval}s")

    while True:
        metrics = parse_log(log_path)
        out_path.write_text(json.dumps(metrics, indent=2))
        print(f"[{metrics['ts']}] iter={metrics['iter']} train_loss={metrics['train_loss']} val_loss={metrics['val_loss']}")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
