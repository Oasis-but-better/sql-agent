"""
CP5 — Prepare mlx-lm training data.

Reads data/dataset/{train,val}.jsonl, strips non-messages fields,
writes data/mlx/{train,valid}.jsonl (each line = {"messages": [...]}).

Role mapping:
  system / user / assistant  → kept as-is (Qwen template handles them)
  tool                       → kept as-is (Qwen maps to <tool_response> user block)

mlx-lm 0.31.3 ChatDataset reads only the "messages" key; extra keys are ignored
but we strip them anyway for cleanliness.
"""

import json
import sys
import pathlib


def convert(src_path: pathlib.Path, dst_path: pathlib.Path) -> tuple[int, int]:
    """Return (written, skipped) counts."""
    written = skipped = 0
    with src_path.open() as fin, dst_path.open("w") as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARN line {lineno}: JSON error {e}", file=sys.stderr)
                skipped += 1
                continue
            msgs = ex.get("messages")
            if not msgs:
                print(f"  WARN line {lineno}: no messages key", file=sys.stderr)
                skipped += 1
                continue
            out = {"messages": msgs}
            fout.write(json.dumps(out) + "\n")
            written += 1
    return written, skipped


def main():
    root = pathlib.Path(__file__).parent.parent
    src_dir = root / "data" / "dataset"
    dst_dir = root / "data" / "mlx"
    dst_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        (src_dir / "train.jsonl", dst_dir / "train.jsonl"),
        (src_dir / "val.jsonl", dst_dir / "valid.jsonl"),
        (src_dir / "test.jsonl", dst_dir / "test.jsonl"),
    ]

    for src, dst in pairs:
        if not src.exists():
            print(f"SKIP {src} — not found")
            continue
        written, skipped = convert(src, dst)
        print(f"{src.name} → {dst.name}: written={written}, skipped={skipped}")


if __name__ == "__main__":
    main()
