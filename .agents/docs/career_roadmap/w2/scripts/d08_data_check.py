#!/usr/bin/env python3
"""Read-only prerequisite inventory; does not import torch or unpickle files."""

import argparse
import importlib.util
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-dir",
        type=Path,
        default=Path(
            os.environ.get("NANOCHAT_BASE_DIR", str(Path.home() / ".cache/nanochat"))
        ),
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    base = a.base_dir.expanduser().resolve()
    data = base / "base_data_climbmix"
    if not data.exists():
        data = base / "base_data"
    shards = sorted(data.glob("*.parquet")) if data.exists() else []
    files = [base / "tokenizer/tokenizer.pkl", base / "tokenizer/token_bytes.pt"]
    report = dict(
        base_dir=str(base),
        data_dir=str(data),
        split_rule="sorted parquet files: last=val, all preceding=train",
        files={
            str(f): dict(
                exists=f.is_file(), bytes=f.stat().st_size if f.is_file() else None
            )
            for f in files
        },
        parquet_files=[str(f) for f in shards],
        train_shards=max(0, len(shards) - 1),
        val_shards=min(1, len(shards)),
        dependencies={
            m: importlib.util.find_spec(m) is not None
            for m in ["torch", "rustbpe", "tiktoken", "pyarrow"]
        },
        content_validated=False,
    )
    report["ready_for_content_check"] = (
        len(shards) >= 2
        and all(f.is_file() for f in files)
        and all(report["dependencies"].values())
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_content_check"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
