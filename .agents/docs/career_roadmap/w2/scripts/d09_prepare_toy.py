#!/usr/bin/env python3
"""Create a fresh, isolated synthetic tokenizer+parquet set for training smoke only."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="must not exist; no existing tokenizer is overwritten",
    )
    a = p.parse_args()
    if a.base_dir.exists():
        p.error("--base-dir must be a fresh path")
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS

    train = [
        f"Train sample {i}. Nanochat inference cache and model shape. 中文测试 {i % 31}.\n"
        for i in range(1200)
    ]
    val = [
        f"Validation document {i}. Check token round trip and batch shapes. 验证文本 {i % 17}.\n"
        for i in range(1200)
    ]
    tokenizer = RustBPETokenizer.train_from_iterator(iter(train), vocab_size=512)
    data = a.base_dir / "base_data_climbmix"
    data.mkdir(parents=True)
    tokenizer.save(a.base_dir / "tokenizer")
    special_ids = {tokenizer.encode_special(s) for s in SPECIAL_TOKENS}
    lengths = [
        0 if i in special_ids else len(tokenizer.decode_single_token_bytes(i))
        for i in range(tokenizer.get_vocab_size())
    ]
    torch.save(
        torch.tensor(lengths, dtype=torch.int32),
        a.base_dir / "tokenizer/token_bytes.pt",
    )
    for i, texts in enumerate([train, val]):
        pq.write_table(
            pa.table({"text": texts}),
            data / f"d09_shard_{i:05d}.parquet",
            row_group_size=128,
        )
    (a.base_dir / "d09_synthetic_provenance.json").write_text(
        json.dumps(
            dict(
                kind="synthetic_smoke_only",
                train_rows=len(train),
                val_rows=len(val),
                vocab_size=tokenizer.get_vocab_size(),
                quality_claim=False,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        f"Created isolated synthetic smoke data at {a.base_dir}; do not report model quality or real-corpus performance."
    )


if __name__ == "__main__":
    main()
