#!/usr/bin/env python3
"""Validate the real tokenizer and optionally one best-fit dataloader batch."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-data', action='store_true', help='reads train+val parquet and validates one real batch')
    a = p.parse_args()
    result = dict(status='error', cases=[])
    try:
        import torch
        from nanochat.tokenizer import get_tokenizer, get_token_bytes, SPECIAL_TOKENS
        tok = get_tokenizer()
        texts = ['Hello nanochat!', '中文与 English 混合。', 'line1\nline2\t  ', '1234567890', '🙂 café', '']
        for text in texts:
            ids = tok.encode(text)
            assert tok.decode(ids) == text, repr(text)
            assert all(0 <= i < tok.get_vocab_size() for i in ids)
            bos_ids = tok.encode(text, prepend=tok.get_bos_token_id())
            assert bos_ids[0] == tok.get_bos_token_id() and tok.decode(bos_ids[1:]) == text
            result['cases'].append(dict(text=text, token_ids=ids, passed=True))
        assert tok.encode(texts) == [tok.encode(t) for t in texts]
        special = {s: tok.encode_special(s) for s in SPECIAL_TOKENS}
        assert len(set(special.values())) == len(SPECIAL_TOKENS)
        byte_lengths = get_token_bytes()
        assert len(byte_lengths) == tok.get_vocab_size()
        assert all(byte_lengths[i].item() == 0 for i in special.values())
        result.update(vocab_size=tok.get_vocab_size(), special_tokens=special, token_bytes_shape=list(byte_lengths.shape))
        if a.check_data:
            from nanochat.dataset import list_parquet_files
            import pyarrow.parquet as pq
            paths = list_parquet_files()
            assert len(paths) >= 2, 'one shard leaves train split empty'
            for path in paths:
                pf = pq.ParquetFile(path)
                assert 'text' in pf.schema_arrow.names and pf.metadata.num_rows > 0, path
                sample = pf.read_row_group(0, columns=['text']).column('text').to_pylist()
                assert sample and all(isinstance(t, str) for t in sample), path
            from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
            for split in ['train', 'val']:
                loader = tokenizing_distributed_data_loader_bos_bestfit(tok, 1, 32, split=split, device='cpu', buffer_size=4)
                x, y = next(loader)
                assert tuple(x.shape) == (1, 32) and x.dtype == torch.long
                assert torch.equal(x[:, 1:], y[:, :-1])
                assert x[0, 0].item() == tok.get_bos_token_id()
                assert x.min() >= 0 and y.min() >= 0 and x.max() < tok.get_vocab_size() and y.max() < tok.get_vocab_size()
                result[split + '_batch'] = dict(shape=list(x.shape), first_ids=x[0, :8].tolist(), passed=True)
        result['status'] = 'ok'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
