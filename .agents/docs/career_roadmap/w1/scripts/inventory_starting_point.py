#!/usr/bin/env python3
"""Locate MiniMind baseline evidence and summarize CSV facts, without rerunning it."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
FILES = ('README.md', 'reports/reproduction.md', 'results/tables/kv_cache.csv', 'results/tables/d14_clean_kv_cache.csv', 'results/tables/generation_benchmark_raw.csv', 'results/tables/d14_clean_benchmark_raw.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minimind-root', type=Path, default=ROOT.parent / 'minimind')
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/inference_infra/results/w1/starting_point_inventory.json')
    args = parser.parse_args()
    candidates = [args.minimind_root / 'experiments/minimind_baseline']
    archive = args.minimind_root / 'experiments/archive'
    if archive.exists():
        candidates.extend(sorted(archive.glob('minimind_baseline*')))
    result = {'minimind_root': str(args.minimind_root), 'candidates': [], 'interpretation': 'File existence and recorded values only; not a new correctness run.'}
    for candidate in candidates:
        record = {'directory': str(candidate), 'files': []}
        for relative in FILES:
            path = candidate / relative
            entry = {'relative_path': relative, 'exists': path.is_file()}
            if path.is_file():
                entry['bytes'] = path.stat().st_size
                entry['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                if path.suffix == '.csv':
                    with path.open(newline='') as stream:
                        reader = csv.DictReader(stream)
                        rows = list(reader)
                        entry['columns'] = reader.fieldnames
                    entry['rows'] = len(rows)
                    if rows and 'max_abs_logit_diff' in rows[0]:
                        entry['max_abs_logit_diff'] = max(float(r['max_abs_logit_diff']) for r in rows)
                    if rows and 'repetition' in rows[0]:
                        groups = {}
                        for row in rows:
                            key = '|'.join(row.get(k, '') for k in ('prompt_tokens', 'generated_tokens', 'batch', 'cache', 'device', 'dtype'))
                            groups.setdefault(key, []).append(row['repetition'])
                        entry['repetitions_per_group'] = groups
            record['files'].append(entry)
        result['candidates'].append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if any(f['exists'] for c in result['candidates'] for f in c['files']) else 2


if __name__ == '__main__':
    raise SystemExit(main())
