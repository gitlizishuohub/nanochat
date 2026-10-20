#!/usr/bin/env python3
"""Validate W2 JSONL, retain failed groups, recompute run statistics into CSV."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

KEYS = ('session_id', 'device', 'model_kind', 'source', 'model_tag', 'checkpoint_step', 'seed',
        'mode', 'prompt_sha256', 'prompt_fill_id', 'attention_backend', 'precision', 'checkpoint_sha256', 'prompt_tokens', 'decode_tokens', 'batch_size')


def p95(values):
    """Nearest-rank empirical P95, so n=5 returns maximum (not a stable tail estimate)."""
    return sorted(values)[math.ceil(.95 * len(values)) - 1]


def positive(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def metrics(row):
    if row.get('schema_version') != 1:
        raise ValueError('unsupported or missing schema_version')
    if row.get('status') != 'ok' or row.get('phase') != 'measure':
        return None
    p, g, b = (row[k] for k in ('prompt_tokens', 'decode_tokens', 'batch_size'))
    if min(p, g, b) <= 0 or p + g > row['model_config']['sequence_len']:
        raise ValueError('invalid context or shape')
    ttft, elapsed, steps = row['ttft_ms'], row['elapsed_ms'], row['decode_step_ms']
    count = row['generated_tokens']
    per_row = row['per_row_generated_tokens']
    if row['actual_decode_steps'] != g or len(steps) != g - 1:
        raise ValueError('truncated/extra decode steps marked ok')
    if len(per_row) != b or sum(per_row) != count or any(v != g for v in per_row):
        raise ValueError('incomplete effective generation marked ok')
    if not all(positive(v) for v in [ttft, elapsed, *steps]) or elapsed < ttft:
        raise ValueError('invalid/nonfinite timing')
    if not math.isclose(elapsed, ttft + sum(steps), rel_tol=.03, abs_tol=.5):
        raise ValueError('elapsed disagrees with TTFT plus decode timing')
    return {'ttft_ms': ttft, 'tpot_ms': statistics.mean(steps) if steps else None,
            'tokens_per_s': count / (elapsed / 1000)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    args = p.parse_args()
    groups, seen, audit = defaultdict(list), set(), []
    warmups = defaultdict(list)
    # Read renamed daily artifacts and pre-migration raw evidence without renaming it.
    files = sorted({*args.raw_dir.glob('d[0-9][0-9]_p*_g*_b*.jsonl'),
                    *args.raw_dir.glob('p*_g*_b*.jsonl')})
    if not files:
        p.error('No dNN_p*_g*_b*.jsonl (or legacy p*_g*_b*.jsonl) raw files; pending matrix is not measured data')
    for file in files:
        for line_number, line in enumerate(file.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row.get('run_id')
            if not identity or identity in seen:
                raise ValueError(f'{file}:{line_number}: missing or duplicated run_id')
            seen.add(identity)
            key = tuple(json.dumps(row.get(k), sort_keys=True) for k in KEYS)
            # Different software/hardware/config never merge merely because shape matches.
            key += (json.dumps(row.get('env'), sort_keys=True), json.dumps(row.get('model_config'), sort_keys=True))
            if row.get('phase') == 'warmup':
                warmups[(key, row.get('session_id'))].append(row.get('status') == 'ok' and row.get('schema_version') == 1)
                continue
            reason, values = row.get('status', 'missing_status'), None
            try:
                values = metrics(row)
            except (KeyError, TypeError, ValueError) as error:
                reason = f'invalid_row:{error}'
            groups[key].append((row, values, reason))
            audit.append({'file': str(file.resolve()), 'line': line_number, 'run_id': identity,
                          'included': values is not None, 'reason': 'ok' if values else reason})
    results = []
    for key, entries in groups.items():
        base = {k: json.loads(v) for k, v in zip(KEYS, key)}
        valid = [values for _, values, _ in entries if values is not None]
        sessions = {row.get('session_id') for row, value, _ in entries if value is not None}
        warmup_ok = bool(sessions) and all(session is not None and len(warmups[(key, session)]) >= 3 and all(warmups[(key, session)]) for session in sessions)
        formal = len(valid) >= 5 and warmup_ok and base['device'] == 'cuda' and base['model_kind'] == 'checkpoint'
        base.update(n_total=len(entries), n_valid=len(valid), n_excluded=len(entries) - len(valid),
                    warmup_status='OK' if warmup_ok else 'MISSING_OR_FAILED',
                    conclusion_status='NO_VALID_SAMPLES' if not valid else ('GPU_BASELINE' if formal else ('CPU_OR_RANDOM_CONTRACT' if base['device'] != 'cuda' or base['model_kind'] != 'checkpoint' else ('INSUFFICIENT_WARMUP' if not warmup_ok else 'INSUFFICIENT_SAMPLES'))),
                    excluded_reasons=json.dumps(Counter(reason for _, value, reason in entries if value is None), ensure_ascii=False),
                    env_json=key[-2], model_config_json=key[-1],
                    run_ids=json.dumps([row['run_id'] for row, value, _ in entries if value is not None]))
        for name in ('ttft_ms', 'tpot_ms', 'tokens_per_s'):
            values = [v[name] for v in valid if v[name] is not None]
            for suffix, fn in [('median', statistics.median), ('p95', p95), ('min', min), ('max', max)]:
                base[f'{name}_{suffix}'] = fn(values) if values else ''
        for name in ('peak_allocated_bytes', 'peak_reserved_bytes'):
            values = [row.get(name) for row, value, _ in entries if value is not None and isinstance(row.get(name), (int, float)) and math.isfinite(row[name]) and row[name] >= 0]
            for suffix, fn in [('median', statistics.median), ('min', min), ('max', max)]:
                base[f'{name}_{suffix}'] = fn(values) if values else ''
        results.append(base)
    if not results:
        p.error('Only warmups present; no measured groups')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text('\n'.join(json.dumps(a, ensure_ascii=False) for a in audit) + '\n')
    print(f'{len(results)} groups -> {args.output}; provenance -> {args.audit}')


if __name__ == '__main__':
    main()
