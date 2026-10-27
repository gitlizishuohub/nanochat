#!/usr/bin/env python3
"""Reproduce CPU random-model measurements or rebuild statistics from retained raw."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
BENCH = ROOT / '.agents/docs/career_roadmap/w2/scripts/d12_benchmark.py'
AGGREGATE = ROOT / '.agents/docs/career_roadmap/w3/scripts/d13_aggregate.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['measure', 'rebuild'], required=True,
                        help='measure runs CPU inference; rebuild only reads existing raw')
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--summary-dir', type=Path, required=True,
                        help='new output directory; existing targets are never overwritten')
    args = parser.parse_args()
    raw = args.raw_dir.resolve()
    summary = args.summary_dir.resolve()
    if summary.exists():
        parser.error('summary-dir exists; choose a new directory')
    if args.mode == 'measure' and raw.exists():
        parser.error('measure raw-dir exists; choose a new directory to avoid appending')
    if args.mode == 'rebuild' and not raw.is_dir():
        parser.error('rebuild raw-dir must be an existing directory')
    if raw == summary or raw in summary.parents or summary in raw.parents:
        parser.error('raw-dir and summary-dir must be separate, non-nested directories')
    if not AGGREGATE.is_file() or (args.mode == 'measure' and not BENCH.is_file()):
        parser.error('required tutorial script is missing')
    summary.mkdir(parents=True)
    if args.mode == 'measure':
        raw.mkdir(parents=True)
    record = dict(mode=args.mode, python=sys.executable, raw_dir=str(raw),
                  summary_dir=str(summary), purpose='CPU learning reproduction; inspect raw identity',
                  status='started', commands=[])
    manifest = summary / 'd20_reproduction.json'

    def save():
        manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')

    def run(command, name, env=None):
        log = summary / f'd20_{name}.log'
        entry = dict(argv=command, cwd=str(ROOT), log=str(log))
        if env is not None:
            entry['environment_overrides'] = {
                key: env[key] for key in ('NANOCHAT_DTYPE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
            }
        record['commands'].append(entry)
        save()
        print('running:', json.dumps(command, ensure_ascii=False), flush=True)
        try:
            with log.open('w') as stream:
                result = subprocess.run(command, cwd=ROOT, env=env,
                                        stdout=stream, stderr=subprocess.STDOUT)
            entry['returncode'] = result.returncode
        except OSError as error:
            entry.update(returncode=1, error=f'{type(error).__name__}: {error}')
        save()
        print(f'{name}: exit={entry["returncode"]}; log={log}', flush=True)
        return entry['returncode']

    save()
    if args.mode == 'measure':
        env = dict(os.environ, NANOCHAT_DTYPE='float32', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for batch in (1, 4):
            command = [sys.executable, str(BENCH), '--device', 'cpu', '--random-model',
                       '--prompt-tokens', '8', '--decode-tokens', '4', '--batch-size', str(batch),
                       '--warmup', '3', '--repeats', '5', '--seed', '42',
                       '--output', str(raw / f'd20_p8_g4_b{batch}_random_cpu.jsonl')]
            code = run(command, f'b{batch}', env)
            if code:
                record['status'] = 'benchmark_failed'
                save()
                return code if code > 0 else 1
    prefix = 'd20_baseline' if args.mode == 'measure' else 'd20_rebuilt'
    command = [sys.executable, str(AGGREGATE), '--raw-dir', str(raw),
               '--output', str(summary / f'{prefix}.csv'),
               '--audit', str(summary / f'{prefix}_audit.jsonl')]
    code = run(command, 'aggregate')
    record['status'] = 'processes_ok' if code == 0 else 'aggregate_failed'
    save()
    print(f'manifest={manifest}; inspect sample counts, warmup and raw before accepting results')
    return code if code >= 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
