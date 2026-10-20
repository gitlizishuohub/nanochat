#!/usr/bin/env python3
"""Capture pytest stdout, stderr, exact argv and exit code using the current Python."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[5]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'experiments/inference_infra/results/w1/d03_pytest')
    parser.add_argument('--select', default=None, help='Optional pytest -k expression, for a targeted rerun')
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output_dir / stamp
    output.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, '-m', 'pytest', '-q', 'tests/test_engine.py', 'tests/test_attention_fallback.py', '-p', 'no:cacheprovider']
    if args.select:
        argv += ['-k', args.select]
    env = dict(os.environ, NANOCHAT_DTYPE='float32', CUDA_VISIBLE_DEVICES='', PYTHONPYCACHEPREFIX=os.environ.get('PYTHONPYCACHEPREFIX', '/tmp/nanochat_w1_pycache'))
    start = time.perf_counter()
    with (output / 'd03_stdout.log').open('w') as stdout, (output / 'd03_stderr.log').open('w') as stderr:
        result = subprocess.run(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    metadata = {'argv': argv, 'cwd': str(ROOT), 'exit_code': result.returncode, 'elapsed_seconds': time.perf_counter() - start, 'NANOCHAT_DTYPE': 'float32', 'CUDA_VISIBLE_DEVICES': '', 'status': 'PASS' if result.returncode == 0 else 'FAIL'}
    (output / 'd03_run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))
    print(f'Logs: {output}')
    print((output / 'd03_stdout.log').read_text())
    if result.returncode:
        print((output / 'd03_stderr.log').read_text(), file=sys.stderr)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
