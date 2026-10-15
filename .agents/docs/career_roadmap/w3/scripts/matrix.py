#!/usr/bin/env python3
"""Freeze P/G/B cases and explicitly run a chosen subset with W2 benchmark."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--meta', type=Path, required=True, help='Exact meta_NNNNNN.json from chosen checkpoint')
    p.add_argument('--model-tag', required=True)
    p.add_argument('--step', required=True, type=int)
    p.add_argument('--source', choices=['base'], default='base')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch', type=int, choices=[1, 4], help='Only execute this B, still preserve all cases')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--execute', action='store_true', help='Actually invoke benchmark; default only plans')
    p.add_argument('--repeats', type=int, default=5)
    args = p.parse_args()
    if args.repeats < 5 or args.step < 0:
        p.error('At least 5 repetitions and a nonnegative step are required')
    meta = json.loads(args.meta.read_text())
    config = meta['model_config']
    limit = int(config['sequence_len'])
    if limit <= 0:
        p.error('sequence_len must be positive')
    root = Path(__file__).resolve().parents[5]
    bench = root / '.agents/docs/career_roadmap/w2/scripts/benchmark.py'
    if not bench.is_file():
        p.error(f'Benchmark missing: {bench}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / 'matrix.jsonl'
    if manifest.exists():
        p.error('Use a new output-dir for each attempt; refusing to overwrite matrix.jsonl')
    for b in (1, 4):
        for prompt in (32, 128, 512):
            for generation in (32, 128):
                case = f'p{prompt}_g{generation}_b{b}'
                output = (args.output_dir / f'{case}.jsonl').resolve()
                command = [sys.executable, str(bench), '--device', args.device,
                           '--source', args.source, '--model-tag', args.model_tag,
                           '--step', str(args.step), '--prompt-tokens', str(prompt),
                           '--decode-tokens', str(generation), '--batch-size', str(b),
                           '--warmup', '3', '--repeats', str(args.repeats),
                           '--seed', str(args.seed), '--output', str(output)]
                supported = prompt + generation <= limit
                status = 'INVALID_CONTEXT' if not supported else ('PENDING_GPU' if args.device == 'cuda' else 'PENDING_CPU')
                row = dict(case_id=case, prompt_tokens=prompt, decode_tokens=generation,
                           batch_size=b, sequence_len=limit, status=status,
                           config=config, meta=str(args.meta.resolve()), source=args.source,
                           model_tag=args.model_tag, checkpoint_step=args.step, seed=args.seed,
                           command=command, raw_file=str(output), execute=args.execute,
                           compute_dtype=os.environ.get('NANOCHAT_DTYPE', 'auto'),
                           note='Memory estimates require measured compute dtype; CPU is contract validation only')
                if supported and args.execute and (args.batch is None or b == args.batch):
                    log = args.output_dir / f'{case}.log'
                    start = time.time()
                    with log.open('w') as stream:
                        result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
                    row.update(status='PROCESS_OK' if result.returncode == 0 else 'PROCESS_ERROR',
                               returncode=result.returncode, log=str(log.resolve()), wall_seconds=time.time() - start)
                with manifest.open('a') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(f'{case}: {row["status"]}', flush=True)
    print(manifest)


if __name__ == '__main__':
    main()
