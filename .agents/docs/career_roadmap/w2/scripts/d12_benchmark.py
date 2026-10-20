#!/usr/bin/env python3
"""Synchronized repeated Engine benchmark. CPU/random results are contracts only."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--random-model', action='store_true', help='tiny random weights; never a trained baseline')
    p.add_argument('--source', choices=['base'], default='base')
    p.add_argument('--model-tag', default='d11_w2_20')
    p.add_argument('--step', type=int)
    p.add_argument('--prompt-tokens', type=int, default=32)
    p.add_argument('--decode-tokens', type=int, default=8)
    p.add_argument('--prompt-fill-id', type=int, default=1, help='ordinary ID repeated after BOS; changing it creates a different workload')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--output', type=Path, required=True, help='append-only JSONL; use a fresh path per session')
    p.add_argument('--seed', type=int, default=42)
    return p


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def driver_version():
    try:
        return subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                       text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class ContractTokenizer:
    """Synthetic ID mapping for random-model API validation, never training."""
    names = ['<|bos|>', '<|user_start|>', '<|user_end|>', '<|assistant_start|>',
             '<|assistant_end|>', '<|python_start|>', '<|python_end|>', '<|output_start|>', '<|output_end|>']

    def encode_special(self, text):
        return 256 + self.names.index(text)

    def get_bos_token_id(self):
        return self.encode_special('<|bos|>')

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode('utf-8', errors='replace')

    def encode(self, text):
        return list(text.encode('utf-8'))


def load(args):
    import torch
    if args.device == 'cpu':
        # Must precede imports that read the process-level precision constant.
        os.environ.setdefault('NANOCHAT_DTYPE', 'float32')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is false')
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.common import COMPUTE_DTYPE
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    identity = {}
    if args.random_model:
        config = GPTConfig(sequence_len=max(128, args.prompt_tokens + args.decode_tokens),
                           vocab_size=512, n_layer=2, n_head=2, n_kv_head=2, n_embd=64, window_pattern='L')
        model = GPT(config).to(device)
        model.init_weights()
        model.eval()
        tokenizer = ContractTokenizer()
    else:
        from nanochat.common import get_base_dir
        from nanochat.checkpoint_manager import load_model, find_last_step
        directory = Path(get_base_dir()) / 'base_checkpoints' / args.model_tag
        step = args.step if args.step is not None else find_last_step(directory)
        args.step = step
        model, tokenizer, _ = load_model(args.source, device, phase='eval', model_tag=args.model_tag, step=step)
        for name in [f'model_{step:06d}.pt', f'meta_{step:06d}.json']:
            identity[name] = sha256(directory / name)
        identity['tokenizer.pkl'] = sha256(Path(get_base_dir()) / 'tokenizer' / 'tokenizer.pkl')
    return torch, model, tokenizer, str(COMPUTE_DTYPE), identity


def measure(args, torch, model, tokenizer):
    from nanochat.engine import Engine
    engine = Engine(model, tokenizer)
    # Same ID prompt for every repeat; no tokenizer work inside timing.
    prompt = [tokenizer.get_bos_token_id()] + [args.prompt_fill_id] * (args.prompt_tokens - 1)
    stop = {tokenizer.get_bos_token_id(), tokenizer.encode_special('<|assistant_end|>')}
    sync = (lambda: torch.cuda.synchronize(model.get_device())) if args.device == 'cuda' else lambda: None
    counts = [0] * args.batch_size
    completed = [False] * args.batch_size
    times = []
    steps = 0
    forced = False
    stream = engine.generate(prompt, num_samples=args.batch_size, max_tokens=args.decode_tokens,
                             temperature=0.0, seed=args.seed)
    sync()
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats(model.get_device())
    start = previous = time.perf_counter()
    try:
        # Pull exactly G yields. Do not resume after final yield: Engine would
        # otherwise compute one unused forward pass before max_tokens check.
        for _ in range(args.decode_tokens):
            try:
                column, masks = next(stream)
            except StopIteration:
                break
            sync()
            now = time.perf_counter()
            times.append((now - previous) * 1000)
            previous = now
            steps += 1
            for i, (token, mask) in enumerate(zip(column, masks)):
                if completed[i]:
                    continue
                if token in stop:
                    completed[i] = True
                elif mask:
                    counts[i] += 1
                else:
                    forced = True
            if any(completed):
                # Keep this failed sample; fixed-length performance rows must
                # have all B rows active, with no terminal/padding columns.
                break
    finally:
        stream.close()
    elapsed = sum(times)
    good = steps == args.decode_tokens and all(x == args.decode_tokens for x in counts) and not forced
    return dict(status='ok' if good else 'early_stop', actual_decode_steps=steps,
                generated_tokens=sum(counts), per_row_generated_tokens=counts,
                ttft_ms=times[0] if times else None, decode_step_ms=times[1:],
                tpot_ms=sum(times[1:]) / len(times[1:]) if len(times) > 1 else None,
                elapsed_ms=elapsed, tokens_per_s=sum(counts) * 1000 / elapsed if elapsed else None,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(model.get_device()) if args.device == 'cuda' else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(model.get_device()) if args.device == 'cuda' else None,
                error=None)


def main():
    p = parser()
    args = p.parse_args()
    if min(args.prompt_tokens, args.decode_tokens, args.batch_size) < 1 or args.warmup < 3 or args.repeats < 5:
        p.error('P/G/B must be positive; warmup>=3 and repeats>=5 required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    common = dict(schema_version=1, session_id=str(uuid.uuid4()), mode='engine', device=args.device,
                  model_kind='random' if args.random_model else 'checkpoint', source=args.source,
                  model_tag=args.model_tag, checkpoint_step=args.step, seed=args.seed,
                  prompt_tokens=args.prompt_tokens, decode_tokens=args.decode_tokens, batch_size=args.batch_size,
                  prompt_fill_id=args.prompt_fill_id,
                  timing_scope='token_ids_to_last_yield; synchronized per yield; tokenizer excluded',
                  batch_semantics='one_prompt_prefill_then_clone',
                  evidence_scope='CPU_FUNCTION_ONLY' if args.device == 'cpu' else ('RANDOM_CONTRACT_ONLY' if args.random_model else 'CUDA_CHECKPOINT_BASELINE'))
    def write(row):
        with args.output.open('a', encoding='utf-8') as f:
            f.write(json.dumps({**common, **row}, ensure_ascii=False, allow_nan=False) + '\n')
    try:
        torch, model, tokenizer, precision, identity = load(args)
        from dataclasses import asdict
        from nanochat.flash_attention import USE_FA3
        if not 0 <= args.prompt_fill_id < model.config.vocab_size:
            raise ValueError('--prompt-fill-id is outside the model vocabulary')
        special_ids = {tokenizer.encode_special(s) for s in ContractTokenizer.names}
        if args.prompt_fill_id in special_ids:
            raise ValueError('--prompt-fill-id must be an ordinary non-special token ID')
        prompt_ids = [tokenizer.get_bos_token_id()] + [args.prompt_fill_id] * (args.prompt_tokens - 1)
        common['prompt_sha256'] = hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()
        if args.prompt_tokens + args.decode_tokens > model.config.sequence_len:
            raise ValueError('P+G exceeds model.config.sequence_len; no automatic truncation')
        def git(*cmd):
            return subprocess.check_output(['git', '-C', str(ROOT), *cmd], text=True).strip()
        common.update(checkpoint_step=args.step, model_config=asdict(model.config), checkpoint_sha256=identity,
                      precision=precision, attention_backend='fa3' if USE_FA3 else 'sdpa',
                      env=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                               device_name=torch.cuda.get_device_name(model.get_device()) if args.device == 'cuda' else platform.processor(),
                               commit=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain')),
                               num_threads=torch.get_num_threads(), platform=platform.platform(),
                               triton=package_version('triton'), driver_version=driver_version()))
        failed = False
        with torch.inference_mode():
            for phase, n in [('warmup', args.warmup), ('measure', args.repeats)]:
                for repeat in range(n):
                    try:
                        row = measure(args, torch, model, tokenizer)
                    except Exception as exc:
                        write(dict(run_id=str(uuid.uuid4()), phase=phase, repeat=repeat, status='error',
                                   error=f'{type(exc).__name__}: {exc}'))
                        print(f'{phase} repeat={repeat}: {type(exc).__name__}: {exc}', file=sys.stderr)
                        return 1
                    write(dict(run_id=str(uuid.uuid4()), phase=phase, repeat=repeat, **row))
                    failed |= row['status'] != 'ok'
        print(f'{args.output}: {args.warmup} warmups + {args.repeats} measurements; scope={common["evidence_scope"]}')
        return 2 if failed else 0
    except Exception as exc:
        write(dict(run_id=str(uuid.uuid4()), phase='setup_or_run', repeat=None, status='error',
                   error=f'{type(exc).__name__}: {exc}'))
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
