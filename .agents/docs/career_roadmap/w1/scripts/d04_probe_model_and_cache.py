#!/usr/bin/env python3
"""Probe actual tiny GPT shapes and cache state on CPU; never loads a checkpoint."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('shapes', 'cache'), default='shapes')
    parser.add_argument('--prompt-tokens', type=int, default=8)
    parser.add_argument('--decode-steps', type=int, default=4)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.prompt_tokens < 2 or args.decode_steps < 1 or args.prompt_tokens + args.decode_steps > 64:
        parser.error('require P>=2, steps>=1 and P+steps<=64')
    # Set before any nanochat/torch import, avoiding GPU kernel discovery/downloads.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['NANOCHAT_DTYPE'] = 'float32'
    sys.path.insert(0, str(ROOT))
    try:
        import torch
        from nanochat.gpt import GPT, GPTConfig
        from nanochat.engine import KVCache
        from nanochat import flash_attention
    except ImportError as exc:
        print(f'DEPENDENCY_MISSING: {exc}. Activate conda nanochat and finish CPU/dev installation.', file=sys.stderr)
        return 2
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = GPTConfig(sequence_len=64, vocab_size=128, n_layer=2, n_head=4, n_kv_head=2, n_embd=64, window_pattern='L')
    model = GPT(config)
    model.init_weights()
    model.eval()
    # Zero output projections at init would hide attention/cache mistakes. Use finite,
    # nonzero random projections and smear state for a more informative smoke probe.
    with torch.no_grad():
        for block in model.transformer.h:
            torch.nn.init.normal_(block.attn.c_proj.weight, std=0.02)
            torch.nn.init.normal_(block.mlp.c_proj.weight, std=0.02)
        model.smear_lambda.fill_(0.15)
    ids = torch.randint(0, config.vocab_size, (1, args.prompt_tokens))
    result = {'purpose': 'CPU random-weight functional probe; no quality/performance claim', 'mode': args.mode, 'config': dataclasses.asdict(config), 'seed': 42, 'torch': torch.__version__, 'USE_FA3': flash_attention.USE_FA3, 'shapes': [], 'steps': []}
    hooks = []
    for name, module in model.named_modules():
        if name in ('transformer.wte', 'transformer.h.0', 'transformer.h.0.attn.c_q', 'transformer.h.0.attn.c_k', 'transformer.h.0.attn.c_v', 'lm_head'):
            def hook(_module, inputs, output, label=name):
                result['shapes'].append({'module': label, 'input': list(inputs[0].shape), 'output': list(output.shape)})
            hooks.append(module.register_forward_hook(hook))
    with torch.inference_mode():
        logits = model(ids)
        assert tuple(logits.shape) == (1, args.prompt_tokens, config.vocab_size)
        assert torch.isfinite(logits).all()
        result['logits_shape'] = list(logits.shape)
        result['logits_dtype'] = str(logits.dtype)
        for handle in hooks:
            handle.remove()
        if args.mode == 'cache':
            cache = KVCache(batch_size=1, num_heads=config.n_kv_head, seq_len=64, head_dim=16, num_layers=config.n_layer, device='cpu', dtype=torch.float32)
            prefix = ids.clone()
            current = ids
            for step in range(args.decode_steps):
                before = cache.get_pos()
                cached = model(current, kv_cache=cache)[:, -1, :]
                reference = model(prefix)[:, -1, :]
                torch.testing.assert_close(cached, reference, atol=1e-5, rtol=1e-4)
                assert cache.get_pos() == prefix.shape[1]
                assert cached.argmax(-1).item() == reference.argmax(-1).item()
                result['steps'].append({'step': step, 'input_tokens': current.shape[1], 'cache_before': before, 'cache_after': cache.get_pos(), 'cache_seqlens': cache.cache_seqlens.tolist(), 'prev_embedding_shape': list(cache.prev_embedding.shape), 'max_abs_logit_diff': (cached-reference).abs().max().item(), 'greedy_token': cached.argmax(-1).item()})
                current = cached.argmax(-1, keepdim=True)
                prefix = torch.cat([prefix, current], dim=1)
            target = KVCache(batch_size=4, num_heads=2, seq_len=64, head_dim=16, num_layers=2, device='cpu', dtype=torch.float32)
            target.prefill(cache)
            pos = cache.get_pos()
            for row in range(4):
                torch.testing.assert_close(target.k_cache[:, row, :pos], cache.k_cache[:, 0, :pos])
                torch.testing.assert_close(target.v_cache[:, row, :pos], cache.v_cache[:, 0, :pos])
                torch.testing.assert_close(target.prev_embedding[row], cache.prev_embedding[0])
            result['k_cache_shape'] = list(target.k_cache.shape)
            result['v_cache_shape'] = list(target.v_cache.shape)
            result['replication_batch'] = 4
            result['replication_position'] = pos
            result['storage_bytes_kv'] = (target.k_cache.numel()+target.v_cache.numel()) * target.k_cache.element_size()
            target.reset()
            assert target.get_pos() == 0 and target.prev_embedding is None
            result['reset_verified'] = True
    filename = 'd04_shapes_probe.json' if args.mode == 'shapes' else 'd05_cache_probe.json'
    output = args.output or ROOT / 'experiments/inference_infra/results/w1' / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
