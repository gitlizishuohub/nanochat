#!/usr/bin/env python3
"""Load nanochat base checkpoint and compare greedy prefixes on two generation paths."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--model-tag", default="d10_w2_smoke")
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--prompt", default="The chemical formula of water is")
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.max_tokens < 1:
        p.error("--max-tokens must be positive")
    report = dict(status="error", model_tag=a.model_tag, step=a.step, prompt=a.prompt)
    try:
        import torch

        if a.device == "cpu":
            os.environ.setdefault("NANOCHAT_DTYPE", "float32")
        from dataclasses import asdict
        from nanochat.checkpoint_manager import load_model
        from nanochat.engine import Engine

        model, tok, meta = load_model(
            "base",
            torch.device(a.device),
            phase="eval",
            model_tag=a.model_tag,
            step=a.step,
        )
        ids = tok.encode(a.prompt, prepend=tok.get_bos_token_id())
        assert (
            len(ids) + a.max_tokens <= model.config.sequence_len
        ), "P+G exceeds sequence_len"
        with torch.inference_mode():
            reference = list(
                model.generate(ids, max_tokens=a.max_tokens, temperature=0.0)
            )
            engine_ids = []
            stream = Engine(model, tok).generate(
                ids, num_samples=1, max_tokens=a.max_tokens, temperature=0.0
            )
            try:
                for _ in range(a.max_tokens):
                    try:
                        column, masks = next(stream)
                    except StopIteration:
                        break
                    engine_ids.append(column[0])
                    if column[0] in {
                        tok.get_bos_token_id(),
                        tok.encode_special("<|assistant_end|>"),
                    }:
                        break
            finally:
                stream.close()
        match = reference[: len(engine_ids)] == engine_ids
        report.update(
            status="ok" if match else "mismatch",
            prompt_ids=ids,
            reference_ids=reference,
            engine_ids=engine_ids,
            common_prefix_match=match,
            reference_text=tok.decode(reference),
            engine_text=tok.decode(engine_ids),
            model_config=asdict(model.config),
            note="Prefix check includes terminal ID; GPT.generate has no Engine stop/tool semantics. Not logits equivalence.",
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
