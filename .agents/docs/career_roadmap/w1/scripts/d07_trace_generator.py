#!/usr/bin/env python3
"""Trace a simplified Engine-style generator; no model, GPU, or timing."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[5]


def generate_columns(max_tokens, events):
    """Preserve the yield/increment/forward/stop order using event labels only."""
    events.extend(["prefill(prompt)", "copy_cache(batch)"])
    num_generated = 0
    while True:
        if num_generated >= max_tokens:
            events.append("stop(max_tokens)")
            return
        column = num_generated + 1
        events.extend([f"sample(column={column})", f"yield(column={column})"])
        yield column
        events.append(f"resume(column={column})")
        num_generated += 1
        events.append(f"forward(column={column})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=3, help="number of yielded columns (>=1)")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/inference_infra/results/w1/d07_generator_trace.json",
    )
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be >=1")

    events = []
    generator = generate_columns(args.tokens, events)
    result = {
        "purpose": "Control-flow teaching model only; no nanochat execution or performance measurement",
        "python_executable": sys.executable,
        "max_tokens": args.tokens,
        "events_after_creation": events.copy(),
        "calls": [],
    }
    print(f"Python: {sys.executable}")
    print(result["purpose"])
    print(f"create generator: {events}")
    for call_index in range(1, args.tokens + 2):
        start = len(events)
        try:
            column = next(generator)
        except StopIteration:
            record = {"next_call": call_index, "outcome": "StopIteration", "events": events[start:]}
        else:
            record = {
                "next_call": call_index,
                "outcome": "yield",
                "column": column,
                "events": events[start:],
            }
        result["calls"].append(record)
        print(f"next #{call_index}: {' -> '.join(record['events'])} => {record['outcome']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
