#!/usr/bin/env python3
"""Plan a bounded real base_train invocation; execution requires --execute."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--steps", type=int, choices=[2, 20], default=2)
    p.add_argument(
        "--model-tag", help="default: d10_w2_smoke for 2 steps, d11_w2_20 for 20"
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="plan+log directory; checkpoint stays in NANOCHAT_BASE_DIR",
    )
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.model_tag is None:
        a.model_tag = "d10_w2_smoke" if a.steps == 2 else "d11_w2_20"
    command = [
        sys.executable,
        "-m",
        "scripts.base_train",
        f"--device-type={a.device}",
        "--depth=4",
        "--max-seq-len=512",
        "--device-batch-size=1",
        "--total-batch-size=512",
        f"--num-iterations={a.steps}",
        "--run=dummy",
        "--eval-every=-1",
        "--eval-tokens=512",
        "--core-metric-every=-1",
        "--sample-every=-1",
        "--window-pattern=L",
        "--warmup-steps=0",
        f"--model-tag={a.model_tag}",
    ]
    day = "d10" if a.steps == 2 else "d11"
    a.output_dir.mkdir(parents=True, exist_ok=True)
    plan = dict(
        command=command,
        cwd=str(ROOT),
        status="PENDING",
        steps=a.steps,
        model_tag=a.model_tag,
        base_dir=os.environ.get(
            "NANOCHAT_BASE_DIR", str(Path.home() / ".cache/nanochat")
        ),
    )
    print(shlex.join(command))
    if a.execute:
        checkpoint = Path(plan["base_dir"]) / "base_checkpoints" / a.model_tag
        if checkpoint.exists():
            p.error(
                f"{checkpoint} already exists; choose a new --model-tag to preserve evidence"
            )
        env = os.environ.copy()
        if a.device == "cpu":
            env.setdefault("NANOCHAT_DTYPE", "float32")
        with (a.output_dir / f"{day}_train.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        plan.update(
            status="ok" if proc.returncode == 0 else "error",
            returncode=proc.returncode,
            checkpoint_dir=str(checkpoint),
        )
    (a.output_dir / f"{day}_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return plan.get("returncode", 0)


if __name__ == "__main__":
    raise SystemExit(main())
