#!/usr/bin/env python3
"""Inventory the active interpreter and repository without installing packages."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]


def command(argv):
    try:
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=20)
        return {"argv": argv, "exit_code": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/inference_infra/results/w1/d02_environment.json')
    args = parser.parse_args()
    packages = {}
    for name in ('torch', 'pytest', 'tiktoken', 'rustbpe', 'pyarrow', 'kernels'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'N/A: distribution not installed in this interpreter'
    result = {
        'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'root': str(ROOT), 'python': sys.executable, 'python_version': sys.version,
        'platform': platform.platform(), 'conda_prefix': os.environ.get('CONDA_PREFIX', 'N/A'),
        'nanochat_base_dir': os.environ.get('NANOCHAT_BASE_DIR', 'N/A: default ~/.cache/nanochat'),
        'packages': packages, 'git_head': command(['git', 'rev-parse', 'HEAD']),
        'git_status': command(['git', 'status', '--short']),
        'uv': command(['uv', '--version']) if shutil.which('uv') else 'N/A: uv not on PATH',
        'nvidia_smi': command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']) if shutil.which('nvidia-smi') else 'N/A: nvidia-smi not on PATH',
    }
    # Import torch in a child process: a broken installation must not erase inventory.
    code = 'import json,torch; print(json.dumps({"torch":torch.__version__,"cuda_build":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}))'
    result['torch_runtime'] = command([sys.executable, '-c', code])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
