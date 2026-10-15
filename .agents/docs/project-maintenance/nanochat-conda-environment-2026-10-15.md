---
source_files:
  - .python-version
  - pyproject.toml
  - uv.lock
  - tests/test_engine.py
  - tests/test_attention_fallback.py
doc_type: reference
---

# nanochat Conda 环境验证记录（2026-10-15）

按用户要求创建名为 `nanochat` 的 <abbr title="Conda · Python 环境与软件包管理工具。命名环境将解释器及依赖与系统 Python 隔离；激活后 python 指向该环境。">Conda</abbr> 环境。环境路径为 `/home/lizishuo/miniconda3/envs/nanochat`，解释器为该目录的 `bin/python`。项目 `.python-version` 指定 3.10，本轮安装 Python 3.10.20。

## 使用入口

从项目根目录执行：

```bash
source /home/lizishuo/miniconda3/etc/profile.d/conda.sh
conda activate nanochat
python -c 'import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())'
```

预期解释器路径以 `/envs/nanochat/bin/python` 结尾，PyTorch 输出 `2.9.1+cpu`，CUDA 可用性为 `False`。教程采用激活环境后的 `python`；`uv run` 默认可能另建项目 `.venv`，不作为本轮教程运行入口。

## 安装依据与实际版本

本轮先执行 `conda create -n nanochat python=3.10 pip -y`，再用 `uv export --frozen --extra cpu --group dev --no-emit-project` 从现有 <abbr title="锁文件 · 记录已解析的依赖版本、来源及包哈希的文件。此处是已有 uv.lock；本轮按其选择 CPU 依赖和 dev 开发工具，没有重新生成锁文件。">锁文件</abbr> 导出安装清单，在该 Conda 环境中执行 `uv pip install --python … --require-hashes -r …`。临时清单和下载文件保存在 `/tmp`。

CPU PyTorch wheel 直接使用锁文件中的官方 URL 和对应哈希。PyArrow 的官方源下载较慢，最后从清华 PyPI 镜像下载同一 wheel；SHA-256 与 `uv.lock` 一致后才安装。没有永久修改 pip/uv 镜像配置，没有修改 `pyproject.toml`、`uv.lock` 或 `.python-version`。

| 项目 | 本轮实测 |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.9.1+cpu |
| NumPy | 1.26.4 |
| pytest | 8.4.2 |
| PyArrow | 21.0.0 |
| rustbpe / tiktoken | 0.1.0 / 0.11.0 |
| kernels / wandb | 0.11.7 / 0.21.3 |
| 计算精度 | float32 |
| 注意力路由 | SDPA，即 PyTorch 的缩放点积注意力回退路径 |
| CUDA | 不可用；安装 CPU 版不提供 GPU 测量能力 |

## 验证结果与复查命令

核心依赖和 `nanochat.gpt`、`nanochat.engine`、`nanochat.flash_attention` 导入成功。`uv pip check` 检查 79 个已安装包，报告依赖兼容。

```bash
uv pip check --cache-dir /tmp/nanochat-uv-cache --python "$CONDA_PREFIX/bin/python"
PYTHONPYCACHEPREFIX=/tmp/nanochat-pycache python -m pytest -q -p no:cacheprovider \
  tests/test_engine.py tests/test_attention_fallback.py
```

实际测试结果为 **13 passed, 10 skipped in 5.16s**，退出码 0。跳过项依赖本机不可用的 FA3。它是 CPU 测试结果，不证明 GPU 后端、真实 checkpoint 推理或训练已通过；也不替代后续随机真实模型与缓存逐步对照实验。

本轮没有下载训练语料、训练 tokenizer 或生成训练 checkpoint。最初由 uv 创建但未安装依赖的临时 `.venv` 已移除，当前项目环境采用上述 Conda 环境。
