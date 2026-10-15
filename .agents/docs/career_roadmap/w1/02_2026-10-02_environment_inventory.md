---
source_files:
  - pyproject.toml
  - nanochat/common.py
  - nanochat/dataset.py
  - .agents/docs/career_roadmap/nanochat_daily_execution_2026-10-01.md
source_symbols:
  - get_base_dir
  - _detect_compute_dtype
doc_type: tutorial
knowledge_scope: W1 D2 Conda环境与硬件事实盘点
---

# W1 D2 · 2026-10-02：记录真正运行实验的环境

今天产出环境卡片，不追求模型跑得快。你需要能回答“用哪个 Python、哪个源码版本、什么设备运行”，并让另一人知道缺什么。预计 60–75 分钟：10 分钟激活环境，15 分钟采集，20 分钟解释字段，15 分钟写 `env.md`，最后 10 分钟核对。

前置：[D1 起点报告](01_2026-10-01_starting_point.md)。没有 GPU 可完成全部盘点；网络不可用时保留已有环境，不反复下载。

## 1. 固定解释器，不另建虚拟环境

本轮应用户要求已经创建 <abbr title="Conda environment · Conda 管理的独立 Python 环境。激活后终端中的 python 和已安装包应来自同一前缀；创建成功仅证明解释器存在，不证明全部项目依赖已安装。">Conda 环境</abbr> `nanochat`，Python 为 3.10.20。2026-10-15 编写本指导时，CPU/dev 依赖安装已经完成，实测 PyTorch 为 `2.9.1+cpu`，CPU/FP32/SDPA 可运行，CUDA 不可用。完整安装与导入验证记录见[Conda 环境记录](../../project-maintenance/nanochat-conda-environment-2026-10-15.md)。这只是编写时的本机事实，读者仍需重新采集自己的环境卡。以后各日均使用这一个环境：

```bash
cd /home/lizishuo/_program_/Project/nanochat
source /home/lizishuo/miniconda3/etc/profile.d/conda.sh
conda activate nanochat
python -c 'import sys; print(sys.executable); print(sys.version)'
echo "$CONDA_PREFIX"
```

两个路径应分别指向 `.../envs/nanochat/bin/python` 与 `.../envs/nanochat`。若 shell 仍指向其他环境，先停止模型运行并修正激活；不要用 `uv run` 产生第二个 `.venv`，也不要因为依赖正在下载而反复创建环境。

## 2. 运行盘点脚本

打开[环境盘点脚本](scripts/inspect_environment.py)，先看 `command()` 如何捕捉异常，再执行：

```bash
python .agents/docs/career_roadmap/w1/scripts/inspect_environment.py --help
python .agents/docs/career_roadmap/w1/scripts/inspect_environment.py
```

结果在 `experiments/inference_infra/results/w1/environment.json`。脚本只读取 Git、包元数据、硬件命令，并在子进程导入 PyTorch；不下载、安装或训练。子进程导入失败会保留错误，而不是让整张环境卡消失。每次重跑覆盖同名文件，可用 `--output` 保存带日期快照。

逐项阅读 JSON：

| 字段 | 如何理解 | 应写入环境报告的内容 |
|---|---|---|
| `git_head.stdout` | 当前提交标识 | 完整 commit；不要把分支名当不可变版本 |
| `git_status.stdout` | 提交之外的工作区变化 | 哪些是原有改动、哪些是自己产物 |
| `python`、`conda_prefix` | 真正执行脚本的环境 | 绝对路径 |
| `packages` | 当前解释器可见的发行包版本 | 缺失写 N/A，不填预期版本 |
| `torch_runtime.exit_code` | PyTorch 导入是否完成 | 0 才继续解读 stdout 的 JSON |
| `cuda_build` | PyTorch 编译时绑定的 CUDA 版本 | `null` 表示 CPU 构建；与驱动版本分开 |
| `cuda_available` | 该进程能否使用 CUDA | false 不能做 GPU 基准 |
| `nvidia_smi` | 外部驱动工具的可见信息 | 缺命令与无 GPU 分开记录 |

<abbr title="CUDA · NVIDIA GPU 的计算运行环境。PyTorch 是否编译了 CUDA、驱动是否可见、当前进程能否分配 GPU 张量是不同条件。CPU 版 PyTorch 即使运行在带卡机器上也不能进行 CUDA 计时。">CUDA</abbr> 可用性必须来自这次进程，不能依据买过显卡或主机型号判断。`nvidia-smi` 不在 PATH 也不能单独证明物理机器无卡。JSON 中的 `recorded_at` 是 UTC 时间戳，报告的实际执行日期按本地 Asia/Shanghai 填写。

## 3. 核对项目依赖和数据根目录

```bash
sed -n '1,145p' pyproject.toml
rg -n 'NANOCHAT_BASE_DIR|def get_base_dir|COMPUTE_DTYPE|NANOCHAT_DTYPE' nanochat/common.py
rg -n 'get_base_dir|parquet' nanochat/dataset.py
```

当前 `pyproject.toml` 要求 Python ≥3.10，固定 `torch==2.9.1`，CPU 与 GPU extra 互斥；dev 组包含 pytest。<abbr title="extra · Python 项目可选依赖集合；cpu/gpu 用于选择对应的 torch 分发。dev group 则提供测试等开发工具。二者用途不同，不能把激活 Conda 视为已经安装这些依赖。">extra 与依赖组</abbr> 是安装选择，不是机器能力探测。

本教程已有环境优先复用。依赖尚未安装完就写 `INSTALLING`，等安装结束后重跑盘点再更新版本。若将来需要复建，先核对本机 uv 对参数的支持：

```bash
uv pip install --help
uv export --help
```

已核对当前 uv 支持 `uv pip install --python` 与 `--group`。涉及 CPU/GPU 源和锁文件的复建，请按[环境记录中的实际安装步骤](../../project-maintenance/nanochat-conda-environment-2026-10-15.md)逐条执行；不能仅凭 `pip install torch` 猜中 2.9.1 CPU 轮子，也不能把 CPU 与 GPU 两套依赖同时装进环境。今天没有必要重装一个已经可用的环境。

`get_base_dir()` 默认使用 `~/.cache/nanochat`，可由 `NANOCHAT_BASE_DIR` 指定。只读盘点当前值：

```bash
python -c 'import os; from pathlib import Path; p=Path(os.environ.get("NANOCHAT_BASE_DIR", str(Path.home()/".cache/nanochat"))); print(p); print("exists:",p.exists())'
```

不要今天就批量下载数据。环境卡只登记 tokenizer、parquet 语料、checkpoint 的候选根路径；是否完整由 W2 专门核查。大文件存储位置与正式文本报告位置分开，`/tmp` 用于缓存和临时验证，不适合保存唯一 checkpoint。

## 4. 写正式环境卡

新建 `experiments/inference_infra/env.md`：

```markdown
# 实验环境
- 实际采集日期与时区：
- 源码 commit / dirty 项：
- Python 路径与版本：
- Conda 环境：nanochat
- 依赖安装状态：INSTALLING / COMPLETE / BLOCKED
- PyTorch 版本 / CUDA build / 当前 CUDA available：
- GPU 名称、驱动、显存：N/A 或真实值
- uv / pytest / kernels：
- 本次盘点 JSON：results/w1/environment.json
- NANOCHAT_BASE_DIR 实际值或默认展开路径：
- tokenizer / 数据 / checkpoint 候选路径：尚未核验
- 可执行范围：CPU 盘点 / CPU 模型 / CUDA（逐项标状态）
- 仍缺条件、原始错误、下一步：
```

<abbr title="dirty worktree · Git 工作区包含尚未提交的改动。记录它是为了说明实际执行源码可能不只由 commit 决定，不表示应清空这些改动。">dirty 状态</abbr> 需要可追溯；只写一个 commit 而隐藏源码改动会让后续性能对比无法复现。

## 5. 失败分流与验收

- Python 不对：重新 source 与 activate，再执行脚本，不继续拿其他项目环境的数据填卡。
- 包元数据显示版本，导入却失败：读 `torch_runtime.stderr`，这是动态库或安装完整性问题，不能判为可运行。
- torch 缺失：等依赖安装完成或记录 BLOCKED；标准库盘点仍算完成，模型验收仍未完成。
- 无 CUDA：在环境卡写 CPU 路线可继续，GPU 项 `PENDING_GPU`；明天执行 CPU 测试。
- 输出目录不可写：可先传 `--output /tmp/nanochat_environment.json` 验证脚本，正式证据最终迁回实验目录并登记实际路径。

结束时勾选：解释器正确、commit/dirty 已留存、版本为实际值、驱动与 CUDA runtime 不混淆、缺项有原因、环境卡能链接到原始 JSON。追加当天记录：`实际日期 / commit / 用时 / 命令 / 文件 / 状态 / D3 待办`。

上一篇：[D1 起点复核](01_2026-10-01_starting_point.md)。下一篇：[D3 推理测试](03_2026-10-03_inference_smoke.md)。
