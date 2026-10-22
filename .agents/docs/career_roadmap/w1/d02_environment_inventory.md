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

# W1 D2：记录真正运行实验的环境

今天自动采集环境事实，不追求模型跑得快。你需要能回答“用哪个 Python、哪个源码版本、什么设备运行”，并让另一人知道缺什么。预计 60–75 分钟：10 分钟激活环境，15 分钟采集，20 分钟解释字段，15 分钟回答当天问题，最后 10 分钟核对。

前置：[D1 起点复核](d01_starting_point.md)。没有 GPU 可完成全部盘点；网络不可用时保留已有环境，不反复下载。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[2.1 本地环境入口](../../llm-training-workflow/02-environment-and-data-preparation.md#reading-2-1)。
- **今天重点**：只读本地环境入口，并从依赖参考中找出解释器、设备、数据根目录和计算精度分别由什么决定。按今天盘点的实际输出理解。
- **暂时跳过**：复建用的 uv 命令只作参考；沿用已有 <abbr title="Conda · 管理独立 Python 解释器及依赖的环境工具；此处使用已有 nanochat 环境。">Conda</abbr> 环境。数据下载放到 D8。

**读完能解释**：Conda 环境、CPU/GPU 设备与计算精度是同一个概念吗？你今天分别用什么输出确认它们？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 当日脚本与产物清单

所有项目内路径均以 nanochat 项目根目录为基准；外部 MiniMind 只读历史证据不改名。`dNN_` 表示全程首次使用或产生日，跨日复用不改前缀；上游源码和测试保留原生名称。

| 角色与脚本 | 用途与执行阶段 | 输入路径或来源 | 运行条件 |
|---|---|---|---|
| 入口：[`.agents/docs/career_roadmap/w1/scripts/d02_inspect_environment.py`](scripts/d02_inspect_environment.py) | 第 2 节采集解释器、依赖和硬件事实 | 项目 `pyproject.toml` 对照；读取当前 Git、环境变量、包元数据和硬件命令，无需模型文件 | 当前 Python；PyTorch 在子进程导入，缺失或失败也记录 |

| 完整项目相对路径 | 用途 | 生成者 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/results/w1/d02_environment.json` | 本次环境原始事实 | `d02_inspect_environment.py` | 自动：正常完成采集后写入 |
| `experiments/inference_infra/d02_env.md` | 环境卡及运行范围 | 第 4 节问答可选整理 | 可选人工整理；直接聊天回答即可，不影响验收 |
| `experiments/inference_infra/reports/d02_checks.log` | 本日终端检查输出与错误 | 各 Bash 代码块的 tee | 自动追加各代码块内的输出；命令及退出码在交流时补充 |

清单是执行后的预期，不表示文件已经存在。已有未加前缀的历史实验文件保留，新执行按上述路径写入。

正文的 `sed`、`rg`、`head`、解释器检查等只向终端打印，不会自动生成独立结果文件；各 Bash 代码块内的输出追加保存到 `experiments/inference_infra/reports/d02_checks.log`；交流时附所用命令、退出码和关键结果，自动结果已有的字段无需手抄。 盘点或探查的结构化结果保存为 <abbr title="JSON / JavaScript Object Notation · 用键值和数组组织数据的文本格式；这里保存脚本实际采集的事实或断言通过后的结果，可供程序读取。">JSON</abbr>。 未执行或失败时按真实状态记录，不能把旧同名文件当作本次成功。

## 1. 固定解释器，不另建虚拟环境

本轮应用户要求已经创建 <abbr title="Conda environment · Conda 管理的独立 Python 环境。激活后终端中的 python 和已安装包应来自同一前缀；创建成功仅证明解释器存在，不证明全部项目依赖已安装。">Conda 环境</abbr> `nanochat`，Python 为 3.10.20。编写本指导时，CPU/dev 依赖安装已经完成，实测 PyTorch 为 `2.9.1+cpu`，CPU/FP32/SDPA 可运行，CUDA 不可用。完整安装与导入验证记录见[Conda 环境记录](../../project-maintenance/nanochat-conda-environment-2026-10-15.md)。这只是编写时的本机事实，读者仍需重新采集自己的环境卡。当前终端会自动激活 nanochat，以后各日直接检查这一个环境，无需重复激活：

日志只在 `{ ...; }` 命令组执行期间记录；<abbr title="Bash process substitution · 进程替换。这里把命令组的输出交给 tee，同时显示并追加到日志；重定向在命令组结束后恢复。">进程替换</abbr>不会持续接管交互终端，因此提示符和 VS Code 终端集成控制字符不会混入日志。命令组在当前 shell 执行，变量设置会保留；代码块返回最后一条命令的退出码，需核对单条命令时立即记录 `$?`。已有的显式文件重定向仍写入各自文件。复制时包含外层花括号和末尾重定向。若之前执行过旧版全局日志命令，先关闭旧终端并新开终端；仅修改文档不会停止旧终端的记录。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
python -c 'import sys; print(sys.executable); print(sys.version)'
echo "$CONDA_PREFIX"
} > >(tee -a experiments/inference_infra/reports/d02_checks.log) 2>&1
```

两个路径应分别指向 `.../envs/nanochat/bin/python` 与 `.../envs/nanochat`。若 shell 仍指向其他环境，先停止模型运行并修正激活；不要用 `uv run` 产生第二个 `.venv`，也不要因为依赖正在下载而反复创建环境。

## 2. 运行盘点脚本

打开[环境盘点脚本](scripts/d02_inspect_environment.py)，先看 `command()` 如何捕捉异常，再执行：

```bash
{
python .agents/docs/career_roadmap/w1/scripts/d02_inspect_environment.py --help
python .agents/docs/career_roadmap/w1/scripts/d02_inspect_environment.py
} > >(tee -a experiments/inference_infra/reports/d02_checks.log) 2>&1
```

结果在 `experiments/inference_infra/results/w1/d02_environment.json`。脚本只读取 Git、包元数据、硬件命令，并在子进程导入 PyTorch；不下载、安装或训练。子进程导入失败会保留错误，而不是让整张环境卡消失。每次重跑覆盖同名文件，可用 `--output` 保存带日期快照。

逐项阅读 JSON：

| 字段 | 如何理解 | 交流时核对的内容 |
|---|---|---|
| `git_head.stdout` | 当前提交标识 | 完整 commit；不要把分支名当不可变版本 |
| `git_status.stdout` | 提交之外的工作区变化 | 哪些是原有改动、哪些是自己产物 |
| `python`、`conda_prefix` | 真正执行脚本的环境 | 绝对路径 |
| `packages` | 当前解释器可见的发行包版本 | 缺失写 N/A，不填预期版本 |
| `torch_runtime.exit_code` | PyTorch 导入是否完成 | 0 才继续解读 stdout 的 JSON |
| `cuda_build` | PyTorch 编译时绑定的 CUDA 版本 | `null` 表示 CPU 构建；与驱动版本分开 |
| `cuda_available` | 该进程能否使用 CUDA | false 不能做 GPU 基准 |
| `nvidia_smi` | 外部驱动工具的可见信息 | 缺命令与无 GPU 分开记录 |

<abbr title="CUDA · NVIDIA GPU 的计算运行环境。PyTorch 是否编译了 CUDA、驱动是否可见、当前进程能否分配 GPU 张量是不同条件。CPU 版 PyTorch 即使运行在带卡机器上也不能进行 CUDA 计时。">CUDA</abbr> 可用性必须来自这次进程，不能依据买过显卡或主机型号判断。`nvidia-smi` 不在 PATH 也不能单独证明物理机器无卡。JSON 中的 `recorded_at` 是 UTC 时间戳，交流时的实际执行日期按本地 Asia/Shanghai 填写。

## 3. 核对项目依赖和数据根目录

```bash
{
sed -n '1,145p' pyproject.toml
rg -n 'NANOCHAT_BASE_DIR|def get_base_dir|COMPUTE_DTYPE|NANOCHAT_DTYPE' nanochat/common.py
rg -n 'get_base_dir|parquet' nanochat/dataset.py
} > >(tee -a experiments/inference_infra/reports/d02_checks.log) 2>&1
```

当前 `pyproject.toml` 要求 Python ≥3.10，固定 `torch==2.9.1`，CPU 与 GPU extra 互斥；dev 组包含 pytest。<abbr title="extra · Python 项目可选依赖集合；cpu/gpu 用于选择对应的 torch 分发。dev group 则提供测试等开发工具。二者用途不同，不能把激活 Conda 视为已经安装这些依赖。">extra 与依赖组</abbr> 是安装选择，不是机器能力探测。

本教程已有环境优先复用。依赖尚未安装完就写 `INSTALLING`，等安装结束后重跑盘点再更新版本。若将来需要复建，先核对本机 uv 对参数的支持：

```bash
{
uv pip install --help
uv export --help
} > >(tee -a experiments/inference_infra/reports/d02_checks.log) 2>&1
```

已核对当前 uv 支持 `uv pip install --python` 与 `--group`。涉及 CPU/GPU 源和锁文件的复建，请按[环境记录中的实际安装步骤](../../project-maintenance/nanochat-conda-environment-2026-10-15.md)逐条执行；不能仅凭 `pip install torch` 猜中 2.9.1 CPU 轮子，也不能把 CPU 与 GPU 两套依赖同时装进环境。今天没有必要重装一个已经可用的环境。

`get_base_dir()` 默认使用 `~/.cache/nanochat`，可由 `NANOCHAT_BASE_DIR` 指定。只读盘点当前值：

```bash
{
python -c 'import os; from pathlib import Path; p=Path(os.environ.get("NANOCHAT_BASE_DIR", str(Path.home()/".cache/nanochat"))); print(p); print("exists:",p.exists())'
} > >(tee -a experiments/inference_infra/reports/d02_checks.log) 2>&1
```

不要今天就批量下载数据。交流时只需说明 tokenizer、parquet 语料、checkpoint 的候选根路径；是否完整由 W2 专门核查。大文件存储位置与自动结果、可选记录的位置分开，`/tmp` 用于缓存和临时验证，不适合保存唯一 checkpoint。

## 4. 当天自验证与交流问题

完成前面的检查后，按下面当天问题自行验证，再直接在聊天里与我交流。每题可用短句、数值和证据路径回答；未执行、失败或不理解的地方如实说明，无需排版或长文。`experiments/inference_infra/d02_env.md` 仅供需要时可选整理，不是必交文件。

1. 这次脚本真正用了哪个 Python 和 Conda 环境？从 JSON 的 `python`、`conda_prefix` 与终端路径交叉确认，是否混入其他环境？
2. 当前 commit 与工作区改动是什么？对照 `git_head.stdout`、`git_status.stdout`，哪些会影响后续实验复现？
3. PyTorch 包版本存在是否就能运行？结合 `packages`、`torch_runtime.exit_code` 和 stderr 说明本次导入结果、失败原因或未完成安装项。
4. `cuda_build`、`cuda_available`、`nvidia_smi` 各证明什么？依据本次采集判定 CPU/GPU 可执行范围，不把工具缺失等同于物理无卡。
5. `NANOCHAT_BASE_DIR` 的实际展开路径是什么？tokenizer、语料、checkpoint 哪些仅有候选位置、哪些尚未核验？D3 的 CPU 测试条件是否齐备？

<abbr title="dirty worktree · Git 工作区包含尚未提交的改动。记录它是为了说明实际执行源码可能不只由 commit 决定，不表示应清空这些改动。">dirty 状态</abbr> 需要可追溯；只写一个 commit 而隐藏源码改动会让后续性能对比无法复现。

## 5. 失败分流与验收

- Python 不对：先检查终端自动激活配置，确认解释器路径正确后再执行脚本，不继续拿其他项目环境的数据填卡。
- 包元数据显示版本，导入却失败：读 `torch_runtime.stderr`，这是动态库或安装完整性问题，不能判为可运行。
- torch 缺失：等依赖安装完成或记录 BLOCKED；标准库盘点仍算完成，模型验收仍未完成。
- 无 CUDA：交流时说明 CPU 路线可继续，GPU 项 `PENDING_GPU`；明天执行 CPU 测试。
- 输出目录不可写：可先传 `--output /tmp/nanochat_d02_environment.json` 验证脚本，正式证据最终迁回实验目录并登记实际路径。

结束时核对解释器、commit/dirty、实际版本、驱动与 CUDA runtime 的区别和缺项原因；交流附原始 JSON 路径与实际日期、命令、状态及 D3 待办，无需另写环境卡，用时可选。

上一篇：[D1 起点复核](d01_starting_point.md)。下一篇：[D3 推理测试](d03_inference_smoke.md)。
