---
source_files:
  - pyproject.toml
  - nanochat/common.py
  - nanochat/dataset.py
  - .agents/docs/career_roadmap/w1/scripts/d02_inspect_environment.py
source_symbols:
  - get_base_dir
  - _detect_compute_dtype
  - COMPUTE_DTYPE
doc_type: tutorial
knowledge_scope: W1 D2 解释器、运行时与资源路径
---

# W1 D2：理解程序究竟在什么环境里运行

同一份源码为什么会在一个终端成功、另一个终端报错？今天从解释器、包、设备、精度和数据路径五个条件解释这个问题，再用盘点脚本确认自己的实际条件。预计 50–75 分钟；概念推演无需操作，实际操作只采集信息、导入模块，不训练、不下载、不需要权重。CPU 环境可以完成本日全部学习，GPU 信息有则记录，没有无需补做。

前置：[D1 项目全景与起点](d01_starting_point.md)。本教程沿用终端自动激活的 `nanochat` Conda 环境，避免为一次盘点再创建环境。

## 当天的流程阅读

用约 10–20 分钟阅读 [2.1 本地环境入口](../../llm-training-workflow/02-environment-and-data-preparation.md#reading-2-1)，了解环境和数据准备在完整流程中的位置，再回到下文。

- **今天重点**：找出解释器、设备、数据根和计算精度分别由谁决定。
- **暂时跳过**：复建用的 uv 命令仅作参考，不据此更换已有 Conda 环境；数据下载留到 D8 按需选择。
- **口述自检**：Conda 环境、CPU/GPU 设备与计算精度为什么是三个不同问题？

## 1. 从“两台 Python”理解环境

### 1.1 激活环境究竟改变了什么

<abbr title="Conda environment · Conda 环境。包含独立 Python 解释器和包的目录；激活主要调整当前 shell 的路径与环境变量，让 python 等命令优先指向该目录。激活不等于依赖完整或 GPU 可用。">Conda 环境</abbr>可以想成一套解释器与依赖目录。假设机器上有：

- `/usr/bin/python3`，没有安装 torch；
- `/home/lizishuo/miniconda3/envs/nanochat/bin/python`，安装了 torch。

两者执行同一个 `import torch` 会得到不同结果，源码并没有变化。终端里的 `python` 是命令名，最终选中哪个文件取决于 PATH；`sys.executable` 才能显示当前进程实际使用的解释器。`CONDA_PREFIX` 是环境声明，路径若与解释器不对应，应先排查激活或命令覆盖，不能只看提示符上的环境名。

[pyproject.toml](../../../../pyproject.toml)描述的是安装要求：当前要求 Python ≥3.10、`torch==2.9.1`；CPU/GPU 可选依赖互斥，dev 组提供 pytest 等工具。<abbr title="Extra / Dependency group · 可选依赖集合与依赖组。此项目 cpu/gpu extra 选择相应 torch 来源，dev group 提供开发工具。它们是安装选择，不是硬件探测结果。">extra 与依赖组</abbr>回答“应安装什么”，盘点结果回答“现在实际有什么”，二者需要对照。

### 1.2 “包显示已安装”为什么还不够

包版本通常来自发行包的 <abbr title="Package metadata · 包元数据。安装时记录的包名、版本等信息；读取版本可不导入包，因此能在动态库损坏等情况下仍显示版本。">元数据</abbr>。假设元数据能读到 `torch=2.9.1`，导入时却报缺少动态库：版本查询成功与模型能运行并不矛盾，它们检查的是不同层。

因此判断顺序是：先确认实际解释器，再查包版本，随后确认运行时能导入，最后才检查具体行为。今天的盘点到运行时导入为止；D3 的 CPU 测试继续检查特定推理和注意力行为，其中的模型替身并不执行真实 GPT。

### 1.3 设备与精度也不是一回事

<abbr title="CUDA · NVIDIA GPU 计算平台。这里分别检查 PyTorch 构建是否含 CUDA 支持、外部驱动信息、当前进程是否可用；三者不能互相替代。">CUDA</abbr>相关信息至少有三个层次：

| 观察 | 实际回答的问题 | 不能据此断言什么 |
|---|---|---|
| `nvidia-smi` 可显示设备和驱动 | 外部驱动工具看到了什么 | 当前 Python 一定能使用 CUDA |
| `torch.version.cuda` 非空 | 当前 torch 构建面向哪个 CUDA 版本 | GPU 一定能被这个进程使用 |
| `torch.cuda.is_available()` 为 true | 当前 torch 进程判断 CUDA 可用 | 你的具体模型、精度和规模一定可运行 |

纸面例子：主机有 NVIDIA 卡，但当前解释器装的是 CPU 版 torch，那么 `nvidia-smi` 可以成功，而 `cuda_build=null`、`cuda_available=false`。正确判断是“这个 Python 不能做 CUDA 计算”，不能推出物理机器无卡。反过来，找不到 `nvidia-smi` 也可能只是命令不在 PATH。

设备决定在哪算；<abbr title="dtype / Data type · 数值类型。float32、bfloat16 等决定数值表示和精度范围；设备相同不意味着 dtype 相同，计算类型也不必等于所有存储张量的类型。">dtype</abbr>决定用什么数值类型算。CPU 主线通常使用 <abbr title="FP32 / 32-bit Floating Point · 32 位浮点数。在本项目默认 CPU 路径用于计算；它不是 GPU 或 CUDA 的同义词。">FP32</abbr>，但环境变量也能改变项目选择，因此要读实际选择逻辑。

## 2. 带读项目机制：精度和文件根从哪里来

### 2.1 common.py 在导入时选择计算精度

[common.py](../../../../nanochat/common.py)先定义 `_detect_compute_dtype()`，随后在模块顶层立即调用：

```python
COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()
```

沿函数分支推演一次，无需运行：

1. 若有 `NANOCHAT_DTYPE`，按 `bfloat16`、`float16`、`float32` 的映射采用指定值。拼写错误会使映射查找失败；不要为做盘点随意设置。
2. 否则，CUDA 可用且设备计算能力至少为 `(8,0)` 时选 `bfloat16`。
3. 更早的 CUDA 设备回退 `float32`；没有 CUDA 也默认 `float32`。

这里选择的是项目的计算类型，不代表所有参数、输出或缓存都必须同一类型。今天不需要操作低精度训练。

为什么强调“导入时”？假设第一次 `import nanochat.common` 得到 `torch.float32`，之后才把 `os.environ["NANOCHAT_DTYPE"]` 改成 `bfloat16`：环境字符串变了，模块常量不会自动重算。其他文件还可能已经导入这个常量；准备新实验时，应在启动新 Python 进程前设置配置。当前盘点只读现有设置，不改变它。

### 2.2 get_base_dir 管的是资源位置

同一文件的实际规则为：`NANOCHAT_BASE_DIR` 非空时使用指定目录，否则使用 `~/.cache/nanochat`，并创建该目录。它不是“查有没有模型”的函数。

[dataset.py](../../../../nanochat/dataset.py)在这个根下组织 `base_data_climbmix` 语料，必要时还会查旧的 `base_data`；词表与 checkpoint 也各有自己的子目录。假设甲、乙使用同一 commit，但根目录分别指向两个位置，即使代码完全一样，也可能读到不同资源或缺文件。

今天只计算候选路径并检查它是否存在，不导入 dataset、不调用下载入口。目录存在只能说明位置存在，词表、语料和权重是否齐备留给 W2 分别核查；没有这些文件不影响 D3 的无权重 CPU 测试。

## 3. 带读盘点脚本，再采集实际事实

打开[环境盘点脚本](scripts/d02_inspect_environment.py)。它按下面顺序构建结果：

```python
packages[name] = importlib.metadata.version(name)
# ... 收集解释器、环境变量、Git 与硬件命令 ...
result['torch_runtime'] = command([sys.executable, '-c', code])
```

第一行只查版本；最后一行启动子进程实际导入 torch。这里使用 `sys.executable`，保证子进程复用当前 Python，不会再次凭 PATH 猜解释器。子进程异常退出时，`command()`保存 `exit_code`、stdout 和 stderr；命令启动失败或超时则保存 `error`。因此坏掉的 torch 不会让其余环境信息全部丢失。

**脚本总体退出 0 只表示盘点写完，不表示 torch 可运行。** 必须继续看 `torch_runtime.exit_code`；不存在该字段时检查 `error`，非 0 时读 stderr。也不要对失败命令的 stdout 强行按 JSON 解析。

### 3.1 实际操作：解释器与环境盘点

以下仅导入 torch 和采集元数据，不运行模型、不更新参数。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
python -c 'import sys; print(sys.executable); print(sys.version)'
d02_python_rc=$?; echo "python_exit_code=$d02_python_rc"
echo "$CONDA_PREFIX"
python .agents/docs/career_roadmap/w1/scripts/d02_inspect_environment.py
d02_inventory_rc=$?; echo "inventory_exit_code=$d02_inventory_rc"
} > >(tee -a experiments/inference_infra/reports/d02_checks.log) 2>&1
```

预期解释器是 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。以本次输出为准；[既有环境记录](../../project-maintenance/nanochat-conda-environment-2026-10-15.md)曾确认 Python 3.10.20、torch 2.9.1+cpu，但历史安装记录不等于当前运行结果。

局部日志仅覆盖花括号内命令，结束后恢复交互终端，不持续收集提示符。退出码紧跟目标命令保存，后面 `echo` 的成功不能冒充前面命令成功；若旧终端已有全局重定向，先新开终端。

### 3.2 怎样解释生成的 JSON

结果写入 `experiments/inference_infra/results/w1/d02_environment.json`，重复执行会覆盖；用 `--output` 可另存快照。<abbr title="JSON / JavaScript Object Notation · 以对象、数组和基本值组织的文本格式；此处保存此次采集的事实，不能把同名历史文件当作新运行结果。">JSON</abbr>里的 `recorded_at` 为 UTC，聊天说明实际日期时使用本地 Asia/Shanghai。

| 字段 | 读法与作用 |
|---|---|
| `python`、`conda_prefix`、`packages` | 检查解释器与安装内容；N/A 是当前解释器找不到发行包，不是用预期版本代填 |
| `git_head.stdout`、`git_status.stdout` | commit 固定已提交源码；工作区改动还可能改变实际行为，两者都要保留 |
| `torch_runtime` | 先看退出码或 error，再看 stdout 内的 torch/CUDA 信息；脚本成功不能替代这项成功 |
| `nvidia_smi` | 区分工具缺失、工具报错与成功输出，不用它单独决定后续 Python 的设备 |
| `nanochat_base_dir` | 记录环境变量或默认说明；它还不是资源完整性检查 |

如果看到 `cuda_build=null`、`cuda_available=false`，同时 torch 子进程退出 0，说明导入成功，但这份构建没有 CUDA 支持；结合本机记录中的 `2.9.1+cpu`，可确认当前使用 CPU 分发。单凭没有 CUDA 不能推断物理机器无 GPU，也不能判断其他平台加速能力。D3 可沿 CPU 主线继续，不需要把 GPU 另列为未完成任务。若 torch 导入失败，则先定位这个具体故障，仍可完成本日概念学习。

### 3.3 实际操作：补上计算类型与资源位置

仅在 torch 导入成功后执行第一条。它读取 common 的实际常量；第二条用标准库计算候选目录，避免为了查位置调用会创建目录的 `get_base_dir()`。

```bash
{
python -c 'from nanochat.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON; print(COMPUTE_DTYPE); print(COMPUTE_DTYPE_REASON)'
d02_dtype_rc=$?; echo "dtype_exit_code=$d02_dtype_rc"
python -c 'import os; from pathlib import Path; p=Path(os.environ.get("NANOCHAT_BASE_DIR") or str(Path.home()/".cache/nanochat")); print(p); print("exists:",p.exists())'
d02_path_rc=$?; echo "path_exit_code=$d02_path_rc"
} > >(tee -a experiments/inference_infra/reports/d02_checks.log) 2>&1
```

例如 `torch.float32` 配合 `auto-detected: no CUDA (CPU/MPS)` 解释了默认精度来源；若是 `set via NANOCHAT_DTYPE=...`，说明环境覆盖生效。输出根目录不存在并非“模型损坏”，只是这个位置当前未建资源目录；今天无需因此下载。

## 4. 当天自验证与交流

直接在聊天回答即可，命令、退出码与 JSON 路径共用一份。实际未执行的采集如实说明；不要求额外环境报告，也不为回答假设题制造故障。

1. **为什么两次 import 不同**：沿第 1 节的两个 Python 例子说明，应先比较哪个输出？为什么包元数据可查仍可能导入失败？再用本次 JSON 判断 D3 是否可以开始 CPU 测试。

   **核对要点**：把解释器、元数据和子进程导入分开；torch 导入成功还不等于模型测试通过。

2. **解释 CUDA 的矛盾表象**：`nvidia-smi` 成功但 `cuda_available=false`，怎样用 `cuda_build` 区分 CPU 构建与需继续排查的 CUDA 构建？

   **核对要点**：给出有先后次序的判断，不从单一输出推断物理机器无卡；假设题无需实际操作。

3. **精度设置何时生效**：同样使用 CPU，为什么两次进程的 `COMPUTE_DTYPE` 仍可能不同？在导入后修改环境变量，已有常量会怎样？

   **核对要点**：能从检测顺序和顶层赋值推导，解释本次实际精度的来源。

4. **相同 commit 是否等于相同条件**：工作区有模型源码改动、资源根目录也换了，还能只用 commit 声称复现条件相同吗？哪些差异值得保留？

   **核对要点**：分别说明源码与输入资源可能改变什么；目录存在不代表配套资源齐全。

## 5. 脚本、产物与常见分流

所有项目内路径相对 nanochat 根目录，以下是执行后的预期，不能据清单声称已经生成。

| 脚本与输入 | 用途与条件 |
|---|---|
| [`.agents/docs/career_roadmap/w1/scripts/d02_inspect_environment.py`](scripts/d02_inspect_environment.py)；输入为当前解释器、包元数据、Git、环境变量与可见硬件命令 | 第 3 节采集；标准库启动，torch 在同解释器子进程导入，缺包或导入失败也记录；不安装依赖 |

| 完整项目相对路径 | 用途与生成者 | 生成条件 |
|---|---|---|
| `experiments/inference_infra/results/w1/d02_environment.json` | 盘点脚本写入原始事实 | 盘点正常写入后自动生成，重跑覆盖 |
| `experiments/inference_infra/reports/d02_checks.log` | Bash 局部 `tee` 保存检查输出、错误与即时退出码 | 执行对应命令组后自动追加；精度/路径命令只打印，靠此日志保存 |
| `experiments/inference_infra/d02_env.md` | 整理环境卡 | 可选人工编写，聊天回答可替代 |

| 现象 | 怎样继续 |
|---|---|
| Python 指向其他环境 | 先确认自动激活配置或使用正确完整解释器路径；不混用别的项目包 |
| 包版本存在但导入失败 | 读 `torch_runtime.stderr` 或 `error`；按真实错误排查，不先重装整套环境 |
| torch 未装完或缺失 | 保留盘点事实，模型操作等运行时就绪再开始；无需反复创建环境 |
| 无 CUDA | 继续 CPU 主线；GPU 仅为自选扩展 |
| 输出目录不可写 | 可传 `--output /tmp/nanochat_d02_environment.json` 临时诊断；要作为正式证据时另保存到实验目录并说明实际路径 |

复建环境时再按[已记录的安装步骤](../../project-maintenance/nanochat-conda-environment-2026-10-15.md)核对当前条件；本日不要求执行 `uv sync` 或重装可用环境。结束时应能根据实际事实解释“当前用什么运行、能继续哪一步、还缺哪项必要条件”，而不是仅展示一份长日志。

上一篇：[D1 项目全景与起点](d01_starting_point.md)。下一篇：[D3 推理测试](d03_inference_smoke.md)。
