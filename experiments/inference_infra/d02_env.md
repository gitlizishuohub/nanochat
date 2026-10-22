# W1 D2 环境盘点报告

实验日期：2026-10-22（Asia/Shanghai）。原始环境采集时间：15:17:51；对应 JSON 时间戳为 `2026-10-22T07:17:51.628863+00:00`。

证据：[环境 JSON](results/w1/d02_environment.json)、[当天终端日志](reports/d02_checks.log)、[D02 教程](../../.agents/docs/career_roadmap/w1/d02_environment_inventory.md)。本报告结合用户自验证回答、原始 JSON 和助手补充检查整理，不代表 D3 已运行。

## 1. Python 与 Conda 环境

本次使用 <abbr title="Conda environment · Conda 管理的独立 Python 解释器与依赖集合；环境名称不代表 GPU 能力。">Conda 环境</abbr> `nanochat`，Python 为 `3.10.20`。

- 解释器：`/home/lizishuo/miniconda3/envs/nanochat/bin/python`。
- Conda 前缀：`/home/lizishuo/miniconda3/envs/nanochat`。

JSON 中解释器与环境前缀一致，用户确认没有混入其他环境路径；现有证据未发现环境混用。

## 2. 源码版本与工作区

采集时 commit 为 `1bb53623e997f715910697370936bc7293288ef7`。

用户回答“无改动”对应源码无改动；原始 JSON 的 Git 状态仍记录了 D01 日志修改及 D01/D02 日志、原始日志副本新增，因此工作区并非完全干净。助手后续检查还看到新生成的环境 JSON。现有状态未显示源码改动；复现时应同时保留 commit 与这些实验记录，不能只写“工作区无改动”。本报告是采集后新增的整理材料。

## 3. PyTorch 导入结果

本次 PyTorch 为 `2.9.1+cpu`；`torch_runtime.exit_code=0`，stderr 为空，导入成功。

包元数据存在只能证明版本可查询，不能单独证明动态库和运行依赖完整。本次子进程实际导入成功，才支持“PyTorch 可导入”的结论；这不等于模型测试通过。

## 4. 设备与 CUDA 范围

当前使用 <abbr title="WSL / Windows Subsystem for Linux · Windows 上的 Linux 运行环境；本次检查只反映当前进程可见的路径、工具与设备。">WSL</abbr> 环境，JSON 平台信息为 WSL2。

| 检查项 | 本次结果 | 可支持的结论 |
|---|---|---|
| `cuda_build` | `null` | 当前 PyTorch 为 CPU 构建 |
| `cuda_available` | `false` | 当前进程无法使用 CUDA |
| `nvidia_smi` | `N/A: nvidia-smi not on PATH` | 当前 PATH 找不到工具；不能据此判断物理机器没有显卡 |

<abbr title="CUDA · NVIDIA GPU 的计算平台；PyTorch 是否包含 CUDA 支持、驱动工具是否可见、当前进程是否可用 GPU 是不同检查项。">CUDA</abbr> 当前不可用，后续可执行 CPU 路线；GPU 项记为 `PENDING_GPU`，尚未验证 GPU 正确性或性能。

## 5. 数据根目录与 D3 准备情况

JSON 的 `nanochat_base_dir` 为 `N/A: default ~/.cache/nanochat`，表示未设置覆盖变量，按 `nanochat/common.py:get_base_dir()` 使用默认路径。助手在当前 WSL 进程补充检查得到：

- `NANOCHAT_BASE_DIR=None`。
- 默认根路径：`/home/lizishuo/.cache/nanochat`。
- 该路径的 `exists()` 为 `False`。

用户补充说明使用的是 WSL。上述检查只说明当前 WSL 中该默认路径不存在，不能外推为 Windows 宿主机、其他挂载路径或其他环境没有数据。其他位置尚未核验，也尚未配置替代数据根目录。

以下路径是当前源码约定的候选位置，均相对于上述数据根目录；本次没有完成真实文件的完整性、可加载性或兼容性核验：

| 内容 | 候选相对路径 |
|---|---|
| 分词器 | `tokenizer/tokenizer.pkl` |
| 分词器辅助数据 | `tokenizer/token_bytes.pt` |
| 当前训练语料 | `base_data_climbmix/*.parquet` |
| 旧语料回退位置 | `base_data/*.parquet` |
| 预训练 checkpoint | `base_checkpoints/<模型标签>/` |
| 微调与强化学习 checkpoint | `chatsft_checkpoints/<模型标签>/`、`chatrl_checkpoints/<模型标签>/` |

<abbr title="Checkpoint · 保存模型权重及相关配置等信息的检查点；目录名称存在并不能证明权重完整或与当前模型兼容。">Checkpoint</abbr> 和真实语料不属于 D3 测试的前置条件。D3 使用测试替身和程序构造的张量，验证生成控制流、缓存容器和注意力回退等行为。

助手使用同一解释器补充执行 `import sys, torch, pytest` 成功，命令退出码为 `0`，确认 PyTorch 为 `2.9.1+cpu`、pytest 为 `8.4.2`。原始 JSON 还记录了 tiktoken、rustbpe、pyarrow 和 kernels 的包版本；这些元数据不替代 D3 的实际测试收集与执行。

当前具备进入 D3 CPU 测试的基础条件；D3 仍待实际执行，尚无本次通过数、跳过数或测试退出码。下一步按 [D03 教程](../../.agents/docs/career_roadmap/w1/d03_inference_smoke.md)运行完整两文件测试，保存真实日志并解释跳过范围。
