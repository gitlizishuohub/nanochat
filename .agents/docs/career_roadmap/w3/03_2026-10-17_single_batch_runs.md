---
source_files:
  - nanochat/engine.py
  - nanochat/gpt.py
  - nanochat/checkpoint_manager.py
  - nanochat/flash_attention.py
  - scripts/infer_bench.py
  - tests/test_engine.py
  - tests/test_attention_fallback.py
doc_type: tutorial
knowledge_role: supporting
knowledge_scope: W3第3日实操与验收
---

# W3 · 2026-10-17 · 单样本基线与无 GPU 分流

[上一日](02_2026-10-16_baseline_matrix.md) · [下一日](04_2026-10-18_batch_runs.md)。

## 开始前：固定终端和记录方式

本日命令均从 nanochat 仓库根执行。先打开终端，激活已创建的 Conda 环境；若新终端提示找不到 `conda`，先运行 `source /home/lizishuo/miniconda3/etc/profile.d/conda.sh` 再激活，不创建另一个 `.venv`。

```bash
conda activate nanochat
cd /home/lizishuo/_program_/Project/nanochat
python -c "import sys; print(sys.executable)"
git status --short
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。依赖是否已装好要另查，激活成功不代表 PyTorch 可用。保留已有改动，遇到未认识的变更不要还原。教程脚本保存在本周 `scripts/`，结果写入 `experiments/inference_infra/`，两者不能混为同一类产物。

在 `experiments/inference_infra/reports/experiment-log.md` 末尾追加当天日志，记录实际执行日期；标题里的日期是原计划槽位，补做时不能倒填为当时已完成。

```text
计划槽位：W3 / 当日标题日期
实际开始与结束：
解释器、git commit、相关源码是否有未提交改动：
命令（完整参数）与退出码：
输入来源 / 产物路径：
观察到的事实：
尚未验证 / 阻塞原因：
下一步 / 下次检查时间：
```

## 今天的目标（约 2–4 小时，计时任务可缩短）

今天检查 B=1 的三个输入长度与两个生成长度中可运行的子集。每个有效 GPU case 至少 5 次正式运行；无 GPU 时验证 CPU 契约和执行入口，GPU 结果留 `PENDING_GPU`。前置：昨天的固定模型、矩阵和协议；不因为日历推进而跳过 W2 加载检查。

<abbr title="TTFT / Time To First Token · 首token时延；本教程从第一次next开始到首个yield完成并同步，包含Engine缓存分配、预填、复制及采样，不包括tokenizer与模型加载。">TTFT</abbr> 包含首个输出前的 Engine 工作。<abbr title="TPOT / Time Per Output Token · 后续输出步平均耗时；本周每次运行用首步之后的逐步同步时间求算术平均，B大于1时为一列token的步时间。">TPOT</abbr> 衡量首 token 之后每步时间，不能把 TTFT 简单当成纯矩阵乘的 prefill 时间。

## 第一步：核对设备和依赖

```bash
python -c "import torch; print(torch.__version__); print('cuda_available=',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
python .agents/docs/career_roadmap/w2/scripts/benchmark.py --help
```

若 import 失败，记录解释器和错误，先完成环境修复，不把“无 torch”误写成“CPU 测试通过”。有 GPU 但返回 false 时检查所装 PyTorch 版本与驱动；今天不自动租卡或联网安装。

## 第二步 A：有 GPU，显式执行 B=1

在同一终端恢复昨天的真实 META、MODEL_TAG、CHECKPOINT_STEP 和 NANOCHAT_BASE_DIR。以下命令使用[矩阵驱动](scripts/matrix.py)，从项目根运行：

```bash
python .agents/docs/career_roadmap/w3/scripts/matrix.py \
  --meta "$META" --source base --model-tag "$MODEL_TAG" --step "$CHECKPOINT_STEP" \
  --batch 1 --device cuda --execute \
  --output-dir experiments/inference_infra/results/raw/w3_b1_attempt01
```

`--execute` 是实际运行开关；去掉它仅列计划。程序逐 case 等待并保存完整 log；`PROCESS_OK` 仅说明子进程退出码为 0，不等于全部正式行都有效。其余 B=4 case 仍列 PENDING。顺序固定有利于首次核查，正式分析仍需考虑温度、时钟和背景负载；不要把一次跑出来的最小值当成绩。

在运行中观察机器负载和日志；一组耗时明显超出预算可用 Ctrl+C 停止，保留已落盘行，日志写中断时间。中断可能导致矩阵缺未落盘 case，不能标成全矩阵完成；恢复用新的 attempt 目录。

## 第二步 B：无 GPU，执行一个小 CPU 契约

先保留昨天 GPU 计划不动，再用[W2 基准脚本](../w2/scripts/benchmark.py) 的随机小模型做短输入验证。<abbr title="Smoke test · 冒烟验证；用很小工作量检查接口、文件格式与执行路径，不证明模型质量或GPU性能。">smoke</abbr> 只检查程序是否形成闭环。

```bash
python .agents/docs/career_roadmap/w2/scripts/benchmark.py \
  --device cpu --random-model --prompt-tokens 32 --decode-tokens 8 --batch-size 1 \
  --warmup 3 --repeats 5 --seed 42 \
  --output experiments/inference_infra/results/raw/p32_g8_b1_cpu_contract.jsonl
python -m pytest tests/test_engine.py tests/test_attention_fallback.py -q
```

随机权重/测试 tokenizer 是接口夹具，不能代替实际 checkpoint 的性能报告。CPU SDPA 测试通过也不证明 FA3 通过，GPU 测试被跳过必须单列 skip 数。CPU 机器资源有限时先只跑 Engine 测试，完整 fallback 单测后续补记。

## 第三步：看一行真实原始记录

用下面只读命令，替换为你刚生成的 case 文件；不要拿 matrix.jsonl 当 benchmark raw。

```bash
python -c "import json,pathlib; p=pathlib.Path('experiments/inference_infra/results/raw/p32_g8_b1_cpu_contract.jsonl'); r=[json.loads(x) for x in p.read_text().splitlines()]; print(json.dumps(r[-1],indent=2,ensure_ascii=False))"
```

检查 `phase` 区分 warmup/measure；正式至少 5 行，`status=ok` 才候选有效。`actual_decode_steps` 是 yield 列数，`generated_tokens` 是排除终止、forced 和完成行 padding 后的有效 token 总数，两者不可互换；B=1 也可能因工具规则而不同。检查 TTFT、decode_step_ms、elapsed_ms 都为正且单位 ms，`env`、model_config、后端、精度都有记录。

## 失败分流、产物与验收

- `early_stop`：保留原始行，检查 assistant_end/bos 或工具强制 token；不能通过修改 EOS 语义把它包装成原协议成功。
- OOM/error：查看同名 log 的异常；模型加载期失败可能没有 raw 行，但进程退出码仍有记录。
- `INVALID_CONTEXT`：今天不运行；应扩大合法 checkpoint 上下文并重建不同实验，而非悄悄截断 P。
- 正式有效不足 5 次：可继续补跑同条件新 attempt，但跨目录合并前审计模型、env 和 run_id；今天可标不足样本。

在 `reports/baseline_b1.md` 写每 case 的计划状态、退出码、measure总数/有效数、raw与log路径。验收：所有 B1 候选都可追踪；至少一个可用设备的功能验证或明确依赖阻塞；没有将 CPU/random 数字写成 GPU baseline。下一天专看批量行为。

## 核对记录与完成标记

本文基于当前仓库相关生成、加载、注意力路由及原基准代码编写；源码阅读不能代替执行。本教程的交付状态不表示读者已经完成当天实验。填写实际日志后再勾选：

- [ ] 输入条件与前一天产物已核实。
- [ ] 本日命令在明确设备执行，或具体阻塞已记录。
- [ ] 日志、原始输出与材料保存到所写位置。
- [ ] 示例数值与真实结果已区分，失败和待测未删除。
- [ ] 能说明本日证据证明什么、尚不能证明什么。
