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
knowledge_scope: W3第5日实操与验收
---

# W3 D5 · 逐行审计和统计复算

[上一日](d18_batch_runs.md) · [下一日](d20_reproduction_entry.md)。

## 开始前：固定终端和记录方式

本日命令均从 nanochat 仓库根执行。先打开终端，激活已创建的 Conda 环境；若新终端提示找不到 `conda`，先运行 `source /home/lizishuo/miniconda3/etc/profile.d/conda.sh` 再激活，不创建另一个 `.venv`。

```bash
conda activate nanochat
cd /home/lizishuo/_program_/Project/nanochat
python -c "import sys; print(sys.executable)"
git status --short
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
exec > >(tee -a experiments/inference_infra/reports/d19_checks.log) 2>&1
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。依赖是否已装好要另查，激活成功不代表 PyTorch 可用。保留已有改动，遇到未认识的变更不要还原。教程脚本保存在本周 `scripts/`，结果写入 `experiments/inference_infra/`，两者不能混为同一类产物。

在 `experiments/inference_infra/reports/d15_experiment-log.md` 末尾追加当天日志，记录实际执行日期；标题里的 W3 Dn 是学习顺序，补做时不能倒填为此前已完成。

```text
学习槽位：W3 / 当日标题编号
实际开始与结束：
解释器、git commit、相关源码是否有未提交改动：
命令（完整参数）与退出码：
输入来源 / 产物路径：
观察到的事实：
尚未验证 / 阻塞原因：
下一步 / 下次检查时间：
```

## 今天的目标（约 3 小时）

今天从 raw 重新计算结果，确保每个表格数字都能追到原始行。<abbr title="JSONL / JSON Lines · 每行一个JSON对象的文本格式；一行对应一次运行，失败行和预热行也单独保存。">JSONL</abbr> 方便追加，但同一文件重复执行会混入多轮；必须检查 run_id 和实验条件。<abbr title="Median · 中位数；数据排序后居中的值，偶数样本取中间两值平均。相比最小值更能表示典型一次运行。">median</abbr> 代表典型运行，不是最好的一次成绩。

前置：至少有一个真正生成的 raw 文件；只有 PENDING_GPU 清单时不能产出性能 CSV。时间分配：30 分钟筛选文件，40 分钟聚合，40 分钟抽样复算，40 分钟检查失败与结论。

## 第一步：选择一个独立条件的 raw 目录

使用[统计审计脚本](scripts/d13_aggregate.py)。它读取所给目录下 `dNN_p*_g*_b*.jsonl`（兼容迁移前 `p*_g*_b*.jsonl`），不递归，也不把计划矩阵当原始测量。CPU 契约示例使用上级 raw 目录；GPU B1/B4 各自 attempt 目录应分别运行，防止未经检查混并。

```bash
python .agents/docs/career_roadmap/w3/scripts/d13_aggregate.py \
  --raw-dir experiments/inference_infra/results/raw \
  --output experiments/inference_infra/results/summary/d19_baseline.csv \
  --audit experiments/inference_infra/results/summary/d19_baseline_audit.jsonl
```

若你实际数据在 `w3_b1_attempt01`，修改 raw-dir，同时用独立输出名 `experiments/inference_infra/results/summary/d19_baseline_b1.csv`（审计另用 `experiments/inference_infra/results/summary/d19_baseline_b1_audit.jsonl`；B4分别使用 `experiments/inference_infra/results/summary/d19_baseline_b4.csv` 和 `experiments/inference_infra/results/summary/d19_baseline_b4_audit.jsonl`）；B4 同理。只有正式合并时才按明确的记录清单复制或汇总到新目录，并检查相同 checkpoint/硬件/软件/后端；不要用 `cat` 盲目拼整个 raw 树。

预期提示 `N groups -> ...`。没有匹配文件时明确报错；这比输出空表伪装完成更有用。脚本不修改 raw；会重写指定 summary 和 audit，因此保留要对照的旧版时先选新输出文件名。

## 第二步：理解脚本的统计定义

<abbr title="P95 · 经验第95百分位；本脚本采用nearest-rank，排序后取ceil(0.95×n)处的值。n=5时等于最大值，仅是少量样本描述，不是稳定尾延迟估计。">P95</abbr> 与 min/max 一起报告。5 次样本的 P95 恰为最大值，不要写成有可靠 95% 服务保障。

每次运行的指标从原始字段重算：

- TTFT：`ttft_ms`，第一次 next 到首 yield 同步后的时间。
- TPOT：`mean(decode_step_ms)`，G−1 个后续 token 列的算术平均；B4 不再除以 4，因为这是用户等待每个生成步的时间。
- tokens/s：`generated_tokens / (elapsed_ms/1000)`，全部有效输出 token 除以首 next 至最后 yield 的端到端时间，包含 TTFT；不包括 prompt token。

表中的中位数与 P95 都先算每次运行的指标，再跨有效正式运行统计。没有将全部 decode step 混起来算某个“总体 TPOT”。上游 infer_bench 使用逐步时间中位数和不含首步的 decode 吞吐，与此处定义不同，不能混表比较。

## 第三步：核查哪些行被排除

脚本要求：phase=measure、status=ok、实际 G 步、G−1 个后续时间、B 行均有效生成 G 个 token、正且有限的耗时、合法上下文、总时间大致等于 TTFT 加后续时间。提前结束、强制 token 导致计数不足、错误、截断和非法时间都写入 audit 的排除原因。预热不计入正式样本，但仍检查 run_id 是否重复，并要求正式GPU结论所用session至少3次成功预热。未知schema_version的正式行标invalid并留审计记录。表中另保留有效行的peak_allocated_bytes和peak_reserved_bytes的median/min/max；CPU字段为空，不填0。

组键包含 session_id、模型权重与tokenizer哈希、来源与 step、seed、模式、设备、精度、后端、P/G/B、完整 env 和 model_config，环境不同或重复session不会仅因为形状相同就合并。没有有效正式样本标 `NO_VALID_SAMPLES`；有有效行的 CPU 或 random 模型标 `CPU_OR_RANDOM_CONTRACT`。GPU checkpoint 有效行少于 5 标 `INSUFFICIENT_SAMPLES`，同session不足3次成功预热或有失败预热则标 `INSUFFICIENT_WARMUP`。`GPU_BASELINE` 只证明样本结构满足条件，仍需人工确认权重哈希、输入和背景负载。

## 第四步：手算一个小例子核对单位

教学例子不是测量结果：假设 B=1、G=4，TTFT=10ms，decode step=[2,3,4]ms，则 elapsed≈19ms，TPOT=3ms，吞吐=4/0.019≈210.53 token/s。B=4 且每行完整4个 token 时，分子是16，但绝不能提前假定 elapsed仍19ms。

抽取一个真实 run_id，用计算器按同样公式复算，然后在 `experiments/inference_infra/reports/d19_statistics_audit.md` 写：文件、行号、run_id、原始值、公式、自动表对应组键、差异。基准程序另存的 tpot/tokens_per_s 与复算值不一致时先核查定义，不改 raw 数字迎合表格。

## 失败分流与验收

- run_id 重复：确定是否重复复制文件，移除重复副本而不是修改 run_id 制造新样本。
- JSON损坏：定位行号并保留损坏原件，追查中断/写入原因；不要删掉后声称样本完整。
- n_valid=0：保留该组失败数，下一次修复后重跑新 attempt。
- 没有 GPU：基线表明确 CPU 契约，不画 GPU 加速图，不把 pending 填0。

验收：d19_baseline.csv 存在或有无数据阻塞说明；每个有效统计带 run_ids；每个排除行有 audit 原因；单位、P95方法、样本数、失败数和设备明确。下一天用真实通过的命令写复现入口。

## 本日脚本与产物清单

下列路径以项目根目录为基准；脚本编号保留首次使用日。外部历史归档与上游 checkpoint 原生文件名保留，按正文真实绝对路径或环境变量解析；不冒充项目新产物。<abbr title="条件生成：只有指定分支执行、前置文件满足且实际写入后，文件才算存在；清单中的目标路径本身不证明生成成功。">条件生成</abbr>须在验收时核对。只打印的检查统一进入本日 checks 日志，不另称生成报告。

| 脚本完整项目相对路径 | 用途 | 阶段 | 输入路径或来源 | 运行条件 |
|---|---|---|---|---|
| [.agents/docs/career_roadmap/w3/scripts/d13_aggregate.py](scripts/d13_aggregate.py) | 逐行审计并重算统计 | 第1–3步 | --raw-dir 的真实测量文件；不读计划矩阵 | 标准库；至少一条正式行，只有预热或无raw时报错 |

| 产物完整项目相对路径 | 用途 | 生成者/步骤 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/reports/d19_checks.log` | 环境、只读检查、命令输出及错误 | 开始前 Bash 的 exec / tee | 自动捕获后续终端输出；退出码人工记入日志 |
| `experiments/inference_infra/reports/d15_experiment-log.md` | 每日事实、退出码和阻塞 | 人工追加；跨日复用首次文件 | 人工编写 |
| `experiments/inference_infra/results/summary/d19_baseline.csv` | 独立条件的统计表 | d13_aggregate.py --output | 条件生成；有测量组时生成，可含无有效样本组 |
| `experiments/inference_infra/results/summary/d19_baseline_audit.jsonl` | 逐行纳入/排除原因 | d13_aggregate.py --audit | 随统计成功自动生成 |
| `experiments/inference_infra/results/summary/d19_baseline_b1.csv` | B1 独立目录汇总 | d13_aggregate.py --output | 条件生成；仅选择 B1 attempt 目录并指定此输出时 |
| `experiments/inference_infra/results/summary/d19_baseline_b1_audit.jsonl` | B1 逐行审计 | d13_aggregate.py --audit | 随对应 B1 统计成功生成 |
| `experiments/inference_infra/results/summary/d19_baseline_b4.csv` | B4 独立目录汇总 | d13_aggregate.py --output | 条件生成；仅选择 B4 attempt 目录并指定此输出时 |
| `experiments/inference_infra/results/summary/d19_baseline_b4_audit.jsonl` | B4 逐行审计 | d13_aggregate.py --audit | 随对应 B4 统计成功生成 |
| `experiments/inference_infra/reports/d19_statistics_audit.md` | 人工抽取 run_id 复算记录 | 第4步 | 人工编写 |

## 核对记录与完成标记

本文基于当前仓库相关生成、加载、注意力路由及原基准代码编写；源码阅读不能代替执行。本教程的交付状态不表示读者已经完成当天实验。填写实际日志后再勾选：

- [ ] 输入条件与前一天产物已核实。
- [ ] 本日命令在明确设备执行，或具体阻塞已记录。
- [ ] 日志、原始输出与材料保存到所写位置。
- [ ] 示例数值与真实结果已区分，失败和待测未删除。
- [ ] 能说明本日证据证明什么、尚不能证明什么。
