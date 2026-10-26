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

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[13.1 四种评价出口](../../llm-training-workflow/13-evaluation-and-metrics.md#reading-13-1)；[13.2 能力与系统性能的边界](../../llm-training-workflow/13-evaluation-and-metrics.md#reading-13-2)。
- **今天重点**：完整认识四种能力评价出口，再回到当天统计复算，明确损失、任务正确率、推理延迟和吞吐各自的统计对象。
- **暂时跳过**：能力指标公式只需能解释分子、分母与范围；不必下载全部评测任务或运行能力评测。

**读完能解释**：<abbr title="BPB / Bits per Byte · 每字节比特数；用有效目标负对数似然和字节数衡量预测误差。">BPB</abbr>、任务准确率与 <abbr title="TPOT / Time per Output Token · 首词元之后的平均输出间隔；具体统计以当日协议为准。">TPOT</abbr> 分别评价什么？为什么一个指标变好不代表另外两个也变好？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 开始前：固定终端和记录方式

本日命令均从 nanochat 仓库根执行。当前终端会自动激活 nanochat 环境，无需重复激活；先核对解释器路径，不创建另一个 `.venv`。

日志只在 `{ ...; }` 命令组执行期间记录；<abbr title="Bash process substitution · 进程替换。这里把命令组的输出交给 tee，同时显示并追加到日志；重定向在命令组结束后恢复。">进程替换</abbr>不会持续接管交互终端，因此提示符和 VS Code 终端集成控制字符不会混入日志。命令组在当前 shell 执行，变量设置会保留；代码块返回最后一条命令的退出码，需核对单条命令时立即记录 `$?`。已有的显式文件重定向仍写入各自文件。复制时包含外层花括号和末尾重定向。若之前执行过旧版全局日志命令，先关闭旧终端并新开终端；仅修改文档不会停止旧终端的记录。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
{
python -c "import sys; print(sys.executable)"
git status --short
} > >(tee -a experiments/inference_infra/reports/d19_checks.log) 2>&1
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。依赖是否已装好要另查，激活成功不代表 PyTorch 可用。保留已有改动，遇到未认识的变更不要还原。教程脚本保存在本周 `scripts/`，结果写入 `experiments/inference_infra/`，两者不能混为同一类产物。

本日验证完成后，直接按文末“本日自验证与交流”逐题和我交流，附关键输出或自动日志/原始结果路径即可。无需手写报告，也不要求固定排版或长文。实际日期、设备、命令参数与退出码以日志和聊天核对；标题里的 W3 Dn 是学习顺序，不倒填完成状态。若想自行整理，可选用 `experiments/inference_infra/reports/d15_experiment-log.md`，后续跨日也可引用本次聊天和自动结果。

## 今天的目标（约 3 小时）

今天从 raw 重新计算结果，确保每个表格数字都能追到原始行。<abbr title="JSONL / JSON Lines · 每行一个JSON对象的文本格式；一行对应一次运行，失败行和预热行也单独保存。">JSONL</abbr> 方便追加，但同一文件重复执行会混入多轮；必须检查 run_id 和实验条件。<abbr title="Median · 中位数；数据排序后居中的值，偶数样本取中间两值平均。相比最小值更能表示典型一次运行。">median</abbr> 代表典型运行，不是最好的一次成绩。

前置：至少有一个真正生成的 raw 文件；只有 PENDING_GPU 清单时不能产出性能 CSV。时间分配：30 分钟筛选文件，40 分钟聚合，40 分钟抽样复算，40 分钟检查失败与结论。

## 第一步：选择一个独立条件的 raw 目录

使用[统计审计脚本](scripts/d13_aggregate.py)。它读取所给目录下 `dNN_p*_g*_b*.jsonl`（兼容迁移前 `p*_g*_b*.jsonl`），不递归，也不把计划矩阵当原始测量。CPU 契约示例使用上级 raw 目录；GPU B1/B4 各自 attempt 目录应分别运行，防止未经检查混并。

```bash
{
python .agents/docs/career_roadmap/w3/scripts/d13_aggregate.py \
  --raw-dir experiments/inference_infra/results/raw \
  --output experiments/inference_infra/results/summary/d19_baseline.csv \
  --audit experiments/inference_infra/results/summary/d19_baseline_audit.jsonl
} > >(tee -a experiments/inference_infra/reports/d19_checks.log) 2>&1
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

脚本要求：phase=measure、status=ok、实际 G 步、G−1 个后续时间、B 行均有效生成 G 个 token、正且有限的耗时、合法上下文、总时间大致等于 TTFT 加后续时间。非预热行中的提前结束、强制 token 导致计数不足、错误、截断和非法时间都写入 audit 的排除原因。预热不计入正式样本，但仍检查 run_id 是否重复，并要求正式GPU结论所用session至少3次成功预热。未知schema_version的正式行标invalid并留审计记录。表中另保留有效行的peak_allocated_bytes和peak_reserved_bytes的median/min/max；CPU字段为空，不填0。

组键包含 session_id、模型权重与tokenizer哈希、来源与 step、seed、模式、设备、精度、后端、P/G/B、完整 env 和 model_config，环境不同或重复session不会仅因为形状相同就合并。没有有效正式样本标 `NO_VALID_SAMPLES`；有有效行的 CPU 或 random 模型标 `CPU_OR_RANDOM_CONTRACT`。有有效行的 GPU checkpoint 组若同session不足3次成功预热或有失败预热，优先标 `INSUFFICIENT_WARMUP`；预热满足但有效正式行少于5时标 `INSUFFICIENT_SAMPLES`。`GPU_BASELINE` 只证明样本结构满足条件，仍需人工确认权重哈希、输入和背景负载。

## 第四步：手算一个小例子核对单位

教学例子不是测量结果：假设 B=1、G=4，TTFT=10ms，decode step=[2,3,4]ms，则 elapsed≈19ms，TPOT=3ms，吞吐=4/0.019≈210.53 token/s。B=4 且每行完整4个 token 时，分子是16，但绝不能提前假定 elapsed仍19ms。

抽取一个真实 run_id，用计算器按同样公式复算，然后在聊天给出文件、行号、run_id、原始值、公式、自动表对应组键和差异；可选整理到 `experiments/inference_infra/reports/d19_statistics_audit.md`，不要求另写报告。基准程序另存的 tpot/tokens_per_s 与复算值不一致时先核查定义，不改 raw 数字迎合表格。

## 失败分流与验收

- run_id 重复：确定是否重复复制文件，移除重复副本而不是修改 run_id 制造新样本。
- JSON损坏：定位行号并保留损坏原件，追查中断/写入原因；不要删掉后声称样本完整。
- n_valid=0：保留该组失败数，下一次修复后重跑新 attempt。
- 没有 GPU：基线表明确 CPU 契约，不画 GPU 加速图，不把 pending 填0。

验收：d19_baseline.csv 存在或有无数据阻塞说明；每个有效统计带 run_ids；每个被排除的非预热行有 audit 原因，预热另从 raw 核查；单位、P95方法、样本数、失败数和设备明确。下一天用真实通过的命令写复现入口。

## 本日脚本与产物清单

下列路径以项目根目录为基准；脚本编号保留首次使用日。外部历史归档与上游 checkpoint 原生文件名保留，按正文真实绝对路径或环境变量解析；不冒充项目新产物。<abbr title="条件生成：只有指定分支执行、前置文件满足且实际写入后，文件才算存在；清单中的目标路径本身不证明生成成功。">条件生成</abbr>须在验收时核对。只打印的检查统一进入本日 checks 日志，不另称生成报告。

| 脚本完整项目相对路径 | 用途 | 阶段 | 输入路径或来源 | 运行条件 |
|---|---|---|---|---|
| [.agents/docs/career_roadmap/w3/scripts/d13_aggregate.py](scripts/d13_aggregate.py) | 逐行审计并重算统计 | 第1–3步 | --raw-dir 的真实测量文件；不读计划矩阵 | 标准库；至少一条正式行，只有预热或无raw时报错 |

| 产物完整项目相对路径 | 用途 | 生成者/步骤 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/reports/d19_checks.log` | 环境、只读检查、命令输出及错误 | 各 Bash 代码块的 tee | 自动追加各代码块内的输出；退出码在聊天核对，或可选补记日志 |
| `experiments/inference_infra/reports/d15_experiment-log.md` | 每日事实、退出码和阻塞 | 自愿整理；跨日可引用聊天和自动结果 | 可选人工笔记，不影响验收 |
| `experiments/inference_infra/results/summary/d19_baseline.csv` | 独立条件的统计表 | d13_aggregate.py --output | 条件生成；有测量组时生成，可含无有效样本组 |
| `experiments/inference_infra/results/summary/d19_baseline_audit.jsonl` | 逐行纳入/排除原因 | d13_aggregate.py --audit | 随统计成功自动生成 |
| `experiments/inference_infra/results/summary/d19_baseline_b1.csv` | B1 独立目录汇总 | d13_aggregate.py --output | 条件生成；仅选择 B1 attempt 目录并指定此输出时 |
| `experiments/inference_infra/results/summary/d19_baseline_b1_audit.jsonl` | B1 逐行审计 | d13_aggregate.py --audit | 随对应 B1 统计成功生成 |
| `experiments/inference_infra/results/summary/d19_baseline_b4.csv` | B4 独立目录汇总 | d13_aggregate.py --output | 条件生成；仅选择 B4 attempt 目录并指定此输出时 |
| `experiments/inference_infra/results/summary/d19_baseline_b4_audit.jsonl` | B4 逐行审计 | d13_aggregate.py --audit | 随对应 B4 统计成功生成 |
| `experiments/inference_infra/reports/d19_statistics_audit.md` | 人工抽取 run_id 复算记录 | 第4步交流后自愿整理 | 可选人工笔记，聊天中的复算可替代 |

## 本日自验证与交流

可直接在聊天中按题号回答，不要求手写报告。每题给出结论和可定位依据（源码位置、命令与退出码、日志片段或自动结果字段）；同一份证据可复用。概念推演须注明假设，实际运行须区分本次与历史结果。未执行写 `PENDING`，失败写真实错误和影响范围，不能据此勾选实验通过。

1. **输入选择与输出归属**：本次raw-dir、CSV、audit的完整路径是什么？扫描是否递归、匹配哪些文件名，为何matrix不能当raw？

   **通过标准**：引用实际聚合命令与glob，不混不同attempt树；无匹配、只有预热或无测量组按真实报错说明，不用空表伪装完成，重复执行会重写summary/audit。

2. **分组身份与防混并**：取两个P/G/B相同的记录，哪些字段决定能否同组？session、权重/词表/输入hash、环境、config分别如何处理？

   **通过标准**：逐项对应KEYS及env/model_config，说明不同session不会自动合并；CSV run_ids追溯有效行，只有人工核查后才能考虑跨attempt对照，不盲目cat目录。

3. **一行完整性检查**：选一个真实measure+ok，怎样核验上下文、G列、G−1间隔、B行有效数和总数、正有限时间及elapsed一致性？

   **通过标准**：引用metrics每条判断与原始字段，status=ok仍需内容验证；时间一致性按脚本允许的rel_tol/abs_tol核对，缺字段/未知schema进入invalid原因。

4. **逐 run 手工复算**：给出一个run_id的TTFT、decode_step_ms、generated_tokens、elapsed_ms，复算TPOT和tokens/s，与自动结果相差多少？

   **通过标准**：提供file/line/run_id、公式和单位；TPOT为后续步算术均值，端到端吞吐含TTFT，不把上游infer_bench的步内中间值/排除TTFT吞吐混入。

5. **跨运行的统计顺序**：自动表怎样从每run指标得到median/min/max/P95？G=1时TPOT怎样保留，能否将所有step混成总体TPOT？

   **通过标准**：对应metrics与聚合循环，解释先每run再跨run及None/空单元格；区别median(tokens/s)与总token/总时长，不将两种统计因数值接近而混写。

6. **P95 的可手算定义**：对教学样本[9,1,5,2,3]ms计算median、nearest-rank P95、min/max；为什么不能把结果写成稳定的95%服务保障？

   **通过标准**：注明假设，排序并按ceil(0.95N)定位，N=5时P95取最大值；同时对照真实组的n_valid，区分经验分位数、置信区间与线上保障。

7. **排除行与 audit 范围**：找一条被排除的非warmup行，reason是什么；若没有排除行，选纳入行逐项解释。预热为什么不出现在audit和n_total中？

   **通过标准**：对照audit的file/line/run_id/included/reason与warmup分支continue；预热另从raw检查，未知schema/early_stop/error不能静默消失，缺真实异常不必制造失败。

8. **重复、损坏与样本账**：n_total、n_valid、n_excluded与有效run_ids怎样对应？重复run_id、缺run_id、坏JSON和只有warmup时脚本怎样处理？

   **通过标准**：说明n_total为该组非warmup条目，重复ID在预热也检查；解析/身份错误可使聚合中止，不能保证仍产出audit，应保留原件和退出码，不能改run_id制造新样本。

9. **分层状态与预热**：实际conclusion_status和warmup_status是什么？NO_VALID_SAMPLES、CPU_OR_RANDOM_CONTRACT、INSUFFICIENT_WARMUP、INSUFFICIENT_SAMPLES、GPU_BASELINE分别需什么条件？

   **通过标准**：按脚本优先级判断：先有效样本，再设备/模型及预热/数量；同身份session至少3次status=ok且schema=1预热，GPU基线还需≥5有效正式行；预热仅按 status/schema 记账，没有执行 metrics() 对正式行的完整检查，GPU_BASELINE仍需人工核身份和负载。

10. **内存、负载与结论范围**：表中allocated/reserved怎样从有效行汇总，CPU空单元格为什么不写0？哪些权重、背景负载、GPU路径仍缺人工核验？

    **通过标准**：引用实际内存median/min/max或未测状态，区分分配器保留与活跃模型内存；只报告现有数据支持的结论，下一天复现入口引用真实通过命令和限制。

## 核对记录与完成标记

本文基于当前仓库相关生成、加载、注意力路由及原基准代码编写；源码阅读不能代替执行。本教程的交付状态不表示读者已经完成当天实验。按当天问题核对真实证据后再勾选；聊天回答即可，无需额外报告：

- [ ] 输入条件与前一天产物已核实。
- [ ] 本日命令在明确设备执行，或具体阻塞已记录。
- [ ] 自动日志和原始输出可定位，实践材料或聊天回答可回溯；可选笔记无需生成。
- [ ] 示例数值与真实结果已区分，失败和待测未删除。
- [ ] 能说明本日证据证明什么、尚不能证明什么。
