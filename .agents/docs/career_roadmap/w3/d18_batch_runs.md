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
knowledge_scope: W3第4日实操与验收
---

# W3 D4 · 批量生成与失败记录

[上一日](d17_single_batch_runs.md) · [下一日](d19_statistics_audit.md)。

## 开始前：固定终端和记录方式

本日命令均从 nanochat 仓库根执行。先打开终端，激活已创建的 Conda 环境；若新终端提示找不到 `conda`，先运行 `source /home/lizishuo/miniconda3/etc/profile.d/conda.sh` 再激活，不创建另一个 `.venv`。

```bash
conda activate nanochat
cd /home/lizishuo/_program_/Project/nanochat
python -c "import sys; print(sys.executable)"
git status --short
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
exec > >(tee -a experiments/inference_infra/reports/d18_checks.log) 2>&1
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

## 今天的目标与前置（约 3 小时）

先确认昨天 B=1 至少有可解释的执行结果，再测 B=4；如果 B1 就 OOM，今天优先记录边界，不通过换成另一模型后仍沿用同名基线。<abbr title="Batch size · 批大小；Engine此处为一次prefill后复制缓存的num_samples，每行共享同一prompt，不能直接等同四个独立请求。">batch</abbr> 增大可能提高总 token 吞吐，也可能增加单步耗时和显存；不能预设“四倍 batch 几乎免费”。

时间分配：30 分钟检查内存预算，60–90 分钟执行，40 分钟检查逐行完成数，30 分钟写失败表。

## 第一步：重看昨天配置与缓存公式

从昨天 log 确认模型、dtype、后端、P/G 和 seed。比较 B=1 与 B=4 时保持这些条件相同。Engine 在 batch=1 预填后复制 KV，因此 B4 的 TTFT 多了缓存复制和更大分配；不是四次完整 prefill。

<abbr title="OOM / Out Of Memory · 内存不足；本处应区分模型加载期、缓存分配期和计算期。失败说明当前配置资源不足，不能作为正常延迟样本。">OOM</abbr> 必须保留错误与输入组合。使用第 2 天公式估计 B4 解码缓存约为 B1 的四倍，但总峰值显存通常不会精确四倍，因为权重等共享部分仍存在。GPU reserved 与 allocated 不同：前者包括分配器保留，后者反映当前张量占用。

## 第二步：执行 B4 的可运行子集

使用[矩阵驱动](scripts/d16_matrix.py)，仍把实际 META 等变量从协议恢复。默认至少 5 次正式运行，CPU 分支不以速度为今天的目标。

```bash
python .agents/docs/career_roadmap/w3/scripts/d16_matrix.py \
  --artifact-day 18 --meta "$META" --source base --model-tag "$MODEL_TAG" --step "$CHECKPOINT_STEP" \
  --batch 4 --device cuda --execute \
  --output-dir experiments/inference_infra/results/raw/w3_b4_attempt01
```

若无 CUDA，先不要执行以上命令，GPU case 保留计划状态；另跑下面短输入的 CPU 合约例子：

```bash
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py \
  --device cpu --random-model --prompt-tokens 32 --decode-tokens 8 --batch-size 4 \
  --warmup 3 --repeats 5 --seed 42 \
  --output experiments/inference_infra/results/raw/d18_p32_g8_b4_cpu_contract.jsonl
```

没有必要在 CPU 上强跑长 P512×G128 并等待数小时。GPU 候选完整性由 matrix 状态表示，CPU 小样本只用于检查 B4 接口与有效 token 记账。

## 第三步：检查有效 token，而非假定 B×G

打开 raw 的 `per_row_generated_tokens`。满长度正常运行时应有 4 项，每项 G，且和为 `generated_tokens`。某行提前终止后 Engine 仍可能产生列；不能把列数乘 B 当全部有效输出。forced token 是工具返回等引入的 token，其工作量和普通采样不同，本周统计排除并将未达到完整条件的行留作异常。

<abbr title="EOS / End Of Sequence · 序列结束标记；Engine实际上按assistant_end或bos等项目专用标记终止行，不能假定所有生成器都共享同一个EOS规则。">EOS</abbr> 在此是结束行为的统称；GPT.generate 简单循环不会采用同一套 Engine 行状态规则。由此带来的有效 token 差异不应自动判作 cache 数值错误。

对比 `actual_decode_steps`、`decode_step_ms` 长度和有效 token 数。G 步输出通常包含一次 TTFT 和 G−1 个后续步；最后 StopIteration 的额外边界不作为已输出 token 的延迟。基准封装每步同步会引入开销，应保持 B1/B4 方法相同。

## 第四步：写一张包含失败的表

保存 `experiments/inference_infra/reports/d18_baseline_batch.md`，每个 P/G 下对 B1 与 B4 并列：

```text
P/G/B：
checkpoint / device / precision / backend：
raw / process log：
measure数 / 完整有效数 / early_stop数 / error数：
实际每行有效token：
OOM发生阶段（从异常位置核查，不能猜）：
峰值allocated/reserved（CPU写N/A）：
是否满足进入正式统计门槛：
```

可以新增 B2 作为资源边界探索，但必须新目录、原始字段 batch_size=2，并在报告声明它不属于原定 12 组；本周 d16_matrix.py 仅接受原计划 B1/B4。不要覆盖 B4 失败行。

## 验收与下一步

B4 在同条件下成功或失败都可以验收“边界记录完成”；只有 >=5 有效 GPU 正式行才能写该 case 基线。若 B1/B4 工作量不等、模型或后端不同，先标不可直接比较。完成后保留全部 raw、log 和失败原因，明天统一自动汇总，不手工抄最快值。

## 本日脚本与产物清单

下列路径以项目根目录为基准；脚本编号保留首次使用日。外部历史归档与上游 checkpoint 原生文件名保留，按正文真实绝对路径或环境变量解析；不冒充项目新产物。<abbr title="条件生成：只有指定分支执行、前置文件满足且实际写入后，文件才算存在；清单中的目标路径本身不证明生成成功。">条件生成</abbr>须在验收时核对。只打印的检查统一进入本日 checks 日志，不另称生成报告。

| 脚本完整项目相对路径 | 用途 | 阶段 | 输入路径或来源 | 运行条件 |
|---|---|---|---|---|
| [.agents/docs/career_roadmap/w3/scripts/d16_matrix.py](scripts/d16_matrix.py) | 固定12个组合，按开关执行子集 | 计划/显式运行 | META 指向真实上游 meta；MODEL_TAG、CHECKPOINT_STEP | 计划需真实 meta；--execute 需权重/tokenizer，CUDA运行需GPU |
| [.agents/docs/career_roadmap/w2/scripts/d12_benchmark.py](../w2/scripts/d12_benchmark.py) | 矩阵内部调用；CPU分支直接调用 | 逐次测量 | checkpoint及合成prompt；CPU示例使用随机夹具 | 依赖 PyTorch；真实基准需匹配 tokenizer |

| 产物完整项目相对路径 | 用途 | 生成者/步骤 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/reports/d18_checks.log` | 环境、只读检查、命令输出及错误 | 开始前 Bash 的 exec / tee | 自动捕获后续终端输出；退出码人工记入日志 |
| `experiments/inference_infra/reports/d15_experiment-log.md` | 每日事实、退出码和阻塞 | 人工追加；跨日复用首次文件 | 人工编写 |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_matrix.jsonl` | 12个候选状态、命令与原始输出目标 | d16_matrix.py | 自动生成；计划中的 raw_file 仅为目标，不保证文件存在 |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p32_g32_b4.log` | 对应组合的子进程输出/错误 | d16_matrix.py | 满足上下文并选中该批量、--execute 时生成 |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p32_g32_b4.jsonl` | 该组合逐次测量行 | d12_benchmark.py（矩阵调用） | 条件生成；模型加载异常写 error 原始行；解释器启动、导入或参数解析失败可能只有 log，无 raw |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p32_g128_b4.log` | 对应组合的子进程输出/错误 | d16_matrix.py | 满足上下文并选中该批量、--execute 时生成 |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p32_g128_b4.jsonl` | 该组合逐次测量行 | d12_benchmark.py（矩阵调用） | 条件生成；模型加载异常写 error 原始行；解释器启动、导入或参数解析失败可能只有 log，无 raw |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p128_g32_b4.log` | 对应组合的子进程输出/错误 | d16_matrix.py | 满足上下文并选中该批量、--execute 时生成 |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p128_g32_b4.jsonl` | 该组合逐次测量行 | d12_benchmark.py（矩阵调用） | 条件生成；模型加载异常写 error 原始行；解释器启动、导入或参数解析失败可能只有 log，无 raw |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p128_g128_b4.log` | 对应组合的子进程输出/错误 | d16_matrix.py | 满足上下文并选中该批量、--execute 时生成 |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p128_g128_b4.jsonl` | 该组合逐次测量行 | d12_benchmark.py（矩阵调用） | 条件生成；模型加载异常写 error 原始行；解释器启动、导入或参数解析失败可能只有 log，无 raw |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p512_g32_b4.log` | 对应组合的子进程输出/错误 | d16_matrix.py | 满足上下文并选中该批量、--execute 时生成 |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p512_g32_b4.jsonl` | 该组合逐次测量行 | d12_benchmark.py（矩阵调用） | 条件生成；模型加载异常写 error 原始行；解释器启动、导入或参数解析失败可能只有 log，无 raw |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p512_g128_b4.log` | 对应组合的子进程输出/错误 | d16_matrix.py | 满足上下文并选中该批量、--execute 时生成 |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p512_g128_b4.jsonl` | 该组合逐次测量行 | d12_benchmark.py（矩阵调用） | 条件生成；模型加载异常写 error 原始行；解释器启动、导入或参数解析失败可能只有 log，无 raw |
| `experiments/inference_infra/results/raw/d18_p32_g8_b4_cpu_contract.jsonl` | 无 GPU 分支的短输入随机模型记录 | d12_benchmark.py | 仅执行 CPU 分支且成功写入后存在 |
| `experiments/inference_infra/reports/d18_baseline_batch.md` | 候选状态、有效数及失败边界 | 末尾验收步骤 | 人工编写 |

## 核对记录与完成标记

本文基于当前仓库相关生成、加载、注意力路由及原基准代码编写；源码阅读不能代替执行。本教程的交付状态不表示读者已经完成当天实验。填写实际日志后再勾选：

- [ ] 输入条件与前一天产物已核实。
- [ ] 本日命令在明确设备执行，或具体阻塞已记录。
- [ ] 日志、原始输出与材料保存到所写位置。
- [ ] 示例数值与真实结果已区分，失败和待测未删除。
- [ ] 能说明本日证据证明什么、尚不能证明什么。
