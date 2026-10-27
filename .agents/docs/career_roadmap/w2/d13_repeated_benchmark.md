---
source_files:
  - .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py
  - .agents/docs/career_roadmap/w3/scripts/d13_aggregate.py
  - nanochat/engine.py
  - nanochat/flash_attention.py
  - scripts/infer_bench.py
source_symbols:
  - measure
  - main
  - metrics
  - Engine.generate
doc_type: tutorial
knowledge_scope: W2第6天重复测量、指标复算与结果追溯
---

# W2 D6：把一次生成变成可复核的重复测量

前一天：[返回前一天](d12_benchmark_schema.md)；后一天：[继续下一天](d14_matrix_smoke.md)；[原始总计划](../nanochat_daily_execution_2026-10-01.md)。

今天回答一个问题：**同一模型生成多次，怎样得到一个能解释、能追溯的结果？** 你会用 CPU 上的微型随机模型完成“生成记录 → 检查单次指标 → 汇总五次正式样本”。无需训练、tokenizer 文件或 checkpoint；可选模型与 GPU 不影响当天完成。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，已读内容按问题回查即可，不新增手写报告。

- **阅读入口**：[12.2 生成控制流与返回格式](../../llm-training-workflow/12-inference-and-generation.md#reading-12-2)；[10.1 日志口径](../../llm-training-workflow/10-logging-checkpoints-and-resume.md#reading-10-1)。
- **今天重点**：生成可能提前终止；要求的长度和实际有效输出长度并不总相等。日志应留下发生过的过程，统计再决定能纳入哪些样本。
- **暂时跳过**：训练和恢复流程；本日不启动训练。
- **读完能解释**：要求输出四个 token，却只得到两个时，用“四个”计算吞吐为什么会夸大结果？

## 1. 先分清“两层循环”

### 1.1 三次预热、五次正式究竟重复什么

设提示长度 P=8、生成列数 G=4、批量 B=1。<abbr title="Token · 模型处理的离散单位，以整数 ID 表示；一个 token 不一定对应一个字或一个单词。">token</abbr> 数按模型输入输出单位计算。一次 <abbr title="Run · 一次完整生成测量。本文一次 run 从固定提示开始，创建新生成器及缓存，最多取 G 次输出；不是一个 token 的生成步骤。">run</abbr> 从同一个提示开始，生成 a、b、c、d。三个后续间隔对应 b、c、d；a 的等待时间是首 token 延迟。

`--warmup 3 --repeats 5` 表示把**整次生成**执行八次：

| 顺序 | phase | 每次工作 | 记录位置 | 是否进入正式指标统计 |
|---|---|---|---|---|
| 第 1–3 次 | warmup | 提示 8 个 token，最多生成 4 列 | raw 中各一行 | 否 |
| 第 4–8 次 | measure | 相同提示和配置，最多生成 4 列 | raw 中各一行 | 有效时进入 |

因此正常情况下 raw 有八行，正式样本数是五；每行又包含一次首 token 耗时和三个后续间隔。不是“预热三个 token，再测五个 token”，也不是将 15 个后续间隔当成 15 次独立生成。

<abbr title="Warmup · 预热。先执行与正式测量相同的工作，使首次库调用、内存分配等初始化影响尽量在正式样本之前发生；固定三次并不保证所有设备都已稳定。">预热</abbr>减少首次调用影响，但不保证设备已经稳定。每次 run 都新建 Engine 和 <abbr title="KV Cache / Key-Value Cache · 键值缓存。保存已经处理位置的注意力键和值，使后续生成复用历史计算；本文每次完整 run 重新创建。">KV Cache</abbr>，预热不会让正式 run 直接沿用上一轮提示缓存。模型加载只发生一次，位于这八次测量之前。

**先手推**：若改成 G=1，仍然有八次 run；每行只有首 token，没有后续间隔，TPOT 应为 `null`，不是 0。保持这个区分再读代码。

### 1.2 一条记录怎样产生三个指标

下面是教学假设，不是你的 CPU 实测：一次 B=1、G=4 的有效生成，首 token 花 30 ms，后续间隔为 `[10, 12, 8]` ms。

| 指标 | 在该 run 中怎样算 | 例子 |
|---|---|---|
| <abbr title="TTFT / Time To First Token · 首 token 延迟。本文从计时开始到第一次输出返回；模型加载与提示分词不在其中。">TTFT</abbr> | 第一次输出前的等待 | 30 ms |
| <abbr title="TPOT / Time Per Output Token · 后续输出间隔。本文先对一个 run 内首 token 之后的 G−1 个输出间隔求平均，再跨 run 汇总。">TPOT</abbr> | 后续间隔的平均值 | (10+12+8)/3 = 10 ms |
| elapsed | 首 token 加后续间隔 | 30+10+12+8 = 60 ms |
| tokens/s | 有效 token 总数 ÷ 包含首 token 的总时长 | 4/0.060 ≈ 66.67 token/s |

今天脚本的吞吐**包含首 token 延迟**。若只算后续生成速度，得到 `3/0.030=100 token/s`，那是另一种统计范围，不能直接替换本行结果。W1 阅读过的原生 `scripts/infer_bench.py` 吞吐排除了首 token；比较前应先统一范围。

若 B=4、每行都完整输出四个 token，分子是 16；仍按整个批次经过的时间计算。一次输出返回一列 B 个 token，TPOT 是列间隔，不是将间隔再除以 B。Engine 对**同一个提示先做一次 prefill，再复制缓存到 B 行**；这不等于四个不同用户的请求分别处理提示。

### 1.3 五条正式样本怎样汇成一个结果

继续用假设的五次有效 TTFT：`[30, 32, 29, 80, 31]` ms，排序为 `[29, 30, 31, 32, 80]`。

- 中位数取中间的 31 ms。它避免只挑最快的 29 ms。
- <abbr title="P95 / 95th percentile · 第95百分位。脚本使用 nearest-rank：排序后取 ceil(0.95N) 位。五个样本时取第五位，即最大值，不能代表稳定的尾延迟估计。">P95</abbr> 用 nearest-rank：`ceil(0.95×5)=5`，所以是 80 ms。
- 同时保留 N=5 和范围 29–80 ms。P95 等于最大值，说明五次样本提供的尾部信息很少。

这里是“先为每个 run 算指标，再对五个 run 求中位数或 P95”。TPOT 也先每 run 求后续间隔均值；不要把所有 run 的原始间隔摊平后当作同一种统计。吞吐列的中位数是五个 run 吞吐的中位数，也不是把所有 token 和时间相加所得的整体吞吐。

## 2. 带读：源码如何兑现这些定义

打开 [重复测量脚本](scripts/d12_benchmark.py)，这次按下面执行顺序阅读，不需要逐行背字段。

### 2.1 main：准备一次，完整生成重复八次

`load(args)` 在循环外构造模型或加载权重，设置 `eval()`。随后计算模型/输入身份和环境信息；这些工作都不在生成计时内。进入 `torch.inference_mode()` 后，核心循环是：

```python
for phase, n in [("warmup", args.warmup), ("measure", args.repeats)]:
    for repeat in range(n):
        row = measure(args, torch, model, tokenizer)
        # 为本次 run 添加 run_id、phase、repeat，并追加一行 JSONL
```

这是省略异常分支的源码节选。一个脚本进程对应一个 `session_id`；每个 run 有自己的 `run_id`。重新启动命令会产生新 session，即使输出追加进同一文件，也不应合并成同一次实验。

异常处理也属于执行过程：`early_stop` 会保存后继续其他 run，最后退出 2；某个 run 抛异常会写 `error` 行并提前退出 1；模型加载等准备异常会写 `setup_or_run` 错误行。因而文件不足八行可能有原因，八行也不保证五次正式成功。

### 2.2 measure：首个 next 才真正进入生成器

`measure()` 创建新 Engine、构造固定 token ID 提示，再调用 `engine.generate(...)` 得到生成器。这时生成器正文尚未执行。计时开始之后：

```python
for _ in range(args.decode_tokens):
    column, masks = next(stream)
    sync()
    now = time.perf_counter()
    times.append((now - previous) * 1000)
    previous = now
```

这也是省略计数和异常处理的节选。第一次 `next()` 执行 [Engine.generate](../../../../nanochat/engine.py) 中的提示前向计算、缓存分配/复制和第一次采样，所以 `times[0]` 是 TTFT。后面的 `next()` 从上次 yield 恢复，处理前一列 token 并得到下一列；`times[1:]` 才是后续间隔。

脚本恰好取 G 次输出，然后关闭生成器。当前 Engine 在最后一次 yield 后仍有准备下一步的前向计算；若继续取到 `StopIteration`，会额外执行一次不需要的前向。W1 原生入口在这次调用结束时捕获异常，没有把这次耗时追加进自己的计时数组；额外执行和计入指标要分开判断。这里的 G=4 是“四列输出”，不是“首 token 后再执行四次”。

CUDA 路径每次输出后同步，避免只测到任务提交时间；CPU 的 `sync()` 为空操作。脚本不执行 `torch.compile`；记录的是当前 eager Engine。计时覆盖生成器工作及其间 Python 控制开销，排除加载权重、提示分词和文件写入。

最后检查实际输出：仅非终止且 `mask=1` 的 token 计入有效数。出现终止、强制输出或不足 G 列，状态为 `early_stop`，不作为完整固定长度样本。

## 3. 动手：CPU 随机模型跑一次完整流程

### 3.1 准备环境与独立结果位置

所有命令从项目根目录执行。终端已自动激活 nanochat，不重复激活。局部 `{ ...; } > >(tee ...)` 只记录该命令组，避免持续截获终端提示符。日志追加写入；每条关键命令后立即打印退出码。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
{
python -c "import sys; print(sys.executable); print(sys.version)"
export NANOCHAT_DTYPE=float32
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py --help
echo "help_exit=$?"
} > >(tee -a experiments/inference_infra/reports/d13_session.log) 2>&1
```

解释器应为 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。精度在模块导入时确定；显式设置 FP32 避免沿用终端的其他值。帮助命令成功只说明入口可用，下面生成记录才是实际实验。

### 3.2 先预测记录形状，再执行

先预测：一个新 session 正常应有八行；每行 `actual_decode_steps=4`、`per_row_generated_tokens=[4]`、`generated_tokens=4`、`decode_step_ms` 长度为 3。

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py --device cpu --random-model --prompt-tokens 8 --decode-tokens 4 --batch-size 1 --warmup 3 --repeats 5 --seed 42 --output experiments/inference_infra/results/raw/d13_p8_g4_b1_random_cpu.jsonl
echo "benchmark_exit=$?"
} > >(tee -a experiments/inference_infra/reports/d13_session.log) 2>&1
```

`--random-model` 构造两层、嵌入维度 64、两头、词表 512 的小模型，使用脚本内的合成 token 映射。它不会下载模型、训练模型或读取真实 tokenizer；数字用于学习执行和统计契约，生成文本没有质量意义。

打开命令生成的 raw：`experiments/inference_infra/results/raw/d13_p8_g4_b1_random_cpu.jsonl`（相对于项目根，执行后才存在），找到本次 session 的 `measure+ok` 行。先从 `ttft_ms` 与 `decode_step_ms` 加出 elapsed，再用有效数复算 tokens/s、用后续间隔复算 TPOT，对照该行结果。不要要求耗时等于前面的 30 ms 教学假设。

脚本仅追加数据。再次运行前换一个仍匹配 `dNN_p*_g*_b*.jsonl` 的文件名，例如加 `_retry1`；若已经追加，依据 session 区分。不要删除失败行后把余下记录描述成一次全成功测量。

### 3.3 从 raw 到审计，再到 CSV

现在打开 [汇总脚本](../w3/scripts/d13_aggregate.py)。先看 `metrics(row)`：只对 `measure+ok` 计算指标，还要核验列数、各行有效数、正数耗时，以及 `elapsed≈TTFT+sum(steps)`。它从原始计时和计数**重新计算** TPOT 与吞吐，不盲信已有汇总字段。

再看 `main()`。原始文件采用 <abbr title="JSONL / JSON Lines · 每行一个独立 JSON 对象。本文一行保存一次 run，可逐行保留成功或失败过程。">JSONL</abbr>；汇总采用 <abbr title="CSV / Comma-Separated Values · 逗号分隔表格。本文每行代表一个实验身份组，列中保存样本账和跨 run 统计值。">CSV</abbr>。脚本先为记录建立身份组，再验证每条正式记录，将结果写入 <abbr title="Audit · 审计记录。本文每条非预热记录对应一条来源记录，含原文件绝对路径、行号、run_id、是否纳入和原因。它支持把 CSV 数字追溯到原始生成。">audit</abbr>，最后按组统计有效样本。预热单独检查，不写入 audit，也不计入 CSV 的 `n_total`。正常八行 raw 对应五行 audit，CSV 中 `n_total=n_valid=5`、`n_excluded=0`、`warmup_status=OK`。

```bash
{
python .agents/docs/career_roadmap/w3/scripts/d13_aggregate.py --raw-dir experiments/inference_infra/results/raw --output experiments/inference_infra/results/summary/d13_w2_baseline.csv --audit experiments/inference_infra/results/summary/d13_w2_audit.jsonl
echo "aggregate_exit=$?"
} > >(tee -a experiments/inference_infra/reports/d13_session.log) 2>&1
```

该命令读取 raw 目录内**所有匹配文件**，不只读 D13 文件。输出文件名带 d13 不代表输入只来自今天；不要将其他天的组数算作今天完成数。CSV 与 audit 每次覆盖生成，raw 保留原记录。

为什么相同 P/G/B 不一定同组？例如昨天与今天各跑一次，`session_id` 不同，就分开统计。若输入 fill ID 改变，`prompt_sha256` 随之改变，也分组。设备、精度、注意力后端、模型配置、权重/元数据/tokenizer 哈希及 `env`（源码版本、dirty、线程数等）同样参与身份判断。这样不会把不同工作负载混成一次五样本结果。

按这个方向核对一条证据链：

1. 在 CSV 依据本次 `session_id` 找到该组，从 `run_ids` 取一个 ID。
2. 在 audit 找该 ID，确认 `included=true`，取 `file` 和 `line`。
3. 回到 raw 对应行，确认 phase/status，再复算指标。预热状态另回 raw 查前三行。

这条链比“看到 CSV 有数字”多回答了两个问题：数字来自哪里、为什么这条记录能纳入。CPU 主线预期 `conclusion_status=CPU_OR_RANDOM_CONTRACT`，表示流程与统计的验证范围；它可以完成今天学习，不产生未来必须补做 GPU 的欠账。

## 4. 结果不符合预期时，沿状态分流

| 现象 | 先检查什么 | 怎样处理 |
|---|---|---|
| raw 八行但正式有效少于五 | 本 session 的 phase/status、audit 的 included/reason | 八行只是记录数。保留排除记录，说明实际样本数；不要挑五个最快值 |
| `early_stop`、退出 2 | 实际列数、各行有效数、终止或强制输出 | 固定长度样本未满足。可换普通 `--prompt-fill-id 2` 并换新文件，作为新输入工作负载；同提示 greedy 重跑通常不会解决终止 |
| `status=error`、退出 1 | raw 的 error 与会话日志 | 准备异常可能只有一行，run 异常会提前结束循环；修复依赖或参数后另开 session |
| `P+G exceeds` | 模型的 sequence_len | 缩短输入/输出长度；脚本不会静默截断 |
| 汇总因重复 run_id 停止 | 是否复制同一 raw 到多个匹配文件 | 保留唯一原始来源，别靠复制凑重复次数 |
| CSV 的 TPOT 为空 | 是否 G=1 | 正常：没有后续间隔。CSV 空值对应 raw 的 null，不能填 0 |
| 汇总没有找到输入 | 文件名及 raw-dir | 文件名须匹配 dNN_p*_g*_b*.jsonl；先确认测量实际生成文件 |

CPU / random 行的显存字段为空是适用范围所致，不是缺少 GPU 作业。若选择修复并重跑，只针对发现的问题，不必为了重复验收再次运行有效实验。

## 5. 可选：已有 D10 权重时复用同一协议

本节只供**已经主动完成 D10 可选两步训练**的人使用。没有文件直接跳过；无需为 D13 开始训练。先设置 D10 实际使用的数据根，不能猜测有 checkpoint：

```bash
{
export NANOCHAT_BASE_DIR=/tmp/nanochat-w2-toy
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py --device cpu --source base --model-tag d10_w2_smoke --step 2 --prompt-tokens 32 --decode-tokens 8 --batch-size 1 --warmup 3 --repeats 5 --seed 42 --output experiments/inference_infra/results/raw/d13_p32_g8_b1_checkpoint_cpu.jsonl
echo "checkpoint_exit=$?"
} > >(tee -a experiments/inference_infra/reports/d13_session.log) 2>&1
```

`/tmp/nanochat-w2-toy` 仅为 D10 合成分支示例，以你实际保留的根为准。要求其中已有 `base_checkpoints/d10_w2_smoke/model_000002.pt`、`meta_000002.json` 和 `tokenizer/tokenizer.pkl`。显式 tag/step 避免脚本默认标签指向旧计划。合成数据权重仍只支持系统流程学习，不能作为语言质量结论。

已有合适 GPU、对应 PyTorch 和权重时，可自选改 `--device cuda`，输出用 `d13_p32_g8_b1_checkpoint_cuda.jsonl`，并在新进程导入前显式选择精度、记录实际驱动与设备。未选择此扩展不会影响当天验收。也不把随机 CPU 的数字与该组直接比较为加速结论。

## 6. 今日自验证与交流

按题号在聊天回答，无需手写 report。开头给一次本次 session、raw/CSV/audit 路径；实测题引用同一份证据即可。未执行的主线操作如实说未执行，可选分支未选不算缺项。

1. 若把本页命令改为 G=1，完整成功应有几行 raw、几条 audit、每行几个后续间隔？为什么 TPOT 为空而 TTFT 有值？

   **核对要点**：分别解释 run 数、正式样本数和一次 run 内的输出数，连到 next/yield 的执行顺序。

2. 取本次一条 `measure+ok`，用原始计时与有效数复算 TPOT、tokens/s 和 elapsed。若有人把首 token 后的吞吐当作这里的吞吐，哪两个统计范围混淆了？

   **核对要点**：给出 run_id、实际值、单位和公式；说明首 token 的计数与耗时必须同范围，不拿教学假设当实测。

3. 从本次 CSV 沿 run_id→audit→raw 追溯一条记录。若同目录另一文件 P/G/B 一样、session 不同，它会怎样进入统计？预热去哪里查？

   **核对要点**：指出实际路径与行号，解释至少一个分组差异及其理由；CSV 的 n_total 不包含预热。

4. 对本次五个正式有效 TTFT 排序，复核 median、P95、样本数和范围。若其中一个很慢，能否删掉再称为“五次正式测量”？

   **核对要点**：展示取位方法；解释五样本 P95 的局限和删除记录的证据问题。有无效样本则按实际数量讨论，不补造五条。

5. 推演：三次预热均成功，五次正式中一条 early_stop。CSV 的 n_total、n_valid、n_excluded 应是什么？如果改 fill ID 后重跑成功，能否把两组成功行拼满五条？

   **核对要点**：区分原始记录、有效样本与身份组；保留失败原因，用新输入 hash 解释为什么分开。无需主动制造异常。

## 7. 脚本与产物查阅表

下表路径以项目根为基准。只打印的环境/帮助检查由局部 tee 保存到当天日志；实际命令运行才生成实验文件。

| 脚本完整项目路径 | 用途与阶段 | 完整输入路径/参数 | 运行条件 |
|---|---|---|---|
| `.agents/docs/career_roadmap/w2/scripts/d12_benchmark.py` | 第 3 节重复生成，第 5 节可选权重复用 | 主线仅 P/G/B 参数；可选读 `$NANOCHAT_BASE_DIR/base_checkpoints/d10_w2_smoke/model_000002.pt`、`$NANOCHAT_BASE_DIR/base_checkpoints/d10_w2_smoke/meta_000002.json`、`$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl` | 主线 CPU 与项目依赖即可；可选文件须已存在 |
| `.agents/docs/career_roadmap/w3/scripts/d13_aggregate.py` | 第 3.3 节验证/汇总 | `experiments/inference_infra/results/raw/dNN_p*_g*_b*.jsonl`，兼容旧 `p*_g*_b*.jsonl` | 至少有实际非预热记录，run_id 唯一 |

| 产物完整项目相对路径 | 用途 | 生成步骤与条件 |
|---|---|---|
| `experiments/inference_infra/results/raw/d13_p8_g4_b1_random_cpu.jsonl` | 主线每 run 原始记录 | 第 3.2 节自动追加；正常八行，异常保留实际行数 |
| `experiments/inference_infra/results/summary/d13_w2_baseline.csv` | 各组有效样本统计与结论范围 | 第 3.3 节 aggregate 自动覆盖生成 |
| `experiments/inference_infra/results/summary/d13_w2_audit.jsonl` | 非预热记录的纳入判断与来源 | 同上；预热回 raw 查 |
| `experiments/inference_infra/reports/d13_session.log` | 环境、命令输出和即时退出码 | 各 Bash 命令组的 tee 自动追加 |
| `experiments/inference_infra/results/raw/d13_p32_g8_b1_checkpoint_cpu.jsonl` | 可选两步权重 CPU 记录 | 仅第 5 节实际执行时追加生成 |
| `experiments/inference_infra/results/raw/d13_p32_g8_b1_checkpoint_cuda.jsonl` | 可选 GPU 权重记录 | 仅自行选择 CUDA 扩展且实际执行后生成 |

当天完成标准是能解释上述循环和统计，并完成主线 raw→audit→CSV 的一次追溯；不要求训练、GPU 或额外 Markdown 报告。下一天复用已确认的证据，进一步研究 P/G/B 改变工作负载会发生什么。
