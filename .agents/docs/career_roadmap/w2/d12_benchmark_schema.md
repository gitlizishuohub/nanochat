---
source_files:
  - .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py
  - .agents/docs/career_roadmap/w3/scripts/d13_aggregate.py
  - nanochat/common.py
  - nanochat/tokenizer.py
  - nanochat/dataset.py
  - nanochat/dataloader.py
  - scripts/tok_train.py
  - scripts/tok_eval.py
  - scripts/base_train.py
  - nanochat/checkpoint_manager.py
  - nanochat/gpt.py
  - nanochat/engine.py
  - scripts/infer_bench.py
  - nanochat/flash_attention.py
doc_type: tutorial
knowledge_scope: W2第5天实操与验收
---

# W2 D5：从一次生成推导指标，再读懂原始记录

前一天：[返回前一天](d11_checkpoint_generation.md)；后一天：[继续下一天](d13_repeated_benchmark.md)；[原始总计划](../nanochat_daily_execution_2026-10-01.md)。

今天学习的是**怎样测量和保存一次推理生成**。前半页用假设数据推演，只需阅读、计算；实际操作在后半页，使用CPU与脚本临时初始化的小型随机模型，**不需要训练、下载数据或已有权重文件**。<abbr title="Checkpoint · 保存模型参数及相关元信息的检查点文件。本页CPU随机分支直接初始化参数，不加载checkpoint，也不进行训练。">Checkpoint</abbr>只在自选的既有权重加载分支中使用。最后故意加载不存在的模型，观察错误如何保存，也是预期练习。GPU性能实验仅在自己有条件且想扩展时选做，不属于本日完成条件。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[10.1 日志口径](../../llm-training-workflow/10-logging-checkpoints-and-resume.md#reading-10-1)；[13.2 能力与系统性能的边界](../../llm-training-workflow/13-evaluation-and-metrics.md#reading-13-2)。
- **今天重点**：了解训练日志的耗时与吞吐口径，再对照今天的推理协议，明确计时起止、分母、预热和失败行。
- **暂时跳过**：本栏目的训练吞吐不是推理指标公式；具体测量定义仍以当天教程和脚本为准。

**读完能解释**：为什么训练日志中的 tokens/s 不能直接与 Engine 的生成吞吐放在一列比较？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 第一步：用六个假设输出算出三个指标

本节只推演，不运行模型。<abbr title="Token · 模型输入和输出的离散单位，由整数ID表示；一个token不一定对应一个汉字、单词或字符。">Token</abbr>数决定输入和输出形状：P表示输入token数，G表示要求输出的最大列数，B表示同一提示的并行样本数。设P=8、G=3、B=2，没有终止或工具插入。Engine每次 `yield` 返回一列B个token。两行输入相同，本脚本采用贪心采样，示意中两行输出也相同：

| 时刻（从计时起点算起） | 第1行 | 第2行 | 这次等待 |
|---|---|---|---|
| 30 ms | a | a | 第1列，等待30 ms |
| 40 ms | b | b | 第2列，距上列10 ms |
| 55 ms | c | c | 第3列，距上列15 ms |

第1列前，模型处理8个输入位置，建立 <abbr title="KV Cache / Key-Value Cache · 保存已处理位置的键和值张量。后续生成复用它们，避免重复计算全部历史位置；这里每次measure重新建立缓存。">KV Cache</abbr>，从提示末位置的预测中为两行选出a。这一阶段叫prefill。恢复生成器后，把两行的a输入模型并复用缓存，预测b；再把b输入，预测c。这两个后续步骤叫decode。整个过程中参数不更新，所以不涉及训练。

现在按本周 [测量脚本](scripts/d12_benchmark.py) 的定义计算：

- <abbr title="TTFT / Time to First Token · 已编码输入到首个输出列就绪的延迟。本脚本包含cache分配、prefill、复制和采样；CUDA还包含边界同步，不含加载与文本分词。">TTFT</abbr> 为 **30 ms**，不是完整生成耗时。
- <abbr title="TPOT / Time Per Output Token · 本脚本取首列之后各输出列间隔的算术平均，单位ms。每列含B个并行输出，间隔不再除以B。">TPOT</abbr> 为 `(10+15)/2=12.5 ms`。一次等待产生一整列，所以这里不除以 B。
- 完整测量时长为 `30+10+15=55 ms`；有效输出共 `2×3=6 token`。本周吞吐为 `6/0.055≈109.09 token/s`，**包含首列等待时间**。如果只算后续decode，则是 `4/0.025=160 token/s`，它回答的是另一段过程的速度。

这就是为什么记录中同时保留 `ttft_ms`、`decode_step_ms=[10,15]`、`elapsed_ms=55` 和 `generated_tokens=6`：之后可以重新计算，而不是只相信一个吞吐数字。G=1时没有后续列间隔，TPOT为 `null`，表示此指标在这次运行中不适用，不能填0。

## 第二步：对照D7，辨别两个脚本的口径

D7读的 [上游 infer_bench](../../../../scripts/infer_bench.py) 与今天的脚本都使用Engine，但统计定义不同。下表是源码事实，不要求在CPU上运行上游CUDA入口。

| 问题 | D7所读 infer_bench | 本周 d12_benchmark.py |
|---|---|---|
| 输入 | 自然文本先分词，再按长度截取 | 起始标记BOS加重复普通ID，默认ID1；不进行文本分词 |
| TPOT | 后续步耗时排序后取 `sorted[n//2]`，偶数样本取中间靠后值 | 后续列间隔的算术平均 |
| tokens/s | `B×后续步数 / 后续步总耗时`，排除TTFT | `实际有效token总数 / 含TTFT的elapsed` |
| 生成何时停 | 继续调用到StopIteration；最后一次恢复可能执行一个未用于输出的forward | 最多取G次yield，然后close，不恢复最后一列后的计算 |
| 输入太长 | 缩短提示以适配上下文 | P+G超出上下文时报错，不偷偷改工作负载 |

例如后续间隔为 `[1,2,9,20] ms`，上游选9 ms，本周取8 ms；仅更换定义就能改变结果。上游最后一次触发StopIteration的调用没有追加到 `step_times`，不能误称它一定计入已报告的decode总时间。本周明确停在末列就绪，避免执行那次多余的forward；也不能据此直接宣称本周吞吐更高。

B的语义两者一致：先对**一个提示做batch=1 prefill，再把缓存复制到B行**。B=4不是给4个不同用户各处理一个提示。固定ID输入让形状容易控制，却不能代表真实问答的端到端延迟。

## 第三步：跟着源码走过计时、计数与落盘

打开 [d12_benchmark.py](scripts/d12_benchmark.py)，按下面四个片段带读；无需先理解所有环境字段。短片段用于阅读，省略号表示省略原参数，不是终端操作。

### 3.1 创建stream还没有执行prefill

`load()`在测量之前完成。随机分支创建两层小型GPT和合成ID映射，并调用 `model.init_weights()`，这是初始化权重，不是训练；checkpoint分支才从磁盘加载既有权重和tokenizer。随后 `measure()`创建Engine与固定ID输入：

```python
stream = engine.generate(...)
sync()
start = previous = time.perf_counter()
```

`generate()`是Python生成器：创建对象时函数体暂停，首次 `next(stream)`才进入Engine、分配cache并运行prefill。因此计时起点放在首次next之前，模型加载与构造提示ID均在计时之外。

### 3.2 每次next得到一列，时间数组把边界保留下来

```python
column, masks = next(stream)
sync()
now = time.perf_counter()
times.append((now - previous) * 1000)
previous = now
```

代入第一节：三次得到的 `times` 是 `[30,10,15]`。第一项给TTFT，后两项给decode间隔；`sum(times)`从起点累加到末列同步完成。相邻时间戳之间也包含Python循环和计数等主机工作，它不是只统计GPU算子的耗时。

CPU分支的 `sync()`什么也不做；CUDA分支调用设备同步，因为CPU提交GPU工作后可能先返回，直接读取主机时钟不等于GPU已完成。这里解释CUDA逻辑不要求你有GPU，未来若改成其他计时方法，需要另行说明范围。

外层最多循环G次，`finally: stream.close()`保证退出时关闭生成器。结合 [Engine.generate](../../../../nanochat/engine.py) 的 `yield` 后方代码看：恢复生成器才会把刚输出的一列送入下一次forward。收满G列就关闭，便不会为一个不需要的第G+1列继续计算。

### 3.3 yield次数和有效token数不是同一件事

`column`是输出ID列表；`masks`中1表示采样、0表示工具逻辑强制插入，**不是“该行是否结束”**。脚本另外用 `completed`识别BOS或assistant_end终止ID，终止token不计入有效输出。

对每行依次判断：已完成则跳过；本次是终止ID则标记完成；普通采样输出计数加1；强制输出则记 `forced=True`，不算有效token。一旦任何行完成，本次measure立即退出。即使其他行还可继续，也不继续把它当“全部B行都输出G个有效token”的固定长度样本。

最后只有 `steps==G`、每行计数均为G且没有forced时才记 `status=ok`；其他正常返回记 `early_stop`。例如B=2、G=3，首列是普通采样输出，第二列为 `[assistant_end,y]`，结果只取2列、每行计数为 `[1,2]`，总数为3。不能用B×G=6替代它。`early_stop`保留原始计时和计数，但不进入固定长度成功样本的汇总。

### 3.4 一个run写一行，异常也要有来处

`main()`先校验参数，再加载模型，随后执行3次预热和至少5次正式测量。每次run都重新调用measure、创建cache；预热不是复用同一份生成cache。`write(row)`用追加模式把公共身份字段与本次结果合并，保存为 <abbr title="JSONL / JSON Lines · 每行一个独立JSON对象。此处一行保存一次run，便于保留预热、成功、提前结束及异常；终端日志不应写入同一文件。">JSONL</abbr>。

有以下退出路径要分清：

| 发生位置 | 原始结果 | 退出码含义 |
|---|---|---|
| 参数校验，如warmup=0 | 尚未调用write，没有新raw行 | argparse退出2 |
| 加载/准备失败，如不存在的checkpoint | 一条 `phase=setup_or_run,status=error`；可能没有环境和计时字段 | 退出1 |
| 某次measure抛异常 | 保存当次phase/repeat及error，停止后续运行 | 退出1 |
| 全部测量返回 | 预热和正式行均保存；可能有early_stop | 全ok退出0；存在early_stop退出2 |

所以不能只看退出码2就断言是哪种问题，要结合日志与是否有raw行。setup错误行字段稀疏是合法状态，不应该编造延迟填满表。

## 第四步：用CPU观察记录，不加载训练权重

### 运行前：固定 shell 与结果位置

所有命令从仓库根目录执行。当前终端会自动激活 nanochat 环境，无需重复激活；先运行以下检查。这里使用本项目的 Conda 环境；不要混用系统 Python。CPU实验建议先 `export NANOCHAT_DTYPE=float32`；脚本只在变量未设置时选择FP32，已有变量会遵从并记录，不能假定CPU必定FP32。GPU实验另按协议显式选dtype。

日志只在 `{ ...; }` 命令组执行期间记录；<abbr title="Bash process substitution · 进程替换。这里把命令组的输出交给 tee，同时显示并追加到日志；重定向在命令组结束后恢复。">进程替换</abbr>不会持续接管交互终端，因此提示符和 VS Code 终端集成控制字符不会混入日志。命令组在当前 shell 执行，变量设置会保留；代码块返回最后一条命令的退出码，需核对单条命令时立即记录 `$?`。已有的显式文件重定向仍写入各自文件。复制时包含外层花括号和末尾重定向。若之前执行过旧版全局日志命令，先关闭旧终端并新开终端；仅修改文档不会停止旧终端的记录。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
{
python -c "import sys; print(sys.executable); print(sys.version)"
} > >(tee -a experiments/inference_infra/reports/d12_session.log) 2>&1
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。若环境不存在或第三方依赖缺失，先解决环境，不把 `--help` 成功当作模型实验通过。后面写入的 `experiments/` 文件是你实际执行教程时产生的正式成果；本指导不预先生成成绩。大数据、tokenizer、权重放在 `NANOCHAT_BASE_DIR`，不要加进 Git。表中的 `$NANOCHAT_BASE_DIR` 代表当前 shell 选定的真实绝对数据根；合成分支示例为 `/tmp/nanochat-w2-toy`，未设置时当前用户默认根为 `/home/lizishuo/.cache/nanochat`。交流时给出实际展开后的绝对路径；`*.parquet` 表示该目录内逐一列出的实际数据文件，旧数据回退目录为 `$NANOCHAT_BASE_DIR/base_data/`。

本页各 Bash 代码块内的输出追加保存在 `experiments/inference_infra/reports/d12_session.log`；命令返回后立即执行 `echo "exit=$?"`，避免丢失退出码。帮助、源码检索、解释器与 Git 检查只打印，不另有检查结果文件。日志是追加写入，重跑时交流中注明会话边界或日志中的起止位置。

先预测：3次预热加5次正式测量，正常完成会写几行？随后运行：

```bash
{
export NANOCHAT_DTYPE=float32
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py --device cpu --random-model --prompt-tokens 8 --decode-tokens 3 --batch-size 2 --warmup 3 --repeats 5 --output experiments/inference_infra/results/raw/d12_schema_cpu.jsonl
run_exit=$?
echo "benchmark_exit=$run_exit"
} > >(tee -a experiments/inference_infra/reports/d12_session.log) 2>&1
```

文件是追加写入；首次正常完成预期8行，重跑会追加新session，不能把整个文件行数当成本次run数。参数正常时仍可能产生early_stop，按实际状态解释，不能改成ok。输出中的 `CPU_FUNCTION_ONLY`表示这份记录验证CPU测量流程，不证明模型已经学会语言，也不证明GPU加速效果；随机模型仍会执行真实前向计算，只是权重未经训练。随机分支仍会记录参数默认的model_tag，它不代表加载过同名权重；要结合 `model_kind=random`、空的checkpoint哈希映射和实际命令判断模型来源。

查看文件前一条正式行，将第一节的手算步骤换成该行真实值；不需要抄出全部字段：

```bash
{
python -c 'import json; from pathlib import Path; rows=[json.loads(s) for s in Path("experiments/inference_infra/results/raw/d12_schema_cpu.jsonl").read_text().splitlines() if s.strip()]; row=next(r for r in rows if r["phase"]=="measure"); print(json.dumps({k:row.get(k) for k in ["session_id","status","ttft_ms","decode_step_ms","elapsed_ms","per_row_generated_tokens","generated_tokens","tokens_per_s","evidence_scope"]}, ensure_ascii=False, indent=2))'
} > >(tee -a experiments/inference_infra/reports/d12_session.log) 2>&1
```

这个短检查选取文件中第一条正式行；追加重跑后它仍可能是旧session，交流时注明session_id。你应能用“计数之和÷总秒数”复算吞吐。若是early_stop，则解释截断原因和有效计数，不把它报为完整G列基线。

再制造一个真实的缺失模型错误。这一步故意省略random-model，不要求准备checkpoint：

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py --device cpu --model-tag absent-w2-schema-check --step 123456 --output experiments/inference_infra/results/raw/d12_schema_error.jsonl
error_exit=$?
echo "expected_error_exit=$error_exit"
} > >(tee -a experiments/inference_infra/reports/d12_session.log) 2>&1
```

预期退出1，JSONL包含真实 `error`。根据实际日志判断是否已走到“模型不存在”，还是更早发生依赖错误；两者都能说明异常落盘，但证据范围不同。这个错误是本次练习要观察的行为，不表示要去训练一个模型来“修好”它。若想验证argparse分支，可在上述命令中加 `--warmup 0` 并换一个新output路径：预期退出2，新文件不生成，因为参数校验早于write。

有D10既有权重时，可以自选用 `--model-tag d10_w2_smoke --step 2` 替换random-model并换一个output路径，检验加载后的测量链路；没有则直接完成CPU随机模型主线。CUDA扩展需自行具备设备与合适的既有权重，使用独立结果路径；不把它列作本日未完成项。

## 第五步：把学到的边界整理成可回查协议

现在再看下面清单：它是前面推导的速查表，不需要逐项背诵。实际CPU练习使用8/3/2，下一天重复基准从8/4/1开始；比较时按行中真实参数分组，不把它们混成同一工作负载。在聊天中说明边界即可，`experiments/inference_infra/d12_protocol.md` 仅为可选整理位置：

```text
协议版本：1
源码commit / dirty摘要：
工作负载：固定token-ID prompt；同一prompt复制B份输出
模型来源 / tag / step / 权重hash / tokenizer hash：
设备 / GPU驱动 / PyTorch / CUDA / dtype / attention backend：
seed：42
P / G / B：本日CPU练习8 / 3 / 2；D13主线8 / 4 / 1
边界：P+G不超过sequence_len；不自动裁短
预热：3次；正式：至少5次；每次新建Engine cache
TTFT范围：token ID输入到首列yield，含分配/prefill/复制/采样，不含加载和tokenizer
TPOT：G-1个同步输出列间隔；每次run先取均值
有效token：每行尚未结束且mask=1且不是terminal的token
吞吐：有效token总数 / elapsed秒；B乘法不再重复计算
汇总：仅measure+ok；median与nearest-rank P95；失败/提前结束保留
CPU或random：仅功能/契约，不用于GPU加速结论
GPU驱动：CPU主线不适用；自选CUDA扩展时另记录
```

注意 B=4 的 prefill 仍只有一个 prompt，而不是四个不同请求的吞吐；不能把这样的 TTFT 与普通 batch-prefill 工作负载混为一谈。每次调用新建 cache 并预分配零张量；这部分开销明确计入 TTFT。

### 原始字段速查与下一天的汇总

<abbr title="Schema · 数据结构约定。这里规定每行字段名称、单位和状态含义；它帮助程序解释记录，但字段齐全本身不证明测量有效。">Schema</abbr>由脚本输出定义。下一天的 [聚合脚本](../w3/scripts/d13_aggregate.py) 只把 `phase=measure,status=ok` 的合法行纳入指标，还会核对计数与时长并重新计算；预热、错误、提前结束均不冒充正式成功样本。它对“每个run的平均TPOT”再算median/P95，不是把所有decode步混到一个数组；本周采用 <abbr title="Nearest-rank P95 · 将n个值升序排列，取第ceil(0.95×n)个。n=5时等于最大值，小样本不能稳定刻画尾延迟。">nearest-rank P95</abbr>，具体重复统计留待D13。

| 类别 | 实际字段与约束 |
|---|---|
| 标识 | schema_version=1、session_id、run_id；每行run_id唯一 |
| 运行角色 | phase=warmup/measure；repeat从0开始；setup_or_run是异常行 |
| 状态 | ok/early_stop/error；只有固定G步、所有B行都有G个有效token才ok |
| 形状 | prompt_tokens、decode_tokens、batch_size、actual_decode_steps、prompt_fill_id、prompt_sha256 |
| 计数 | generated_tokens、per_row_generated_tokens；前者为后者之和 |
| 延迟 | ttft_ms、decode_step_ms、tpot_ms、elapsed_ms；毫秒单位 |
| 吞吐 | tokens_per_s；实际有效token数除以含TTFT总秒数 |
| 内存 | peak_allocated_bytes、peak_reserved_bytes；CPU为null，不能写0假装测得 |
| 归因 | model_kind、source、model_tag、checkpoint_step、model_config、checkpoint_sha256 |
| 可复现 | seed、env、precision、attention_backend、timing_scope、batch_semantics、evidence_scope |
| 异常 | error；setup失败行允许缺少延迟和环境字段，汇总必须容忍 |

GPU峰值在模型已加载之后重置，allocated仍包含活跃模型内存；reserved是分配器保留值，不是“缓存比无缓存节省显存”。每次重复不清空CUDA allocator，所以reserved可能保留前次状态，交流时说明此边界。

两个D12示例文件名是 `d12_schema_cpu.jsonl`、`d12_schema_error.jsonl`，用于逐行学习。聚合脚本只扫描 `dNN_p*_g*_b*.jsonl` 与旧 `p*_g*_b*.jsonl`，因此不会自动汇总这两个示例；D13会用匹配命名另做重复实验。

## 失败分流与验收

- JSONL不能解析：检查是否把普通终端日志追加到了同一个raw文件；stdout日志与JSONL分开保存。
- 字段缺失：首先检查status/phase，setup错误缺timing是合法的；ok行缺字段才需修复。
- G=1：只有TTFT没有后续间隔，TPOT应null；不要除以G或强填0。
- 验收：能复算一次CPU记录的计数/时长/吞吐并解释状态，能说明旧新脚本口径差异，错误样例来自真实命令。没有GPU或checkpoint不影响本日完成；CPU结果按其功能范围交流，不填入GPU性能基线。

## 今日脚本与产物清单

每日编号使用全程 d01–d21；跨日复用保留首次实际执行日。以下项目相对路径均以仓库根目录为基准，脚本清单直接列出本日执行入口。

| 脚本完整项目路径 | 用途 | 执行阶段 | 输入 | 运行条件 |
|---|---|---|---|---|
| `.agents/docs/career_roadmap/w2/scripts/d12_benchmark.py` | 观察成功/提前结束行和错误行 | 第四步 | CPU主线：脚本内随机初始化GPT与合成ID映射，无磁盘权重；错误分支：`$NANOCHAT_BASE_DIR/base_checkpoints/absent-w2-schema-check/model_123456.pt`及同目录meta（故意不存在） | nanochat环境依赖可用；无需GPU/训练；缺模型非零退出是预期 |

| 产物完整项目相对路径 | 用途 | 生成者 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/d12_protocol.md` | 固定工作负载、计时与统计边界 | 第五步 | 可选人工整理；可复用本聊天确认的协议与自动记录，文件若生成仍保留 d12 |
| `experiments/inference_infra/results/raw/d12_schema_cpu.jsonl` | CPU随机模型逐run记录 | d12_benchmark.py | 执行第四步后自动追加；正常完成3次预热+5次正式行；异常可能提前退出，不预填结果 |
| `experiments/inference_infra/results/raw/d12_schema_error.jsonl` | 真实错误样例 | d12_benchmark.py | 条件自动；通过参数校验进入实验后写错误行；argparse 错误不写 |
| `experiments/inference_infra/reports/d12_session.log` | 当天终端输出与手动记录的退出码 | 各 Bash 代码块的 tee | 自动；仅追加各代码块内的输出 |

原生 `tokenizer.pkl`、`token_bytes.pt`、`model_<step>.pt`、`meta_<step>.json` 与 `optim_<step>_rank0.pt` 是loader的格式契约，选做既有checkpoint加载时保留其名称；CPU随机分支不读取这些文件。已有历史结果不原位重命名，新的命令使用本页命名。未执行的CPU检查如实说明，可选Markdown、既有checkpoint与GPU扩展未做均不影响本日完成。

## 今日自验证与交流问题

按题号在聊天中回答即可。概念题说明理由或推导；实测题引用CPU日志或JSONL中的关键片段。开头一次注明参数、session_id与证据路径，不必逐题重抄。未执行的操作如实说明；历史结果注明日期与版本。无需另写协议报告，也无需训练或运行GPU实验。

1. 对照 D7 审计的旧入口，当前 measure 为什么选择“第 G 列 yield”为结束点并立即关闭生成器？如果改成等待 StopIteration，或者把文本分词也算进去，还能沿用同一计时协议吗？

   **核对要点**：定位实际计时起止与 close，解释被纳入或排除的工作如何变化；已编码 ID 的生成延迟与文本请求延迟分别命名，不能把计时范围变化当成实现加速。

2. 假设 B=4、G=4，TTFT=10ms，后续输出列间隔为 `[2,3,9]ms`，且所有行完整有效。请推导本周TPOT、总时长与吞吐，再判断D7上游脚本会得到相同的TPOT和吞吐吗？G=1时怎样表示TPOT？

   **核对要点**：展示均值、总时间、有效计数与单位换算；旧脚本采用排序后的中间项及排除TTFT的后续吞吐。TPOT不除B，吞吐不重复乘B；G=1保留缺值。

3. 一次生成确实 yield 了 G 列，但其中一行提前产生终止 ID，或一列含工具强制插入的 token。这次结果还能当固定 G 的有效样本吗？为何不能直接用 B×G 作吞吐分子？

   **核对要点**：沿 `counts/completed/forced` 判断有效输出，区分终止 token、采样 token 与 forced token；解释脚本的停止条件和 `ok/early_stop` 判定，保留异常结果而不补齐数字。

4. 两次测量 P/G/B 完全相同，第二次却更换了 tokenizer、fill ID 或计时范围，能直接计算加速比吗？你会保留哪些最少的身份与协议证据来判断可比性？

   **核对要点**：解释模型/词表/输入身份和实际精度、后端、版本、计时边界对比较的作用，结合脚本已有哈希与环境记录；只列对判断有用的证据，避免逐字段抄表。说明同 prompt 复制 B 行与 B 条独立请求的差别，缺条件时限定结论。

5. 从CPU随机模型记录选一条正式行，用原始计数和时间复算吞吐，说明它的状态及证据范围。再对照缺模型的命令：实际在什么阶段失败，错误是否写进JSONL？若改成 `--warmup 0`，为何可能连原始行都没有？

   **核对要点**：CPU行复算一致，显存null表示未测量；随机初始化不等于训练，model_tag不单独证明权重来源。引用真实退出码与错误行的phase/status/error，区分缺模型与更早的依赖失败；按参数校验和try顺序解释落盘差异。错误可追溯不等于性能测量成功。

离开前保留本日会话日志与自动结果；若追加重跑，注明会话边界。交流时可附 `git rev-parse HEAD`、`git status --short` 及实际设备/精度以定位证据，不必另填通用表。下一天可直接复用本聊天中已确认的结论与证据路径。共享前检查数据路径和日志是否包含私人信息；若提交成果，只纳入自己的轻量记录与脚本，不提交权重、数据、环境目录或缓存。
