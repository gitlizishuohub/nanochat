---
source_files:
  - nanochat/engine.py
  - nanochat/gpt.py
  - nanochat/flash_attention.py
  - .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py
  - .agents/docs/career_roadmap/w3/scripts/d13_aggregate.py
source_symbols:
  - Engine.generate
  - KVCache.prefill
  - measure
  - metrics
doc_type: tutorial
knowledge_scope: W2第7天，解释批量生成、缓存容量与小矩阵证据
---

# W2 D7：理解批量生成，用小矩阵串起 W2

前一天：[返回前一天](d13_repeated_benchmark.md)；后一天：[W3第一天](../w3/d15_resume_draft.md)；[原始总计划](../nanochat_daily_execution_2026-10-01.md)。

今天解决一个具体问题：**B 从 1 变成 4，吞吐升高，是否说明每条样本都生成得更快？** 先用数字回答，再从 Engine 和统计脚本解释原因，最后用 CPU 小矩阵把本周的“生成→记录→复算→汇总”连起来。

第 1–3 节是概念推演与源码带读，只需阅读，不要求执行模型。第 4 节的动手主线使用 CPU 随机小模型，不训练、不下载模型，也不需要 checkpoint 或 GPU。完成这条主线即可完成 W2 的学习；已有 checkpoint 的重载、GPU 性能测量都是可选扩展。

## 当天的流程阅读

用约 10–20 分钟按问题回查，阅读不新增手写报告或实验产物。

- **阅读入口**：[5.3 batch、epoch 和每卡分配](../../llm-training-workflow/05-data-encoding-and-batching.md#reading-5-3)；[9.1 数据分工与梯度累积](../../llm-training-workflow/09-distributed-optimizer-and-communication.md#reading-9-1)；[9.2 优化器中的通信与分片](../../llm-training-workflow/09-distributed-optimizer-and-communication.md#reading-9-2)。
- **今天重点**：区分训练的批量、跨卡分工与今天同提示的多样本生成。world size 是参与分布式执行的进程数；梯度累积是在参数更新前累计多次小批量梯度；Engine 的 num_samples 是一次生成中的样本行数。
- **暂时跳过**：9.2 的集合通信和矩阵分片只理解职责，不推导算法，也不启动多卡实验。

**读完能解释**：把 Engine 的 `num_samples` 改成 4，为什么不会变成四卡训练，也不会发生四次梯度累积？

## 1. 用数字区分总吞吐和每行等待时间

### 1.1 B4 产出四行，但每行仍在等待下一列

沿用 D12 的记号：P 是输入 token 数，G 是每行要求输出的 token 数，B 是并行样本行数。先假设 P=8、G=4，所有行都正常生成，没有提前结束。下面是**纸面算例，不是测量结果**：

| 配置 | 首列等待 TTFT | 后续三列各等待 | 一次完成耗时 | 总输出 token | 总吞吐 |
|---|---:|---:|---:|---:|---:|
| B1 | 30 ms | 10、10、10 ms | 60 ms | 1×4=4 | 4÷0.060≈66.7 token/s |
| B4 | 45 ms | 15、15、15 ms | 90 ms | 4×4=16 | 16÷0.090≈177.8 token/s |

B4 每次输出一列，列里有四个 token；累计四列后，每行都有四个 token。它在 90 ms 内总共交付了 16 个 token，所以总吞吐比 B1 高。但每一行都等了 90 ms 才拿到完整四个 token，比 B1 的 60 ms 更久。

<abbr title="TPOT / Time Per Output Token · 后续输出 token 的平均等待时间。本文先排除首列，再对后续输出列的时间间隔取平均；一次列输出包含 B 行，不能再除以 B 后声称是每行延迟。">TPOT</abbr> 描述每行后续 token 的平均等待间隔。B4 的 TPOT 是 `(15+15+15)/3=15 ms`，不能再除以 4 写成 3.75 ms：每行收到下一 token 实际仍要等 15 ms。若用总耗时除总 token 得到 `90/16=5.625 ms/token`，那是另一种批量产出率的倒数，包含首列成本，也不是每行 TPOT。

因此比较 B1/B4 时，先分别回答“单位时间总共交付多少 token”和“某一行要等多久”，再谈哪一种改善。B4 是否更快只能由你关心的指标决定；CPU 实测也完全可能因为开销上升而没有吞吐提升。

### 1.2 本项目 B4 的四行从哪里来

Engine 接收的是一个提示 token 列表。它先只对这一份提示执行 <abbr title="Prefill · 提示处理阶段。模型一次处理已有输入，生成对应 K/V 缓存与末位置的预测分数；这里以 batch=1 执行，再复制给多个样本行。">prefill</abbr>，再把缓存复制到 B 行；此后每轮 <abbr title="Decode · 增量生成阶段。使用已有 K/V 缓存处理新 token，逐轮产生后续输出；本文一次输出列包含 B 个样本位置。">decode</abbr> 一起推进这 B 行。这与四个用户各自发送不同提示、分别承担 prefill 工作的服务场景有区别。

本页脚本使用 `temperature=0.0`，每行通过最大分数选择 token。相同提示和确定性计算下，四行通常内容相同；本实验检查批量形状、计数与计时，并不检验答案多样性或语言质量。

## 2. 从源码看 B 为什么同时影响计时和内存

以下是阅读片段，省略部分参数；无需复制执行。完整实现见 [Engine.generate 与 KVCache](../../../../nanochat/engine.py)。

### 2.1 一次 prefill，再分配 B 行 decode 缓存

```python
kv_cache_prefill = KVCache(batch_size=1, seq_len=len(tokens), ...)
logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
logits = logits[:, -1, :].expand(num_samples, -1)

kv_length_hint = len(tokens) + max_tokens
kv_cache_decode = KVCache(batch_size=num_samples, seq_len=kv_length_hint, ...)
kv_cache_decode.prefill(kv_cache_prefill)
del kv_cache_prefill
```

第一行分配单提示缓存，模型处理 P 个位置后，`logits[:, -1, :]` 取最后位置的预测分数。`expand` 将这份分数呈现给 B 行；它没有为 B 个不同提示重新计算 prefill。

随后新建的 decode 缓存容量是 P+G，批量维是 B。在 `KVCache.prefill()` 里，关键赋值是：

```python
self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
self.cache_seqlens.fill_(other_pos)
```

源缓存的批量维为 1，赋值会把已有 P 个位置的数据复制到目标缓存的每一行；`cache_seqlens` 记录各行已经有多少个有效位置。目标拥有实际分配的 B 行存储，不是 B 行永久共享同一份 K/V。接着旧缓存被删除；复制期间两份缓存会同时存在，删除后也不能据此假定分配器保留内存立即下降。

[D12 的 measure](scripts/d12_benchmark.py) 在第一次 `next(stream)` 前开始计时。Python 生成器此时才进入函数体，所以分配、prefill、复制和第一次采样都落入首列 TTFT。B 增大可能增加首列成本，也可能增加后续每列成本；只看总吞吐，无法判断成本变化发生在哪一段。

### 2.2 用张量形状计算容量，不把它当作总内存

`KVCache.__init__` 各分配一个 K 和一个 V 张量，形状都是 `(L, B, T, Hkv, D)`。L 是层数，T 是分配容量，Hkv 是 K/V 头数，D 是每头维度，即 `n_embd // n_head`。设每元素占 s 字节，则：

```text
K/V 张量容量 = 2 × L × B × T × Hkv × D × s 字节
```

系数 2 来自 K、V 两份张量。这里应使用 `n_kv_head`，不是随意把查询头数 `n_head` 填进去；T 是已分配容量，也不是当前已用的位置数。

以本页 CPU 随机模型为例：L=2、Hkv=2、D=64/2=32，FP32 每元素 4 字节；P=8、G=4，所以 decode 容量 T=12。纸面复算得到：

| 部分 | B1 | B4 |
|---|---:|---:|
| decode K/V | 2×2×1×12×2×32×4=12,288 bytes | 49,152 bytes |
| 单提示 prefill K/V | 8,192 bytes | 8,192 bytes |
| 复制时两份 K/V 合计 | 20,480 bytes | 57,344 bytes |

decode K/V 随 B 放大四倍，但复制时的这两份合计没有放大四倍。这仍然**只是 K/V 张量容量**：没有计入模型参数、预测分数、中间计算张量、序列长度计数，以及缓存对象中的 `prev_embedding` 等状态。

CUDA 的 allocated 峰值是分配器跟踪的活跃张量内存峰值，reserved 峰值还包含其保留空间；两者都不能直接用上表替代。CPU 分支将这两个 CUDA 字段写为 `null`，表示没有测量该指标，不表示内存占用为零。你可以在 CPU 上完成容量推导，无需 GPU 实测。

## 3. 从一次生成到一个可追溯的统计结果

### 3.1 为什么 G=4 时只有三个后续间隔

在 [measure](scripts/d12_benchmark.py) 中，每次 `next(stream)` 返回一个 B 行输出列，然后记录距上一次取样的时间。第一项 `times[0]` 是 TTFT，`times[1:]` 才是后续列间隔。关键计算是：

```python
elapsed = sum(times)
generated_tokens = sum(counts)
tpot_ms = sum(times[1:]) / len(times[1:])
tokens_per_s = sum(counts) * 1000 / elapsed
```

上面 TPOT 只展示 G>1 的分支；G=1 时实际写入 `None`。B4/G4 的一条完整记录应有四个逐行计数 `[4,4,4,4]`，总计 16；输出列数 `actual_decode_steps=4`，后续间隔数为 3。总耗时单位是 ms，因此吞吐计算要乘 1000，换成每秒。

每条样本都必须完整生成才记为 `ok`。提前终止或工具强制 token 不能与固定长度成功行混在一起；失败记录应保留。脚本恰好取 G 次输出后关闭生成器，避免恢复最后一次 yield 后多算一次不使用的 forward。这里也没有把 tokenizer 或模型加载耗时计入生成区间。

### 3.2 汇总先复算每条，再比较同一条件的重复

[自动统计脚本](../w3/scripts/d13_aggregate.py) 的 `metrics()` 先校验形状、token 数与时间关系，再从原始字段复算 TPOT、吞吐；主流程按 session、模型身份、精度、后端、P/G/B 和环境等条件分组。这意味着 B1/B4 会成为不同组，而同一配置重新启动得到的新 session 也会单列。

分组以后，脚本对各 run 的指标求中位数，而不是把所有 token 和所有时间先混在一起。例如两条同条件 run 各生成 4 个 token，分别耗时 10、30 ms：逐 run 吞吐是 400、约 133.3 token/s，中位数约为 266.7；若改算总 token/总时长，则是 `8/0.040=200 token/s`。两者的统计口径不同，不能互换字段名。

<abbr title="Audit · 审计追踪记录。本文记录每条非预热记录的输入文件、行号、run_id、是否计入和排除原因，帮助从汇总回查原始证据。">audit</abbr> 文件使你能从 summary 回到具体 raw 行。预热会被单独检查，不作为正式统计样本；五次正式测量足以验证这条统计链，但其经验 P95 只是五个值的最大值，不能据此声称掌握了稳定的尾延迟。

## 4. 动手：用 CPU 跑 B1/B4，再解释输出

### 4.1 准备与日志

下面开始才需要执行命令。终端沿用自动激活的 nanochat 环境，所有命令从仓库根目录执行；显式选择 FP32，不混用系统 Python。日志只在各 `{ ...; }` 命令组中交给 `tee`，不持续重定向交互终端。已有旧版全局重定向的终端应先关闭再新开。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
export NANOCHAT_DTYPE=float32
{
python -c "import sys; print(sys.executable); print(sys.version)"
} > >(tee -a experiments/inference_infra/reports/d14_session.log) 2>&1
```

解释器应为 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。环境缺失时先修复环境；本页无需准备语料、tokenizer 文件或训练权重。帮助、解释器和源码检查只打印，其输出保存在会话日志，没有单独结果文件。

### 4.2 先预测，再运行两组

先写出或口述 B1、B4 各自应有的有效 token 总数、输出列数、后续间隔数。再执行 [重复基准脚本](scripts/d12_benchmark.py)：

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py --device cpu --random-model --prompt-tokens 8 --decode-tokens 4 --batch-size 1 --warmup 3 --repeats 5 --output experiments/inference_infra/results/raw/d14_p8_g4_b1_gate_cpu.jsonl
rc=$?; echo "B1 exit=$rc"
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py --device cpu --random-model --prompt-tokens 8 --decode-tokens 4 --batch-size 4 --warmup 3 --repeats 5 --output experiments/inference_infra/results/raw/d14_p8_g4_b4_gate_cpu.jsonl
rc=$?; echo "B4 exit=$rc"
} > >(tee -a experiments/inference_infra/reports/d14_session.log) 2>&1
```

每组分别检查打印的退出码：0 表示全部记录成功，2 表示存在提前结束等不完整样本，1 表示发生设置或运行错误。后续 `echo` 的成功不能替代前一条 Python 命令的结果。CPU 运行可能等待一段时间；计时值受本机线程、调频和其他进程影响，不要求接近第 1 节的假设数字。

raw 是追加写入。首次运行使用上述路径；若重跑，使用新文件名，例如 `d14_p8_g4_b1_retry1_cpu.jsonl`，避免把多次执行误认为一个 session。两组各有 3 条 warmup、5 条 measure 才是预期；不能只数总行数判断成功。打开一条 `phase=measure,status=ok` 的 B4 记录，检查：

| 字段 | 应如何理解 |
|---|---|
| `per_row_generated_tokens` | `[4,4,4,4]`，每行四个有效 token |
| `generated_tokens` | 16，是所有行合计 |
| `actual_decode_steps` / `decode_step_ms` | 4 列输出 / 3 个后续间隔 |
| `elapsed_ms` | TTFT 加三个后续间隔，允许日志精度误差 |
| `tpot_ms` | 三个后续间隔的均值，不除 B |
| `tokens_per_s` | `16×1000/elapsed_ms` |
| CUDA 内存字段 | `null`，本分支没有测 CUDA 内存 |

若记录为 `early_stop` 或 `error`，先看其状态与 `error` 内容，保留原文件再修复问题，不靠改标记让它变成 `ok`。两组结果一高一低都可以是有效观察：先确认模型配置、prompt 身份、环境与精度可比，再解释总吞吐和每行延迟，不能从 CPU 随机模型推出 GPU 或真实服务的加速倍数。

### 4.3 自动汇总并回查一条证据

执行 [统计入口](../w3/scripts/d13_aggregate.py)：

```bash
{
python .agents/docs/career_roadmap/w3/scripts/d13_aggregate.py --raw-dir experiments/inference_infra/results/raw --output experiments/inference_infra/results/summary/d14_w2_gate.csv --audit experiments/inference_infra/results/summary/d14_w2_gate_audit.jsonl
rc=$?; echo "aggregate exit=$rc"
} > >(tee -a experiments/inference_infra/reports/d14_session.log) 2>&1
```

该入口读取 raw 目录下所有匹配 `dNN_p*_g*_b*.jsonl` 的文件，也兼容旧命名。因此 CSV 可能含 D13 或历史重跑的组；通过 audit 的完整输入路径、`run_id` 与 CSV 的 `session_id` 选出今天的两组。预期每组 `n_valid=5`、`warmup_status=OK`，`conclusion_status=CPU_OR_RANDOM_CONTRACT`。这个状态表示证据适用于 CPU/随机模型功能检查，**不是本周学习失败**，也无需为了改成 GPU 状态而购买硬件或训练模型。

任选一条 included=true 的 B4 audit 记录，回到它指向的 raw 文件与行号，手算一次 TPOT、总吞吐，再确认 run_id 在对应 CSV 组里。最后将 B1/B4 两组的吞吐和延迟并排比较，用第 1 节的方法解释。summary 保存的是一组 run 的统计，单条 raw 的数值不必等于组中位数。

### 4.4 可选扩展，不影响本日完成

若想观察提示长度影响，可把 P 改为 32，B 仍分别取 1/4，输出改为 `experiments/inference_infra/results/raw/d14_p32_g4_b1_gate_cpu.jsonl` 和 `experiments/inference_infra/results/raw/d14_p32_g4_b4_gate_cpu.jsonl`；随后重跑汇总。若今天只做 B1/B4 两组，直接结束即可，无需把未选的矩阵格记成待办。

若已经有 [D10 可选实验](d10_training_smoke.md) 生成并由 [D11](d11_checkpoint_generation.md) 核对的 checkpoint，可在基准命令中去掉 `--random-model`，显式加 `--source base --model-tag d10_w2_smoke --step 2`，并用新的 raw 路径 `experiments/inference_infra/results/raw/d14_p8_g4_b1_checkpoint_cpu.jsonl`（B4 相应改为 b4）。复用实际 `NANOCHAT_BASE_DIR`，不要为了本日额外训练。示例标签指合成语料两步模型，只验证加载和生成机制，不代表语言能力；若用其他来源，填写真实标签、步数和来源。

本页不要求 CUDA 实验。未来确实有设备与模型时，可按已学的计时协议单独测量；在那之前，性能结论只写到本次实际设备、模型与输入覆盖的范围。

## 5. W2 回顾与交流问题

W2 的学习链是：D08/D09 理解文本怎样成为 token 和训练批次，D10 理解参数更新与训练循环，D11 理解保存/重建状态，D12 定义可比较的生成记录，D13 控制重复和预热，D14 用批量变式检验自己能否解释统计。理解训练机制不要求亲自训练；这周也不要求学会训练一个有用的语言模型。

在聊天里回答下面 5 题即可，人工报告可选。概念题用本页算例作答；实操题引用自己实际运行的片段。若尚未执行 CPU 命令，直接说明“概念已读，CPU 实操未执行”，这描述进度，不生成虚构成绩。

1. B4 比 B1 吞吐高，但 TTFT 和 TPOT 都更高，这是否矛盾？用一个具体数值解释，并说明为何不能把 TPOT 除以 4。

   **核对要点**：区分总 token 产出率、每行等待间隔和一次完成耗时；B、G 与间隔数对应正确。

2. Engine 的 B4 与四个不同用户请求相比，复用了哪部分计算、为每行准备了什么存储？为什么这也不等于四卡训练或四次梯度累积？

   **核对要点**：能沿单提示 prefill→缓存复制→B 行 decode 解释；参数是否更新、是否跨进程执行说得清，不要求推导通信算法。

3. 仅把 P 从 8 改成 32，G=4、B=4 保持不变，decode K/V 容量怎样变化？能否据此断言总内存也按同一比例变化？

   **核对要点**：从实际容量 T=P+G 代入第 2 节公式，区分 K/V 预分配、单提示缓存和其他张量，不拿 CPU 的 null 当零显存。

4. 从今天一条有效 B4 raw 记录复算 TPOT 和吞吐，再用 audit 找到它在 summary 中的归属。如果 CSV 里出现多于两个组，你会先核对什么？

   **核对要点**：单位、有效计数、run_id 和输入来源能对应；知道共享目录与 session 分组的作用，区分单条数值和组中位数。未运行则先用本页假设值练算并说明未有实测证据。

5. 两个 run 各输出 4 个 token，耗时 10、30 ms。解释逐 run 吞吐的中位数与总 token/总时间为何不同；再用一句话描述本周 CPU 随机模型实验能支持的结论。

   **核对要点**：独立复算两种口径，不混换名称；结论明确设备、模型与功能范围，理解 CPU 完成记录链即可进入 W3，没有训练或 GPU 也能完成本周。

## 今日脚本与产物清单

以下完整项目相对路径均以仓库根目录为基准。仅在实际执行对应命令后才生成产物；本教程不预填成绩。脚本复用首次编号，不增加周 README。

| 脚本完整项目路径 | 用途 / 阶段 | 完整输入路径与条件 |
|---|---|---|
| `.agents/docs/career_roadmap/w2/scripts/d12_benchmark.py` | 第 4.2 节 CPU B1/B4；4.4 可选扩展 | 默认随机分支无文件输入，仅 P/G/B；需要 nanochat CPU 环境。可选 checkpoint 分支读取 `$NANOCHAT_BASE_DIR/base_checkpoints/d10_w2_smoke/model_000002.pt`、`$NANOCHAT_BASE_DIR/base_checkpoints/d10_w2_smoke/meta_000002.json` 和 `$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl`，不读取优化器状态 |
| `.agents/docs/career_roadmap/w3/scripts/d13_aggregate.py` | 第 4.3 节逐条复算、分组和审计 | 读取 `experiments/inference_infra/results/raw/` 中符合命名模式的实际 JSONL；先有本次 raw 再运行 |

`$NANOCHAT_BASE_DIR` 仅用于可选的已有文件分支，表示实际数据根；未设置时本用户默认 `/home/lizishuo/.cache/nanochat`，若先前 D10 使用 `/tmp/nanochat-w2-toy`，此处继续使用同一个实际根。交流时展开为真实绝对路径，不仅贴变量名。默认 random 分支不需要设置它。

| 产物完整项目相对路径 | 用途 | 生成者及条件 |
|---|---|---|
| `experiments/inference_infra/results/raw/d14_p8_g4_b1_gate_cpu.jsonl` | B1 原始记录 | 第 4.2 节 d12_benchmark.py 自动追加，实际执行才生成 |
| `experiments/inference_infra/results/raw/d14_p8_g4_b4_gate_cpu.jsonl` | B4 原始记录 | 同上，B4 命令 |
| `experiments/inference_infra/results/raw/d14_p32_g4_b1_gate_cpu.jsonl` | 可选 P32/B1 | 第 4.4 节选择扩展且实际运行时自动生成 |
| `experiments/inference_infra/results/raw/d14_p32_g4_b4_gate_cpu.jsonl` | 可选 P32/B4 | 同上，B4 命令 |
| `experiments/inference_infra/results/raw/d14_p8_g4_b1_checkpoint_cpu.jsonl`、`experiments/inference_infra/results/raw/d14_p8_g4_b4_checkpoint_cpu.jsonl` | 可选已有 checkpoint 的批量记录 | 第 4.4 节对应替换参数命令，满足文件条件且实际执行才生成 |
| `experiments/inference_infra/results/summary/d14_w2_gate.csv` | 分组统计 | 第 4.3 节 d13_aggregate.py 自动生成，重跑覆盖 |
| `experiments/inference_infra/results/summary/d14_w2_gate_audit.jsonl` | 回查原始行与排除原因 | 同上，自动生成，重跑覆盖 |
| `experiments/inference_infra/reports/d14_session.log` | 命令输出、实际退出码 | 第 4 节各局部 tee 自动追加；检查重跑时的会话边界 |
| `experiments/inference_infra/reports/d14_w2_gate.md` | 可选周结整理 | 人工编写；直接在聊天回答即可，无需创建 |

本页不新增训练产物。CPU 运行异常应记录真实错误与下一步修复；没有选择 checkpoint、P32 或 GPU 扩展不算缺项。保留日志与自动结果后，继续 [W3 简历初稿](../w3/d15_resume_draft.md)，描述自己实际理解和验证过的能力。
