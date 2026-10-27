---
source_files:
  - scripts/infer_bench.py
  - nanochat/engine.py
  - nanochat/checkpoint_manager.py
  - nanochat/gpt.py
source_symbols:
  - bench_generate
  - build_prompt
  - Engine.generate
  - Engine.generate_batch
  - load_model
  - GPT.generate
doc_type: tutorial
knowledge_scope: W1 D7 从生成执行过程理解计时、计数与基准审计
---

# W1 D7：带着三个生成 token，读懂推理基准在测什么

有人告诉你：“这个模型能生成 100 token/s。”你还需要知道什么，才能判断它快不快？至少要问：是否包含处理输入的时间？是一个回答还是多个回答的总速度？提前结束后怎么计数？今天沿 nanochat 的执行过程，学会自己回答这些问题。

本日把 **代码执行 → 计时范围 → 输出计数 → 指标含义** 连起来。你不需要先读懂整份源码；下面先解释概念，摘出关键代码，走完一个例子，再回原文件验证。预计 90–110 分钟，可分两次：概念与生成器约 25 分钟，源码与计时约 30 分钟，统计与输入约 25 分钟，交流与回顾约 15 分钟。这是建议预算，不是完成记录。

**执行范围**：第 1 节是纸面推演，第 2 节运行 Python 标准库教学脚本，第 3–6 节阅读真实源码并计算示例。本日不运行模型、不更新参数，也不需要训练过的权重或 GPU。源码中的 CUDA 基准是学习对象，不是当天需要执行或补做的任务。

前置：[D4 模型前向](d04_model_shapes.md)、[D5 缓存](d05_cache_lifecycle.md)、[D6 后端](d06_attention_backends.md)。先记住：模型把 token ID 变成预测分数；缓存复用历史 K/V；计算可以使用不同注意力后端。忘记时回查，不必重做已通过的实验。

## 当天的流程阅读

先用约 10–15 分钟读指定小节获得背景，再顺着本文学习。

- **阅读入口**：[13.2 能力与系统性能的边界](../../llm-training-workflow/13-evaluation-and-metrics.md#reading-13-2)；[12.2 生成控制流与返回格式](../../llm-training-workflow/12-inference-and-generation.md#reading-12-2)。
- **今天重点**：回答是否准确与回答花多长时间是两类指标；逐列输出和最终有效文本也有区别。
- **暂时跳过**：能力评估公式、带宽利用率公式、强化学习与训练命令。先掌握计时和计数。

**带着问题回来**：文本生成正常，为什么性能数字仍可能不可比？正文逐步建立答案，末尾换情景检验。

## 1. 用假设例子推演一次生成，再理解怎样计时

### 1.1 P、G、B 改变了什么

P 是提示 token 数，G 是要求生成的最大列数，B 是并行样本数。<abbr title="token · 词元。模型处理的离散编号，不一定等于一个汉字、单词或字符；数量以分词器编码为准。">token</abbr> 数不能用字符数代替。

下面只做纸面推演，无需执行。假设 **P=8、G=3、B=1**，没有提前终止或工具调用，模型依次选出 a、b、c；字母是代指 token ID 的占位符：

1. 一次处理 8 个输入位置，建立缓存，用最后位置的预测分数选出 a。这叫 <abbr title="prefill · 预填充。处理整个提示前缀、建立 KV Cache，并得到预测首个输出所需的分数。">prefill</abbr>。
2. 把 a 输入模型，结合提示缓存，得到预测 b 的分数并选出 b。
3. 把 b 输入模型，得到预测 c 的分数并选出 c。后面这种复用缓存、逐 token 推进的过程叫 <abbr title="decode · 增量解码。本文指首个输出之后，输入上一个 token 并复用历史缓存来预测下一个 token。">decode</abbr>。

**生成 a 不需要先把 a 输入模型**：它由提示最后位置的输出预测。因此三个输出只有两个后续输出间隔。

这里描述的是推理：用已有参数计算下一个 token，参数不更新。训练则需要根据目标计算损失、反向传播并更新参数。本节既没有要求训练，也没有要求你实际加载模型完成上述三步。

本基准的 B 是**同一个提示的 B 个样本**。Engine 先做 batch=1 prefill，再复制缓存到 B 行，不是给 B 个不同请求分别 prefill。这里 B=4 的吞吐不能直接外推为在线服务处理四个不同用户的速度。

### 1.2 三个指标回答不同问题

| 指标 | 想回答的问题 | 必须说明什么 |
|---|---|---|
| <abbr title="TTFT / Time To First Token · 首词元延迟。从约定起点到首个输出可用的时间，是否包含分词、排队、缓存分配要看计时边界。">TTFT</abbr> | 第一个 token 等多久？ | 从哪里开始、到哪里结束 |
| <abbr title="TPOT / Time Per Output Token · 每输出词元时间。当前入口用首 token 后各步间隔的较高中间值表示；其他实现可能采用不同统计。">TPOT</abbr> | 后续 token 间隔多长？ | 平均值、中位数还是其他统计 |
| <abbr title="throughput · 吞吐。单位时间产出的 token 数，需要定义分子计哪些输出、分母覆盖哪些时间。">tokens/s</abbr> | 单位时间总共产出多少？ | 计数与计时是否覆盖同一段过程 |

做一个**假设算例，不是实测**：首 token 等 30 ms，后两个各等 10 ms。后续 decode 吞吐是 `2/0.020=100 token/s`；从起点到第三个输出的整体速率是 `3/0.050=60 token/s`。同一次生成得到不同数字，因为覆盖的工作不同。

因此，只看到“100 token/s”还不能判断一次回答有多快：这个例子中的 100 排除了首 token 的 30 ms，包含首 token 后才是 60 token/s。阅读性能结果时，需要先确认它统计哪段时间；这里无需执行额外操作。

## 2. CPU 教学实验：为什么调用 generate 还没有开始生成

### 2.1 理解 yield 与 next

普通函数通常从入口执行到 return。包含 yield 的 Python <abbr title="generator · 生成器。调用先得到可恢复执行的对象；next 推进到下一个 yield，yield 返回值并暂停，下次 next 从暂停处继续。">生成器</abbr>可以分段执行：

- `stream = generate(...)`：创建对象，函数体尚未执行。
- `next(stream)`：启动执行，到 `yield value` 时交出 value 并暂停。
- 再次 `next(stream)`：从上次 yield 的后面恢复，不是从第一行重跑。
- 函数结束：`next(stream)` 抛出 `StopIteration`，表示没有下一个值。

“创建对象”“得到最后一个值”“确认结束”可能是三个不同的时刻。nanochat 的计时问题正好落在这些边界上。

### 2.2 先预测，再观察四次 next

打开[生成器追踪脚本](scripts/d07_trace_generator.py)。它只用 Python 标准库，用事件标签模拟 Engine 的顺序；没有模型、张量或耗时测量。输出中的 forward 只是标签。

下面是核心教学摘录，完整可执行版本见脚本：

```python
events.extend(["prefill(prompt)", "copy_cache(batch)"])
num_generated = 0
while True:
    if num_generated >= max_tokens:
        events.append("stop(max_tokens)")
        return
    column = num_generated + 1
    events.extend([f"sample(column={column})", f"yield(column={column})"])
    yield column
    events.append(f"resume(column={column})")
    num_generated += 1
    events.append(f"forward(column={column})")
```

**运行前预测**：创建生成器后 events 是否已有内容？第三次 next 返回第三列时，forward 3 执行了吗？第四次 next 会直接停止吗？不必写报告，预测后对照日志。

命令在项目根目录执行，沿用已自动激活的 nanochat 环境，不重复激活。`--tokens 3` 模拟返回三列，`--output` 指定追踪记录。复制整个命令组，日志只在花括号内部记录，不持续接管终端。若旧终端仍启用了全局日志重定向，先新开终端。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
python -c 'import sys; print(sys.executable)'
python .agents/docs/career_roadmap/w1/scripts/d07_trace_generator.py --tokens 3 \
  --output experiments/inference_infra/results/w1/d07_generator_trace.json
trace_status=$?
printf 'generator trace exit=%s\n' "$trace_status"
} > >(tee -a experiments/inference_infra/reports/d07_checks.log) 2>&1
```

正常运行应观察以下**事件关系**，而不是某个性能数字：

| 调用 | 本次推进的工作 | 结果 |
|---|---|---|
| 创建生成器 | 无函数体事件 | 一个生成器对象 |
| 第 1 次 next | prefill → 复制缓存 → sample 1 → yield 1 | 第 1 列 |
| 第 2 次 next | 恢复 → forward 1 → sample 2 → yield 2 | 第 2 列 |
| 第 3 次 next | 恢复 → forward 2 → sample 3 → yield 3 | 第 3 列 |
| 第 4 次 next | 恢复 → forward 3 → 检查上限并停止 | StopIteration |

如果预测错了，找 yield 与停止检查的位置：yield 暂停时，下面的递增和 forward 尚未执行；下次恢复先执行它们，之后才回到循环顶部检查上限。

**改变一个条件**：把 `--tokens 3` 改为 `--tokens 1`，输出改为 `experiments/inference_infra/results/w1/d07_generator_trace_g1.json`，沿用同一个命令组运行。先预测需要几次 next、几次 forward。这检验控制流理解，不能证明真实模型已运行。

## 3. 源码带读：把事件标签对应到真实实现

### 3.1 先认识调用链和文件职责

打开[scripts/infer_bench.py](../../../../scripts/infer_bench.py)的 main、bench_generate，再看[nanochat/engine.py](../../../../nanochat/engine.py)的 Engine.generate：

```text
main：加载模型与 tokenizer → 准备 prompt_tokens → 预热
  bench_generate：创建生成器 → 计时并反复 next
    Engine.generate：管理缓存、逐列返回、判断停止
      model.forward：执行模型计算，输出预测分数
```

计时器包在哪一层，决定覆盖哪些工作。D4 的 GPT.generate 是另一条无缓存参考路径；名字同样含 generate，不能把它当成本基准调用的实现。

### 3.2 第一次 next：首 token 前做了哪些事

以下是 Engine.generate 的**关键原句摘录，中间有省略，不是独立运行脚本**：

```python
logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
logits = logits[:, -1, :].expand(num_samples, -1)
# 中间分配 decode cache，并复制提示缓存。
kv_cache_decode.prefill(kv_cache_prefill)
# 循环内采样，随后构建输出列并更新行状态。
next_ids = sample_next_token(logits, rng, temperature, top_k)
sampled_tokens = next_ids[:, 0].tolist()
# 省略逐行处理。
yield token_column, token_masks
```

对应 P=8、G=3、B=1 的例子：

1. 分配提示的 <abbr title="KV Cache / Key-Value Cache · 键值缓存。保存已处理位置的注意力 K/V，供后续解码复用；属于运行状态，不是模型参数。">KV Cache</abbr>，把 8 个 ID 组成张量送入 forward。
2. logits 是各位置对词表中每个 token 的预测分数。`logits[:, -1, :]` 的三个轴分别选“全部样本、最后位置、全部词表项”，用最后位置分数预测 a；expand 为 B 个样本提供分数视图，不是重新执行 B 次前向。
3. 为 B 行分配 decode 缓存，复制提示缓存并初始化行状态。B 增大时，这部分工作和内存需求会变化。
4. 选择 token；temperature=0 时使用 argmax，即选择分数最高的 ID。tolist 将选中 ID 转为 Python 列表，再更新每行状态。
5. yield 返回一列 ID 和来源 mask。

因此第一次 next 包含缓存分配、prefill、复制、采样与 CPU 状态处理，远不止一个注意力运算。

### 3.3 后续 next 与结束探测

Engine 在 yield 后的真实顺序是：

```python
yield token_column, token_masks
num_generated += 1
ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]
```

第二次 next 恢复后输入 a，得到预测 b 的分数；第三次输入 b，得到预测 c 的分数。

**本例取满三列后，benchmark 还会调用第四次 next**：先输入 c 做一次 forward，再回到循环顶部发现达到上限，抛出 StopIteration。没有第四列被返回。这解释了教学脚本中的尾部 forward。

如果调用者拿到第三列就停止消费，执行路径会不同。这里解释当前 benchmark 如何穷尽生成器，不表示所有生成接口都必须多做一次 forward。回原文件核对：停止检查在循环顶部，而 forward 在 yield 后面。

## 4. 放回计时器：TTFT 到底包含什么

### 4.1 从起止语句推导，别根据变量名猜测

bench_generate 的 TTFT 原句：

```python
torch.cuda.synchronize(device)
generator = engine.generate(prompt_tokens, num_samples=batch_size,
                            max_tokens=decode_tokens, temperature=temperature)
t_start = time.perf_counter()
next(generator)
torch.cuda.synchronize(device)
ttft = time.perf_counter() - t_start
```

perf_counter 测 CPU 侧经过时间。开表前先等之前的设备工作完成；创建生成器后开表，推进到首个 yield，再等设备工作完成才读表。

为什么要 <abbr title="CUDA synchronize · CUDA 设备同步。等待所选设备已提交的 GPU 工作完成；同步本身也有成本，比较两版要保持边界一致。">CUDA 同步</abbr>？CPU 通常提交 GPU 任务后就能继续执行，若直接读 CPU 时钟，可能只量到提交动作，计算仍未结束。这是“发出任务”与“确认完成”的区别。内部某些操作也会同步，但不能据此随意删掉边界同步。

从顺序推导包含关系：

| 操作 | 在 TTFT 内吗 | 理由 |
|---|---|---|
| 加载模型、构造提示并分词 | 否 | main 在调用 benchmark 前完成 |
| 创建生成器 | 否 | 开表之前，函数体尚未执行 |
| 缓存分配、prefill、复制到 B 行 | 是 | 首次 next 内、首个 yield 前 |
| 首 token 选择、转列表、状态处理 | 是 | 首个 yield 前 |
| 第一次 next 后的同步 | 是 | 读表之前 |
| 输入 a 以预测 b 的 forward | 否 | 下次 next 才恢复 |

这里测的是**输入已为 token ID 时，Engine 产出第一列的延迟**。用户请求还可能有分词、排队、网络与展示，此处没测到。它也比纯 prefill 算子时间更宽。main 将 batch=1 的 TTFT 赋给 prefill_time 计算近似 prefill 指标，变量名不会改变边界。

### 4.2 step_times 之和为什么不是全程耗时

后续计时原句：

```python
step_times = []
while True:
    t0 = time.perf_counter()
    try:
        next(generator)
    except StopIteration:
        break
    torch.cuda.synchronize(device)
    step_times.append(time.perf_counter() - t0)
```

第二、三次 next 成功，各记录一个间隔；第一列在 TTFT 内。因此没有提前结束时，G 个输出对应 G−1 个后续间隔。

第四次 next 执行尾部 forward，却在异常分支直接 break，没有 append，也没走到后面的显式同步。**这段工作没进入已记录的间隔，不能把实际耗时当作零。** 内部操作可能已等待部分 GPU 工作，也不能代替明确的全程边界。

测“完整消费到结束”的时间，需要另设覆盖消费过程的计时与最终同步；它和“直到最后一列可用”的时间又有区别。今天识别差别，W2 再核对实现。

## 5. 从执行过程推导统计与输入问题

### 5.1 原入口怎样算 TPOT 和吞吐

main 的公式是：

```python
num_steps = len(step_times)
tpot = sorted(step_times)[num_steps // 2]
tok_per_sec = batch_size * num_steps / sum(step_times)
```

step_times 单位是秒；吞吐分子只按后续列计数，分母排除首 token 与结束探测。

假设间隔为 `[0.004,0.006,0.012,0.018]` 秒，四项排序后 `4//2=2`，Python 从 0 编号，因此取 12 ms。两个中间值平均则是 9 ms，四项算术平均是 10 ms。原入口采用**较高的中间项**，不是后两种定义。

这四步也不是四次完整运行：它们只经历一次初始缓存分配与首 token 阶段，不能提供四个独立 TTFT 样本。

<abbr title="warmup · 预热。正式测量前执行相关路径，让初始化、内存分配器和算子选择等准备状态趋稳；是否充分仍需观察。">预热</abbr>用于减少首次准备成本的影响。原入口为 prefill 做一次 G=2 预热、一次正式测量；对每个 B 做一次 G=8 预热、一次正式测量。W2 约定至少 3 次预热、5 次正式重复并存原始值，这是项目协议，不保证任何设备上都已充分预热。

### 5.2 一列 B 个 ID，不一定是 B 个有效输出

做一个**假设反例，不是默认配置实测**：B=4，一行在第 2 列出现 `<|assistant_end|>`，其余三行继续到第 5 列。第一行结束后的 ID 不应继续算作有效回答，但 Engine.generate 仍逐行处理，直到所有行完成才整体停。

默认 temperature=0、各行同提示时通常输出相同、一起结束；上例审查其他采样配置下的边界。需要区分：

- **完成状态**：由终止 ID 与每行 completed 状态决定。终止 token 本身计不计也需约定；generate_batch 会过滤终止 token 及其后内容。
- **来源 mask**：token_masks 中 1 表示采样、0 表示工具强制注入，不能表示哪一行已完成。mask=0 也不表示没有输出，只是来源不同。

benchmark 丢弃 next 返回的列和 mask，没有重建完成状态。因此 B×步数只是按列推得的数量。要报告有效文本吞吐，先定义计数对象，再存逐行证据以便复算。

### 5.3 输入长度可能被自动改掉

main 会截短提示：

```python
max_prompt = config.sequence_len - args.decode_tokens
prompt_len = min(args.prompt_tokens, max_prompt)
prompt_tokens = build_prompt(tokenizer, prompt_len)
```

例如 sequence_len=64、请求 P=60、G=8，最大 P 是 56，实际为 `min(60,56)=56`。一版处理 56、另一版处理 60 时，速度差异混入了工作量变化。

原 JSON 已存实际 prompt_len，但没同时保存请求值；应把 requested/actual 放在一起。build_prompt 的长度包含 BOS，不应再加一。零/负长度未被上述代码完整拒绝，`P+G<=sequence_len` 之外还要检查参数范围。G=1 没有后续间隔，原入口跳过该组，不能把 TPOT 写成 0 ms。

## 6. 回查源码，把审计发现对应到 W2 的测量协议

### 6.1 回查原句，检验能否解释

现在已知道要找什么。运行短命令组后，在原文件找到计时起止、yield 后 forward、统计公式与截短输入；分别说明“改变它会改变哪个指标”。源码变化后重查，不把定位文件当作理解。

```bash
{
rg -n 'def bench_generate|def build_prompt|synchronize|t_start|step_times|tpot =|tok_per_sec =|max_prompt|prompt_len =' scripts/infer_bench.py
rg -n 'def generate|num_generated|state.completed|yield token_column|model.forward|token_masks.append' nanochat/engine.py
rg -n '^def load_model|^def build_model|model_config|vocab_size' nanochat/checkpoint_manager.py
} > >(tee -a experiments/inference_infra/reports/d07_checks.log) 2>&1
```

被阅读的上游 GPU 入口要求 CUDA、单进程、合法 nanochat <abbr title="checkpoint · 检查点。该入口需要模型权重及配套架构元数据等资产，还要匹配 tokenizer；其他项目的单个权重文件不能直接替换。">checkpoint</abbr> 与匹配 tokenizer。load_model 按保存配置构建模型，再载入权重和 tokenizer；词表大小相同也不能单独证明 ID 语义映射一致。这些是解释上游入口的条件，本日不执行它，不需要补齐这些资源，也不标 GPU 待办；不要为本日运行包含下载与长训练的 runs/runcpu.sh。

### 6.2 把理解转成可核验要求

| 问题 | 已理解的原因 | W2 要核对什么 |
|---|---|---|
| 同名指标测不同工作 | 计时、同步、停止消费的位置不同 | 明确 TTFT、后续间隔、全程边界与单位 |
| 一次运行难判断波动 | 步内间隔不是完整运行样本 | 至少 3 次预热、5 次正式重复，留原始值 |
| B×列数可能高估有效量 | 提前完成与工具注入的口径不同 | 逐行 ID、来源 mask、完成/停止原因与有效计数 |
| 请求长度不等于实际长度 | P 会被 clamp | requested/actual P、G、B 及合法性检查 |
| 缺比较条件 | 模型、设备、精度、后端可能改变 | 版本、环境、seed、实际 dtype/后端，失败组也保留 |
| 峰值显存被误读 | 重置统计不释放已有模型 | 说明 peak allocated，不能称纯 KV 增量 |

<abbr title="peak allocated memory · 峰值已分配张量内存。包括已有模型和运行中的活跃张量；不同于缓存池 reserved、驱动看到的进程占用或纯 KV 增量。">peak allocated</abbr> 来自 reset_peak_memory_stats 与 max_memory_allocated；重置的是统计，不是释放模型。

跨运行可报中位数与范围。若用 <abbr title="P95 / 95th percentile · 第 95 百分位数，需声明最近秩或插值等算法。五个样本估计尾部很粗糙，不是置信区间。">P95</abbr>，要说明算法与样本数；五次重复不足以精确描述尾延迟。今天不推导分位数算法。

这些是从审计得出的检查项，实际覆盖程度应逐项对照 [D12 测量协议](../w2/d12_benchmark_schema.md)与[D13 重复测量](../w2/d13_repeated_benchmark.md)的脚本：例如 W2 选择严格校验 P 而不自动截短，并将提前终止记录单独保留。W2 的 CPU 随机模型主线无需 checkpoint；本日不改上游 benchmark，也不提前运行下一周实验。

## 7. 当天自验证与交流问题

正文已带读完整例子，下面换条件检查迁移理解。在聊天按题号回答即可，报告可选；命令、退出码与证据路径共用一份。若没做第 2 节教学实验，说明“未执行教学追踪”；其余可用纸面推演回答。历史证据注明日期与版本，不为答题重跑历史实验，GPU 不在本日实操或待补范围。

1. **预测生成器执行**：对 --tokens 1，预测创建后、首次 next 后、结束探测后各做了什么，再引用实际 JSON 解释预测与观察的关系。真实 Engine 的停止检查与 forward 各在哪里？

   **核对要点**：区分暂停与恢复、定位尾部 forward；教学标签不是模型证据。

2. **改变计时边界**：把 TTFT 终点移到第一次 next 之前会漏什么？保留 next 却删去之后的显式同步又有什么问题？若测用户提交文本到首 token，还需覆盖哪些当前没测的环节？

   **核对要点**：用生成器执行、异步与测量层级解释，不只背全称。

3. **复算统计**：假设 B=2，单次后续间隔为 [2,3,7,12] ms，按原源码求 TPOT 与 decode tokens/s。为什么不能当成四次独立 TTFT？若一行中途结束，分子又需什么额外证据？

   **核对要点**：索引、单位换算、样本层级和逐行计数；mask 不是完成状态。

4. **判断可比性**：同一模型 sequence_len=96、G=16，A 请求 P=88、B 请求 P=72，求实际 P，说明比较前要纠正什么条件或记录。若只汇总 step_times，又遗漏哪些阶段？

   **核对要点**：沿 clamp 推导，区分 requested/actual；指出首 token 与结束探测。

5. **安排修补优先级**：从第 6 节选两项优先要求，各说明保存什么、怎样判断满足、解决什么误判。至少指出一个必须注明的计时或计数定义变化。

   **核对要点**：有理由、有可检查判据，不只列字段名。

## 8. W1 回顾：把本周知识连成执行链

尝试口述：token ID → 前向与预测分数 → 提示缓存 → 缓存复制与增量解码 → 后端执行 → 逐列返回 → benchmark 计时统计。今天新增的是解释最后两段如何受实现影响。

不重答已通过的题：形状不清回 D4，缓存推进不清回 D5，后端和未覆盖分支不清回 D6，缺测试证据才补 D3。能沿执行顺序解释计时和计数即可完成本日学习；标准库实验可以补充控制流证据，不用于性能结论。W1 不要求训练、checkpoint 或 GPU 性能结果。

交流已理解内容、实际执行的 CPU 证据、具体疑问与下一周目标，注明日期、commit 和状态。W2 先查 tokenizer/数据，再理解训练与保存；训练和既有 checkpoint 重载是有明确条件的选择，之后可用 CPU 随机模型学习测量协议，不必先获得训练权重。

## 执行参考：脚本、产物与失败处理

操作信息集中于此，按需回查。路径相对项目根目录；上游保留原名，新增脚本/产物以首次使用日 dNN_ 开头，跨日复用不改名。

| 脚本 | 用途与阶段 | 完整输入路径或来源 | 条件 |
|---|---|---|---|
| [d07_trace_generator.py](scripts/d07_trace_generator.py) | 第 2 节观察执行顺序 | --tokens 与内部教学事件，无外部文件 | Python 标准库；不需 torch、GPU 或权重 |
| [scripts/infer_bench.py](../../../../scripts/infer_bench.py) | 第 3–6 节只读对象 | nanochat/engine.py、nanochat/checkpoint_manager.py、nanochat/gpt.py | 阅读只需文件；运行需 CUDA、单进程与匹配资产 |
| [d04_probe_model_and_cache.py](scripts/d04_probe_model_and_cache.py) | W1 可选补形状/缓存证据 | 内部随机权重；nanochat/gpt.py、nanochat/engine.py | 同 D4/D5，CPU 与项目依赖 |
| [d03_run_inference_tests.py](scripts/d03_run_inference_tests.py) | W1 可选补测试证据 | tests/test_engine.py、tests/test_attention_fallback.py | 同 D3，已有证据不重跑 |

| 完整项目相对路径 | 用途 | 生成者与条件 |
|---|---|---|
| `experiments/inference_infra/results/w1/d07_generator_trace.json` | 三列追踪，含解释器与逐次事件 | D7 脚本 --tokens 3 成功时自动写入/覆盖 |
| `experiments/inference_infra/results/w1/d07_generator_trace_g1.json` | 一列变式 | 同脚本 --tokens 1 并指定该 --output 路径时自动写入/覆盖 |
| `experiments/inference_infra/reports/d07_checks.log` | 命令组 stdout/stderr、打印的状态 | Bash 块 tee 自动追加，不自动记命令文本 |
| `experiments/inference_infra/reports/d07_benchmark_audit.md` | 可选整理聊天与结论 | 人工可选，不是验收前置 |
| `experiments/inference_infra/results/w1/d04_shapes_probe.json` | 可选形状补测 | D4 脚本 --mode shapes 断言通过后自动生成/覆盖 |
| `experiments/inference_infra/results/w1/d05_cache_probe.json` | 可选缓存补测 | D4 脚本 --mode cache 断言通过后自动生成/覆盖 |
| `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_stdout.log` | 可选测试 stdout | D3 捕获脚本运行时自动生成 |
| `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_stderr.log` | 可选测试 stderr，可为空 | 同上 |
| `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_run.json` | 可选测试退出码等元数据 | 同脚本子进程正常返回后自动生成 |

清单是运行后预期，不代表已存在。<abbr title="JSON / JavaScript Object Notation · 以键值、数组组织数据的文本格式；本日追踪文件保存事件顺序，不含性能数字。">JSON</abbr> 可打开核对；失败或未执行不把旧文件当本次成功。UTC 时间戳格式为 YYYYmmddTHHMMSSffffffZ，以 D3 实际打印目录为准。可选脚本完整用法见对应日期，只在缺证据时补测。

| 情况 | 处理 |
|---|---|
| 解释器不符合预期 | 回 D2 查环境，不在此自动切换 |
| --tokens 为零或负 | 应非零退出，改成正整数 |
| 输出写入失败 | 留 stderr、退出码，查 --output 目录和权限 |
| 事件与预测不同 | 沿 yield、递增、forward、顶部检查重新推演 |
| 无 CUDA/checkpoint | 按本日标准库实验与源码带读继续，无需补资源或列 GPU 待办 |
| 能定位但不能解释 | 回带读例子，不以文件清单代替理解 |

教学脚本只验证生成器控制流，静态审计只对应当前实现；它们不证明真实模型数值、GPU 性能或读者学习效果已验收。

上一篇：[D6 后端路由](d06_attention_backends.md)。下一阶段：[W2 D1 数据前置核查](../w2/d08_data_prerequisites.md)。
