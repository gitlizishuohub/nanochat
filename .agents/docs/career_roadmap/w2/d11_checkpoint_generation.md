---
source_files:
  - scripts/base_train.py
  - nanochat/checkpoint_manager.py
  - nanochat/gpt.py
  - nanochat/engine.py
  - nanochat/common.py
  - nanochat/tokenizer.py
  - .agents/docs/career_roadmap/w2/scripts/d10_train_smoke.py
  - .agents/docs/career_roadmap/w2/scripts/d10_checkpoint_probe.py
source_symbols:
  - save_checkpoint
  - load_checkpoint
  - build_model
  - load_model_from_dir
  - GPT.generate
  - Engine.generate
doc_type: tutorial
knowledge_scope: W2第4天，理解checkpoint重建、恢复训练与双路径生成检查
---

# W2 D4：理解 checkpoint，并核对两条生成路径

前一天：[返回前一天](d10_training_smoke.md)；后一天：[继续下一天](d12_benchmark_schema.md)；[原始总计划](../nanochat_daily_execution_2026-10-01.md)。

今天解决一个问题：**关闭 Python 后，怎样重新得到同一个模型？重新得到模型，是否就能接着原来的训练？** 先分清保存了什么，再理解加载和生成的代码。

本日的概念、例子和源码带读都不要求 GPU、模型权重或训练经历。[D10](d10_training_smoke.md) 没有执行训练也能学完。第 4 节是可选的 CPU 重载实验，只在已有配套 checkpoint 时执行；本页无需新增训练，也无需为了获得实验资格从头训练二十步。完成纸面推演后可直接进入 D12，未选实操无需补做。

## 当天的流程阅读

用约 10–20 分钟读指定小节，再回到下面的例子；只需口述理解，不新增手写报告。

- **阅读入口**：[4.3 从配置重建模型并加载参数](../../llm-training-workflow/04-model-construction-and-weight-loading.md#reading-4-3)；[10.2 文件职责与保存时点](../../llm-training-workflow/10-logging-checkpoints-and-resume.md#reading-10-2)；[10.3 真实恢复与跨阶段加载](../../llm-training-workflow/10-logging-checkpoints-and-resume.md#reading-10-3)。
- **今天重点**：模型配置、参数、优化器状态和数据读取状态分别解决什么问题；推理加载与继续训练为什么需要不同的东西。
- **暂时跳过**：不实际运行恢复训练，不展开分布式恢复和逐位复现。能够加载模型不等于这些行为已经验证。

**读完能解释**：给定模型参数文件，程序还需要知道什么，才能让一段文字通过模型？第 1 节先带你推导答案。

## 1. 从一个小例子分清：保存模型，保存了什么

### 1.1 配置决定容器，权重决定里面的数值

<abbr title="Checkpoint · 检查点。程序保存到文件的一组模型参数和相关状态；能否用于推理或恢复训练，取决于具体保存内容与加载方式。">checkpoint</abbr> 不是一个统一的魔法文件名。nanochat 把相关内容分成多个文件，职责如下。

| 内容 | 解决的问题 | 本项目的存放位置 |
|---|---|---|
| 模型配置 | 要创建几层、每层多宽、词表多大？ | `meta_<step>.json` 中的 `model_config` |
| 模型参数，也叫权重 | 已经学到的矩阵和向量分别是什么值？ | `model_<step>.pt` |
| 优化器状态 | 下一次更新参数时，如何延续之前积累的更新历史？ | `optim_<step>_rank0.pt`，此处为单进程例子 |
| 数据与训练循环状态 | 数据读到哪里、做了几步、损失统计累计到哪里？ | metadata 中的 `dataloader_state_dict`、`step`、`loop_state` 等 |
| 分词器 | 文字怎样变成 ID，输出 ID 又代表什么文字？ | 数据根下的 `tokenizer/tokenizer.pkl`，不在模型标签目录内 |

做一个**纸面例子，无需执行**：假设一个小模型词表只有 5 个 ID，每个 ID 用 4 个数表示，那么输入嵌入矩阵的形状就是 `5 × 4`。配置告诉程序需要这样一张表；参数文件给出这 20 个数的实际值。换成 `5 × 8` 的模型骨架，旧表便装不进去；重建同样形状但使用新随机数，也得不到原来那个模型。

因此，重新加载要按“读取配置 → 创建匹配的模型 → 装入权重”的顺序完成。**创建结构相同的模型，与恢复原来的参数，是两个动作。**

再设想两份分词器词表都只有 5 项，但 A 把 ID 2 解释为“水”，B 把 ID 2 解释为“火”。维度检查会通过，文字含义却已经换了。由此可以推导：词表大小相同是必要的兼容性检查，不能代替训练时使用的分词器身份记录。

### 1.2 能做推理，为什么还不等于恢复了训练

推理是使用参数预测下一个 ID，不执行参数更新。假设两个训练过程当前参数完全相同，但优化器积累的历史不同；下一批数据也可能不同。再进行一次更新，它们便可能走向不同的参数。

这里的<abbr title="Optimizer state · 优化器状态。除当前参数以外，为后续更新保留的历史量，例如动量、梯度的一阶/二阶统计及更新计数。具体内容由所用优化器决定。">优化器状态</abbr>解释了“只加载权重”遗漏什么；数据读取状态解释了“还要接上哪批数据”。训练调度所需的步数、总步数和相关配置也要一致。原生训练入口会读取相应状态，但不能仅凭这些字段存在，就声称已经验证逐位一致的恢复；当前保存调用没有额外保存随机数生成器或<abbr title="GradScaler · 梯度缩放器。本项目在 FP16 训练中使用它动态调整梯度缩放比例，减少低精度下的数值问题；其状态与模型参数、优化器状态是不同内容。">低精度梯度缩放器</abbr>状态，也需按实际训练条件判断复现边界。

对照 [训练包装器](scripts/d10_train_smoke.py)：它构造的是 `python -m scripts.base_train ... --num-iterations=...`，没有传 `--resume-from-step`。假设另用新标签调用它并指定二十步，这表示**额外从头训练二十步**，不是先做两步、再接着做十八步；这里只判断命令含义，无需执行。本日直接复用已有的两步 checkpoint 即可学习重载，二十步不会让加载机制更完整。

## 2. 带读源码：文件怎样重新变成模型

下面的片段摘取关键语句，省略日志和兼容分支；用于阅读，不需要粘贴执行。完整实现见 [checkpoint_manager.py](../../../../nanochat/checkpoint_manager.py)。

### 2.1 先看保存端，确定加载端能够拿到什么

在 [base_train.py](../../../../scripts/base_train.py) 搜索 `save_checkpoint(`，可以看到调用结构：

```python
save_checkpoint(
    checkpoint_dir,
    step,
    orig_model.state_dict(),
    optimizer.state_dict(),
    {
        "step": step,
        "model_config": model_config_kwargs,
        # 本教程省略其他 metadata 字段
    },
    rank=ddp_rank,
)
```

这里的 <abbr title="Metadata · 元数据。描述 checkpoint 的配置、保存步数、训练设置与恢复状态等信息；本项目用 JSON 存放它，模型参数数值另存为 PT 文件。">metadata</abbr> 字典只展示两个字段，完整调用还保存第 1 节列出的训练状态。`state_dict()` 是把对象当前需要保存的状态组织为字典；模型字典主要包含参数，优化器字典包含更新历史。随后 `save_checkpoint` 把它们写入不同文件：rank 0 写参数和 JSON metadata，各 rank 写自己的优化器分片。六位补零使 step 2 对应 `model_000002.pt`，不是 `model_2.pt`。

原生训练循环在运行末尾、或满足定期保存条件时调用这一段。**文件路径是保存规则，文件存在才是实际保存的证据。** 计划命令成功、目录创建成功都不说明权重已经产生。

### 2.2 加载时沿四个动作追踪

`load_model("base", ..., model_tag=..., step=...)` 最终进入 `build_model`。沿下面四个动作阅读，就能对应第 1 节的概念：

1. `load_checkpoint(..., load_optimizer=False)` 读取权重和 metadata。此处用于构建推理模型，因此不加载优化器；同一个目录里可以有优化器文件，但这次调用不使用它。
2. `GPTConfig(**meta_data["model_config"])` 与 `GPT(model_config)` 创建结构。`**` 将字典中的字段展开成配置参数，如 `n_layer=4`；不是自动从权重推测层数。
3. `to_empty` 分配设备存储，`init_weights()` 初始化包括旋转位置编码在内的状态，再执行 `load_state_dict(model_data, strict=True, assign=True)` 装入保存的参数。这里的初始化不意味着丢弃旧权重；下一句会装载它们。加载器也会处理旧参数名和部分旧配置兼容性。
4. `phase="eval"` 令模型进入评估模式，随后从当前数据根加载分词器，检查词表大小。它没有读取“训练时分词器哈希”来自动证明两者一一对应。

`strict=True` 会要求参数键与模型匹配，形状不匹配也会报错；它保护结构兼容性，不能判断分词器是否把同一个 ID 解释成另一段文字。CPU/MPS 加载还会把保存的 BF16（16 位浮点格式）参数转为 FP32（32 位浮点格式）；计算精度另由环境选择，所以复现实验时也要记录实际精度。

### 2.3 原生恢复训练为什么走另一条分支

再回到 `base_train.py`，依次搜索 `resuming`、`optimizer.load_state_dict`、`dataloader_resume_state_dict`：它使用 `--resume-from-step` 开启恢复，加载模型和该 rank 的优化器状态，把 metadata 中的数据状态交给 loader，并恢复 step 与循环统计。

这条调用链说明“恢复”包含哪些动作；它不等于本日需要实际运行这些动作。重新加载做生成时，优化器不参与 forward；继续训练时，只有 forward 所需的文件还不够。调度还依赖当前启动配置，不能随意改总步数或数据后宣称与未中断训练等价。

## 3. 同一个模型，为什么要比较两种生成方式

### 3.1 用三个输出 ID 走一遍执行过程

假设输入 ID 是 `[5, 8]`，模型依次选出 `[11, 12, 13]`，且没有工具调用或提前终止。这些数字是**控制流示意，不是当前分词器的真实编码，也不需要执行模型**。

| 要选哪个输出 | `GPT.generate` 本次送入 forward 的内容 | `Engine.generate` 本次送入 forward 的内容 |
|---|---|---|
| 11 | `[5, 8]` | `[5, 8]`，同时建立缓存 |
| 12 | `[5, 8, 11]` | `[11]`，结合前面的缓存 |
| 13 | `[5, 8, 11, 12]` | `[12]`，结合已经更新的缓存 |

[GPT.generate](../../../../nanochat/gpt.py) 的关键循环是 `logits = self.forward(ids)`，选出 ID 后拼到 `ids` 末尾；下一轮重算整个前缀。[Engine.generate](../../../../nanochat/engine.py) 先以 batch 1 处理提示、建立缓存，再复制到采样行；之后只输入新 ID。此处 probe 使用 `num_samples=1`，便于与无缓存路径对照。

<abbr title="KV Cache / Key-Value Cache · 保存先前位置的注意力键和值，供后续逐 token 推理复用。本项目还用 prev_embedding 保留跨调用的 smear 历史状态；缓存正确性不只取决于 K/V 张量。">KV Cache</abbr> 避免重算历史位置，但要求历史状态接续正确。两条路径若配置、参数、输入和计算条件一致，在相同选择规则下应当得到一致的输出；这是对“重载后模型与缓存路径能否一起工作”的快速检查。

### 3.2 阅读 probe 的一行比较，明确它到底检查什么

打开 [双路径检查脚本](scripts/d10_checkpoint_probe.py)，先读这三处：

```python
ids = tok.encode(a.prompt, prepend=tok.get_bos_token_id())
reference = list(model.generate(ids, max_tokens=a.max_tokens, temperature=0.0))
match = reference[: len(engine_ids)] == engine_ids
```

第一句给两条路径同一份输入 ID，包含序列起始标记 BOS；第二句用 <abbr title="Greedy decoding · 贪心生成。每一步选择当前预测分数最高的 token；此处用 temperature=0 选择 argmax，不进行随机采样。并列或近似并列时仍可能受数值差异影响。">greedy</abbr> 排除随机采样因素；第三句比较 Engine 实际输出与 reference 的**同长前缀**。

为什么不要求完整列表长度相同？`GPT.generate` 按指定上限输出，不处理 Engine 的终止规则；Engine 遇到 `<|assistant_end|>` 或 BOS 会标记完成。probe 使用流式 `Engine.generate`，把终止 ID 加入列表后才停止，因而比较**包括终止 ID**。例如假设终止 ID 为 99，reference 为 `[11, 99, 7, 8]`，Engine 为 `[11, 99]`，比较 `[11, 99] == [11, 99]` 才符合脚本语义。不要套用 `generate_batch` 会去掉终止 ID 的行为。

还需注意：Engine 遇到配对的 Python 工具标记时，可能把计算器结果放入 `forced_tokens` 队列，输出掩码 0 表示强制注入、1 表示模型采样。probe 收到 `masks` 后没有写入报告，也没有排除工具路径；普通文本提示不保证模型绝不会生成工具标记。出现差异时先查这类控制流，不能直接断言缓存算错。

### 3.3 相同 token 为什么不证明 logits 与缓存完全相同

<abbr title="Logits · 模型对各候选 token 输出的未归一化预测分数。贪心生成只取最大分数对应的 ID，不保留整组分数作为一致性证据。">logits</abbr> 为 `[2.0, 1.0]` 和 `[20.0, 19.9]` 时，贪心规则都选择 ID 0，但两组数值显然不同。由此可以推出：`common_prefix_match=true` 证明本次输入下这两组已观测 token 前缀一致，不能证明每步 logits、所有缓存状态或所有提示都正确。

若第 4 个输出不同，先固定同一权重、配置、分词器、输入 ID、精度及后端，排除终止和工具注入差异，再检查**前三个相同输出形成的共同前缀**之后那一步 logits。直接比较已经分叉后很远的输出，混入了不同输入造成的影响。本日理解这一定位思路即可，无需新增完整数值测试。

## 4. 可选 CPU 实操：已有 D10 checkpoint 时重新加载

没有 checkpoint 时，完成前面例子与文末问题即可；本节标为“未选实操”，不标学习欠账。选择实操时只进行加载和生成，**不运行训练、不更新参数、不要求 GPU**。已有 D10 probe 结果也可复用，说明来源即可，不必为了新日志重跑。

### 4.1 核对已有文件和环境

本例复用 D10 的 `d10_w2_smoke`、step 2；若 D10 使用其他标签或数据根，改为那一次的真实值，不能指向另一份刚好同名的文件。`NANOCHAT_BASE_DIR` 未设置时原生默认根为 `/home/lizishuo/.cache/nanochat`；D10 合成分支若使用 `/tmp/nanochat-w2-toy`，本节应继续使用它，临时目录被清理后就没有那些文件。

终端会自动激活 Conda 的 `nanochat` 环境，无需再次激活。以下先查看文件，不自动创建 tokenizer 或下载权重。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/lizishuo/.cache/nanochat}"
export NANOCHAT_DTYPE=float32
python -c "import sys; print(sys.executable); print(sys.version)"
printf 'base_dir=%s\n' "$NANOCHAT_BASE_DIR"
ls -l "$NANOCHAT_BASE_DIR/base_checkpoints/d10_w2_smoke/model_000002.pt" "$NANOCHAT_BASE_DIR/base_checkpoints/d10_w2_smoke/meta_000002.json" "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
} > >(tee -a experiments/inference_infra/reports/d11_session.log) 2>&1
```

解释器应为 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`；三个文件应都存在。缺文件就停在这里，改用纸面学习；不需要补训。`token_bytes.pt` 是训练/评估的配套产物，本次 probe 不读取它。局部 <abbr title="tee · 把收到的输出同时写到终端与文件。本页仅在花括号命令组内重定向，命令组结束后恢复正常终端输出。">tee</abbr> 保存命令输出，避免把整个交互终端的提示符和控制字符长期写入日志。各块是追加写入，重跑时注明证据属于哪次运行。

### 4.2 先预测，再读实际 JSON

运行前先预测：在两条路径都没有提前终止时，`reference_ids` 与 `engine_ids` 应各最多有 8 个 ID；提前终止时 Engine 列表可能更短。P 是实际编码后的输入 ID 数，含 BOS，必须满足 `P + 8 ≤ model.config.sequence_len`，不能直接拿字符数代入。

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d10_checkpoint_probe.py --device cpu --model-tag d10_w2_smoke --step 2 --prompt 'The chemical formula of water is' --max-tokens 8 --output experiments/inference_infra/reports/d11_checkpoint_generation_2.json
probe_rc=$?
printf 'probe_exit=%s\n' "$probe_rc"
(exit "$probe_rc")
} > >(tee -a experiments/inference_infra/reports/d11_session.log) 2>&1
```

脚本只在 `status="ok"` 时退出 0；`mismatch` 或加载错误返回 1，也写 JSON。按下面顺序解释结果，别只看一句输出文本：

1. 看 `status` 与 `error`：是加载失败，还是加载后序列不同？`status="ok"` 是本次比较成功，不是模型质量合格。
2. 看 `model_config`：确认配置确由保存的 metadata 重建，特别是层数、宽度、词表和上下文长度，而不是自己临时猜出一组配置。
3. 看 `prompt_ids` 和两组生成 ID：它们是否符合刚才的长度预测？若有终止 ID，按第 3.2 节解释截断。
4. 看 `common_prefix_match`：用 ID 逐项核对这个布尔值。两步训练可能生成无意义文本；这与本次加载及前缀检查是不同问题。

### 4.3 固定身份：为什么只写 tag 和 step 还不够

`load_model_from_dir` 没传标签时会猜测模型标签；没传 step 时会找最高编号的模型文件。本例显式指定两者，避免增加新文件后悄悄换模型。但标签只是目录名，目录内容仍能被替换，所以还要识别文件内容。

下面读取 metadata，并保存三份文件的 <abbr title="SHA-256 · 根据文件字节计算的内容摘要，用来核对是否为同一份文件。摘要识别内容，但不能单独证明文件的训练来源或它与某次训练的配套关系。">SHA-256</abbr>；仅在第 4.1 节确认输入齐全后执行：

```bash
{
python -m json.tool "$NANOCHAT_BASE_DIR/base_checkpoints/d10_w2_smoke/meta_000002.json" > experiments/inference_infra/reports/d11_checkpoint_identity.log 2>&1
meta_rc=$?
printf 'metadata_exit=%s\n' "$meta_rc"
sha256sum "$NANOCHAT_BASE_DIR/base_checkpoints/d10_w2_smoke/model_000002.pt" "$NANOCHAT_BASE_DIR/base_checkpoints/d10_w2_smoke/meta_000002.json" "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" >> experiments/inference_infra/reports/d11_checkpoint_identity.log 2>&1
hash_rc=$?
printf 'hash_exit=%s\n' "$hash_rc"
cat experiments/inference_infra/reports/d11_checkpoint_identity.log
} > >(tee -a experiments/inference_infra/reports/d11_session.log) 2>&1
```

把真实绝对数据根、tag/step、三项 hash 与 D10 的来源记录关联：使用的是合成文本短跑，还是原语料短跑？现在计算的 tokenizer hash 只识别**当前文件**，若没有训练时的配套记录，不能倒推它必定就是训练那一份。metadata 中保存了配置和训练状态，不包含可自动证明这层关联的 tokenizer hash。

## 5. 遇到问题时，按失败阶段判断

| 现象 | 已知道什么 | 下一步 |
|---|---|---|
| 文件不存在 | 尚未完成加载 | 核对真实数据根、标签和六位 step；无文件就采用纸面学习 |
| 参数键/形状报错 | 配置与参数不能正常装配 | 核对 metadata 是否配错；不要随意修改层数来迎合权重 |
| 词表大小报错 | tokenizer 与模型的基本维度不兼容 | 找回原配套 tokenizer，不能改 metadata 硬凑 |
| `status="mismatch"` | 本次同长前缀比较不同 | 保留 JSON、首个分歧位置，按第 3.3 节分清输入、控制流和数值条件 |
| 输出不通顺但前缀一致 | 本次路径检查通过，语言质量尚不能由此判断 | 两步 smoke 本就不验证知识和语言能力；不要为提高本题成绩追加训练 |

GPU 是后续自行选择的扩展条件，不是本日必须补齐的事项；本节也没有测量速度。

## 6. 今日脚本与产物清单（执行时查阅）

所有项目相对路径都以 `/home/lizishuo/_program_/Project/nanochat` 为根。本日只改学习方式，不改变原生 checkpoint 格式。清单中的文件只在对应命令实际执行后产生；纯阅读与推演不要求产生任何文件。

| 脚本完整项目路径 | 用途与阶段 | 完整输入路径/条件 |
|---|---|---|
| `.agents/docs/career_roadmap/w2/scripts/d10_checkpoint_probe.py` | 第 4.2 节可选 CPU 重载和双路径检查 | `$NANOCHAT_BASE_DIR/base_checkpoints/d10_w2_smoke/model_000002.pt`、`$NANOCHAT_BASE_DIR/base_checkpoints/d10_w2_smoke/meta_000002.json`、`$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl`；配套文件已存在且 `P+G` 不超上下文 |
| `.agents/docs/career_roadmap/w2/scripts/d10_train_smoke.py` | 第 1.2 节仅阅读其命令构造；本日不执行 | 阅读源码不需要数据；其 D10 训练输入和生成条件见 [D10](d10_training_smoke.md) |

| 产物完整项目相对路径 | 用途 | 生成条件 |
|---|---|---|
| `experiments/inference_infra/reports/d11_session.log` | 本日所选操作输出与 probe 退出码 | 各 Bash 命令组局部 tee 自动追加；不含组外终端操作 |
| `experiments/inference_infra/reports/d11_checkpoint_generation_2.json` | 双路径 ID、配置和检查结果 | 执行 probe 后自动保存；失败时也含错误状态 |
| `experiments/inference_infra/reports/d11_checkpoint_identity.log` | metadata 与三项文件 hash | 第 4.3 节重定向保存；失败也会记录错误，须看退出码 |
| `experiments/inference_infra/reports/d11_checkpoint_manifest.md` | 可选整理身份与学习结论 | 人工选写；直接在聊天回答即可，无需为完成本日创建 |

以下是**已有 D10 训练曾成功保存时**可复用的输入，D11 不会生成或更新它们。以合成分支数据根 `/tmp/nanochat-w2-toy` 为例；若实际使用默认根，则对应前缀是 `/home/lizishuo/.cache/nanochat`：

| 完整绝对路径（示例） | 用途与来源 |
|---|---|
| `/tmp/nanochat-w2-toy/base_checkpoints/d10_w2_smoke/model_000002.pt` | 参数；D10 包装器实际训练后由 `scripts/base_train.py` → `save_checkpoint` 保存 |
| `/tmp/nanochat-w2-toy/base_checkpoints/d10_w2_smoke/meta_000002.json` | 配置、step 和状态；同一保存调用 |
| `/tmp/nanochat-w2-toy/base_checkpoints/d10_w2_smoke/optim_000002_rank0.pt` | 单进程优化器状态；训练保存产生，推理 probe 不读取 |
| `/tmp/nanochat-w2-toy/tokenizer/tokenizer.pkl` | D09 等分词准备步骤已产生且训练时配套使用的 tokenizer；本日只读取 |

权重和 tokenizer 留在原数据根，不提交 Git；只分享自己的轻量日志时也应检查私人路径。跨日脚本保留首次使用编号；已有 `d11_w2_20` 历史产物无需改名，本页的 step 2 例子不代表历史二十步已运行。

D12 的 `d12_benchmark.py` 默认标签仍是 `d11_w2_20`，这只是脚本默认值，不代表你必然有这份权重。无 checkpoint 时按 [D12](d12_benchmark_schema.md) 的 CPU 随机模型主线学习；若选择复用本例的真实 checkpoint，需显式传 `--model-tag d10_w2_smoke --step 2`，并保持同一数据根。

## 7. 今日自验证与交流问题

可以在聊天中逐题简答，不要求另写报告。未选实操时用上面的假设例子解释；选了实操时一次说明实际 tag/step、来源、CPU/精度与证据路径，再引用必要字段即可。可选实验未执行无需标欠账，不能把假设数列写成实测结果。

1. 只有 `model_000002.pt`，为什么还不能直接把文字交给模型？结合 `5×4` 的例子，解释配置和 tokenizer 各补上哪一部分。若换成同样大小的词表，什么检查可能通过、什么含义仍可能错？

   **核对要点**：区分结构、参数和 ID 含义，说明严格加载及词表大小检查的能力边界。

2. 一份 checkpoint 能成功用于推理，为什么不等于恢复了原训练？从保存与加载的代码中选出优化器、数据进度和训练循环各一项，再判断“用新标签调用包装器、指定二十步但不传 resume”的含义。

   **核对要点**：沿真实调用说明状态用途、resume 参数与启动配置，不要求运行恢复训练。

3. 假设 reference 为 `[11,99,7,8]`，Engine 为 `[11,99]`，99 是终止 ID。probe 应怎样比较，结果说明了什么？若你做了实操，再用实际 JSON 核对自己的判断。

   **核对要点**：按 Engine 长度截取 reference，保留终止 ID；区分 token 前缀一致、logits 一致和模型语言质量。

4. 下周同一 tag/step 的结果突然变了，你会用哪些已有证据先判断是否加载了同一组文件？若文件身份相同但第 4 个输出首次分歧，接下来应检查哪一步输入和哪类控制流？

   **核对要点**：真实数据根、显式 tag/step 与三项 hash；hash 不能代替训练来源记录；在共同前缀处核对精度/后端、终止和工具强制输出，再考虑 logits。

**完成本日的判断**：能解释文件职责、跟随加载链、手推两个生成路径并说清比较边界，就已完成学习主线。已有配套文件时选择 CPU 实操，可再得到一次真实重载证据；不要求训练、GPU 性能结果或新增手写报告。
