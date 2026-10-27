---
source_files:
  - nanochat/tokenizer.py
  - nanochat/dataset.py
  - nanochat/dataloader.py
  - scripts/tok_train.py
  - scripts/tok_eval.py
  - scripts/base_train.py
  - nanochat/gpt.py
  - nanochat/flash_attention.py
  - .agents/docs/career_roadmap/w2/scripts/d09_prepare_toy.py
  - .agents/docs/career_roadmap/w2/scripts/d09_roundtrip.py
source_symbols:
  - RustBPETokenizer.encode
  - _document_batches
  - tokenizing_distributed_data_loader_with_state_bos_bestfit
  - CausalSelfAttention.forward
  - GPT.forward
doc_type: tutorial
knowledge_scope: W2 D2 从文本往返与装箱推导输入目标及 CPU 批次检查
---

# W2 D2：从一段文字，推导出模型要预测的每一个目标

模型吃的是整数 ID，不是汉字；训练时交给它的也不是一句话，而是“输入这一串 ID，预测下一串 ID”。今天沿着 **文字 → tokenizer → 文档装箱 → x/y 批次** 走一遍，学会判断数据是否忠实保留原文，以及监督目标有没有错位。

**本日主线在 CPU 上完成，不需要 GPU、模型权重或已经训练过的 GPT。** 第 1–3 节是概念推演和源码带读，无需执行；第 4 节才是实际命令。没有现成 tokenizer 时，准备脚本会在 CPU 上从少量合成文本学习一份小词表，这是“训练分词器”，不会创建 GPT、反向传播或更新语言模型参数。本日没有 GPU 补做要求。

前一天：[返回前一天](d08_data_prerequisites.md)；后一天：[继续下一天](d10_training_smoke.md)；[原始总计划](../nanochat_daily_execution_2026-10-01.md)。建议概念与源码约 35 分钟，CPU 检查约 20–30 分钟，交流约 15 分钟；时间是建议，不是完成记录。

## 当天的流程阅读

用约 10–20 分钟读下面指定的小节，再回到本文；阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[5.1 预训练：装满 T+1 然后 shift](../../llm-training-workflow/05-data-encoding-and-batching.md#reading-5-1)；[5.3 batch、epoch 和每卡分配](../../llm-training-workflow/05-data-encoding-and-batching.md#reading-5-3)。
- **今天重点**：手工形成 x/y；理解一行数据可能装有多篇文档，因此“批次有几行”不等于“有几个句子”。
- **暂时跳过**：5.2 的对话装箱留到 D15；多卡通信实现留到 D14。本日只解释理解批次所需的单位。

带着一个问题阅读：原始行为什么要比输入多一个 token？第 2 节会从具体数值推导答案。

## 1. 概念推演：往返成功到底证明了什么

### 1.1 字符、字节、token 是三种计数

<abbr title="tokenizer · 分词器。依据固定词表和编码规则，把文字转换为整数 ID，并把 ID 序列解码回文字。本文使用 RustBPETokenizer，词表由 rustbpe 学习，实际编码由 tiktoken 执行。">tokenizer</abbr> 可以把常见片段合成一个 <abbr title="token · 词元。对应词表内一个整数编号，可能表示一个字符、多个字符或一个字符的部分 UTF-8 字节。">token</abbr>。例如字符串 `A中` 有 2 个 Python 字符，UTF-8 编码为 4 字节：`41 E4 B8 AD`。具体分成几个 token 取决于词表：不能仅凭“两个字符”推算 token 数，也不能把这里的字节值当作本项目实际 token ID。

假设一个教学词表把它编码为 `[17, 23, 24]`，三个 ID 分别表示字节片段 `41`、`E4 B8`、`AD`。单独解码后两个片段可能看不到完整的“中”，合起来才恢复完整字符。这里的 ID 和切分是**假设例子**，实际结果必须看你的 tokenizer。

因此 <abbr title="round-trip · 往返验证。整段原文 encode 成 ID 序列，再 decode 成字符串，要求与原文逐字符相等；不要求每个 token 单独解码为完整字符。">round-trip</abbr> 要检查的是：

```python
ids = tok.encode(text)
assert tok.decode(ids) == text
```

它排除这组输入上的文字丢失、替换或空白变化；它不证明切分效率高、词表适合整个语料，也不证明模型学会了语言。比如 Python 字符串 `"\n"` 是一个换行，而 `"\\n"` 是反斜杠和 n 两个普通字符；两者不相等，末尾两个空格也不能在比较前自行 `.strip()` 掉。

### 1.2 BOS 是额外边界，不是原文里的字

<abbr title="BOS / Beginning of Sequence · 序列起点标记。nanochat 将它显式加在每篇文档前面；实际 ID 必须通过 tokenizer 获取。">BOS</abbr> 让后续代码能辨认文档开始。以空字符串为例：普通编码得到 `[]`；显式添加 BOS 后得到 `[bos_id]`。标记是我们额外放进去的，比较原文时应剥去它，而不是要求带标记的全文解码等于空字符串。

[nanochat/tokenizer.py](../../../../nanochat/tokenizer.py) 的 `RustBPETokenizer.encode` 中，这几句足以解释行为（原句摘录，省略无关分支）：

```python
ids = self.enc.encode_ordinary(text)
if prepend is not None:
    ids.insert(0, prepend_id)
```

先编码普通文字，再把指定的特殊 ID 插入第 0 位。`get_bos_token_id()` 返回当前词表的 BOS；更换词表后整数可能改变，不能硬编码。批量输入走 `encode_ordinary_batch`，再对每一行插入 BOS。检查脚本会比较“批量编码”与“逐条编码”的结果，避免只验证了其中一条接口。

## 2. 概念推演：T+1 个位置怎样变成 T 个预测任务

### 2.1 把一个具体行展开

下面的数字只表示假设词表中的 ID，不需要加载模型。设 T=5，一行原始数据为：

```text
原始 row = [BOS, 11, 12, 13, 14, 15]  # 6 个位置
输入 x   = [BOS, 11, 12, 13, 14]      # 前 5 个
目标 y   = [ 11, 12, 13, 14, 15]      # 后 5 个
```

| 列（从 0 开始） | 当前输入 x | 该位置应预测的目标 y |
|---|---|---|
| 0 | BOS | 11 |
| 1 | 11 | 12 |
| 2 | 12 | 13 |
| 3 | 13 | 14 |
| 4 | 14 | 15 |

所以输入 13 的位置要预测 14。最后的 15 不进入 x，却为最后一列提供目标；少准备它，最后一列就没有“下一个 token”可用。这正是 T+1 的原因。对齐后应满足 `x[1:] == y[:-1]`；若直接令 `y=x`，任务就变成抄写当前 token。

B 表示批次的行数，T 表示每行输入长度；B=2、T=5 时，x/y 各为 `[2,5]`，共有 10 个预测位置。原始行总共要放 12 个 ID，模型训练的目标位置仍是 10 个。全局批量还要计入进程数和梯度累积次数；例如 1 个进程、累积 3 次上述批次，就对应一次更新中的 30 个目标 token，而不是“30 句话”。这里仅计算单位，不要求实际训练。

### 2.2 y 已在内存里，模型会偷看答案吗

这要区分两种 <abbr title="mask · 掩码。用于选择哪些位置可见或参与计算。注意力掩码限制信息流，损失掩码选择计入训练误差的位置，两者职责不同。">mask</abbr>：

- **注意力掩码**：模型当前位置只能看见当前及之前的输入，不能看后面的答案。实际 `CausalSelfAttention.forward` 使用 `causal=True`；如果配置了滑动窗口，还会限制可见的历史范围。假设窗口覆盖全部历史，对上表第 2 列而言，可见的是 `[BOS,11,12]`，不是后面的 13、14。
- **损失掩码**：选择哪些目标参与误差计算。`GPT.forward` 的交叉熵使用 `ignore_index=-1`，目标等于 -1 时才忽略。今天的预训练 loader 用合法词表 ID 填满目标，没有 padding，也不生成 -1 或单独的 mask 张量；所以各个目标位置都参与误差计算。

[模型源码](../../../../nanochat/gpt.py) 中的关键原句是：

```python
y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
# 以下位于 GPT.forward 的损失分支，省略中间计算。
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
```

这里不要求读懂注意力算法或运行损失。你只需分清“看不见未来输入”和“哪些目标不计误差”是两件事。D15 对话训练中的助手输出选择属于后者，不要套用到今天的预训练 loader。

## 3. 真实源码带读：文本怎样装满一行

### 3.1 文件先决定 train 与 val

[nanochat/dataset.py](../../../../nanochat/dataset.py) 的 `list_parquet_files()` 返回排序后的 parquet 路径。数据 loader 的 `_document_batches` 再执行：

```python
parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
```

最后一个文件属于 val，其余属于 train。故只有一个文件会让 train 为空，不是“它会自动从一份文件划分训练/验证”。这个迭代器会反复遍历数据；若 train 根本没有文件，调用者等不到 yield，会一直没有批次。CPU 检查脚本因此先要求至少两份文件，并检查第一 row group 的 `text` 列。

### 3.2 先加 BOS，再选择能放入的文档

[nanochat/dataloader.py](../../../../nanochat/dataloader.py) 的 `tokenizing_distributed_data_loader_with_state_bos_bestfit` 先设置 `row_capacity = T + 1`。`refill_buffer` 用这句把一组原文变成带 BOS 的文档 ID 列表：

```python
token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
```

然后 <abbr title="Best-Fit Cropping · 尽量选择能完整放入剩余位置的最长文档；都放不下时，裁切缓冲区中最短文档的前缀填满。本文长度均包含显式 BOS。">best-fit</abbr> 在缓冲区找满足条件的最长文档：

```python
if doc_len <= remaining and doc_len > best_len:
    best_idx = i
    best_len = doc_len
```

例如还剩 5 个位置，候选长度为 3、5、8，就选择长度 5；不是取最先到来的长度 3。若还剩 2 个位置，候选最短也有 3 个，就走另一分支：

```python
shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
doc = doc_buffer.pop(shortest_idx)
row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
```

`pop` 取走整篇文档，只使用前 2 个 ID；剩余 1 个没有放回缓冲区。这解释了“没有 padding，位置全满”为什么仍可能丢弃数据。源码注释中的裁切比例依赖语料、长度与配置，不能当作你今天数据的实测值。

一行也可能装下多篇文档，例如 `[BOS,11,12,BOS,21,22]`。切片后目标里会出现中间的 BOS；它仍是合法预测目标。**BOS 提供边界标记，不会自行构造跨文档注意力隔离**；今天的 loader 也没有返回一张按文档分块的注意力 mask。

### 3.3 最后才把原始行切为 x/y

装满原始行后，真实代码执行：

```python
cpu_inputs.copy_(row_buffer[:, :-1])
cpu_targets.copy_(row_buffer[:, 1:])
```

这就是第 2 节的“取前 T 个”和“取后 T 个”。函数随后将缓冲区复制到指定设备并 yield；包装函数 `tokenizing_distributed_data_loader_bos_bestfit` 省去恢复状态，只返回 x/y。虽然内部变量叫 `gpu_buffer`，指定 `device="cpu"` 时它实际也是 CPU 张量，名字不意味着要求 GPU。

这些缓冲区会在下一轮复用；如需长期保存某一批用于逐元素比较，应先 clone，而不是只保留原张量引用。本日检查脚本在下一次迭代之前完成断言，并把 `first_ids` 转成普通列表保存。

## 4. CPU 实际操作：把推演对应到真实结果

### 4.1 确认解释器与日志位置

所有命令从仓库根目录执行。终端已自动激活 nanochat，不重复激活。本日处理的 ID 张量是整数类型 `torch.long`，无需选择模型浮点精度。

日志只记录 `{ ...; }` 命令组内部；Bash <abbr title="process substitution · 进程替换。这里把局部命令组的输出交给 tee，显示并追加日志；命令组结束后不再接管交互终端。">进程替换</abbr>不会持续记录提示符。复制完整花括号和末尾重定向；若旧终端曾启用全局重定向，先新开终端。每条关键命令后立即打印退出码，成功标准同时看状态、断言与退出码。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
python -c "import sys; print(sys.executable); print(sys.version)"
printf 'interpreter exit=%s\n' "$?"
} > >(tee -a experiments/inference_infra/reports/d09_session.log) 2>&1
```

预期解释器为 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。路径或依赖不符时先记录具体错误，不用系统 Python 替换后声称同一环境已通过。本页命令、帮助与检查的打印内容都保存在 `experiments/inference_infra/reports/d09_session.log`；日志追加写入，重跑时注明会话边界。

### 4.2 二选一：复用已有数据，或生成学习用的小数据

先看 [D08 数据盘点](d08_data_prerequisites.md) 的结论。

**已有数据分支**：设置 `export NANOCHAT_BASE_DIR=你已核验的实际绝对路径`，直接进入 4.3。只使用自己生成或可信来源的 tokenizer；其中 `tokenizer.pkl` 的加载涉及 pickle，不从未知来源复制。

**没有数据分支**：使用[隔离合成数据脚本](scripts/d09_prepare_toy.py)，不下载文件、不加载 GPT。它在 CPU 上为 1,200 条合成 train 文本学习小型 <abbr title="BPE / Byte Pair Encoding · 字节对编码。此实现从字节片段学习合并规则，用固定词表将后续文字编码成 ID；学习词表不涉及 GPT 参数。">BPE</abbr> 词表（请求词表大小 512），再生成 1,200 条不同模板的 val 文本及字节长度表。BPE 在这里学习哪些字节片段合并为一个 token；这就是命令内部的 `train_from_iterator`，不是语言模型训练。train/val 模板重复，只用于检查数据管线，不能证明语言质量。

`--base-dir` 必须是尚不存在的全新目录。示例目录已存在就换一个新后缀，并同步修改下面 export；不要覆盖或删除旧数据。

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d09_prepare_toy.py --base-dir /tmp/nanochat-w2-toy
prepare_status=$?
printf 'prepare exit=%s\n' "$prepare_status"
if [ "$prepare_status" -eq 0 ]; then
  export NANOCHAT_BASE_DIR=/tmp/nanochat-w2-toy
  python .agents/docs/career_roadmap/w2/scripts/d08_data_check.py --output experiments/inference_infra/reports/d09_toy_inventory.json
  printf 'inventory exit=%s\n' "$?"
fi
} > >(tee -a experiments/inference_infra/reports/d09_session.log) 2>&1
```

只有 prepare 成功才设置数据根并盘点；失败后先排查，不继续 4.3。`/tmp` 可能被系统清理；要长期保存可改用有权限的大数据盘。交流时给实际绝对路径，不能只写环境变量名。未设置变量时当前用户默认根是 `/home/lizishuo/.cache/nanochat`；本日无需往该默认根下载数据。当前数据目录是 `$NANOCHAT_BASE_DIR/base_data_climbmix/`，不存在时源码会回退到旧的 `$NANOCHAT_BASE_DIR/base_data/`，已有数据分支要记录实际使用的目录。

### 4.3 先检查文字往返

[真实 tokenizer 检查脚本](scripts/d09_roundtrip.py) 会加载选定根目录下的 tokenizer。运行前预测：空字符串会有几个普通 token？中英文混合或 emoji 的每个 token 都必须是完整字符吗？

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d09_roundtrip.py --output experiments/inference_infra/reports/d09_token_roundtrip.json
printf 'tokenizer check exit=%s\n' "$?"
} > >(tee -a experiments/inference_infra/reports/d09_session.log) 2>&1
```

打开 JSON，按以下顺序解释结果：

1. `status=ok`、退出码为 0：所有内置断言通过，而不是“运行到了打印 JSON”。脚本捕获异常时同样会写 JSON，但状态为 error、退出码为 1。
2. `cases` 有六组：英文、中英混合、换行与空白、数字、emoji/重音字符、空字符串。每组 `passed=true` 表示整段 encode→decode 相等；观察实际 `token_ids`，不与第 1 节假设 ID 比较。
3. 脚本还检查带 BOS 时先去掉第一个 ID 再还原文字、批量与逐条编码一致、九个特殊 ID 互不冲突。
4. `token_bytes_shape` 应覆盖实际词表；特殊 ID 的字节计数为零。这张表用于后续按原文字节计算指标；本日检查没有把所有普通 token 的字节长度逐一重算，不能宣称已证明整张表内容正确。

### 4.4 再检查真实 loader 的 CPU 批次

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d09_roundtrip.py --check-data --output experiments/inference_infra/reports/d09_token_data_roundtrip.json
printf 'loader check exit=%s\n' "$?"
} > >(tee -a experiments/inference_infra/reports/d09_session.log) 2>&1
```

这次复用全部往返断言，还用真实 loader 为 train 与 val 各取一个 `[1,32]` CPU 批次。训练入口默认 `buffer_size=1000`；教学检查显式用 4，减少装填等待。它没有调用模型前向、反向或优化器。

正常结果应新增 `train_batch`、`val_batch`，各有 `shape=[1,32]`、`first_ids`、`passed=true`。`first_ids` 只保存输入前八个 ID；第一个应与 JSON 的 BOS 特殊 ID 相同。`passed=true` 还代表脚本内部确认 x/y 为整数、`x[:,1:]==y[:,:-1]`、两者全部 ID 在词表范围。完整 y 没写进 JSON，因此解释错位时引用断言代码和第 2 节推导，不要声称 JSON 已逐项展示 y。

只通过文字往返、但没有两个 split 的 batch 证据，不能声称真实 loader 已通过。两个批次都通过也只证明这次取样的格式与对齐，不覆盖全语料空行、重复、泄漏或语言质量；后续风险应按各自证据判断。

## 5. 失败时怎样定位，而不是盲目重跑

| 现象 | 先解释原因，再决定动作 |
|---|---|
| 缺 rustbpe/tiktoken/pyarrow/torch | 对照解释器与缺包名称；记下未执行范围，修复既有 Conda 环境后再试，不换假 tokenizer 冒充真实检查 |
| prepare 提示路径已存在 | 这是保护旧数据；换全新路径并同步 export，或确认旧目录确实是要复用的可信数据 |
| 找不到 tokenizer 或 token_bytes | 核对已展开的数据根与两个文件；token_bytes 必须属于同一份词表，不复制另一词表的同名文件拼凑 |
| status=error 但有 JSON | 看 `error` 的异常类型和退出码；JSON 存在本身不代表成功 |
| loader 没有输出 | 先确认至少两份 parquet、train 非空、row group/text 有效；不要无限等待空迭代器。检查脚本可提前拦下部分格式问题，不保证所有坏数据都不会卡住 |

缺依赖或输入时，可以先完成概念与源码部分，把未执行的 **CPU 检查**注明原因。GPU 不在本日必做或待补列表中；无需为完成今天租 GPU 或训练语言模型。下一天的实际训练操作只在你选择执行并具备条件时开展。

## 今日脚本与产物清单

每日编号使用全程 d01–d21；跨日复用保留首次实际执行日。以下项目相对路径均以仓库根目录为基准，脚本清单直接列出本日执行入口。

| 脚本完整项目路径 | 用途 | 执行阶段 | 输入 | 运行条件 |
|---|---|---|---|---|
| `.agents/docs/career_roadmap/w2/scripts/d09_prepare_toy.py` | CPU 学习小型 BPE 词表并构造隔离合成数据 | 4.2 无现成数据分支 | 脚本内合成模板；--base-dir 指定全新绝对路径 | 仅无可用数据分支；全新目录；CPU 依赖齐全，不需要 GPT 权重 |
| `.agents/docs/career_roadmap/w2/scripts/d08_data_check.py` | 复用前一天的库存检查 | 准备后 | `/tmp/nanochat-w2-toy/tokenizer/`、`/tmp/nanochat-w2-toy/base_data_climbmix/` | 目录创建后 |
| `.agents/docs/career_roadmap/w2/scripts/d09_roundtrip.py` | 验证文本往返与 loader 批次 | 4.3、4.4 | `$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl`、`$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt`、`$NANOCHAT_BASE_DIR/base_data_climbmix/*.parquet` | 可信 tokenizer；--check-data 另要求非空 train/val |

| 产物完整项目相对路径 | 用途 | 生成者 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/reports/d09_toy_inventory.json` | 合成目录盘点 | d08_data_check.py | 条件自动；仅合成分支 |
| `experiments/inference_infra/reports/d09_token_roundtrip.json` | 六组 tokenizer 断言 | d09_roundtrip.py | 自动；成功或捕获的异常均写 JSON |
| `experiments/inference_infra/reports/d09_token_data_roundtrip.json` | train/val 批次断言 | d09_roundtrip.py --check-data | 条件自动；需执行数据检查 |
| `experiments/inference_infra/reports/d09_token_roundtrip.md` | 往返与错位示意说明 | 聊天交流后 | 可选人工整理；可直接在聊天中回答本日问题 |
| `experiments/inference_infra/reports/d09_session.log` | 当天终端输出与手动记录的退出码 | 各 Bash 代码块的 tee | 自动；仅追加各代码块内的输出 |

合成分支的外部数据根示例为 `/tmp/nanochat-w2-toy`（仓库外，须是全新目录）；脚本条件自动生成下列文件，已有数据分支不生成它们：

| 完整绝对路径（示例数据根） | 用途 | 生成者与方式 |
|---|---|---|
| `/tmp/nanochat-w2-toy/base_data_climbmix/d09_shard_00000.parquet` | 合成 train | d09_prepare_toy.py；条件自动 |
| `/tmp/nanochat-w2-toy/base_data_climbmix/d09_shard_00001.parquet` | 合成 val | d09_prepare_toy.py；条件自动 |
| `/tmp/nanochat-w2-toy/tokenizer/tokenizer.pkl` | 原生 tokenizer | d09_prepare_toy.py；条件自动 |
| `/tmp/nanochat-w2-toy/tokenizer/token_bytes.pt` | 原生 token 字节长度 | d09_prepare_toy.py；条件自动 |
| `/tmp/nanochat-w2-toy/d09_synthetic_provenance.json` | 合成数据来源说明 | d09_prepare_toy.py；条件自动 |

原生 `tokenizer.pkl`、`token_bytes.pt` 保留加载器约定的文件名；用带日编号的外围文件归属日期。本日不生成模型或优化器 checkpoint。已有历史结果不原位重命名，新的命令使用本页命名。未执行的 CPU 检查说明缺少的依赖或输入；不预先声称产物已生成。GPU、语言模型训练与可选 Markdown 均不作为本日验收前置。

## 今日自验证与交流问题

直接在聊天中按题号回答，人工 Markdown 报告可选。概念题用推导说明；实际操作题引用本次 JSON 与日志即可。开头只需一次注明数据分支、实际数据根与证据路径，缺少的 CPU 操作说明原因，不把未运行写成通过。建议约 15–20 分钟。

1. `A中` 有 2 个字符、4 个 UTF-8 字节。若你实际 tokenizer 将它切成 3 个 token，是否出错？某个 token 单独解码显示替换字符时，应怎样判断整段往返是否成功？

   **核对要点**：区分三种计数，依据整段编码后再解码的逐字符相等性；结合实际中文或 emoji case 的 `text/token_ids/passed`，不把示例 ID 当实测。

2. 普通空字符串与显式 prepend BOS 的结果有什么不同？若往返检查失败后有人建议先 `.strip()` 再比较，你是否接受？

   **核对要点**：从 encode 先编码、再插入标记的顺序推导；区分额外边界与原文，对空白/换行作出保真判断。

3. 新的原始行是 `[BOS,31,32,33,34]`，T=4。写出 x、y，指出输入 32 的目标；若 B=3，这个批次包含多少个预测位置？

   **核对要点**：切片与长度正确，说明 T+1 的用途；批次单位是目标 token 位置，不能改说成句子数。

4. 上一题中，输入 32 的位置为何不能看见后面的 33？若把该列 y 改成 -1，又改变了什么？今天的真实 loader 会自动产生这样的 -1 吗？

   **核对要点**：区别注意力信息可见范围与损失位置选择；对应 `causal=True`、`ignore_index=-1` 和 loader 只填合法 ID 的事实，不要求运行模型。

5. 一行剩余 4 个位置，候选文档（含 BOS）长度为 2、4、7，会选哪篇？接着另一行只剩 1 个位置，最短文档长 2，会留下怎样的数据？

   **核对要点**：推导 longest-fit 与 shortest-crop 两个分支；说明文档被 pop 后剩余尾部不回填，位置全满不能推出语料完整保留，BOS 也不是文档隔离掩码。

6. 根据你这次的输出，能否声称“文本往返与 train/val 批次对齐已检查”？如果是，只举两项关键证据并指出一项仍无法推断的结论；未运行则说明对应缺项。

   **核对要点**：同时看 `status`、退出码、两个 split 的 batch 结果，并能回到实际断言；不能由合成数据和单批检查推断语言质量、全语料质量或模型训练已完成。

离开前保留自动结果与本日局部日志；重跑注明会话边界。可选报告位置是 `experiments/inference_infra/reports/d09_token_roundtrip.md`，不写报告不影响交流。下一天复用已确认的结论与路径，无需为了重复证明同一件事再生成数据。分享证据时检查私人路径；不提交 tokenizer、语料、权重或环境目录。
