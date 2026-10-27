---
source_files:
  - nanochat/common.py
  - nanochat/tokenizer.py
  - nanochat/dataset.py
  - nanochat/dataloader.py
  - scripts/tok_train.py
  - scripts/tok_eval.py
  - scripts/base_train.py
  - nanochat/checkpoint_manager.py
source_symbols:
  - get_base_dir
  - RustBPETokenizer.from_directory
  - list_parquet_files
  - parquets_iter_batched
  - get_token_bytes
doc_type: tutorial
knowledge_scope: W2第1天的数据与分词器前置依赖
---

# W2 D1：找到 nanochat 真正使用的 tokenizer 与数据

前一天：[W1第七天](../w1/d07_benchmark_audit.md)；后一天：[继续下一天](d09_token_roundtrip.md)；[原始总计划](../nanochat_daily_execution_2026-10-01.md)。

今天学习一个具体问题：**仓库里有了 Python 代码，为什么还不能直接把一段文字交给模型？** 读完本页，你应能解释语料、分词器和模型参数分别提供什么，以及程序怎样找到它们。

**当天范围**：第 1–3 节是概念与源码推演，无需执行模型；第 4 节只用 CPU 运行标准库盘点脚本。没有 GPU、没有训练过模型、没有任何权重或外部语料，都可以完成今天的学习。盘点发现资源缺失本身就是有效结果，不要求补跑训练。本日没有 GPU 扩展任务。

## 当天的流程阅读

先读本页第 1 节，再用约 10–20 分钟对照以下小节；参考文档用于补充来源，不必照其中的示例运行下载或训练。

- **阅读入口**：[2.2 语料下载、划分与适配](../../llm-training-workflow/02-environment-and-data-preparation.md#reading-2-2)；[3.1 训练词表与配套产物](../../llm-training-workflow/03-tokenizer-and-conversation-protocol.md#reading-3-1)。
- **今天重点**：训练/验证分片为什么分开；分词器文件和每词元字节数文件各自提供什么。
- **暂时跳过**：对话协议留到 D15；大规模语料准备与模型训练均非今天操作。

**读完能解释**：文本、token ID、模型参数三者怎样连起来？哪一类文件缺失会影响哪个环节？无需另写报告。

## 1. 概念推演：一段文字进入模型之前发生什么

### 1.1 语料、分词器、模型参数是三种不同资源

<abbr title="Corpus · 语料。供编码、训练或评估使用的一组文本；文本的来源和划分决定可支持的实验结论。">语料</abbr>提供原始文本。<abbr title="Tokenizer · 分词器。把文本转换为整数token ID序列，并把ID序列解码为文本；ID对应的文本片段由具体词表决定。">分词器</abbr>定义文本与整数之间的转换。模型参数保存模型从数据中学到的数值关系。代码定义怎样计算，文件提供计算要使用的数据或状态。

以下是**人为指定的教学例子，不是 nanochat 的实测分词结果**：

```text
原始文本："cat dog"
假设词表：ID 7 对应 "cat"，ID 12 对应 " dog"
编码结果：[7, 12]
```

模型接收的是整数 `[7, 12]`。它先在输入向量表中查询第 7 行、第 12 行，再用模型参数计算预测。这个输入向量表叫 <abbr title="Embedding · 嵌入表。模型中按token ID查询向量的参数表；表的每一行与具体分词器的一个ID对应。">embedding</abbr>。在预测下一个 token 时，模型输出的各项词表分数同样按 token ID 解释。

假设另一个分词器词表大小完全相同，但 ID 7 表示 `"dog"`。加载时张量大小可以完全吻合，原模型却仍把第 7 行按旧含义处理。**大小相同只能保证索引范围可能兼容，不能保证每个 ID 的语义配套。** 这也是不能随便拿其他项目的 tokenizer 替换 nanochat tokenizer 的原因。

| 资源 | 它回答的问题 | 不能由它单独证明的事情 |
|---|---|---|
| 语料 Parquet 文件 | 要处理哪些文本？ | 模型是否学会这些文本 |
| `tokenizer/tokenizer.pkl` | 文本如何变成 ID，ID 如何变回文本？ | 模型是否经过训练、参数是否配套 |
| 模型权重文件 | 模型用哪些参数做预测？ | 当前 tokenizer 的 ID 含义是否与训练时一致 |

<abbr title="Parquet · 按列组织的数据文件格式。nanochat的数据读取代码从中读取text列；把普通文本文件改成.parquet后缀不会改变其内部格式。">Parquet</abbr> 是文件格式，不是文本后缀的约定。今天盘点只检查文件是否存在，明天才读取其内容，检查能否按预期取出文本。

### 1.2 “训练分词器”和“训练语言模型”不是同一个操作

分词器的 <abbr title="BPE / Byte Pair Encoding · 字节对编码。此处从字节基础单元出发，根据文本中的相邻片段统计学习合并规则，形成可编码和解码的词表。">BPE</abbr> 训练，是根据文本统计学习怎样合并片段、分配 ID；语言模型训练则在预测误差的指导下更新模型参数。训练 tokenizer 不会产生一个能回答问题的语言模型。

本项目的 `scripts/tok_train.py` 负责前者，`scripts/base_train.py` 负责后者。今天两个入口都不执行。D09 若使用小型合成数据准备脚本，会在 CPU 上构建本项目可读取的小词表和数据文件；那一步也不训练语言模型，更不需要已有模型权重。

推理生成则使用模型参数做预测，不通过训练误差更新参数。今天只是理解生成与训练共同依赖的文本编码接口，不进行推理生成。

### 1.3 为什么除了 tokenizer.pkl 还要 token_bytes.pt

回到假设例子，`"cat"` 占 3 个 UTF-8 字节，`" dog"` 占 4 个字节，因此对应 ID 的字节长度可以缓存成：

```text
token_bytes[7]  = 3
token_bytes[12] = 4
```

`tokenizer.pkl` 保存编码规则；`token_bytes.pt` 保存**这套词表每个 ID 对应的原始字节数**。后者用于 <abbr title="BPB / Bits Per Byte · 每字节比特数。把预测误差按原始文本字节数归一化的评估指标；token_bytes.pt帮助将token计数转换为字节计数。">BPB</abbr> 评估：同样一段文字被不同词表分成不同数量的 token，按字节归一化才能减少分词粒度不同带来的比较偏差。它不是权重，也不是语料的词频统计。

`tok_train.py` 的真实处理如下；仅为源码阅读，无需执行：

```python
if token_id in special_ids:
    token_bytes.append(0) # special tokens are not counted
else:
    num_bytes = len(tokenizer.decode_single_token_bytes(token_id))
    token_bytes.append(num_bytes)
```

这里读取单个 token 的**原始字节**，因为单个 token 可能只是一个多字节字符的一部分，单独解码成字符串再计数可能失真。特殊 token 在这份统计中计为 0。两个文件必须来自同一套词表，否则即使长度相同，也可能把某个 ID 的误差除以另一个 token 的字节数。

## 2. 源码带读：程序怎样决定读哪个目录

### 2.1 数据根目录由环境变量决定，不由库存 JSON 决定

[common.py 中的 get_base_dir](../../../../nanochat/common.py) 摘录：

```python
if os.environ.get("NANOCHAT_BASE_DIR"):
    nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
else:
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache")
    nanochat_dir = os.path.join(cache_dir, "nanochat")
os.makedirs(nanochat_dir, exist_ok=True)
return nanochat_dir
```

按执行顺序读：先检查 `NANOCHAT_BASE_DIR` 是否有非空值，有则采用；否则使用当前用户的 `~/.cache/nanochat`；最后确保根目录存在。当前用户未设置变量时，默认路径是 `/home/lizishuo/.cache/nanochat`。这段函数有创建目录的副作用，今天盘点脚本没有调用它。

[tokenizer.py 的 get_tokenizer](../../../../nanochat/tokenizer.py) 在这个根下固定拼接 `tokenizer/`，而 `RustBPETokenizer.from_directory` 再读取其中的 `tokenizer.pkl`。因此完整位置是：

```text
数据根/tokenizer/tokenizer.pkl
数据根/tokenizer/token_bytes.pt
数据根/base_data_climbmix/*.parquet
```

盘点脚本的 `--base-dir` 只改变**这次检查**的位置。例如检查 `/tmp/nanochat-w2-toy`，不会自动让 nanochat 也改读那里；真正使用自定义根的后续程序需要在当前 shell 设置 `NANOCHAT_BASE_DIR`。库存 JSON 是检查结果，训练和数据加载器不会读取它来修改路径。

### 2.2 旧目录回退取决于“不存在”，而非“没有文件”

[dataset.py 的 list_parquet_files](../../../../nanochat/dataset.py) 先使用 `base_data_climbmix/`，只有目录不存在才尝试旧 `base_data/`。把 warning 打印省略后，关键条件是：

```python
if not os.path.exists(data_dir):
    # 中间省略：可选的迁移提示输出
    data_dir = os.path.join(base_dir, "base_data")
```

假设旧 `base_data/` 有两份数据，但你新建了空的 `base_data_climbmix/`：条件为假，程序就使用空新目录，不再回退。因此“为了消除路径错误，先建一个空目录”可能让旧数据从程序视角消失。修正时先确认预期使用哪套数据，避免直接搬移或覆盖已有资源。

今天的检查脚本沿用这一选择条件。JSON 中 `data_dir` 才是实际扫描的目录，应把它和 `base_dir` 一起看。

## 3. 概念推演：训练和验证到底怎样划分

<abbr title="Shard · 分片。把语料分成多个文件，每份保存一部分记录；本项目按排序后的文件位置划分训练和验证。">分片</abbr>不是一个模型，也不是一个批次。当前 [dataset.py 的 parquets_iter_batched](../../../../nanochat/dataset.py) 采用下面的切片：

```python
parquet_paths = list_parquet_files()
parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
```

`list_parquet_files()` 已按文件名排序。`[:-1]` 取末尾之前的所有文件，`[-1:]` 取最后一份。因此假设目录为：

```text
a.parquet → train
b.parquet → train
c.parquet → val
```

这是文件位置规则，不会根据文件名是否包含 `train` 或 `val` 猜用途。若只有 `a.parquet`，训练切片为空，验证切片有一份。把同一真实文件复制成两个名字，虽然会让两个切片都有文件，却让验证重复训练内容，无法判断模型对未见文本的表现；这称为 <abbr title="Data leakage · 数据泄漏。评估所用信息进入训练过程，使评估无法反映对未见数据的真实表现；这里指相同文本同时出现在训练和验证中。">数据泄漏</abbr>。

[dataloader.py 的 _document_batches](../../../../nanochat/dataloader.py) 沿用同样的划分，按行组读取 `text` 列；之后才由 tokenizer 编码，组成模型输入和目标。今天的资源关系就连起来了：

```text
选数据根 → 选数据目录 → 排序并划分文件 → 读text列 → 分词器编码 → 模型输入
```

本节是在推演项目的数据流，不要求启动这个加载器。关于拼接与输入/目标错位，D09 会接着讲。

## 4. CPU 实际操作：让盘点结果回答下一步能做什么

### 4.1 固定解释器和局部日志

所有命令从仓库根目录执行。当前终端会自动激活 `nanochat`，无需重复激活；检查输出应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。盘点脚本只用标准库，不导入 torch，不加载模型，不需要设置计算精度。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
python -c "import sys; print(sys.executable); print(sys.version)"
} > >(tee -a experiments/inference_infra/reports/d08_session.log) 2>&1
```

这里的花括号命令组只把组内输出交给 <abbr title="tee · 同时把输入写到终端和文件的命令。这里与Bash进程替换配合，只记录花括号内命令的stdout和stderr；追加日志不会清空旧记录。">tee</abbr>，不会持续接管终端。旧版全局日志若仍在运行，需要关闭旧终端并新开；仅修改教程不会停止它。重跑时在聊天注明日志会话范围。

### 4.2 先预测，再盘点

打开 [D08 盘点脚本](scripts/d08_data_check.py)。它只查看目录、文件大小和第三方包能否被发现；不会下载、解包、加载 pickle 或创建默认数据根目录。它会创建你指定的输出目录并写 JSON。

运行前先预测：若没有 tokenizer 和语料，库存是否仍应该生成？答案是会；“缺什么”就是这个脚本的结果，不应通过随意建空文件把它改成成功。

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d08_data_check.py --help
python .agents/docs/career_roadmap/w2/scripts/d08_data_check.py --output experiments/inference_infra/reports/d08_data_inventory.json
d08_inventory_status=$?
echo "inventory_exit=$d08_inventory_status"
} > >(tee -a experiments/inference_infra/reports/d08_session.log) 2>&1
```

退出码紧接盘点命令记录到日志。`0` 表示前置条件足以进入后续内容检查，`2` 表示有缺口；两者都可完成今天的盘点学习。其他退出码要看真实错误，例如输出路径不可写，不能统一解释成资源缺失。

### 4.3 读一个假设库存，判断哪些结论成立

以下是**字段关系示例，不是你的实际结果**：

```json
{
  "train_shards": 1,
  "val_shards": 1,
  "dependencies": {"torch": true, "rustbpe": true, "tiktoken": true, "pyarrow": true},
  "content_validated": false,
  "ready_for_content_check": true
}
```

这里 `ready_for_content_check` 为真，还意味着脚本找到两份 tokenizer 文件。它的真实判断是：

```python
report["ready_for_content_check"] = (
    len(shards) >= 2
    and all(f.is_file() for f in files)
    and all(report["dependencies"].values())
)
```

这个表达式只回答“下一步检查所需的东西看起来齐了吗”。两份文件可能是空文件；Parquet 可能没有 `text` 列；依赖包虽然可发现，导入仍可能失败。因此 JSON 的 `content_validated` 始终是 `false`，不能把退出 0 写成“数据可训练”或“训练成功”。

对照你自己的 JSON，按以下顺序解释，而非抄一遍所有字段：

1. 看 `base_dir`、`data_dir`，确认扫描的是预期目录。`files` 与 `parquet_files` 给出本次发现的展开后的绝对路径。
2. 看两份 tokenizer 文件与分片数量，判断缺哪一类资源。`train_shards=0`、`val_shards=1` 通常表示只找到一份文件。
3. 看 `dependencies`，确认明天内容检查所需的包是否可发现。它不代表已成功导入，更不代表二进制依赖兼容。
4. 用 `ready_for_content_check` 选择后续分支，并保留 `content_validated=false` 的边界。

### 4.4 资源不同，明天的入口不同

| 本次盘点情况 | 今天怎样完成 | D09 怎样接续 |
|---|---|---|
| 已有本项目数据和 tokenizer | 说明来源、实际路径和是否来自同一实验，不覆盖它们 | 用 D09 内容检查核实编码与文件内容 |
| 没有数据，但相关依赖可用 | 如实解释缺失项，今天无需补资源 | 按 D09 使用[隔离合成数据准备脚本](scripts/d09_prepare_toy.py)，创建小型功能样本 |
| 第三方依赖也缺失 | 指出具体包和它影响的环节，完成概念推演与标准库盘点 | 进入内容操作前先解决所需依赖 |

D09 的合成数据只验证文件格式和程序接口，不代表真实语料质量。其准备脚本要求全新数据根，避免覆盖已有 tokenizer；具体调用和产物留在 D09 首次使用时说明。今天不提前执行。

**无需下载来完成本日任务**。若以后单独选择真实语料路线，要注意 `python -m nanochat.dataset -n 1` 的 `1` 指训练分片数量，还会另下载末尾验证分片，实际是两份；本页不提供执行该下载的步骤。

## 5. 自验证：用变式判断是否理解

可直接在聊天按题号回答；共用一次解释器、数据根、库存 JSON 路径和 `inventory_exit`，无需逐题复述环境。人工 Markdown 报告是可选笔记。概念题用本文推演作答即可；若未运行盘点，明确“尚未执行”，不要推测文件存在。GPU、真实模型训练和 checkpoint 不在本日验收范围，不构成待补任务。

1. 假设两个分词器都有 512 个 ID，但 ID 7 的含义由 `cat` 变成 `dog`。为什么当前模型加载时的词表大小断言可能通过，预测却会出错？

   **核对要点**：联系输入 embedding 和输出 ID 的解释；[checkpoint_manager.py](../../../../nanochat/checkpoint_manager.py) 只断言词表大小，不能证明 ID 语义配套。无需真实替换文件。

2. 旧目录有数据，新目录存在但为空。程序会读哪一个？若只给盘点脚本传了 `--base-dir`，为什么后续程序仍可能读到别处？

   **核对要点**：分别解释目录回退条件与环境变量的作用，给出不覆盖旧资源的检查思路；修改 JSON 不改变程序数据根。

3. 只有一个 Parquet 时，train/val 各有几份？把它复制成两份，解决了哪个表面条件，又破坏了什么评估前提？

   **核对要点**：根据切片推导；分清“能得到训练文件列表”和“拥有独立验证数据”。

4. 假设 `tokenizer.pkl` 能正确编解码，但 `token_bytes.pt` 来自另一套同样大小的词表，会影响哪类计算？为何不能只看文件存在和数组长度？

   **核对要点**：用第 1.3 节某个 ID 的字节数说明映射错配如何影响 BPB；不要求手推完整评估公式。

5. 根据这次库存选择 D09 分支，并举一项即使退出 0 仍无法证明的事实。如果你的退出码为 2，它是否表示今天学习失败？

   **核对要点**：引用决定分支的实际字段；区分存在、可发现与内容正确，说明退出码服务于资源分流，不是训练或学习成绩。

## 6. 脚本、产物与复查入口

以下完整项目相对路径以 `/home/lizishuo/_program_/Project/nanochat` 为根；表中的 `$NANOCHAT_BASE_DIR` 指实际选定的数据根，未设置时为 `/home/lizishuo/.cache/nanochat`。盘点 `--base-dir` 会覆写本次扫描根。`*.parquet` 代表实际列出的每个文件，不是字面文件名。

| 脚本完整项目路径 | 用途与阶段 | 完整输入位置 | 运行条件 |
|---|---|---|---|
| `.agents/docs/career_roadmap/w2/scripts/d08_data_check.py` | 第 4 节盘点数据与依赖 | `$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl`、`$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt`、`$NANOCHAT_BASE_DIR/base_data_climbmix/*.parquet`；新目录不存在时回退 `$NANOCHAT_BASE_DIR/base_data/*.parquet` | 标准库即可；CPU，无模型加载；缺资源退出 2 |

| 产物完整项目相对路径 | 用途 | 生成者与条件 |
|---|---|---|
| `experiments/inference_infra/reports/d08_data_inventory.json` | 本次资源库存 | 第 4.2 节脚本自动生成；资源有缺口也写入 |
| `experiments/inference_infra/reports/d08_session.log` | 当天终端输出与盘点退出码 | 实际运行各 Bash 命令组时由 tee 追加；不记录组外交互 |
| `experiments/inference_infra/reports/d08_data_prerequisites.md` | 来源、缺口和分支笔记 | 可选人工整理；不影响当天完成，可在聊天中回答 |

本页命令不会生成 tokenizer、语料或权重。原生 `tokenizer.pkl`、`token_bytes.pt` 的文件名是读取契约，不能加日编号来替代；实验目录和本日结果才按 d08 命名。大型资源留在数据根，不纳入 Git；不把未实际执行的操作写成已完成。

需要回查源码时，可执行以下短检索。它们只打印定位，输出保存在本日会话日志，不生成额外结果文件：

```bash
{
rg -n 'def get_base_dir|NANOCHAT_BASE_DIR' nanochat/common.py
rg -n 'DATA_DIR|list_parquet_files|parquet_paths\[:-1\]|parquet_paths\[-1:' nanochat/dataset.py
rg -n 'tokenizer_dir|token_bytes|vocab-size|max-chars' scripts/tok_train.py nanochat/tokenizer.py
} > >(tee -a experiments/inference_infra/reports/d08_session.log) 2>&1
```

离开前应能用自己的话串起“语料文本 → 分词器 ID → 模型输入”，并解释实际库存决定的下一步。下一天：[检查真实编解码与数据内容](d09_token_roundtrip.md)。
