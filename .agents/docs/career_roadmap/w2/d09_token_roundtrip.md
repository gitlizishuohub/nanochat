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
  - nanochat/gpt.py
  - nanochat/engine.py
  - scripts/infer_bench.py
  - nanochat/flash_attention.py
doc_type: tutorial
knowledge_scope: W2第2天实操与验收
---

# W2 D2：逐项验证文本、token 与训练批次

前一天：[返回前一天](d08_data_prerequisites.md)；后一天：[继续下一天](d10_training_smoke.md)；[原始总计划](../nanochat_daily_execution_2026-10-01.md)。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口：**[5.1 预训练：装满 T+1 然后 shift](../../llm-training-workflow/05-data-encoding-and-batching.md#reading-5-1)；[5.3 batch、epoch 和每卡分配](../../llm-training-workflow/05-data-encoding-and-batching.md#reading-5-3)。
- **今天重点：**对照 round-trip 和真实 loader 批次，手工完成一次输入/目标错位，并分清每卡行数、序列长度与有效批量。
- **暂时跳过：**5.2 的对话装箱留到 D15；多卡通信实现留到 D14 的概念预览。

**读完能解释：**给定一行 T+1 个词元，输入和目标如何形成？全局批量的单位为什么不能直接说成句子数？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 开始前：固定 shell 与结果位置

所有命令从仓库根目录执行。首次打开终端先运行以下准备；`conda activate` 若已可用，可省略 `source` 一行。这里使用本项目的 Conda 环境；不要混用系统 Python。CPU实验建议先 `export NANOCHAT_DTYPE=float32`；脚本只在变量未设置时选择FP32，已有变量会遵从并记录，不能假定CPU必定FP32。GPU实验另按协议显式选dtype。

```bash
source /home/lizishuo/miniconda3/etc/profile.d/conda.sh
conda activate nanochat
cd /home/lizishuo/_program_/Project/nanochat
python -c "import sys; print(sys.executable); print(sys.version)"
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
exec > >(tee -a experiments/inference_infra/reports/d09_session.log) 2>&1
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。若环境不存在或第三方依赖缺失，先解决环境，不把 `--help` 成功当作模型实验通过。后面写入的 `experiments/` 文件是你实际执行教程时产生的正式成果；本指导不预先生成成绩。大数据、tokenizer、权重放在 `NANOCHAT_BASE_DIR`，不要加进 Git。表中的 `$NANOCHAT_BASE_DIR` 代表当前 shell 选定的真实绝对数据根；合成分支示例为 `/tmp/nanochat-w2-toy`，未设置时当前用户默认根为 `/home/lizishuo/.cache/nanochat`。交流时给出实际展开后的绝对路径；`*.parquet` 表示该目录内逐一列出的实际数据文件，旧数据回退目录为 `$NANOCHAT_BASE_DIR/base_data/`。

本 shell 后续终端输出同步保存在 `experiments/inference_infra/reports/d09_session.log`；命令返回后立即执行 `echo "exit=$?"`，避免丢失退出码。帮助、源码检索、解释器与 Git 检查只打印，不另有检查结果文件。日志是追加写入，重跑时交流中注明会话边界或日志中的起止位置。

## 今日目标与时间盒

用 20 分钟准备数据，25 分钟做可逆编码检查，20 分钟检查训练批次，15 分钟自验证并交流。完成标志是有实际断言输出，或有明确错误与恢复方法。先读 [昨天的数据盘点](d08_data_prerequisites.md)，确认你选择 existing 还是 isolated-synthetic。

## 第一步：没有数据时，创建独立的 smoke 数据

使用 [隔离合成数据脚本](scripts/d09_prepare_toy.py)。这个脚本不下载文件，不修改生产代码，不覆盖任何已存在的目标目录。以下路径仅供学习实验；若已存在，换一个新后缀，不能删除旧数据再重跑。

```bash
python .agents/docs/career_roadmap/w2/scripts/d09_prepare_toy.py --help
python .agents/docs/career_roadmap/w2/scripts/d09_prepare_toy.py --base-dir /tmp/nanochat-w2-toy
export NANOCHAT_BASE_DIR=/tmp/nanochat-w2-toy
python .agents/docs/career_roadmap/w2/scripts/d08_data_check.py --output experiments/inference_infra/reports/d09_toy_inventory.json
```

脚本创建 1,200 条 train 和 1,200 条 val 合成句子、两份含 `text` 列的 parquet、本项目 RustBPETokenizer 训练产生的词表，以及准确的 token_bytes。`d09_synthetic_provenance.json` 记录来源。train 与 val 使用不同模板，但这种重复数据只用于功能 smoke，不能拿 val loss 评价真实语言能力。临时根目录重启后可能丢失；要长期保存可换到自己有权限的大数据盘路径，并在交流时说明。

**已有数据分支**：跳过准备命令，设置 `export NANOCHAT_BASE_DIR=你已核验的实际绝对路径`。不要把示例 `/tmp` 路径设置到真实数据实验上。

## 第二步：逐项验证 round-trip

<abbr title="Round-trip · 往返验证。本文检查完整字符串经 encode 转为 token IDs 再 decode 后逐字符相等；不要求每个单独 token 都能解码为合法 UTF-8 字符。">round-trip</abbr> 的断言运行在 [真实 tokenizer 检查脚本](scripts/d09_roundtrip.py)：

```bash
python .agents/docs/career_roadmap/w2/scripts/d09_roundtrip.py --output experiments/inference_infra/reports/d09_token_roundtrip.json
```

打开 JSON，确认 `status=ok` 且有六条 `cases`：英文、中英混合、换行与空白、数字、emoji 与重音字符、空字符串。逐条看 `text` 和 `token_ids`：token 数不等于汉字数，也不等于 UTF-8 字节数。空字符串普通 encode 应为空；加 BOS 后才有开头标记。脚本另外验证 batch encode 与逐条 encode 一致、九个特殊 token ID 不冲突、token_bytes 的长度等于实际词表大小且特殊 token 字节数为零。

<abbr title="BOS / Beginning of Sequence · 序列起点标记。本文文档编码时由 prepend 显式添加；它的 ID 是 tokenizer 定义的特殊 ID，不应假定固定为某个整数。">BOS</abbr> 需要与普通文本区别处理：`decode(encode(text, prepend=bos))` 会包括特殊标记，因此脚本比较 `bos_ids[1:]`，不把标记当原文。

## 第三步：验证真实 dataloader 一个批次

```bash
python .agents/docs/career_roadmap/w2/scripts/d09_roundtrip.py --check-data --output experiments/inference_infra/reports/d09_token_data_roundtrip.json
rg -n 'row_capacity|best_idx|cpu_inputs.copy_|cpu_targets.copy_' nanochat/dataloader.py
```

该脚本用每个 split 的真实 loader 取一批 `[1,32]`；训练时 loader 默认 buffer_size 为 1000，教学检查用 4 降低等待。它读取 parquet 第一 row group，检查 `text` 列非空且元素为字符串，然后检查：

1. `x/y` 形状 `[B,T]=[1,32]`、整数类型；
2. `x[:,1:] == y[:,:-1]`，说明 labels 向右错位一个 token；
3. 每行输入以 BOS 起头，ID 都在词表范围；
4. train 与 val 都能产出，不会卡在空 train split。

<abbr title="Best-Fit Cropping · 从文档缓冲区中优先选择能完整塞入剩余空间的最长文档，塞不下时裁切。本文每行构造 T+1 个 token，再错位生成 x/y；没有 padding 不代表没有被裁切内容。">best-fit</abbr> 不等于简单按文件拼接；看清 T+1 到 `[B,T]` 的变化，再用聊天文字给出一个 6-token 的错位例子。

## 第四步：自行核对后交流

直接用文末问题讨论两个 JSON、中文样本、BOS 和 x/y 错位。附实际输出与退出码，不只给出“passed”；`experiments/inference_infra/reports/d09_token_roundtrip.md` 仅为可选整理位置。

## 失败分流与验收

- `No module named rustbpe/tiktoken/pyarrow/torch`：先核对解释器；若 Conda 环境确实缺依赖，记录 `BLOCKED_DEPENDENCY`，不要把合成流程改成自制假 tokenizer 后冒充通过。
- `pickle` 或词表异常：只使用自己生成或信任来源的 tokenizer；回到隔离目录重新准备，保留原数据。
- `get_token_bytes` 报缺失：今天准备脚本会生成它；已有数据则回溯同一 tokenizer 的生成记录。
- dataloader 长时间没有输出：先查 train split 文件数、row group、text 是否全空；不要无限等待没有数据的 iterator。
- 验收：实际输出 `status=ok`、每个断言可解释；依赖受阻时交流中说明缺包名称及未执行范围，明天不可宣称已有合法 checkpoint。

## 今日脚本与产物清单

每日编号使用全程 d01–d21；跨日复用保留首次实际执行日。以下项目相对路径均以仓库根目录为基准，脚本清单直接列出本日执行入口。

| 脚本完整项目路径 | 用途 | 执行阶段 | 输入 | 运行条件 |
|---|---|---|---|---|
| `.agents/docs/career_roadmap/w2/scripts/d09_prepare_toy.py` | 构造隔离合成数据与 tokenizer | 第一步（可选） | 脚本内合成模板；--base-dir 指定全新绝对路径 | 仅无可用数据分支；torch/pyarrow/rustbpe/tiktoken 齐全 |
| `.agents/docs/career_roadmap/w2/scripts/d08_data_check.py` | 复用前一天的库存检查 | 准备后 | `/tmp/nanochat-w2-toy/tokenizer/`、`/tmp/nanochat-w2-toy/base_data_climbmix/` | 目录创建后 |
| `.agents/docs/career_roadmap/w2/scripts/d09_roundtrip.py` | 验证文本往返与 loader 批次 | 第二、三步 | `$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl`、`$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt`、`$NANOCHAT_BASE_DIR/base_data_climbmix/*.parquet` | 可信 tokenizer；--check-data 另要求非空 train/val |

| 产物完整项目相对路径 | 用途 | 生成者 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/reports/d09_toy_inventory.json` | 合成目录盘点 | d08_data_check.py | 条件自动；仅合成分支 |
| `experiments/inference_infra/reports/d09_token_roundtrip.json` | 六组 tokenizer 断言 | d09_roundtrip.py | 自动；成功或捕获的异常均写 JSON |
| `experiments/inference_infra/reports/d09_token_data_roundtrip.json` | train/val 批次断言 | d09_roundtrip.py --check-data | 条件自动；需执行数据检查 |
| `experiments/inference_infra/reports/d09_token_roundtrip.md` | 往返与错位示意说明 | 第四步 | 可选人工整理；可直接在聊天中回答本日问题 |
| `experiments/inference_infra/reports/d09_session.log` | 当天终端输出与手动记录的退出码 | 开场 shell 的 tee | 自动；仅捕获启用后的输出 |

合成分支的外部数据根示例为 `/tmp/nanochat-w2-toy`（仓库外，须是全新目录）；脚本条件自动生成下列文件，已有数据分支不生成它们：

| 完整绝对路径（示例数据根） | 用途 | 生成者与方式 |
|---|---|---|
| `/tmp/nanochat-w2-toy/base_data_climbmix/d09_shard_00000.parquet` | 合成 train | d09_prepare_toy.py；条件自动 |
| `/tmp/nanochat-w2-toy/base_data_climbmix/d09_shard_00001.parquet` | 合成 val | d09_prepare_toy.py；条件自动 |
| `/tmp/nanochat-w2-toy/tokenizer/tokenizer.pkl` | 原生 tokenizer | d09_prepare_toy.py；条件自动 |
| `/tmp/nanochat-w2-toy/tokenizer/token_bytes.pt` | 原生 token 字节长度 | d09_prepare_toy.py；条件自动 |
| `/tmp/nanochat-w2-toy/d09_synthetic_provenance.json` | 合成数据来源说明 | d09_prepare_toy.py；条件自动 |

原生 `tokenizer.pkl`、`token_bytes.pt`、`model_<step>.pt`、`meta_<step>.json` 与 `optim_<step>_rank0.pt` 是 loader 的格式契约，保留其名称；用带日编号的数据文件或模型标签归属日期。已有历史结果不原位重命名，新的命令使用本页命名。没有执行或条件不满足的必需检查均标 PENDING；可选 Markdown 未整理不影响验收。

## 今日自验证与交流问题

自行验证后，直接在聊天中逐题简答即可，不要求正式报告、排版或长文。每个结论引用相应日志、自动结果中的字段或源码位置；没有执行的检查明确标为 `PENDING`，遇到失败给出真实退出码与错误。需要本地留存时再选用上表的可选 Markdown。

1. 六组往返样本中，选一个中文或 emoji 样本贴出文本和 token ID；为什么 token 数、字符数与字节数可能不同？引用 `d09_token_roundtrip.json` 的实际结果。

2. 空字符串普通 encode 与加 BOS 有什么不同？BOS 的 ID 从哪里取得，为什么不能把带 BOS 的直接解码结果与原文比较？

3. 从 `d09_token_data_roundtrip.json` 解释 x/y 的形状与错位关系，再用6个 token 的短序列写出对应输入和目标；best-fit 为什么不等于无裁切拼接？

4. 两个 split 是否都产生合法批次，特殊 ID 与 token_bytes 检查是否通过？附两条检查命令的退出码；若失败，指出第一个失败断言、缺包或缺文件。

5. 这次数据根与来源是什么？合成数据通过证明了哪些功能，还有哪些语言能力、真实数据或训练结果没有验证？

离开前保留本日会话日志与自动结果；若追加重跑，注明会话边界。交流时可附 `git rev-parse HEAD`、`git status --short` 及实际设备/精度以定位证据，不必另填通用表。下一天可直接复用本聊天中已确认的结论与证据路径。共享前检查数据路径和日志是否包含私人信息；若提交成果，只纳入自己的轻量记录与脚本，不提交权重、数据、环境目录或缓存。
