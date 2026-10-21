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
knowledge_scope: W2第1天实操与验收
---

# W2 D1：找到 nanochat 真正使用的 tokenizer 与数据

前一天：[W1第七天](../w1/d07_benchmark_audit.md)；后一天：[继续下一天](d09_token_roundtrip.md)；[原始总计划](../nanochat_daily_execution_2026-10-01.md)。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口：**[2.2 语料下载、划分与适配](../../llm-training-workflow/02-environment-and-data-preparation.md#reading-2-2)；[3.1 训练词表与配套产物](../../llm-training-workflow/03-tokenizer-and-conversation-protocol.md#reading-3-1)。
- **今天重点：**理解训练/验证分片、分词器文件和每词元字节数文件各自的来源与用途，再核对今天脚本找到的真实路径。
- **暂时跳过：**3.2 对话协议留到 D15；不要照参考示例启动大规模下载。

**读完能解释：**分词器、语料分片和模型权重各解决什么问题？词表大小相同为什么还不够证明分词器配套？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 开始前：固定 shell 与结果位置

所有命令从仓库根目录执行。首次打开终端先运行以下准备；`conda activate` 若已可用，可省略 `source` 一行。这里使用本项目的 Conda 环境；不要混用系统 Python。CPU实验建议先 `export NANOCHAT_DTYPE=float32`；脚本只在变量未设置时选择FP32，已有变量会遵从并记录，不能假定CPU必定FP32。GPU实验另按协议显式选dtype。

```bash
source /home/lizishuo/miniconda3/etc/profile.d/conda.sh
conda activate nanochat
cd /home/lizishuo/_program_/Project/nanochat
python -c "import sys; print(sys.executable); print(sys.version)"
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
exec > >(tee -a experiments/inference_infra/reports/d08_session.log) 2>&1
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。若环境不存在或第三方依赖缺失，先解决环境，不把 `--help` 成功当作模型实验通过。后面写入的 `experiments/` 文件是你实际执行教程时产生的正式成果；本指导不预先生成成绩。大数据、tokenizer、权重放在 `NANOCHAT_BASE_DIR`，不要加进 Git。表中的 `$NANOCHAT_BASE_DIR` 代表当前 shell 选定的真实绝对数据根；合成分支示例为 `/tmp/nanochat-w2-toy`，未设置时当前用户默认根为 `/home/lizishuo/.cache/nanochat`。交流时给出实际展开后的绝对路径；`*.parquet` 表示该目录内逐一列出的实际数据文件，旧数据回退目录为 `$NANOCHAT_BASE_DIR/base_data/`。

本 shell 后续终端输出同步保存在 `experiments/inference_infra/reports/d08_session.log`；命令返回后立即执行 `echo "exit=$?"`，避免丢失退出码。帮助、源码检索、解释器与 Git 检查只打印，不另有检查结果文件。日志是追加写入，重跑时交流中注明会话边界或日志中的起止位置。

## 今日目标与时间盒

今天的完成标志是“知道文件实际在哪里、训练/验证如何分开、缺少什么”，还不要求训练成功。安排 15 分钟复核 W1、20 分钟检查路径、20 分钟读代码、15 分钟自验证并交流。前置是 [W1 基准审计](../w1/d07_benchmark_audit.md)；若 W1 还没有完成，先读该页的两条生成路径与设备边界。

## 第一步：先用只读脚本盘点

打开 [数据前置检查脚本](scripts/d08_data_check.py)，先看参数，再执行：

```bash
python .agents/docs/career_roadmap/w2/scripts/d08_data_check.py --help
python .agents/docs/career_roadmap/w2/scripts/d08_data_check.py --output experiments/inference_infra/reports/d08_data_inventory.json
```

该脚本只列目录与包可发现性，不加载 pickle、不访问网络、不创建默认缓存目录。退出 `0` 仅说明“可进入内容检查”；退出 `2` 表示条件缺失，是计划内分流。终端返回后立刻用 `echo $?` 记下退出码，再打开 JSON。重点逐项读：

| 字段 | 应如何解释 |
|---|---|
| `base_dir` | `NANOCHAT_BASE_DIR` 有值则使用该值，否则 `~/.cache/nanochat` |
| `tokenizer.pkl` | nanochat 的编码对象，不是任意 Hugging Face tokenizer 文件 |
| `token_bytes.pt` | 每个 token 的字节长度统计，训练评估入口也会读取；有 pkl 不代表够用 |
| `train_shards` | 按文件名排序后除最后一份外的数量 |
| `val_shards` | 有文件时仅最后一份；只有一份会导致训练 split 为空 |
| `dependencies` | 可发现第三方包，不等于包可成功加载或 ABI 兼容 |
| `content_validated` | 这里始终为 false；明天才检验内容 |

## 第二步：在源码中核对路径，避免检查错目录

```bash
rg -n 'def get_base_dir|NANOCHAT_BASE_DIR' nanochat/common.py
rg -n 'DATA_DIR|list_parquet_files|parquet_paths\[:-1\]|parquet_paths\[-1:' nanochat/dataset.py
rg -n 'tokenizer_dir|token_bytes|vocab-size|max-chars' scripts/tok_train.py nanochat/tokenizer.py
```

在编辑器依次打开三个位置。`get_base_dir()` 本身会创建目录，因此今天库存脚本没有调用它。`dataset.py` 先找 `base_data_climbmix`；仅当目录不存在才回退旧 `base_data`，**新目录存在但为空时不会回退**。脚本显式 `--base-dir` 只改变盘点路径；后续训练读取的是环境变量，不会读取库存 JSON。决定使用自定义路径后，必须在当前 shell 设置，例如 `export NANOCHAT_BASE_DIR=/tmp/nanochat-w2-toy`。

<abbr title="Parquet · 按列组织的数据文件格式。本文每份文件必须含非空 text 字符串列；目录中最后一份文件作为验证 split。文件存在不代表结构正确。">Parquet</abbr> 是容器格式，不能把纯文本改后缀后当作训练数据。<abbr title="Tokenizer · 把文本编码为整数 token ID 并解码回文本的组件。nanochat 的 tokenizer 词表与模型 embedding/output 维度必须一致，ID 语义也必须对应。">tokenizer</abbr> 与权重是一套契约；MiniMind tokenizer 或仅同词表大小的其他 tokenizer 都不能直接替换。

## 第三步：选择有资源或离线分支

**已有 nanochat 数据**：交流时说明来源、已有文件、是否可信、是否与 tokenizer 同一实验。不要重训或覆盖它，明天用内容检查脚本验证。

**没有外部数据，但依赖已安装**：明天使用本周 [隔离合成数据准备脚本](scripts/d09_prepare_toy.py)，在全新数据根目录制造 smoke 数据。它是合法数据格式和本项目 tokenizer，不包含真实预训练质量证据。

**依赖也没有**：今天停在清单与路径检查，记录具体缺哪个包；不要下载整个 `runs/runcpu.sh` 需要的数据。官方 `python -m nanochat.dataset -n 1` 会下载一份训练 shard 并另加末尾验证 shard，可能体积较大；只有你确认存储和网络后再选这条路线，本教程离线主线不要求下载。

## 第四步：自行核对后交流

完成盘点后，直接回答文末的本日问题，附库存 JSON 路径与真实退出码。来源、依赖缺口与分支选择都可以在聊天中说明；如想留本地笔记，可选整理到 `experiments/inference_infra/reports/d08_data_prerequisites.md`，无需长文或排版。

## 常见失败与验收

- `FileNotFoundError`：检查拼写、挂载和环境变量；不要为了让错误消失随手建一个空新数据目录，因为它会挡住旧数据回退。
- 只有一个 shard：补充独立验证文件或使用隔离合成数据；不能复制同一真实 shard 并声称验证独立。
- pkl 存在而 token_bytes 缺失：核对原 tokenizer 生成过程；不得用另一词表生成的 token_bytes 凑数。
- 验收：库存 JSON 可打开；聊天回答逐项说明来源与缺口并引用库存证据；能说清最后文件作为 val 的规则。今天不以“文件都存在”宣称数据可训练。

## 今日脚本与产物清单

每日编号使用全程 d01–d21；跨日复用保留首次实际执行日。以下项目相对路径均以仓库根目录为基准，脚本清单直接列出本日执行入口。

| 脚本完整项目路径 | 用途 | 执行阶段 | 输入 | 运行条件 |
|---|---|---|---|---|
| `.agents/docs/career_roadmap/w2/scripts/d08_data_check.py` | 盘点数据与依赖 | 第一步 | `$NANOCHAT_BASE_DIR/tokenizer/`、`$NANOCHAT_BASE_DIR/base_data_climbmix/`（回退 `$NANOCHAT_BASE_DIR/base_data/`）；--base-dir 可覆写盘点根 | 标准库即可；缺失前置退出 2 |

| 产物完整项目相对路径 | 用途 | 生成者 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/reports/d08_data_inventory.json` | 数据与依赖库存 | d08_data_check.py | 自动；有缺口也写 JSON |
| `experiments/inference_infra/reports/d08_data_prerequisites.md` | 来源、缺口与后续分支 | 第四步 | 可选人工整理；可直接在聊天中回答本日问题 |
| `experiments/inference_infra/reports/d08_session.log` | 当天终端输出与手动记录的退出码 | 开场 shell 的 tee | 自动；仅捕获启用后的输出 |

原生 `tokenizer.pkl`、`token_bytes.pt`、`model_<step>.pt`、`meta_<step>.json` 与 `optim_<step>_rank0.pt` 是 loader 的格式契约，保留其名称；用带日编号的数据文件或模型标签归属日期。已有历史结果不原位重命名，新的命令使用本页命名。没有执行或条件不满足的必需检查均标 PENDING；可选 Markdown 未整理不影响验收。

## 今日自验证与交流问题

自行验证后，直接在聊天中逐题简答即可，不要求正式报告、排版或长文。每个结论引用相应日志、自动结果中的字段或源码位置；没有执行的检查明确标为 `PENDING`，遇到失败给出真实退出码与错误。需要本地留存时再选用上表的可选 Markdown。

1. 当前真正使用的数据根是哪一个绝对路径？对照 `d08_data_inventory.json` 的 `base_dir` 与当前 `NANOCHAT_BASE_DIR`，说明脚本 `--base-dir` 为什么不会自动改变训练路径。

2. `tokenizer.pkl` 与 `token_bytes.pt` 分别来自哪里，是否存在、可信且属于同一套词表？引用库存字段；今天哪些内容仍未验证？

3. 实际 train/val 文件各有哪些？为什么只有一份 shard 会使训练为空？新数据目录存在但为空时，旧目录会不会被使用？给出源码检索位置。

4. 你选择 existing、isolated-synthetic 还是 blocked？引用依赖检查与退出码，说明下一条准备命令，以及磁盘或网络条件是否已确认。

离开前保留本日会话日志与自动结果；若追加重跑，注明会话边界。交流时可附 `git rev-parse HEAD`、`git status --short` 及实际设备/精度以定位证据，不必另填通用表。下一天可直接复用本聊天中已确认的结论与证据路径。共享前检查数据路径和日志是否包含私人信息；若提交成果，只纳入自己的轻量记录与脚本，不提交权重、数据、环境目录或缓存。
