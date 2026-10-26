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

- **阅读入口**：[2.2 语料下载、划分与适配](../../llm-training-workflow/02-environment-and-data-preparation.md#reading-2-2)；[3.1 训练词表与配套产物](../../llm-training-workflow/03-tokenizer-and-conversation-protocol.md#reading-3-1)。
- **今天重点**：理解训练/验证分片、分词器文件和每词元字节数文件各自的来源与用途，再核对今天脚本找到的真实路径。
- **暂时跳过**：3.2 对话协议留到 D15；不要照参考示例启动大规模下载。

**读完能解释**：分词器、语料分片和模型权重各解决什么问题？词表大小相同为什么还不够证明分词器配套？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 开始前：固定 shell 与结果位置

所有命令从仓库根目录执行。当前终端会自动激活 nanochat 环境，无需重复激活；先运行以下检查。这里使用本项目的 Conda 环境；不要混用系统 Python。CPU实验建议先 `export NANOCHAT_DTYPE=float32`；脚本只在变量未设置时选择FP32，已有变量会遵从并记录，不能假定CPU必定FP32。GPU实验另按协议显式选dtype。

日志只在 `{ ...; }` 命令组执行期间记录；<abbr title="Bash process substitution · 进程替换。这里把命令组的输出交给 tee，同时显示并追加到日志；重定向在命令组结束后恢复。">进程替换</abbr>不会持续接管交互终端，因此提示符和 VS Code 终端集成控制字符不会混入日志。命令组在当前 shell 执行，变量设置会保留；代码块返回最后一条命令的退出码，需核对单条命令时立即记录 `$?`。已有的显式文件重定向仍写入各自文件。复制时包含外层花括号和末尾重定向。若之前执行过旧版全局日志命令，先关闭旧终端并新开终端；仅修改文档不会停止旧终端的记录。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
{
python -c "import sys; print(sys.executable); print(sys.version)"
} > >(tee -a experiments/inference_infra/reports/d08_session.log) 2>&1
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。若环境不存在或第三方依赖缺失，先解决环境，不把 `--help` 成功当作模型实验通过。后面写入的 `experiments/` 文件是你实际执行教程时产生的正式成果；本指导不预先生成成绩。大数据、tokenizer、权重放在 `NANOCHAT_BASE_DIR`，不要加进 Git。表中的 `$NANOCHAT_BASE_DIR` 代表当前 shell 选定的真实绝对数据根；合成分支示例为 `/tmp/nanochat-w2-toy`，未设置时当前用户默认根为 `/home/lizishuo/.cache/nanochat`。交流时给出实际展开后的绝对路径；`*.parquet` 表示该目录内逐一列出的实际数据文件，旧数据回退目录为 `$NANOCHAT_BASE_DIR/base_data/`。

本页各 Bash 代码块内的输出追加保存在 `experiments/inference_infra/reports/d08_session.log`；命令返回后立即执行 `echo "exit=$?"`，避免丢失退出码。帮助、源码检索、解释器与 Git 检查只打印，不另有检查结果文件。日志是追加写入，重跑时交流中注明会话边界或日志中的起止位置。

## 今日目标与时间盒

今天的完成标志是“知道文件实际在哪里、训练/验证如何分开、缺少什么”，还不要求训练成功。安排 15 分钟复核 W1、20 分钟检查路径、20 分钟读代码、15 分钟自验证并交流。前置是 [W1 基准审计](../w1/d07_benchmark_audit.md)；若 W1 还没有完成，先读该页的两条生成路径与设备边界。

## 第一步：先用只读脚本盘点

打开 [数据前置检查脚本](scripts/d08_data_check.py)，先看参数，再执行：

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d08_data_check.py --help
python .agents/docs/career_roadmap/w2/scripts/d08_data_check.py --output experiments/inference_infra/reports/d08_data_inventory.json
} > >(tee -a experiments/inference_infra/reports/d08_session.log) 2>&1
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
{
rg -n 'def get_base_dir|NANOCHAT_BASE_DIR' nanochat/common.py
rg -n 'DATA_DIR|list_parquet_files|parquet_paths\[:-1\]|parquet_paths\[-1:' nanochat/dataset.py
rg -n 'tokenizer_dir|token_bytes|vocab-size|max-chars' scripts/tok_train.py nanochat/tokenizer.py
} > >(tee -a experiments/inference_infra/reports/d08_session.log) 2>&1
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
| `experiments/inference_infra/reports/d08_session.log` | 当天终端输出与手动记录的退出码 | 各 Bash 代码块的 tee | 自动；仅追加各代码块内的输出 |

原生 `tokenizer.pkl`、`token_bytes.pt`、`model_<step>.pt`、`meta_<step>.json` 与 `optim_<step>_rank0.pt` 是 loader 的格式契约，保留其名称；用带日编号的数据文件或模型标签归属日期。已有历史结果不原位重命名，新的命令使用本页命名。没有执行或条件不满足的必需检查均标 PENDING；可选 Markdown 未整理不影响验收。

## 今日自验证与交流问题

可直接在聊天中按题号回答，不要求手写报告。每题给出结论和可定位依据（源码位置、命令与退出码、日志片段或自动结果字段）；同一份证据可复用。概念推演须注明假设，实际运行须区分本次与历史结果。未执行写 `PENDING`，失败写真实错误和影响范围，不能据此勾选实验通过。

1. 今天用哪个 Python、数据根和精度设置？库存脚本的盘点路径与后续训练实际读取的路径一致吗？

   **通过标准**：引用 `d08_session.log` 的解释器输出和当前环境变量，给出展开后的绝对路径；对照 `d08_data_inventory.json.base_dir` 与 `common.py:get_base_dir`，解释 `--base-dir` 只影响本次库存、不会修改 shell 环境或训练路径。

2. 分词器、语料分片和模型权重各承担什么职责？为什么相同词表大小仍不足以说明两个 tokenizer 可以替换？

   **通过标准**：分别说明文本与 ID 映射、训练/验证文本来源、模型参数；依据 `tokenizer.py:RustBPETokenizer` 与 `checkpoint_manager.py:build_model` 解释维度检查只能核对词表大小，不能证明每个 ID 的含义和特殊标记协议相同。

3. 库存 JSON 怎样表示 tokenizer 两个文件是否存在？你能据此确认它们属于同一词表、内容正确且来源可信吗？

   **通过标准**：读取 `files` 中两个真实路径的 `exists`、`bytes`，说明 `tokenizer.pkl` 保存编码对象，`token_bytes.pt` 保存逐 ID 字节长度；来源和配套性另引生成记录，未知时明确未知，不能把非零文件大小当作内容验收。

4. 假设同时存在新旧数据目录，新目录缺失、存在且为空、存在且有文件三种情况分别会读哪里？

   **通过标准**：对照 `dataset.py:list_parquet_files` 和 `d08_data_check.py:main` 的存在性分支逐一推导；说明只有新目录不存在才回退，创建空目录会挡住旧数据，不能凭库存退出码决定目录回退。

5. 实际哪些 parquet 文件归 train，哪个归 val？假设文件名为 `a.parquet`、`b.parquet`、`c.parquet`，分配结果是什么？

   **通过标准**：引用库存的 `parquet_files`、`train_shards`、`val_shards` 和 `split_rule`，按文件名排序列出实际分组；概念例子给出 train=a,b、val=c，并定位 `dataset.py:parquets_iter_batched` 的切片。

6. 目录只有零份或一份 shard 时，库存会报告多少 train/val，为什么不能继续声称训练输入就绪？

   **通过标准**：依据 `d08_data_check.py` 计算零份为 0/0、一份为 0/1，说明 `ready_for_content_check` 至少要求两份；结合 loader 的训练 split 解释空数据可能导致取批次无法前进，复制同一真实 shard 不能证明验证独立。

7. `dependencies` 全为 true、库存退出 0，分别证明了什么？还缺哪些内容检查？

   **通过标准**：列出脚本检测的 torch、rustbpe、tiktoken、pyarrow，说明 `find_spec` 仅表示可发现，不能证明可导入或兼容；解释 `ready_for_content_check` 的完整条件及 `content_validated=false`，未检查 parquet 的 text、token ID、往返或批次。

8. 这次库存命令是否真的执行，输出与退出码一致吗？缺失项具体影响明天的哪一步？

   **通过标准**：提供 `d08_data_inventory.json` 路径、库存命令和即时退出码；按实际 `ready_for_content_check` 区分 0 与 2，并逐项引用缺文件、分片或依赖字段；没有本次运行时可复用注明日期的历史证据，未覆盖部分保持 `PENDING`。

9. 你今天选已有数据、隔离合成数据还是依赖受阻分支？下一步应使用哪个现有入口？

   **通过标准**：根据库存缺口给出选择依据：已有可信数据进入 `d09_roundtrip.py`，无数据但依赖齐全使用 `d09_prepare_toy.py --base-dir` 的全新目录，缺依赖先定位具体包；说明不会覆盖既有数据，合成分支只证明功能。

10. 今天哪些结论能验收，哪些必须留到 D09 或之后？为什么不必为了通过 D08 启动官方大数据下载？

    **通过标准**：引用库存 JSON、会话日志及路径源码，区分路径/存在性盘点与内容、训练、质量的未验证边界；解释 `dataset.py` 的 `-n 1` 还会追加末尾验证 shard，本日离线主线不要求下载，不能用帮助成功或文件存在代替实验通过。

离开前保留本日会话日志与自动结果；若追加重跑，注明会话边界。交流时可附 `git rev-parse HEAD`、`git status --short` 及实际设备/精度以定位证据，不必另填通用表。下一天可直接复用本聊天中已确认的结论与证据路径。共享前检查数据路径和日志是否包含私人信息；若提交成果，只纳入自己的轻量记录与脚本，不提交权重、数据、环境目录或缓存。
