---
source_files:
  - nanochat/engine.py
  - nanochat/gpt.py
  - nanochat/checkpoint_manager.py
  - nanochat/flash_attention.py
  - scripts/infer_bench.py
  - tests/test_engine.py
  - tests/test_attention_fallback.py
doc_type: tutorial
knowledge_role: supporting
knowledge_scope: W3第6日实操与验收
---

# W3 · 2026-10-20 · 写出他人能执行的复现入口

[上一日](05_2026-10-19_statistics_audit.md) · [下一日](07_2026-10-21_project_briefing.md)。

## 开始前：固定终端和记录方式

本日命令均从 nanochat 仓库根执行。先打开终端，激活已创建的 Conda 环境；若新终端提示找不到 `conda`，先运行 `source /home/lizishuo/miniconda3/etc/profile.d/conda.sh` 再激活，不创建另一个 `.venv`。

```bash
conda activate nanochat
cd /home/lizishuo/_program_/Project/nanochat
python -c "import sys; print(sys.executable)"
git status --short
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。依赖是否已装好要另查，激活成功不代表 PyTorch 可用。保留已有改动，遇到未认识的变更不要还原。教程脚本保存在本周 `scripts/`，结果写入 `experiments/inference_infra/`，两者不能混为同一类产物。

在 `experiments/inference_infra/reports/experiment-log.md` 末尾追加当天日志，记录实际执行日期；标题里的日期是原计划槽位，补做时不能倒填为当时已完成。

```text
计划槽位：W3 / 当日标题日期
实际开始与结束：
解释器、git commit、相关源码是否有未提交改动：
命令（完整参数）与退出码：
输入来源 / 产物路径：
观察到的事实：
尚未验证 / 阻塞原因：
下一步 / 下次检查时间：
```

## 今天的目标与边界（约 2–3 小时）

本日交付 `experiments/inference_infra/reports/w3_reproduction_v1/reproduction_entry.md`，写清项目目的、模型来源、最短运行步骤和当前结果状态，让读者能从明确的前置条件独立复现。后续公开项目入口可引用该正文；本日先验证正文中的实际命令与文件。

前置：W2 环境与模型记录、W3 原始日志和统计状态。<abbr title="Quickstart · 最短复现步骤；从明确的起始环境开始，用最少命令得到可检查产物，不等于完整训练流程。">Quickstart</abbr> 必须写读者实际需要准备的条件，不能假定有你的个人 shell 变量或缓存。

## 第一步：创建复现正文草稿

使用[材料模板工具](scripts/prepare_materials.py)，附上你实际拥有的证据。没有 baseline.csv 时省略该 evidence，正文写待测。

```bash
python .agents/docs/career_roadmap/w3/scripts/prepare_materials.py \
  --kind entry --output-dir experiments/inference_infra/reports/w3_reproduction_v1 \
  --evidence experiments/inference_infra/results/summary/baseline.csv \
  --evidence experiments/inference_infra/protocol.md
```

打开生成的 `reproduction_entry.md`，不要只留下方括号模板。脚本检查文件与哈希不检查内容真实性，`exists:false` 必须处理或作为缺口说明。写清项目目的是建立 nanochat 推理基线和后续受控优化验证，当前不是已完成自研模型或部署服务。

## 第二步：填硬件、模型和数据依赖

逐项填写解释器路径、Python/PyTorch版本、CPU/GPU型号、内存/显存、compute dtype、后端、相关 commit。说明 checkpoint 来自 W2 哪一步，若是小数据 smoke 就明确不保证语言质量。随机模型只验证契约，不能写成训练产物；MiniMind 权重不兼容 nanochat。

<abbr title="Tokenizer · 分词器；把文本映射为模型词表中的token ID。词表大小相同不保证ID语义映射相同，复现必须匹配实际训练使用的tokenizer。">tokenizer</abbr> 是模型输入契约的一部分。说明 `NANOCHAT_BASE_DIR`、tokenizer 和 checkpoint 存放位置，尤其要标哪些文件不在 Git 中以及合法来源。此处不添加自动下载逻辑，不把用户私有数据或完整个人路径写进可公开材料。

## 第三步：把最短命令写到可以逐行复制

依次写：激活环境 → 进入项目 → 设置真实缓存根目录 → 检查 meta/权重 → matrix 默认计划 → 选 B 执行 → aggregate。直接链接[矩阵驱动](scripts/matrix.py)、[统计审计](scripts/aggregate.py)与[W2基准](../w2/scripts/benchmark.py)，每条命令用真实运行过的参数。

不要把整份训练脚本复制进复现正文。下载与训练不是今天短路径的默认步骤；在“从零准备”段引用 W2 指导，注明训练耗时/资源未知时仍待实测。GPU 不可用时提供 CPU 短输入命令并明确它证明什么，不要求读者先去租卡。

记录预计输出文件：matrix 的 12 个状态、每 case raw JSONL、log、baseline.csv、audit。说明 `--execute` 才启动 benchmark，matrix 默认仅计划；说明 `PROCESS_OK` 与有效正式样本数是不同验收。

## 第四步：用新终端核查独立性

打开一个新终端，按正文从第一行开始，先执行无需 GPU 的路径。不要把旧终端里的 META、MODEL_TAG 等变量自动视为存在；缺变量应在正文加定义。<abbr title="Exit code · 进程退出码；0通常表示程序正常结束，非0表示失败，但0不替代实验有效性判断。">退出码</abbr> 与 stdout/stderr 一起记录。

运行真实命令后记录解释器、起始目录、输入是否已缓存和新产物目录。若今天资源不允许重跑 GPU，只验证默认计划、--help、CPU路径，正文写“GPU执行入口已准备，尚未本日复测”。不能把代码检查称为 GPU 复现成功。

## 第五步：请读者回答四个问题

1. 我需要什么硬件、依赖和外部文件？
2. 我从哪个目录执行哪条命令？
3. 什么输出算成功，什么状态表示仍待 GPU？
4. 出错时看哪个 log，能否从 raw 重建统计？

自己暂代读者时也逐题作答；答不出来的地方就是需要补充的正文。将“我这里能跑”改成可以验证的文件、条件和命令。

## 失败分流与验收

正文太长时保留最短成功路径和失败入口，细节引用每日教程；不要删掉模型来源和失败限制。环境路径只适用本机时标明需要替换的项，不将本地绝对路径当普遍命令。模板中的待填项未完成就不能声称可由他人独立复现。

验收：一份完整复现正文、一次新终端 CPU/离线检查记录、所有结果状态真实。下一天把这些证据压缩成三分钟介绍与一张证据图。

## 核对记录与完成标记

本文基于当前仓库相关生成、加载、注意力路由及原基准代码编写；源码阅读不能代替执行。本教程的交付状态不表示读者已经完成当天实验。填写实际日志后再勾选：

- [ ] 输入条件与前一天产物已核实。
- [ ] 本日命令在明确设备执行，或具体阻塞已记录。
- [ ] 日志、原始输出与材料保存到所写位置。
- [ ] 示例数值与真实结果已区分，失败和待测未删除。
- [ ] 能说明本日证据证明什么、尚不能证明什么。
