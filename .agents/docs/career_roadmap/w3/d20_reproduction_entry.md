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

# W3 D6 · 写出他人能执行的复现入口

[上一日](d19_statistics_audit.md) · [下一日](d21_project_briefing.md)。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[15.1 运行前的条件](../../llm-training-workflow/15-running-examples-and-delivery.md#reading-15-1)；[15.3 交付资产与部署边界](../../llm-training-workflow/15-running-examples-and-delivery.md#reading-15-3)。
- **今天重点**：把复现入口需要的环境、模型、词表、配置和结果范围列全，逐项对应本项目已有文件与仍未执行的条件。
- **暂时跳过**：15.2 只对照入口与前置条件，不复制运行整套训练；不存在的部署形态不写成已交付。

**读完能解释**：一个同学拿到你的复现入口，还必须拿到哪些资产，才能运行同一模型并复核同一结果？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 开始前：固定终端和记录方式

本日命令均从 nanochat 仓库根执行。当前终端会自动激活 nanochat 环境，无需重复激活；先核对解释器路径，不创建另一个 `.venv`。

日志只在 `{ ...; }` 命令组执行期间记录；<abbr title="Bash process substitution · 进程替换。这里把命令组的输出交给 tee，同时显示并追加到日志；重定向在命令组结束后恢复。">进程替换</abbr>不会持续接管交互终端，因此提示符和 VS Code 终端集成控制字符不会混入日志。命令组在当前 shell 执行，变量设置会保留；代码块返回最后一条命令的退出码，需核对单条命令时立即记录 `$?`。已有的显式文件重定向仍写入各自文件。复制时包含外层花括号和末尾重定向。若之前执行过旧版全局日志命令，先关闭旧终端并新开终端；仅修改文档不会停止旧终端的记录。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
{
python -c "import sys; print(sys.executable)"
git status --short
} > >(tee -a experiments/inference_infra/reports/d20_checks.log) 2>&1
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。依赖是否已装好要另查，激活成功不代表 PyTorch 可用。保留已有改动，遇到未认识的变更不要还原。教程脚本保存在本周 `scripts/`，结果写入 `experiments/inference_infra/`，两者不能混为同一类产物。

本日验证完成后，直接按文末“本日自验证与交流”逐题和我交流，附关键输出或自动日志/原始结果路径即可。无需手写报告，也不要求固定排版或长文。实际日期、设备、命令参数与退出码以日志和聊天核对；标题里的 W3 Dn 是学习顺序，不倒填完成状态。若想自行整理，可选用 `experiments/inference_infra/reports/d15_experiment-log.md`，后续跨日也可引用本次聊天和自动结果。

## 今天的目标与边界（约 2–3 小时）

本日交付 `experiments/inference_infra/reports/w3_reproduction_v1/d20_reproduction_entry.md`，写清项目目的、模型来源、最短运行步骤和当前结果状态，让读者能从明确的前置条件独立复现。后续公开项目入口可引用该正文；本日先验证正文中的实际命令与文件。

前置：W2 环境与模型记录、W3 原始日志和统计状态；此前人工报告可用聊天和自动记录替代。复现入口本身是供他人操作的实践材料，保留完整可执行命令，不额外附写实验报告。<abbr title="Quickstart · 最短复现步骤；从明确的起始环境开始，用最少命令得到可检查产物，不等于完整训练流程。">Quickstart</abbr> 必须写读者实际需要准备的条件，不能假定有你的个人 shell 变量或缓存。

## 第一步：创建复现正文草稿

使用[材料模板工具](scripts/d15_prepare_materials.py)，附上你实际拥有的证据。没有 d19_baseline.csv 时省略该 evidence，正文写待测。

```bash
{
python .agents/docs/career_roadmap/w3/scripts/d15_prepare_materials.py \
  --kind entry --output-dir experiments/inference_infra/reports/w3_reproduction_v1 \
  --evidence experiments/inference_infra/results/summary/d19_baseline.csv \
  --evidence experiments/inference_infra/d12_protocol.md
} > >(tee -a experiments/inference_infra/reports/d20_checks.log) 2>&1
```

`d12_protocol.md` 若未整理成文件，省略对应 evidence 参数，直接从此前聊天、矩阵和原始记录恢复真实条件，不为模板补写报告。打开生成的 `d20_reproduction_entry.md`，不要只留下方括号模板。脚本检查文件与哈希不检查内容真实性，`exists:false` 必须处理或作为缺口说明。写清项目目的是建立 nanochat 推理基线和后续受控优化验证，当前不是已完成自研模型或部署服务。

## 第二步：填硬件、模型和数据依赖

逐项填写解释器路径、Python/PyTorch版本、CPU/GPU型号、内存/显存、compute dtype、后端、相关 commit。说明 checkpoint 来自 W2 哪一步，若是小数据 smoke 就明确不保证语言质量。随机模型只验证契约，不能写成训练产物；MiniMind 权重不兼容 nanochat。

<abbr title="Tokenizer · 分词器；把文本映射为模型词表中的token ID。词表大小相同不保证ID语义映射相同，复现必须匹配实际训练使用的tokenizer。">tokenizer</abbr> 是模型输入契约的一部分。说明 `NANOCHAT_BASE_DIR`、tokenizer 和 checkpoint 存放位置，尤其要标哪些文件不在 Git 中以及合法来源。此处不添加自动下载逻辑，不把用户私有数据或完整个人路径写进可公开材料。

## 第三步：把最短命令写到可以逐行复制

依次写：确认自动激活的解释器 → 进入项目 → 设置真实缓存根目录 → 检查 meta/权重 → matrix 默认计划 → 选 B 执行 → aggregate。直接链接[矩阵驱动](scripts/d16_matrix.py)、[统计审计](scripts/d13_aggregate.py)与[W2基准](../w2/scripts/d12_benchmark.py)，每条命令用真实运行过的参数。

不要把整份训练脚本复制进复现正文。下载与训练不是今天短路径的默认步骤；在“从零准备”段引用 W2 指导，注明训练耗时/资源未知时仍待实测。GPU 不可用时提供 CPU 短输入命令并明确它证明什么，不要求读者先去租卡。

复用并链接第 16–19 日的完整产物清单；新终端复测必须使用新目录和 `--artifact-day 20`，新产生的矩阵、case raw/log 使用 `d20_`，不能覆盖原证据。复测输出目录在聊天或本日检查日志中逐项给出完整路径。说明 `--execute` 才启动 benchmark，matrix 默认仅计划；说明 `PROCESS_OK` 与有效正式样本数是不同验收。

## 第四步：用新终端核查独立性

打开一个新终端，按正文从第一行开始，先执行无需 GPU 的路径。不要把旧终端里的 META、MODEL_TAG 等变量自动视为存在；缺变量应在正文加定义。<abbr title="Exit code · 进程退出码；0通常表示程序正常结束，非0表示失败，但0不替代实验有效性判断。">退出码</abbr> 与 stdout/stderr 一起记录。

运行真实命令后记录解释器、起始目录、输入是否已缓存和新产物目录。若今天资源不允许重跑 GPU，只验证默认计划、--help、CPU路径，正文写“GPU执行入口已准备，尚未本日复测”。不能把代码检查称为 GPU 复现成功。

最低复查可只运行帮助与以下 CPU 短输入；均由本日终端捕获检查输出。它会创建新的 CPU 记录，不覆盖第 17 日证据：

```bash
{
python .agents/docs/career_roadmap/w3/scripts/d16_matrix.py --help
python .agents/docs/career_roadmap/w3/scripts/d13_aggregate.py --help
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py \
  --device cpu --random-model --prompt-tokens 32 --decode-tokens 8 --batch-size 1 \
  --warmup 3 --repeats 5 --seed 42 \
  --output experiments/inference_infra/results/raw/d20_p32_g8_b1_cpu_reproduction.jsonl
} > >(tee -a experiments/inference_infra/reports/d20_checks.log) 2>&1
```

## 第五步：请读者回答四个问题

请读者直接按文末第1、2、4、5题检验正文是否交代清楚运行入口、外部资产、统计重建和失败恢复，不另做一套相同问答。

自己暂代读者时，只看复现正文作答；必须依靠旧终端变量或作者临时解释才能回答的地方，回正文补齐。

## 失败分流与验收

正文太长时保留最短成功路径和失败入口，细节引用每日教程；不要删掉模型来源和失败限制。环境路径只适用本机时标明需要替换的项，不将本地绝对路径当普遍命令。模板中的待填项未完成就不能声称可由他人独立复现。

验收：一份完整复现正文、一次新终端 CPU/离线检查记录、所有结果状态真实。下一天把这些证据压缩成三分钟介绍与一张证据图。

## 本日脚本与产物清单

下列路径以项目根目录为基准；脚本编号保留首次使用日。外部历史归档与上游 checkpoint 原生文件名保留，按正文真实绝对路径或环境变量解析；不冒充项目新产物。<abbr title="条件生成：只有指定分支执行、前置文件满足且实际写入后，文件才算存在；清单中的目标路径本身不证明生成成功。">条件生成</abbr>须在验收时核对。只打印的检查统一进入本日 checks 日志，不另称生成报告。

| 脚本完整项目相对路径 | 用途 | 阶段 | 输入路径或来源 | 运行条件 |
|---|---|---|---|---|
| [.agents/docs/career_roadmap/w3/scripts/d15_prepare_materials.py](scripts/d15_prepare_materials.py) | 生成可编辑材料与文件哈希清单 | 第1步 | 正文 --evidence 指定的真实文件；允许缺失并登记 | Conda nanochat；新输出目录，无需 GPU |
| [.agents/docs/career_roadmap/w3/scripts/d16_matrix.py](scripts/d16_matrix.py) | 复用计划与显式执行入口 | 第3–4步 | 真实 META、模型 tag/step 与协议 | 只 --help 无需模型；生成计划需 meta；执行另需 checkpoint |
| [.agents/docs/career_roadmap/w3/scripts/d13_aggregate.py](scripts/d13_aggregate.py) | 复用统计入口 | 第3–4步 | 已有带 run_id 的原始 JSONL | 无原始行时不生成性能表 |
| [.agents/docs/career_roadmap/w2/scripts/d12_benchmark.py](../w2/scripts/d12_benchmark.py) | 复用 CPU 或真实模型基准 | 第3–4步 | CPU随机夹具或已验收 checkpoint/tokenizer | CPU路径需 PyTorch；GPU路径需 CUDA |

| 产物完整项目相对路径 | 用途 | 生成者/步骤 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/results/raw/d20_p32_g8_b1_cpu_reproduction.jsonl` | 新终端 CPU 复查记录 | d12_benchmark.py | 条件生成；仅执行本日 CPU 命令且成功写入后存在 |
| `experiments/inference_infra/reports/d20_checks.log` | 环境、只读检查、命令输出及错误 | 各 Bash 代码块的 tee | 自动追加各代码块内的输出；退出码在聊天核对，或可选补记日志 |
| `experiments/inference_infra/reports/d15_experiment-log.md` | 每日事实、退出码和阻塞 | 自愿整理；跨日可引用聊天和自动结果 | 可选人工笔记，不影响验收 |
| `experiments/inference_infra/reports/w3_reproduction_v1/d20_reproduction_entry.md` | 本日可编辑材料正文 | d15_prepare_materials.py 对应 --kind | 自动生成骨架，人工完善与验收 |
| `experiments/inference_infra/reports/w3_reproduction_v1/d20_entry_evidence_inventory.json` | 输入存在性、SHA-256及首次归属日 | d15_prepare_materials.py | 自动生成；文件存在不等于结论通过 |

## 本日自验证与交流

按题号在聊天中回答即可，预计交流约 15–25 分钟，不要求另写实验报告。本日的复现入口正文是实际交付材料，仍需完整填写并保存。开头一次说明正文路径与本日复查的日期/版本，后面复用关键日志和 raw；历史成功注明归属，未执行写 `PENDING`。不要求为问答补跑 GPU。

1. 请展示已经填写的复现入口。一个没看过本项目的人，能从正文知道项目现在做到了哪一步，并找到一条最短可执行路径吗？

   **核对要点**：引用实际正文中目的/当前状态、起点、完整命令与成功判据的关键部分，不能只交模板或脚本名；没有 GPU 结果就明确 CPU 路径的作用。必要依赖和待填项不得藏在口头补充里，每日报告与讲稿仍可选。

2. 如果读者只有 Git 仓库，没有你的缓存目录，他缺少哪些资产？正文是否足以让他识别并准备匹配的一套？

   **核对要点**：围绕模型能否重建与正确编码说明权重、meta、tokenizer 的来源和身份固定方法；区分已训练 checkpoint 与随机夹具。指出不在 Git 的资产、读者须替换的本机路径和缺失时的停止点；清单 `exists/hash` 只识别文件，不能代替内容与加载验收。

3. 实际从新终端按正文执行到哪一步？有没有某个变量或前置条件只有旧终端里才存在？

   **核对要点**：附本次关键命令、退出码与输出，核对正文能自行定义缓存根、tag/step 和 META 等必要条件。只运行帮助、计划或 CPU 契约时如实限定；新结果使用独立目标且不覆盖旧证据，不能把旧日志称为本日 GPU 复现。

4. 读者拿到你保留的 raw 后，能否只按正文重建同一统计结果，并反查一条有效样本？

   **核对要点**：给出入口中的聚合命令、明确的输入范围与独立输出位置，实际数据可用时用 CSV/audit/run_id 对应一条记录。解释为何只给 summary 或 matrix 不够；没有测量数据时说明目前只能交付何种执行入口，不虚构 CSV。

5. 假设读者说“命令结束了，但结果不符合成功判据”，你的入口能帮他找到下一步吗？用一个具体失败场景检验。

   **核对要点**：任选缺资产、上下文拒绝、OOM、提前结束或样本不足，说明从哪份 log/raw/status 判断以及何时停止、怎样保留失败后重试；场景可以是假设，无需故意造错。正文有对应指引，不能靠作者临时补命令，也不把退出码为 0 当全部实验通过。

## 核对记录与完成标记

本文基于当前仓库相关生成、加载、注意力路由及原基准代码编写；源码阅读不能代替执行。本教程的交付状态不表示读者已经完成当天实验。按当天问题核对真实证据后再勾选；聊天回答即可，无需额外报告：

- [ ] 输入条件与前一天产物已核实。
- [ ] 本日命令在明确设备执行，或具体阻塞已记录。
- [ ] 自动日志和原始输出可定位，实践材料或聊天回答可回溯；可选笔记无需生成。
- [ ] 示例数值与真实结果已区分，失败和待测未删除。
- [ ] 能说明本日证据证明什么、尚不能证明什么。
