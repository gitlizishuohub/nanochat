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
knowledge_scope: W3第1日实操与验收
---

# W3 D1 · 证据驱动的一页简历初稿

[原总计划](../nanochat_daily_execution_2026-10-01.md) · [下一日](d16_baseline_matrix.md)。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[3.2 角色、工具与监督位置](../../llm-training-workflow/03-tokenizer-and-conversation-protocol.md#reading-3-2)；[5.2 SFT：先截对话、再装箱补齐](../../llm-training-workflow/05-data-encoding-and-batching.md#reading-5-2)；[11.1 起点与训练目标](../../llm-training-workflow/11-supervised-fine-tuning.md#reading-11-1)；[11.2 示范数据到输入和目标](../../llm-training-workflow/11-supervised-fine-tuning.md#reading-11-2)。
- **今天重点**：在写贡献之前补齐对话模型的来源：角色协议、助手目标、对话装箱，以及基础权重如何进入监督微调。先读概念和对齐示例，再看串联流程。
- **暂时跳过**：今天只学习 <abbr title="SFT / Supervised Fine-Tuning · 监督微调；用示范对话中的助手目标更新基础模型。">SFT</abbr> 原理，不启动微调；详细优化器继承和停止边界留作 D21 后选读。

**读完能解释**：为什么训练示范里有用户文本，用户目标却可以不计损失？你能把预训练和 SFT 的贡献说清楚吗？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 开始前：固定终端和记录方式

本日命令均从 nanochat 仓库根执行。先打开终端，激活已创建的 Conda 环境；若新终端提示找不到 `conda`，先运行 `source /home/lizishuo/miniconda3/etc/profile.d/conda.sh` 再激活，不创建另一个 `.venv`。

```bash
conda activate nanochat
cd /home/lizishuo/_program_/Project/nanochat
python -c "import sys; print(sys.executable)"
git status --short
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
exec > >(tee -a experiments/inference_infra/reports/d15_checks.log) 2>&1
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。依赖是否已装好要另查，激活成功不代表 PyTorch 可用。保留已有改动，遇到未认识的变更不要还原。教程脚本保存在本周 `scripts/`，结果写入 `experiments/inference_infra/`，两者不能混为同一类产物。

本日验证完成后，直接按文末“本日自验证与交流”逐题和我交流，附关键输出或自动日志/原始结果路径即可。无需手写报告，也不要求固定排版或长文。实际日期、设备、命令参数与退出码以日志和聊天核对；标题里的 W3 Dn 是学习顺序，不倒填完成状态。若想自行整理，可选用 `experiments/inference_infra/reports/d15_experiment-log.md`，后续跨日也可引用本次聊天和自动结果。

## 今天要留下什么（约 2–3 小时）

今天先做求职材料，不等待 GPU。练习形成一页中文简历初稿和证据对应关系；可先在聊天逐条核对事实，再按需整理可编辑稿；没有本人教育、联系方式或贡献范围时保留待填，不从目录名猜学校、年级或履历。把材料读给一个没有看过项目的人，应该能分清已完成工作、正在做的实验和计划中的优化。

时间分配：30 分钟核对证据，40 分钟交流贡献表述，40 分钟试着压缩一页，20 分钟逐句反查；正式排版与报告整理可选。前置是完成 W1 的起点核查；W2 尚未完成时，nanochat 一栏明确写“进行中”。

## 第一步：找到归档后的真实 MiniMind 证据

原总计划中的 `experiments/minimind_baseline/` 已迁移。当前本机证据位于相邻 MiniMind 仓库的 `experiments/archive/minimind_baseline-before-embedding-20261014/`。这份归档代表之前实验，不能当成新源码已再次通过验收。

```bash
MM_ARCHIVE=/home/lizishuo/_program_/Project/minimind/experiments/archive/minimind_baseline-before-embedding-20261014
sed -n '1,100p' "$MM_ARCHIVE/reports/reproduction.md"
head -n 12 "$MM_ARCHIVE/results/tables/kv_cache.csv"
head -n 5 "$MM_ARCHIVE/results/tables/generation_benchmark_raw.csv"
```

先读复现报告，确认它声明的设备是 CPU，再看 KV 表是否同时有 token 对照与 logits 误差。<abbr title="KV Cache / Key-Value Cache · 键值缓存；保存历史位置的注意力 K/V，减少生成时重复计算。本文已有 MiniMind 的一致性证据不能替代 nanochat 的验证。">KV Cache</abbr> 是重复利用历史计算的状态，正确性证据比“能生成一句话”更严格。归档 README 记录的 8 步 token 一致、最大 logits 差约 `2.38e-6`，只可在确认对应 CSV 之后引用；不把归档摘要的数字不经核实复制到简历。

若路径不存在：用 `rg --files /home/lizishuo/_program_/Project/minimind/experiments` 查找报告；找到新位置后登记迁移。确实缺失则写“证据待补”，今天仍可完成没有成绩数字的初稿。

## 第二步：生成可填写草稿与文件存在性清单

需要整理为文件时，可选使用[材料准备脚本](scripts/d15_prepare_materials.py)；直接聊天核对也可以。它只写新模板和证据哈希，不替你验证能力或贡献。

```bash
python .agents/docs/career_roadmap/w3/scripts/d15_prepare_materials.py \
  --kind resume --output-dir experiments/inference_infra/reports/w3_resume_v1 \
  --evidence "$MM_ARCHIVE/reports/reproduction.md" \
  --evidence "$MM_ARCHIVE/results/tables/kv_cache.csv"
```

脚本可选用于整理简历；未使用时可直接在聊天按教育、MiniMind、nanochat、技能四部分核对初稿与证据。采用文件草稿时打开生成的 `d15_resume_draft.md`，逐项填写并删除不适用字段。<abbr title="SHA-256 · 文件内容摘要算法；本处用于辨认核查的是哪个文件版本，不证明其中的实验结论正确。">SHA-256</abbr> 只是文件指纹；`experiments/inference_infra/reports/w3_resume_v1/d15_resume_evidence_inventory.json` 的 `exists: true` 不等于所有实验通过。无效路径会保留 `exists: false`，应人工检查，不隐藏。

## 第三步：逐条重写成可追问的贡献

每条按“做了什么 → 如何证明 → 条件和限制”写两行以内。例如只有 CPU 复现证据时，可写“基于 tiny MiniMind 完成 CPU 训练恢复和缓存一致性验证；通过新进程恢复日志与逐步 logits 对照核查”。只有你实际参与并能解释这些过程才采用这句话；如果只是阅读他人产物，应写“复核已有实验”。

nanochat 目前只完成源码阅读时写“梳理 Engine 预填充与解码路径，制定重复测量协议”，不能写“完成高性能推理引擎”。<abbr title="Prefill · 预填充；一次处理整段提示词形成缓存和首个预测。Decode · 解码；随后按 token 逐步生成。">Prefill / decode</abbr> 分别对应整段输入处理和逐步生成，是后续解释性能的基本区分。没有 GPU 实测时明确 `PENDING_GPU`，没有优化对照时删除“提升 X%”。

建议教育 3–4 行、每个项目 3 条、技能 2 行；每条保留一个可解释事实。不要把上游 README 的训练成本、能力分数或机器配置写成本人成绩。

## 第四步：按需检查一页和导出（可选）

先用编辑器 Markdown 预览检查标题、超长路径和换行；有现成浏览器打印功能时选择“打印 → 另存为 PDF”，纸张 A4，关闭页眉页脚，再看是否一页。没有现成导出器时保留可编辑文档即可，不为今天强制安装排版工具。导出之后打开 PDF 核对中文字体、链接、页数，只有真的生成并打开过才记录 PDF 已验收。

## 失败分流与验收

- 模板目录已存在：脚本会拒绝覆盖；继续手工编辑已有草稿，或用 `w3_resume_v2` 新目录。
- 材料太长：删重复技术栈与日程描述；不要删去设备与工作阶段来制造成绩。
- 不知道某条能否写：回到证据原文，分清“亲自完成 / 阅读复核 / 待做”。

验收：每个成绩有对应文件与行；没有未测量的加速比；教育和联系方式已本人核对或保留待填；能给出可追问的简历初稿内容；一页可编辑稿和 PDF 可在需要投递时整理。下一天冻结基线条件，以后简历更新只能从真实原始数据提取。

## 本日脚本与产物清单

下列路径以项目根目录为基准；脚本编号保留首次使用日。外部历史归档与上游 checkpoint 原生文件名保留，按正文真实绝对路径或环境变量解析；不冒充项目新产物。<abbr title="条件生成：只有指定分支执行、前置文件满足且实际写入后，文件才算存在；清单中的目标路径本身不证明生成成功。">条件生成</abbr>须在验收时核对。只打印的检查统一进入本日 checks 日志，不另称生成报告。

| 脚本完整项目相对路径 | 用途 | 阶段 | 输入路径或来源 | 运行条件 |
|---|---|---|---|---|
| [.agents/docs/career_roadmap/w3/scripts/d15_prepare_materials.py](scripts/d15_prepare_materials.py) | 生成可编辑材料与文件哈希清单 | 第2步 | 正文 --evidence 指定的真实文件；允许缺失并登记 | Conda nanochat；新输出目录，无需 GPU |

| 产物完整项目相对路径 | 用途 | 生成者/步骤 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/reports/d15_checks.log` | 环境、只读检查、命令输出及错误 | 开始前 Bash 的 exec / tee | 自动捕获后续终端输出；退出码在聊天核对，或可选补记日志 |
| `experiments/inference_infra/reports/d15_experiment-log.md` | 每日事实、退出码和阻塞 | 自愿整理；跨日可引用聊天和自动结果 | 可选人工笔记，不影响验收 |
| `experiments/inference_infra/reports/w3_resume_v1/d15_resume_draft.md` | 本日可编辑材料正文 | d15_prepare_materials.py 对应 --kind | 选择材料脚本后自动生成；正式文件整理可选，先核对内容 |
| `experiments/inference_infra/reports/w3_resume_v1/d15_resume_evidence_inventory.json` | 输入存在性、SHA-256及首次归属日 | d15_prepare_materials.py | 选择材料脚本后自动生成；文件存在不等于结论通过 |
| `experiments/inference_infra/reports/w3_resume_v1/d15_resume_draft.pdf` | 一页简历导出 | 编辑器或浏览器打印导出 | 人工导出；可选，未导出不得称存在 |

## 本日自验证与交流

按以下问题自行核查后直接发我；每项给关键输出、文件/行号或原始字段即可。没有执行的部分说明原因和未验证范围，不要求写成报告。

1. 从简历或口头初稿中选一条 MiniMind 成绩：对应哪个归档文件、哪一行？设备和源码版本是什么，归档结果能否证明当前源码也通过？
2. 哪些工作是你亲自完成，哪些只是阅读复核？分别给出一句准确的贡献表述，并指出支持它的日志或原始记录。
3. nanochat 当前能写到哪个阶段？检查是否出现未经实测的 GPU 性能、加速比或上游成绩，说明你删改了哪句话及理由。
4. 本人信息、证据或一页排版还有哪些未核对？直接说待填或未验证；PDF 未导出时不算已有产物。

## 核对记录与完成标记

本文基于当前仓库相关生成、加载、注意力路由及原基准代码编写；源码阅读不能代替执行。本教程的交付状态不表示读者已经完成当天实验。按当天问题核对真实证据后再勾选；聊天回答即可，无需额外报告：

- [ ] 输入条件与前一天产物已核实。
- [ ] 本日命令在明确设备执行，或具体阻塞已记录。
- [ ] 自动日志和原始输出可定位，实践材料或聊天回答可回溯；可选笔记无需生成。
- [ ] 示例数值与真实结果已区分，失败和待测未删除。
- [ ] 能说明本日证据证明什么、尚不能证明什么。
