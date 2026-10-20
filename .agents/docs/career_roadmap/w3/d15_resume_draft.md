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

在 `experiments/inference_infra/reports/d15_experiment-log.md` 末尾追加当天日志，记录实际执行日期；标题里的 W3 Dn 是学习顺序，补做时不能倒填为此前已完成。

```text
学习槽位：W3 / 当日标题编号
实际开始与结束：
解释器、git commit、相关源码是否有未提交改动：
命令（完整参数）与退出码：
输入来源 / 产物路径：
观察到的事实：
尚未验证 / 阻塞原因：
下一步 / 下次检查时间：
```

## 今天要留下什么（约 2–3 小时）

今天先做求职材料，不等待 GPU。交付一份可编辑的一页中文简历草稿和证据清单；没有本人教育、联系方式或贡献范围时保留待填，不从目录名猜学校、年级或履历。把材料读给一个没有看过项目的人，应该能分清已完成工作、正在做的实验和计划中的优化。

时间分配：30 分钟核对证据，40 分钟填写事实，40 分钟压缩一页，20 分钟逐句反查。前置是完成 W1 的起点核查；W2 尚未完成时，nanochat 一栏明确写“进行中”。

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

使用[材料准备脚本](scripts/d15_prepare_materials.py)，它只写新模板和证据哈希，不替你验证能力或贡献。

```bash
python .agents/docs/career_roadmap/w3/scripts/d15_prepare_materials.py \
  --kind resume --output-dir experiments/inference_infra/reports/w3_resume_v1 \
  --evidence "$MM_ARCHIVE/reports/reproduction.md" \
  --evidence "$MM_ARCHIVE/results/tables/kv_cache.csv"
```

打开生成的 `d15_resume_draft.md`。按教育、MiniMind、nanochat、技能四部分逐项填写，删除不适用字段。<abbr title="SHA-256 · 文件内容摘要算法；本处用于辨认核查的是哪个文件版本，不证明其中的实验结论正确。">SHA-256</abbr> 只是文件指纹；`experiments/inference_infra/reports/w3_resume_v1/d15_resume_evidence_inventory.json` 的 `exists: true` 不等于所有实验通过。无效路径会保留 `exists: false`，应人工检查，不隐藏。

## 第三步：逐条重写成可追问的贡献

每条按“做了什么 → 如何证明 → 条件和限制”写两行以内。例如只有 CPU 复现证据时，可写“基于 tiny MiniMind 完成 CPU 训练恢复和缓存一致性验证；通过新进程恢复日志与逐步 logits 对照核查”。只有你实际参与并能解释这些过程才采用这句话；如果只是阅读他人产物，应写“复核已有实验”。

nanochat 目前只完成源码阅读时写“梳理 Engine 预填充与解码路径，制定重复测量协议”，不能写“完成高性能推理引擎”。<abbr title="Prefill · 预填充；一次处理整段提示词形成缓存和首个预测。Decode · 解码；随后按 token 逐步生成。">Prefill / decode</abbr> 分别对应整段输入处理和逐步生成，是后续解释性能的基本区分。没有 GPU 实测时明确 `PENDING_GPU`，没有优化对照时删除“提升 X%”。

建议教育 3–4 行、每个项目 3 条、技能 2 行；每条保留一个可解释事实。不要把上游 README 的训练成本、能力分数或机器配置写成本人成绩。

## 第四步：检查一页和导出

先用编辑器 Markdown 预览检查标题、超长路径和换行；有现成浏览器打印功能时选择“打印 → 另存为 PDF”，纸张 A4，关闭页眉页脚，再看是否一页。没有现成导出器时保留可编辑文档即可，不为今天强制安装排版工具。导出之后打开 PDF 核对中文字体、链接、页数，只有真的生成并打开过才记录 PDF 已验收。

## 失败分流与验收

- 模板目录已存在：脚本会拒绝覆盖；继续手工编辑已有草稿，或用 `w3_resume_v2` 新目录。
- 材料太长：删重复技术栈与日程描述；不要删去设备与工作阶段来制造成绩。
- 不知道某条能否写：回到证据原文，分清“亲自完成 / 阅读复核 / 待做”。

验收：每个成绩有对应文件与行；没有未测量的加速比；教育和联系方式已本人核对或保留待填；一页可编辑稿存在。下一天冻结基线条件，以后简历更新只能从真实原始数据提取。

## 本日脚本与产物清单

下列路径以项目根目录为基准；脚本编号保留首次使用日。外部历史归档与上游 checkpoint 原生文件名保留，按正文真实绝对路径或环境变量解析；不冒充项目新产物。<abbr title="条件生成：只有指定分支执行、前置文件满足且实际写入后，文件才算存在；清单中的目标路径本身不证明生成成功。">条件生成</abbr>须在验收时核对。只打印的检查统一进入本日 checks 日志，不另称生成报告。

| 脚本完整项目相对路径 | 用途 | 阶段 | 输入路径或来源 | 运行条件 |
|---|---|---|---|---|
| [.agents/docs/career_roadmap/w3/scripts/d15_prepare_materials.py](scripts/d15_prepare_materials.py) | 生成可编辑材料与文件哈希清单 | 第2步 | 正文 --evidence 指定的真实文件；允许缺失并登记 | Conda nanochat；新输出目录，无需 GPU |

| 产物完整项目相对路径 | 用途 | 生成者/步骤 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/reports/d15_checks.log` | 环境、只读检查、命令输出及错误 | 开始前 Bash 的 exec / tee | 自动捕获后续终端输出；退出码人工记入日志 |
| `experiments/inference_infra/reports/d15_experiment-log.md` | 每日事实、退出码和阻塞 | 人工追加；跨日复用首次文件 | 人工编写 |
| `experiments/inference_infra/reports/w3_resume_v1/d15_resume_draft.md` | 本日可编辑材料正文 | d15_prepare_materials.py 对应 --kind | 自动生成骨架，人工完善与验收 |
| `experiments/inference_infra/reports/w3_resume_v1/d15_resume_evidence_inventory.json` | 输入存在性、SHA-256及首次归属日 | d15_prepare_materials.py | 自动生成；文件存在不等于结论通过 |
| `experiments/inference_infra/reports/w3_resume_v1/d15_resume_draft.pdf` | 一页简历导出 | 编辑器或浏览器打印导出 | 人工导出；可选，未导出不得称存在 |

## 核对记录与完成标记

本文基于当前仓库相关生成、加载、注意力路由及原基准代码编写；源码阅读不能代替执行。本教程的交付状态不表示读者已经完成当天实验。填写实际日志后再勾选：

- [ ] 输入条件与前一天产物已核实。
- [ ] 本日命令在明确设备执行，或具体阻塞已记录。
- [ ] 日志、原始输出与材料保存到所写位置。
- [ ] 示例数值与真实结果已区分，失败和待测未删除。
- [ ] 能说明本日证据证明什么、尚不能证明什么。
