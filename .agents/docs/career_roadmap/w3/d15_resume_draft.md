---
source_files:
  - .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py
  - .agents/docs/career_roadmap/w3/scripts/d15_prepare_materials.py
doc_type: tutorial
knowledge_role: supporting
knowledge_scope: W3第1日学习与实践
---

# W3 D1 · 证据驱动的一页简历初稿

[原总计划](../nanochat_daily_execution_2026-10-01.md) · [下一日](d16_baseline_matrix.md)。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[3.2 角色、工具与监督位置](../../llm-training-workflow/03-tokenizer-and-conversation-protocol.md#reading-3-2)；[5.2 SFT：先截对话、再装箱补齐](../../llm-training-workflow/05-data-encoding-and-batching.md#reading-5-2)；[11.1 起点与训练目标](../../llm-training-workflow/11-supervised-fine-tuning.md#reading-11-1)；[11.2 示范数据到输入和目标](../../llm-training-workflow/11-supervised-fine-tuning.md#reading-11-2)。
- **今天重点**：在写贡献之前补齐对话模型的来源：角色协议、助手目标、对话装箱，以及基础权重如何进入监督微调。先读概念和对齐示例，再看串联流程。
- **暂时跳过**：今天只学习 <abbr title="SFT / Supervised Fine-Tuning · 监督微调；用示范对话中的助手目标更新基础模型。">SFT</abbr> 原理，不启动微调；详细优化器继承和停止边界留作 D21 后选读。

**读完能解释**：为什么训练示范里有用户文本，用户目标却可以不计损失？你能把预训练和 SFT 的贡献说清楚吗？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 今天学什么，做到哪里算完成

今天学习把项目理解转成可追问的贡献描述，约 60–90 分钟。**不需要训练、GPU 或 checkpoint**。从 W1/W2 自己已经读懂或运行过的一项工作出发；只有阅读证据，也可以写清阅读所得。MiniMind 归档只是有材料时的可选练习，不是前置。

本日交付是一份能在聊天展示的简历初稿内容。排版、一页 PDF 和额外报告可选；姓名、学校、联系方式缺信息就留待本人填写。

## 1. 先区分“项目有这个功能”和“我做了这件事”

上游实现了缓存，不意味着使用者实现了缓存；运行过一项测试，也只支持测试覆盖的输入与断言。写贡献时需要同时回答三件事：**本人动作是什么、用什么方法检查、结果能支持什么**。

用下面的假设经历练习改写，不把它当作你已完成的事实：

| 已有证据 | 可以怎样写 | 为什么 |
|---|---|---|
| 只阅读 Engine 生成路径 | “梳理 prefill 与逐 token 解码的控制流，解释首个输出与后续间隔的区别” | 能展示源码推演，尚无运行结论 |
| 亲自运行 CPU 随机模型，保留 raw | “在 CPU 随机小模型上复核固定长度生成与逐次计时，保留有效 token 计数和原始记录” | 有运行方法与范围，未声称模型质量 |
| 自己新增了计数检查且测试能发现原缺陷 | “补充……检查，用……反例验证” | 只有实际新增和验证过才采用 |

例如“实现高性能推理引擎，性能提升 4 倍”需要本人实现、可比基线和真实对照；只有 B=4 的一份结果并不足以支持。改成第二行后，面试官仍可追问计数、计时和失败处理，内容并不空。

## 2. 用一个技术问题把描述落到代码

选“吞吐是否含首 token 延迟”。先做纸面推演：B=1，生成三个 token，首个等待 30ms，后两次各 10ms。整体是 `3/0.050=60 token/s`；只算后续部分是 `2/0.020=100 token/s`。两个数字回答不同问题，不能挑更大的当成绩。

再带着这个区别读 [W2 基准的 measure](../w2/scripts/d12_benchmark.py)：

```python
elapsed = sum(times)
# 返回值中的两个表达式：
# tpot_ms = sum(times[1:]) / len(times[1:])  （存在后续间隔时）
# tokens_per_s = sum(counts) * 1000 / elapsed
```

`times[0]` 对应首个输出等待，包含在 `elapsed` 中；`counts` 是实际有效输出数量。<abbr title="TTFT / Time To First Token · 从约定计时起点到第一个输出的等待时间；此包装器从首个 next 前开始，排除模型加载与文本编码。">TTFT</abbr> 的起点必须交代，才能解释“含首 token”。上面的代码说明当前包装器采用哪种口径，但**阅读代码不证明你已执行它**。

现在可以形成一条有内容的阅读描述：“对照基准计时表达式，区分包含首 token 的整体吞吐与后续解码吞吐，说明原始计数与时间范围需要一致。”若已运行，再补自己 raw 中一个例子；无需为写简历另跑一次。

## 3. 从自己已有证据中挑一项

进入仓库，用本机已有环境；终端已自动激活时不重复激活。只读检查不会训练或下载：

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
python -c "import sys; print(sys.executable)"
git status --short
rg --files experiments/inference_infra
} > >(tee -a experiments/inference_infra/reports/d15_checks.log) 2>&1
```

找一份本人实际理解的测试日志、缓存探针或 W2 raw；没有文件就使用本日纸面推演与源码解释。先看结果原文，再给出结论，不能从文件名推断测试通过。历史 MiniMind 材料只有能找到并核查时才采用，不要求补建归档。

按“动作 → 方法 → 范围”在聊天写一条初稿，再用“你具体怎么验证的”追问自己。能回答输入、对照、断言和一个未覆盖的情况，这条才有技术内容。

## 4. 压缩成简历初稿（模板可选）

先写教育、nanochat 项目、技能；有其他真实项目再加入。nanochat 项目先保留 2–3 条，每条只说一个可展示事实。技能对应具体动作，例如“能用 Python 读取 JSONL 并核查有效样本”，比孤立列“熟悉 Python”更容易验证。

需要文件骨架时运行下面的标准库工具；它只生成材料，不启动模型：

```bash
{
python .agents/docs/career_roadmap/w3/scripts/d15_prepare_materials.py \
  --kind resume --output-dir experiments/inference_infra/reports/w3_resume_v1
D15_RC=$?
printf 'materials_exit=%s\n' "$D15_RC"
} > >(tee -a experiments/inference_infra/reports/d15_checks.log) 2>&1
```

打开生成稿，填入刚才核对过的内容。若希望登记文件，可在运行前增加 `--evidence` 和实际存在的路径；清单记录 <abbr title="SHA-256 · 文件内容指纹，用于识别本次核查的版本；不证明实验结论或作者身份。">SHA-256</abbr>，缺失项保留 `exists:false`。对照[材料工具](scripts/d15_prepare_materials.py)的 `inventory` 循环：`is_file()` 决定是否计算哈希；`verification` 明写 `FILE_PRESENT_ONLY`。因此脚本退出 0 只表示材料成功生成。

已有目标文件会被拒绝覆盖，退出 2；继续编辑原稿或改用 `w3_resume_v2`。一页排版和 PDF 可在需要投递时做，未导出就不称 PDF 已完成；不为此安装工具。

## 本日脚本与产物清单

路径相对于项目根目录。表中是条件生成目标；未运行分支不会产生文件。

| 脚本完整路径 | 用途与输入 | 条件 |
|---|---|---|
| [.agents/docs/career_roadmap/w2/scripts/d12_benchmark.py](../w2/scripts/d12_benchmark.py) | 本日只读 measure 的计时表达式 | 无需运行模型 |
| [.agents/docs/career_roadmap/w3/scripts/d15_prepare_materials.py](scripts/d15_prepare_materials.py) | 可选简历骨架；可附真实 evidence 路径 | 标准库，新目标目录 |

| 产物完整路径 | 生成者 | 含义与条件 |
|---|---|---|
| `experiments/inference_infra/reports/d15_checks.log` | 各命令组 tee | 仅记录组内输出，立即捕获的退出码也写入；不持续接管终端 |
| `experiments/inference_infra/reports/w3_resume_v1/d15_resume_draft.md` | 材料工具，之后本人编辑 | 可选文件稿；聊天初稿可替代 |
| `experiments/inference_infra/reports/w3_resume_v1/d15_resume_evidence_inventory.json` | 材料工具 | 可选证据指纹；不表示成绩通过 |
| `experiments/inference_infra/reports/w3_resume_v1/d15_resume_draft.pdf` | 编辑器或浏览器人工导出 | 可选，须实际打开核查后才称已交付 |

## 本日自验证与交流

在聊天展示初稿即可，不要求每日手写报告。以下问题使用本人真实证据或明确标注的推演；未选择训练/GPU扩展不登记欠账。

1. 把“我实现了 KV Cache”改成符合自己参与范围的一句话。别人怎样验证这句话？

   **核对要点**：分清上游实现、阅读解释、亲自运行与自己新增；有一个具体方法和证据入口。

2. 为什么本日 100 token/s 的推演不能直接写成“整体生成速度 100 token/s”？

   **核对要点**：分别指出计数和时间区间，算出包含首 token 的 60；例子不是实测。

3. 展示教育、项目和技能三部分的实际初稿。删去最容易夸大的一句，再给出改句。

   **核对要点**：身份信息待本人核实；CPU、随机模型、历史归档的范围写清；不引用上游成绩为本人成绩。

4. 如果证据清单 `exists:true`，但你答不出对应实验怎么做，下一步应怎样处理？

   **核对要点**：回读原记录，限定为复核或暂删该成绩；哈希不能代替能力证明，不为填简历编造运行经历。

## 完成核对

- [ ] 能用一个算例解释所选技术描述。
- [ ] 初稿动词与本人参与、真实证据一致。
- [ ] 可选材料和实际交付分清，没有未测收益数字。

本文核对的是教程与当前源码；读者完成状态以自己的回答和记录为准。
