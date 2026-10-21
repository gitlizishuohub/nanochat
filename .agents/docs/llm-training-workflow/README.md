# LLM 训练流程与术语

正文按学习前置关系连续编号；先读主线，再查实现细节。每章先掌握基本过程，标有“深入”的段落可在源码对照时回读。

## 学习主线

| 顺序 | 章节 | 本章解决的问题 |
| --- | --- | --- |
| 1 | [训练主线与可选分支](01-training-pipeline-and-optional-branches.md) | 有哪些阶段，哪些可以跳过？ |
| 2 | [环境与数据准备](02-environment-and-data-preparation.md) | 解释器、数据与缓存从哪里来？ |
| 3 | [分词器与对话协议](03-tokenizer-and-conversation-protocol.md) | 文本如何编码，对话边界如何表示？ |
| 4 | [模型构建、初始化与参数加载](04-model-construction-and-weight-loading.md) | 模型结构、初始化和参数加载有何区别？ |
| 5 | [编码、目标对齐与批量组织](05-data-encoding-and-batching.md) | 如何形成输入、目标和每步批量？ |
| 6 | [模型前向与预测损失](06-forward-pass-and-prediction-loss.md) | 模型如何计算预测分数与损失？ |
| 7 | [反向传播、优化器与训练预算](07-backpropagation-and-parameter-updates.md) | 梯度如何变成参数更新，训练何时结束？ |
| 8 | [混合精度与训练加速](08-precision-and-training-acceleration.md) | 精度和编译如何改变计算方式？ |
| 9 | [多卡分工与优化器通信](09-distributed-optimizer-and-communication.md) | 数据、梯度和优化器状态如何跨卡分工？ |
| 10 | [日志、检查点与断点恢复](10-logging-checkpoints-and-resume.md) | 保存了什么，恢复能接续到什么程度？ |
| 11 | [监督微调：从基础模型到对话模型](11-supervised-fine-tuning.md) | 基础模型如何通过示范变成对话模型？ |
| 12 | [推理、采样与 KV Cache](12-inference-and-generation.md) | 如何选取下一词元并复用历史计算？ |
| 13 | [验证与能力评估](13-evaluation-and-metrics.md) | 各类指标究竟评价什么？ |
| 14 | [可选 GSM8K 强化学习](14-gsm8k-reinforcement-learning.md) | 如何用数学回答奖励进行可选更新？ |
| 15 | [运行示例与模型交付](15-running-examples-and-delivery.md) | 运行前需要什么，最后交付哪些资产？ |

第 8、9 章是更新机制的专题展开：单卡初读先掌握职责和条件，深入算法与通信细节可回查。第 14 章 <abbr title="RL / Reinforcement Learning · 强化学习；利用模型回答的奖励进一步更新参数，本栏目将其作为可选扩展。">RL</abbr> 为可选分支，交付 <abbr title="SFT / Supervised Fine-Tuning · 监督微调；使用示范回答训练已有基础模型，得到对话模型。">SFT</abbr> 模型可以跳过该章。

## 与每日实验配合

- D1–D3：先看第 1 章全景，第 2 章按当前 <abbr title="Conda · 管理独立 Python 解释器和依赖的环境工具；此处使用名为 nanochat 的既有环境。">Conda</abbr> 环境复查。
- D4–D7：配合第 4、6、12 章理解模型前向与两条生成路径。
- D8–D11：配合第 2、3、5、7、10 章准备数据和最小检查点。
- D12–D21：按需查第 12、13、15 章；SFT 和 RL 不作为本期推理路线的必做实验。

每日操作从[W1 D1](../career_roadmap/w1/d01_starting_point.md)进入。这里提供原理与源码解释，不用先通读全部章节再动手。

## 辅助查阅

- [源码导航与调用关系](references/source-references-and-call-flow.md)：按职责找实现。
- [术语目录](glossary/README.md)：九类术语，保留正文往返链接。
- [原文历史验收记录](../project-maintenance/llm-training-workflow-original-validation-2026-10-12.md)：2026-10-12 的文档与静态检查，不属于训练步骤。
- [项目文档导航](../README.md) · [根目录入口](../../../LLM模型训练完整流程与术语对照.md)

## 来源与验证边界

原技术正文依据 `source_revision: fbc049f379926ecdb49071a129586278245d71c2`，核对日期为 2026-10-12；2026-10-13 拆分，2026-10-15 调整术语与图示。2026-10-21 按学习依赖重排、拆分职责并新增 SFT 流程说明，仅对新增流程做局部源码核对，没有重新验收所有默认值或执行训练、GPU 实验。

英文文件名与中文正文保留。源码链接含本工作区绝对路径和历史行号；阅读时核对当前实现。环境操作沿用每日教程的 Conda 入口，上游 uv 示例只作独立环境参考。悬停、公式和 Mermaid 显示取决于阅读器，源文本检查不等于实际渲染验收。
