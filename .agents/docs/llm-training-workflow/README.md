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

推荐随 D1–D21 学习：每天打开当天教程的“当天的流程阅读”，按指定小节读，再动手核验。每天约 10–20 分钟，回读以解决当天问题为止；不要先读完 15 章再开始 D1。表中只列本日实际需要的小节。

| 每日教程 | 流程小节 | 阅读重点 |
| --- | --- | --- |
| [D1](../career_roadmap/w1/d01_starting_point.md#当天的流程阅读) | [1.1](01-training-pipeline-and-optional-branches.md#reading-1-1) | 先读开头的阶段表和主线图，把分词器、基础训练、对话微调、评估和生成放到同一张图上。 |
| [D2](../career_roadmap/w1/d02_environment_inventory.md#当天的流程阅读) | [2.1](02-environment-and-data-preparation.md#reading-2-1) | 只读本地环境入口，并从依赖参考中找出解释器、设备、数据根目录和计算精度分别由什么决定。 |
| [D3](../career_roadmap/w1/d03_inference_smoke.md#当天的流程阅读) | [4.1](04-model-construction-and-weight-loading.md#reading-4-1)；[4.2](04-model-construction-and-weight-loading.md#reading-4-2) | 掌握配置、随机初始化、已有权重三个概念，理解随机小模型为什么也能做接口测试。 |
| [D4](../career_roadmap/w1/d04_model_shapes.md#当天的流程阅读) | [6.1](06-forward-pass-and-prediction-loss.md#reading-6-1)；[6.2](06-forward-pass-and-prediction-loss.md#reading-6-2) | 先沿 [B,T] → [B,T,C] → [B,T,V] 走通，再对照完整形状表认识注意力和前馈层。 |
| [D5](../career_roadmap/w1/d05_cache_lifecycle.md#当天的流程阅读) | [12.1](12-inference-and-generation.md#reading-12-1)；[12.2](12-inference-and-generation.md#reading-12-2) | 对照今天的缓存探查，跟踪提示预填充、缓存复制、新词元解码和位置推进；区分缓存状态与可学习权重。 |
| [D6](../career_roadmap/w1/d06_attention_backends.md#当天的流程阅读) | [6.2](06-forward-pass-and-prediction-loss.md#reading-6-2)；[8.1](08-precision-and-training-acceleration.md#reading-8-1) | 带着昨天的缓存路径回读因果掩码和滑动窗口；精度小节只区分参数、计算及损失的数值类型。 |
| [D7](../career_roadmap/w1/d07_benchmark_audit.md#当天的流程阅读) | [13.2](13-evaluation-and-metrics.md#reading-13-2)；[12.2](12-inference-and-generation.md#reading-12-2) | 先分清能力指标与系统性能，再回查终止词元、返回序列和有效生成长度，帮助审计计时边界。 |
| [D8](../career_roadmap/w2/d08_data_prerequisites.md#当天的流程阅读) | [2.2](02-environment-and-data-preparation.md#reading-2-2)；[3.1](03-tokenizer-and-conversation-protocol.md#reading-3-1) | 理解训练/验证分片、分词器文件和每词元字节数文件各自的来源与用途，再核对今天脚本找到的真实路径。 |
| [D9](../career_roadmap/w2/d09_token_roundtrip.md#当天的流程阅读) | [5.1](05-data-encoding-and-batching.md#reading-5-1)；[5.3](05-data-encoding-and-batching.md#reading-5-3) | 对照 round-trip 和真实 loader 批次，手工完成一次输入/目标错位，并分清每卡行数、序列长度与有效批量。 |
| [D10](../career_roadmap/w2/d10_training_smoke.md#当天的流程阅读) | [6.3](06-forward-pass-and-prediction-loss.md#reading-6-3)；[7.1](07-backpropagation-and-parameter-updates.md#reading-7-1)；[7.4](07-backpropagation-and-parameter-updates.md#reading-7-4) | 把昨日目标对齐接到有效目标损失、反向、梯度累积和参数更新，结合两步训练区分 backward 与 optimizer.step。 |
| [D11](../career_roadmap/w2/d11_checkpoint_generation.md#当天的流程阅读) | [4.3](04-model-construction-and-weight-loading.md#reading-4-3)；[10.2](10-logging-checkpoints-and-resume.md#reading-10-2)；[10.3](10-logging-checkpoints-and-resume.md#reading-10-3) | 对照本次保存与重新加载的真实文件，分清模型配置、参数、优化器状态和恢复状态。 |
| [D12](../career_roadmap/w2/d12_benchmark_schema.md#当天的流程阅读) | [10.1](10-logging-checkpoints-and-resume.md#reading-10-1)；[13.2](13-evaluation-and-metrics.md#reading-13-2) | 了解训练日志的耗时与吞吐口径，再对照今天的推理协议，明确计时起止、分母、预热和失败行。 |
| [D13](../career_roadmap/w2/d13_repeated_benchmark.md#当天的流程阅读) | [12.2](12-inference-and-generation.md#reading-12-2)；[10.1](10-logging-checkpoints-and-resume.md#reading-10-1) | 回读实际输出长度、终止条件与日志统计范围，逐项对应今天的预热、正式记录和自动汇总。 |
| [D14](../career_roadmap/w2/d14_matrix_smoke.md#当天的流程阅读) | [5.3](05-data-encoding-and-batching.md#reading-5-3)；[9.1](09-distributed-optimizer-and-communication.md#reading-9-1)；[9.2](09-distributed-optimizer-and-communication.md#reading-9-2) | 先复盘批量的定义，再概览多卡的数据与优化器分工，区分训练跨卡扩展和今天同提示的多样本生成。 |
| [D15](../career_roadmap/w3/d15_resume_draft.md#当天的流程阅读) | [3.2](03-tokenizer-and-conversation-protocol.md#reading-3-2)；[5.2](05-data-encoding-and-batching.md#reading-5-2)；[11.1](11-supervised-fine-tuning.md#reading-11-1)；[11.2](11-supervised-fine-tuning.md#reading-11-2) | 在写贡献之前补齐对话模型的来源：角色协议、助手目标、对话装箱，以及基础权重如何进入监督微调。 |
| [D16](../career_roadmap/w3/d16_baseline_matrix.md#当天的流程阅读) | [4.1](04-model-construction-and-weight-loading.md#reading-4-1)；[4.3](04-model-construction-and-weight-loading.md#reading-4-3)；[12.2](12-inference-and-generation.md#reading-12-2) | 把模型配置、所选权重、配套词表和缓存容量连起来，核对今天固定的输入长度、生成长度与样本数。 |
| [D17](../career_roadmap/w3/d17_single_batch_runs.md#当天的流程阅读) | [8.1](08-precision-and-training-acceleration.md#reading-8-1)；[8.3](08-precision-and-training-acceleration.md#reading-8-3) | 结合今天的真实设备和 dtype，区分精度选择、编译准备与正式测量。 |
| [D18](../career_roadmap/w3/d18_batch_runs.md#当天的流程阅读) | [12.2](12-inference-and-generation.md#reading-12-2)；[5.3](05-data-encoding-and-batching.md#reading-5-3) | 回看一次提示预填充后复制到多样本行的行为，结合有效输出数理解批量对总耗时、总吞吐及缓存容量的影响。 |
| [D19](../career_roadmap/w3/d19_statistics_audit.md#当天的流程阅读) | [13.1](13-evaluation-and-metrics.md#reading-13-1)；[13.2](13-evaluation-and-metrics.md#reading-13-2) | 完整认识四种能力评价出口，再回到当天统计复算，明确损失、任务正确率、推理延迟和吞吐各自的统计对象。 |
| [D20](../career_roadmap/w3/d20_reproduction_entry.md#当天的流程阅读) | [15.1](15-running-examples-and-delivery.md#reading-15-1)；[15.3](15-running-examples-and-delivery.md#reading-15-3) | 把复现入口需要的环境、模型、词表、配置和结果范围列全，逐项对应本项目已有文件与仍未执行的条件。 |
| [D21](../career_roadmap/w3/d21_project_briefing.md#当天的流程阅读) | [1.1](01-training-pipeline-and-optional-branches.md#reading-1-1)；[11.5](11-supervised-fine-tuning.md#reading-11-5)；[14.1](14-gsm8k-reinforcement-learning.md#reading-14-1)；[15.3](15-running-examples-and-delivery.md#reading-15-3) | 先用主线图串讲已做的实验，再把 SFT 产物与可选数学奖励训练放到正确位置。 |

阶段目标：D1–D7 认识全景、模型和缓存；D8–D14 接通数据、训练、保存与可信测量；D15–D21 补齐对话阶段、评价及交付的概念，并复用已读内容解释实验。

到 D21 应能串讲流程并指出证据边界，不要求完成全部深入内容。之后按方向补读：第 4 章初始化数值、第 7 章优化器算法和预算公式、第 8 章 FP8、第 9 章多卡通信、第 11 章 SFT 继承与停止细节、第 14 章 RL 更新公式。术语表始终按需查阅。这些是后续阅读方向，不表示要启动相应训练或 GPU 实验。

## 辅助查阅

- [源码导航与调用关系](references/source-references-and-call-flow.md)：按职责找实现。
- [术语目录](glossary/README.md)：九类术语，保留正文往返链接。
- [原文历史验收记录](../project-maintenance/llm-training-workflow-original-validation-2026-10-12.md)：原文的文档与静态检查，不属于训练步骤。
- [项目文档导航](../README.md) · [根目录入口](../../../LLM模型训练完整流程与术语对照.md)

## 来源与验证边界

原技术正文依据 `source_revision: fbc049f379926ecdb49071a129586278245d71c2`，后续拆分正文、调整术语与图示，按学习依赖重排章节、划分职责，并新增 SFT 流程说明，仅对新增流程做局部源码核对，没有重新验收所有默认值或执行训练、GPU 实验。

英文文件名与中文正文保留。源码链接含本工作区绝对路径和历史行号；阅读时核对当前实现。环境操作沿用每日教程的 Conda 入口，上游 uv 示例只作独立环境参考。悬停、公式和 Mermaid 显示取决于阅读器，源文本检查不等于实际渲染验收。
