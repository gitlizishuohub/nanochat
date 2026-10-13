# LLM 训练流程与术语

按训练流程分章阅读，术语表按主题分成六篇。正文的术语链接保留悬停释义，点击后进入对应术语页；术语表可返回具体正文位置。

源码依据沿用原文：`source_revision: fbc049f379926ecdb49071a129586278245d71c2`，`review_date: 2026-10-12`。拆分日期：2026-10-13。技术正文、表格、命令和 Mermaid 图保持原内容；本次没有重新核验源码、执行训练或评估。

[项目文档导航](../README.md) · [根目录入口](../../../LLM模型训练完整流程与术语对照.md)

## 训练流程

| 章节 | 内容 |
| --- | --- |
| 1 | [源码依据与调用关系](01-source-references-and-call-flow.md) |
| 2 | [必需主线与实际可选分支](02-training-pipeline-and-optional-branches.md) |
| 3 | [环境与数据准备](03-environment-and-data-preparation.md) |
| 4 | [tokenizer 与对话协议](04-tokenizer-and-conversation-protocol.md) |
| 5 | [模型构建、默认配置、随机初始化和加载](05-model-construction-and-weight-loading.md) |
| 6 | [编码、截断、padding、labels、batch 与分布式数据](06-data-encoding-and-batching.md) |
| 7 | [内部前向、模块作用、张量与预测损失](07-forward-pass-and-prediction-loss.md) |
| 8 | [反向、混合精度、累积、优化器和学习率](08-backpropagation-and-parameter-updates.md) |
| 9 | [日志、权重保存、checkpoint与断点恢复](09-logging-checkpoints-and-resume.md) |
| 10 | [可选GSM8K强化学习](10-gsm8k-reinforcement-learning.md) |
| 11 | [验证、评估、生成与交付](11-evaluation-generation-and-delivery.md) |
| 13 | [交付检查与未验证部分](13-original-validation-and-limitations.md) |

## 术语对照

[术语表说明](12-glossary-index.md)

- [12.1 训练阶段与实现边界](12-1-training-stages-and-implementation-boundaries.md)
- [12.2 数据与对话协议](12-2-data-and-conversation-protocol.md)
- [12.3 批量与分布式](12-3-batching-and-distributed-training.md)
- [12.4 构建与模型前向](12-4-model-construction-and-forward-pass.md)
- [12.5 损失与参数更新](12-5-loss-and-parameter-updates.md)
- [12.6 保存、评估与交付](12-6-checkpoints-evaluation-and-delivery.md)

## 阅读说明

悬停提示、公式和 Mermaid 外观取决于 Markdown 阅读器，本次未进行阅读器显示验收。源码链接继续使用本工作区绝对路径与原行号，换工作区后需调整。第 13 篇是原文在 2026-10-12 的检查记录，其中“本次”“根目录交付”“HEAD 未变”等描述属于原文生成时的历史记录，不代表这次拆分。
