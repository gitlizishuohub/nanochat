# 项目知识文档导航

项目规范入口：[根 AGENTS.md](../../AGENTS.md)。这里保存项目知识文档；实验源码与正式结果沿用项目原布局。

| 主题 | 文档 | 阅读边界 |
|---|---|---|
| 项目全貌与学习路线 | [nanochat 项目详细阅读与实战指南](nanochat项目详细阅读与实战指南.md) | 保留原文；命令、默认值与技术结论需按当前源码复核 |
| LLM 训练流程与术语 | [分章目录](llm-training-workflow/README.md) | 15 章学习主线，术语与源码导航独立查阅；原正文沿用原有核对记录，新增 SFT 流程做了局部静态核对 |
| Agent 配置与迁移验收 | [迁移检查记录](project-maintenance/init-project-review-2026-10-13.md) | 记录布局、真源、验证结果及尚未验证部分 |

以下路线文档尚未提交，保留在本地。读取前确认文件存在：

- `career_roadmap/nanochat_daily_execution_2026-10-01.md`：推理基础设施逐日计划；从旧 `docs/` 迁入，外部 MiniMind 文档链接已按实际位置修复。计划日期和建议不能作为实验已完成的证据。

项目根目录 [LLM模型训练完整流程与术语对照.md](../../LLM模型训练完整流程与术语对照.md) 保留为拆分栏目的简短入口；正文位于 `llm-training-workflow/`。

上游 [README](../../README.md)、`dev/` 中的原有项目文档及 `graphify-out/` 正式生成结果保留原位置。知识图谱 manifest 是旧快照，包含历史路径；本次未重建图谱。
