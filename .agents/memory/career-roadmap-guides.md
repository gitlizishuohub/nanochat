---
name: career-roadmap-guides
description: 维护 nanochat W1–W3 每日教程及配套实验脚本时召回目录和阅读方式约定。
type: feedback
created: 2026-10-15
updated: 2026-10-15
---

# W1–W3 教程组织约定

用户决定：将 `.agents/docs/career_roadmap/nanochat_daily_execution_2026-10-01.md` 的 W1–W3 展开为三个周目录内的详细每日指导；每周实验脚本放在该周 `scripts/`。不增加周 README，指导正文在实际操作位置直接用相对链接指向脚本，并说明运行方式、结果解释和验收。

环境选择：用户在同一任务中明确要求先创建 Conda 环境，并命名为 `nanochat`。后续教程以 `conda activate nanochat` 后运行 `python` 为主；不要默认切回项目 `.venv` 或使用会另建 `.venv` 的 `uv run`。

Why: 用户要求每天可以手把手操作，明确纠正了额外 README 中转层；脚本应作为实际交付文件，而非只提供代码片段。

How to apply: 维护 `w1/`、`w2/`、`w3/` 时按学习依赖编号，每天完整解释前置、步骤、失败分流和产物。指导与脚本留在周目录，实际执行产生的正式成果放在 `experiments/inference_infra/`；原日历不表示对应任务已经完成。脚本验证与真实模型/GPU 实测分别报告。

Evidence: 2026-10-15 本聊天用户明确要求“有实验脚本也要给试验脚本，放在每周的脚本目录中”和“readme不合要求……指导文件中直接链接指向脚本即可”，随后补充“先创建适用于nanochat的虚拟环境”“conda命名为nanochat”；原总计划位于 `.agents/docs/career_roadmap/nanochat_daily_execution_2026-10-01.md`。

Invalidates: 用户明确改变文档结构、交付范围或实验产物目录时更新；源计划和源码变化后重新核对教程命令。
