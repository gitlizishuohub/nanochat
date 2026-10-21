---
name: llm-training-docs
description: 查找 LLM 训练流程与术语文档或维护其拆分栏目时读取。
type: project
created: 2026-10-13
updated: 2026-10-21
---

# LLM 训练文档位置

事实或决策：用户要求将过大的《LLM模型训练完整流程与术语对照.md》在 docs 中新开栏目拆分。正文现位于 `.agents/docs/llm-training-workflow/`，根目录同名文件保留为简短导航入口。用户随后明确要求英文命名，新栏目文件名统一采用英文，中文正文保留。

Why: 原文约 196 KB，术语表占约 85 KB；按主题拆分，方便阅读与局部维护；2026-10-15 按用户要求对齐当前技能，修复排序、标题层级、悬停标记及 Mermaid 边标签。

How to apply: 从栏目 README 查找章节；本栏目新增或重命名文件沿用英文命名；2026-10-21 用户授权逐项修复组织问题：正文按依赖重排为 01–15，独立 SFT 在 11，生成/评估在 12/13，可选 RL 在 14，运行交付在 15。术语入口为 `glossary/README.md`，九类术语只查阅、不接入正文前后链；源码入口为 `references/source-references-and-call-flow.md`，原文历史验收移到 `.agents/docs/project-maintenance/llm-training-workflow-original-validation-2026-10-12.md`。独立正文以一级标题起始；正文术语使用原生 `<abbr title="…">` 悬停，邻近独立“释义”链接跳转；维护 `ref-` / `term-` 往返锚点，不将术语本身改成链接。Mermaid 的分支说明放在节点中，箭头不附文字。原技术正文仍以 2026-10-12 核对记录为依据；新增 SFT 串联说明于 2026-10-21 局部静态核对，未执行训练或 GPU 测试。README 包含 D1–D21 逐日小节映射；用户希望随每日实验渐进阅读，因此实际入口在 21 篇教程开头的“当天的流程阅读”，不要求先通读全栏目。章节小节使用 reading- 锚点，修改标题时保留这些链接目标。

Evidence: 用户在 2026-10-13 明确要求“文件过大，在docs中新开栏目拆封”；同日用户补充“英文命名”；2026-10-15 用户要求制定计划并按当前技能修改；2026-10-21 用户要求“逐个修改”已指出的章节组织问题；结果见 [栏目目录](../docs/llm-training-workflow/README.md) 及根目录同名入口。

Invalidates: 用户调整文档位置、章节结构或重新核验源码时更新本条；源码变化后不得沿用原核对结果作为当前结论。
