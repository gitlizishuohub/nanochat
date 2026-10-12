# 技能注册与触发

补充 [根 AGENTS.md](../../AGENTS.md)。

公用技能引用唯一真源，不复制进仓库。默认新技能在公共 Vault 维护；只有用户明确要求项目级技能且工作流依赖本仓库时才创建项目实体目录。

五个基线技能在 Vault 的 `skills/` 下同级维护，`init-project` 通过挂载脚本引用其他四个技能。旧 `bundled-skills/` 路径用于兼容，不登记为新项目的真源。

## 基线技能

已运行 `init-project/scripts/mount_baseline_skills.py --project <项目根目录>`；下列五项均为已核验可读取的符号链接，真源为 Vault 根级技能。绝对真源路径依赖当前机器的 Vault 挂载；换机后重新核验并登记实际结果。

| 技能 | 项目挂载 | canonical 真源 | 状态 | 触发条件 |
|---|---|---|---|---|
| `init-project` | [SKILL.md](../skills/init-project/SKILL.md) | `/mnt/d/Obsidian/Vault/skills/init-project/SKILL.md` | mounted | 初始化、迁移或修复规范与挂载；日常任务不重跑 |
| `code-api-annotation` | [SKILL.md](../skills/code-api-annotation/SKILL.md) | `/mnt/d/Obsidian/Vault/skills/code-api-annotation/SKILL.md` | mounted | 创建、修改 `src/` 下源码或补充其注释 |
| `code-knowledge-workflow` | [SKILL.md](../skills/code-knowledge-workflow/SKILL.md) | `/mnt/d/Obsidian/Vault/skills/code-knowledge-workflow/SKILL.md` | already-mounted | 全量讲解源码、持续代码问答或归档知识 |
| `fine-grained-git-commit` | [SKILL.md](../skills/fine-grained-git-commit/SKILL.md) | `/mnt/d/Obsidian/Vault/skills/fine-grained-git-commit/SKILL.md` | mounted | 根规范或 `.agents/` 随改提交，或用户要求提交/保存 |
| `annotate-markdown-terms` | [SKILL.md](../skills/annotate-markdown-terms/SKILL.md) | `/mnt/d/Obsidian/Vault/skills/annotate-markdown-terms/SKILL.md` | mounted | 为指定 Markdown 文档添加专有名词、缩写或技术概念解释 |

`file-context-qa`、`full-knowledge-code-analyzer`、`knowledge-doc-generator` 三项旧知识技能的项目挂载已退役；其功能入口为 `code-knowledge-workflow`。已保留任务开始时用户完成的整合方向。

公共技能提及 `docs/` 或 `.memory/` 时，分别映射为 `.agents/docs/` 与 `.agents/memory/`；用户已明确指定的文档交付位置优先于默认目录。

触发时完整读取对应 SKILL.md。冲突保留原内容；无法创建符号链接时登记真源引用，不复制技能本体。

## 仓库特异技能

无。新增时登记位置与可判定的触发条件。
