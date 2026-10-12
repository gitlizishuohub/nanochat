# init-project 迁移检查记录（2026-10-13）

本次按用户“参照最新的 init-project 完善本项目，并仔细检查”的请求，迁移项目的 Agent 规范、知识文档、记忆和基线技能。未修改训练或推理源码。

## 依据与布局

真源为 `/mnt/d/Obsidian/Vault/skills/init-project/SKILL.md`。技能正文及初始化、挂载脚本与本机 `windows-vault` 别名版本的 SHA-256 一致；正文 SHA-256 为 `b05b83e1b5e798470f95a8e568a0c66c666ca9dd7dab13cd9ce95f97b5ee0738`。本次不使用旧 `bundled-skills/` 作为新登记真源。

- 根 [AGENTS.md](../../../AGENTS.md) 是唯一项目入口；细则位于 `.agents/project_rules/`。
- 知识文档与索引位于 `.agents/docs/`；旧根目录实战指南原文迁入，旧 `docs/career_roadmap/` 整体迁入。
- 根目录《LLM模型训练完整流程与术语对照.md》记录了此前用户明确指定的根目录交付要求，因此保留该位置；仅修复其两个规范与记忆链接。
- 记忆位于 `.agents/memory/`，旧两条初始化导航记忆原文归档，不再用作现行入口；健康工具更新至当前技能配套版本。
- 空旧工作区已移除；临时脚本与检查数据使用 `/tmp`。原有上游 `dev/` 文档及 `graphify-out/` 正式结果保留。
- 五项技能均为指向 Vault 同级技能的符号链接；详见 [技能登记](../../project_rules/skills.md)。已保留任务前完成的知识技能整合方向。
- 取消旧记忆与旧工作区忽略项，根规范及 Agent 文件可以正常纳入 Git。任务前已有两份未提交知识文档保持未提交，不夹带到配置提交。

## 已执行验证

- `python .agents/memory/scripts/memory_check.py .agents/memory/MEMORY.md`：通过；索引 15 行、389 字节。
- `python .agents/memory/scripts/memory_doctor.py .agents/memory --require-evidence`：通过；无断链、孤立条目或 schema 错误。此检查仅验证结构。
- `PYTHONDONTWRITEBYTECODE=1 python /mnt/d/Obsidian/Vault/skills/init-project/tests/test_project_memory.py`：16 个配套测试通过，覆盖迁移冲突、文件保留、健康工具刷新、记忆严格字段与独立进程续接检查。

- 自定义迁移核对：284 个本地文件链接全部有效（含源码行号）；原先失效的 3 个 MiniMind 链接已指向实际 `.agents/docs/` 位置。24 个外部网页链接未联网核验。
- 文件保留核对：实战指南与原 Git 版本字节一致；两份旧记忆归档与迁移前 SHA-256 一致；两份任务前未提交文档的全部内容，仅发生已登记的路径纠正。
- 技能核对：五个项目链接与登记真源逐项 `samefile` 检查通过；两个健康工具与配套真源字节一致；五个旧布局位置均不存在。
- ignore 核对：20 个 Agent 文件或技能链接均未被 Git 忽略；源码、依赖、上游 README 和图谱文件相对于本次任务前 HEAD 无改动。
- 重复运行初始化：8 个既有配置/工具文件全部保留，无新建或刷新；重复挂载：5 项均为 `already-mounted`。运行前后全部 Agent 文件、根规范、ignore 与符号链接快照一致。
- 实际使用记忆：后续提交步骤再次从索引召回 `protected-agent-paths`，读取其证据与失效条件，并据当前权限采用精确操作提权；这是本次任务内的召回，不是未来独立模型会话的验证。
- `git diff --check`：通过。提交时只纳入与文档位置迁移或 Agent 配置完善直接相关的精确路径，分开两个逻辑目的。

## 验证边界

未启动训练、下载数据、安装依赖、执行 GPU 基准或重建知识图谱；未验证旧学习指南的全部技术结论和外部网页。图谱 manifest 包含旧布局快照路径，不能当作当前 Agent 规则。配套工具测试和本次记忆召回不能证明下一次独立模型会话必然遵守记忆协议；跨会话行为需后续相关任务实际召回、核验证据及条件变化时更新。

五项技能使用本机绝对 Vault 路径，换机须重新核验真源。当前沙盒将 `.agents/` 和 `.git/` 设为只读，迁移和提交使用针对已授权操作的 `require_escalated`；普通沙盒曾返回 EXDEV/EROFS，不能据此推断项目目录在普通终端中也只读。
