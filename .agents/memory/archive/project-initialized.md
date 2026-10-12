---
name: project-initialized
description: nanochat 已于 2026-09-12 完成 Agent 工作区、记忆协议和 Git 基线初始化。
type: project
created: 2026-09-12
updated: 2026-09-12
---

# 项目初始化状态

| 组件 | 状态 | 位置 |
|---|---|---|
| 项目规则 | 已创建 | `.agents/AGENTS.md` |
| 基线技能 | 已挂载六项符号链接 | `.agents/skills/` |
| Agent 工作区 | 已创建 | `.agent_workspace/` |
| 长期记忆 | 已创建 | `.memory/` |
| Git | 已存在并连接 `origin/master` | `.git/` |

Why: 统一 Agent 的工作区、记忆与提交纪律，并让 `.agents/` 归项目自身管理。

How to apply: 每次任务先读取 `.agents/AGENTS.md` 和本索引；所有临时产物进入 `.agent_workspace/`，跨会话事实按类型写入 `.memory/`。

相关：[[agents-readme]]
