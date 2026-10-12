---
name: protected-agent-paths
description: 更新项目 Agent 配置遇到只读或跨设备错误时，核查当前沙盒路径权限。
type: project
created: 2026-10-13
updated: 2026-10-13
---

# 受保护的 Agent 路径

2026-10-13 的 Codex 沙盒将本项目 `.agents/` 和 `.git/` 设为只读，即使项目根目录允许写入。普通工具执行迁移时，`docs` 到 `.agents/docs` 的 rename 报 EXDEV，创建基线技能链接报 EROFS；针对已授权的迁移使用 `exec_command` 的 `require_escalated` 经自动审批后成功。

Why: 根目录可写不代表所有子目录可写；权限失败时不能把挂载脚本的 reference-only 状态报告为已挂载。

How to apply: 先核对当前会话权限与目标状态。对用户已授权的规范迁移或精确 Git 提交请求相应的工具提权；不要复制技能本体、用强制暂存掩盖 ignore，或重跑初始化处理日常记忆更新。正常文件搬动遇到真实跨设备边界时采用保留内容的移动方式。

Evidence: 本次执行环境明确列出 `.agents`、`.git` 只读；迁移工具实际返回 EXDEV 和 EROFS，随后 require_escalated 执行初始化与五项技能挂载成功；详见 `.agents/docs/project-maintenance/init-project-review-2026-10-13.md`。

Invalidates: 会话权限配置、沙盒挂载方式或项目路径改变时重验；不推断普通终端也存在同样限制。
