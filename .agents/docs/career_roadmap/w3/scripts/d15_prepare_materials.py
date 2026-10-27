#!/usr/bin/env python3
"""Create honest, editable application/reproduction templates; inspect evidence locally."""

import argparse
import hashlib
import json
from pathlib import Path

TEMPLATES = {
    "d15_resume_draft.md": """# [姓名]｜[邮箱]｜[代码主页]

教育：[学校、专业、学历、毕业时间，仅填本人已核实信息]。
相关课程：[确实修过且能回答问题的课程]。

## nanochat：生成流程与测量验证学习

- 本人动作：[阅读解释 / 运行复核 / 新增实现，按真实参与选择；不能把上游代码写成本人实现]。
- 一个具体问题：[例如首 token 是否计入吞吐、缓存和全历史生成是否数值一致]。
- 方法与证据：[设备、模型类型、输入条件、日志或 raw 的具体行、有效样本数]。
- 能支持的结论：[按实际记录填写；CPU 随机模型结果仅支持该环境下的功能与测量契约检查]。
- 可选扩展：[只有实际选择并执行了训练或 GPU 测量才填写，否则删除本行]。

## 其他项目（可选）

[有真实参与和证据时添加 MiniMind 或其他经历；写清历史版本和复核范围，无材料则删除本节]。

## 技能与补充

[只写可演示的 Python/PyTorch/Linux/Git 能力，每项能落到一个本人动作；压缩至一页]。
""",
    "d20_reproduction_entry.md": """# nanochat CPU 生成与统计：复现入口草稿

## 目的与当前状态

验证随机小模型的生成接口、固定长度计数和重复测量链路；不训练、不依赖 checkpoint 或外部分词器文件。
实际完成状态：[填写本人执行范围、日期和结果；生成模板不代表完成复现]。

## 起点

仓库路径与版本：[填写]。解释器/Python/PyTorch、CPU、线程数：[填写]。
依赖：[已有可导入 torch 的项目环境；首次环境准备见 D02]。
compute dtype/实际注意力后端：[从 raw 填写]。

## 路径 A：重新运行 CPU 随机小模型

[从 D20 教程复制完整、亲自核验的命令；包含 cd、解释器核对、NANOCHAT_DTYPE、线程数、新输出目录、d12_benchmark.py 全部参数以及退出码捕获]。
成功判据：每个 B 的 raw 有 3 条 warmup 和至少 5 条 measure，状态和实际 token 计数合格。
随机初始化不是训练；该路径不需要 META、MODEL_TAG 或 NANOCHAT_BASE_DIR。

## 路径 B：仅从已有 raw 重建统计

输入目录及包含文件：[填写明确范围，保留原文件名和内容，不混入重复 run_id]。
[复制 d13_aggregate.py 的完整 --raw-dir/--output/--audit 命令，使用独立输出位置]。
成功判据：核对分组、n_valid、warmup_status 和 audit 的 file/line/run_id；CPU_OR_RANDOM_CONTRACT 不自动保证样本充分。

## 结果与边界

本次 log/raw/CSV/audit：[填写实际路径]。
模型类型、P/G/B、有效样本数与一次反查：[填写]。
重复运行会产生新的 session/run_id 和时间；统计重建应保留原始身份与有效样本选择。
真实 checkpoint 或 GPU 扩展：[只在选择时填写所需资产与实际状态；未选择可删除，不是本日欠账]。

## 失败恢复

缺 torch：先修正解释器/环境，不启动训练。
early_stop/error：保留原始行和退出码，排查后用新路径重试；不伪造固定长度成功。
无匹配 raw/重复 run_id：检查输入目录和文件名，移除输入中的重复副本，不篡改原证据。
""",
    "d21_project_briefing.md": """# 三分钟项目介绍草稿

## 0:00–0:30 问题与本人角色

我在 nanochat 学习生成流程和测量方法，关注的问题是：[一个具体问题]。
本人实际工作：[阅读解释 / 运行复核 / 新增实现，选择真实动作]。

## 0:30–1:15 用一个例子讲机制

[例如 P=8、G=4 时，prefill 产生首个预测，随后有三个输出间隔；解释源码如何实现，不以概念清单代替因果过程]。

## 1:15–2:15 用原始记录支撑结论

设备、模型类型、P/G/B、有效样本数：[从实际 raw/CSV 填写]。
一次证据反查：[结论 → CSV → audit → raw 的 file/line/run_id]。
没有执行证据时明确只完成推演，不能照抄示例称已运行。

## 2:15–3:00 限制与下一步

当前结论适用范围：[填写；CPU 随机模型不能推导训练模型质量或 GPU 性能]。
下一步：[针对真实证据缺口选择一个可执行动作及成功判据，不自动安排训练/租卡]。

## 可选证据节点

概念与源码 → CPU 生成 → raw → 统计审计 → 复现入口 → 可选进一步实验。
节点分别标记：已运行复核 / 仅阅读推演 / 未选择的扩展；只链接实际存在的文件。
有真实 MiniMind 或训练记录时可另加，历史成果不能自动归为当前工作。
""",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--kind", choices=["resume", "entry", "briefing", "all"], default="all"
    )
    p.add_argument("--evidence", action="append", type=Path, default=[])
    args = p.parse_args()
    selected = {
        "resume": ["d15_resume_draft.md"],
        "entry": ["d20_reproduction_entry.md"],
        "briefing": ["d21_project_briefing.md"],
        "all": list(TEMPLATES),
    }[args.kind]
    targets = [args.output_dir / name for name in selected]
    kinds_by_day = {"d15": "resume", "d20": "entry", "d21": "briefing"}
    manifests = [
        args.output_dir / f"{name[:3]}_{kinds_by_day[name[:3]]}_evidence_inventory.json"
        for name in selected
    ]
    if any(path.exists() for path in [*targets, *manifests]):
        p.error(
            "Target exists: use a new output-dir; refusing to overwrite your edited materials"
        )
    inventory = []
    for path in args.evidence:
        entry = {"path": str(path.resolve()), "exists": path.is_file()}
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            entry.update(
                bytes=path.stat().st_size,
                sha256=digest.hexdigest(),
                verification="FILE_PRESENT_ONLY; human must check claims and authorship",
            )
        inventory.append(entry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in targets:
        path.write_text(TEMPLATES[path.name], encoding="utf-8")
        print(path)
    for target, manifest in zip(targets, manifests):
        payload = {
            "first_use_day": int(target.name[1:3]),
            "material": target.name,
            "evidence": inventory,
        }
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(manifest)


if __name__ == "__main__":
    main()
