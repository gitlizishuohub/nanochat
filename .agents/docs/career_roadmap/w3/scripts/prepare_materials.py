#!/usr/bin/env python3
"""Create honest, editable application/reproduction templates; inspect evidence locally."""
import argparse
import hashlib
import json
from pathlib import Path

TEMPLATES = {
'resume_draft.md': '''# [姓名]｜[邮箱]｜[代码主页]

教育：[学校、专业、学历、毕业时间，逐项核实]。
相关课程：[仅列确实修过且能回答问题的课程]。

## MiniMind：CPU 训练与推理闭环

- [按归档证据确认后填写本人负责部分、模型配置、训练恢复及 KV 一致性工作]。
- 证据：[归档 reports/reproduction.md、results/tables/kv_cache.csv 的具体行]。
- 边界：小数据 CPU 功能验证；不能推出模型质量或 GPU 加速成绩。

## nanochat：推理基线与测量协议（进行中）

- [确认已运行后填写：源码路径梳理、重复运行、逐次 JSONL 与自动统计]。
- 当前设备：[填写]；checkpoint 来源/step：[填写]；有效样本数：[填写或待测]。
- GPU 基线：[PENDING_GPU 或真实原始行路径]；优化加速比：[未测量，删除此项]。

## 技能与补充

[只写可演示的 Python/PyTorch/Linux/Git 等能力及证据，控制全文一页]。
''',
'reproduction_entry.md': '''# nanochat 推理基线：当前复现入口草稿

## 目的

在固定 nanochat checkpoint 上建立可追溯的推理基线；当前状态：[填写]。

## 资源

Python/Conda 环境：[填写]。设备、RAM/VRAM、PyTorch、compute dtype、注意力后端：[填写]。
checkpoint source/model_tag/step、权重哈希和 tokenizer 来源：[填写]。

## 最短运行顺序

1. 激活 conda nanochat，进入仓库根目录。
2. 按 W2 指导核查 checkpoint 与 tokenizer；缺失时停止实测。
3. 按 W3 matrix.py 生成计划，先检查矩阵再显式 --execute。
4. aggregate.py 生成 baseline.csv 与逐行审计记录。

[将实际运行过的完整命令复制到这里，包括 NANOCHAT_BASE_DIR；不能只留脚本文件名]。

## 结果与边界

CPU 功能验证：[日志路径和退出码]。
GPU 基线：[真实 raw 路径或 PENDING_GPU]。
无效上下文、OOM、提前结束：[矩阵与原始错误行路径]。
统计仅使用同条件下至少 5 次有效正式运行；P95 采用 nearest-rank。

## 从失败恢复

缺数据/tokenizer/checkpoint：[返回 W2 对应日]。
OOM：[记录失败，创建新的较小 B 实验；保留失败目录]。
依赖缺失：[记录解释器和模块；不能把跳过当通过]。
''',
'project_briefing.md': '''# 三分钟项目介绍草稿

## 0:00–0:30 问题与角色

我希望理解训练产物如何进入可验证的推理系统；我的实际工作是：[填写本人负责范围]。

## 0:30–1:15 已有证据

MiniMind 归档证明：[数据/训练/恢复/KV 的具体证据]。边界：CPU、小数据、不证明助手能力。

## 1:15–2:15 nanochat 方法和现状

两条路径：[说明 GPT 全历史生成与 Engine 缓存生成的区别]。
计时：[首 next 边界、同步、warmup、重复、有效 token 定义]。
当前证据：[CPU 日志或真实 GPU 基线；没有数据就说 PENDING_GPU]。

## 2:15–3:00 限制与下一步

限制：[硬件/样本数/计时开销/模型质量]。
下一步：先补可信基线，再用 profiler 定位候选瓶颈；尚未声称 RMSNorm 或 KV 优化收益。

## 证据图（手动把每条替换为实际产物）

MiniMind 数据契约 → 训练恢复 → KV 一致性 → nanochat checkpoint → raw JSONL → 自动统计 → trace/优化（待做）。
'''}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--kind', choices=['resume', 'entry', 'briefing', 'all'], default='all')
    p.add_argument('--evidence', action='append', type=Path, default=[])
    args = p.parse_args()
    selected = {'resume': ['resume_draft.md'], 'entry': ['reproduction_entry.md'], 'briefing': ['project_briefing.md'], 'all': list(TEMPLATES)}[args.kind]
    targets = [args.output_dir / name for name in selected]
    manifest = args.output_dir / 'evidence_inventory.json'
    if any(path.exists() for path in [*targets, manifest]):
        p.error('Target exists: use a new output-dir; refusing to overwrite your edited materials')
    inventory = []
    for path in args.evidence:
        entry = {'path': str(path.resolve()), 'exists': path.is_file()}
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            entry.update(bytes=path.stat().st_size, sha256=digest.hexdigest(),
                         verification='FILE_PRESENT_ONLY; human must check claims and authorship')
        inventory.append(entry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in targets:
        path.write_text(TEMPLATES[path.name], encoding='utf-8')
        print(path)
    manifest.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
