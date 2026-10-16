---
source_files:
  - nanochat/gpt.py
  - nanochat/engine.py
  - nanochat/flash_attention.py
  - nanochat/checkpoint_manager.py
  - .agents/docs/career_roadmap/w2/scripts/benchmark.py
  - .agents/docs/career_roadmap/w3/scripts/aggregate.py
doc_type: reference
---

# W1–W3 每日指导交付验证（2026-10-15）

交付为 21 篇每日教程和 13 个 Python 脚本：W1 为 7+4，W2 为 7+6，W3 为 7+3。三个周目录不设 README；指导在实际操作处直接链接脚本。按文件名排序阅读，周末与次周之间有直接链接：

- [W1 第一天：起点复核](../career_roadmap/w1/D1_starting_point.md)。
- [W2 第一天：数据前置条件](../career_roadmap/w2/D1_data_prerequisites.md)。
- [W3 第一天：简历初稿](../career_roadmap/w3/D1_resume_draft.md)。

教程包含前置条件、操作命令、源码位置、结果解读、失败分流、产物模板与每日验收。实验正式输出仍由读者执行后写入 `experiments/inference_infra/`；本轮脚本验证输出放在 `/tmp`，不冒充按日完成的正式实验。

## 实际环境与验证范围

使用 [nanochat Conda 环境](nanochat-conda-environment-2026-10-15.md)：Python 3.10.20、PyTorch 2.9.1+cpu、float32、SDPA。相关生产源码没有修改；验证时 HEAD 为 `b0855cf551333a43ec6a63e70375d0e76a64e60b`，新增教程与脚本尚未提交。原有总计划和后来出现的 `.vscode/` 均未纳入本次修改。

<abbr title="契约验证 · 检查输入、输出、状态推进与记录格式是否符合约定。这里的 CPU 随机模型验证不衡量模型质量，也不能推导 CUDA 性能。">CPU 契约验证</abbr>已完成以下检查：

| 检查 | 实际结果 | 临时证据 |
|---|---|---|
| 原有推理相关 pytest | 13 passed、10 skipped；跳过项需要 FA3 | 本聊天实际命令输出；环境记录含复查命令 |
| 13 个脚本 | 语法解析、编译和 `--help` 均通过 | 交付脚本可逐个重跑 |
| 21 篇文档 | 本地链接存在、围栏成对、abbr 闭合、source_files 存在、无周 README | `/tmp/check_nanochat_guides.py` |
| 环境与 MiniMind 起点盘点 | 两脚本成功，找到迁移后的旧证据 | `/tmp/nanochat_w1_environment.json`、`/tmp/nanochat_w1_start.json` |
| 真正的随机小 GPT 形状 | logits `[1,8,128]` | `/tmp/nanochat_w1_shapes.json` |
| 缓存与无缓存逐步对照 | 位置 8/9/10/11，greedy 一致，最大 logits 绝对差约 `7.45e-9`；B4 的 K/V、prev_embedding 复制及 reset 通过 | `/tmp/nanochat_w1_cache.json` |
| pytest 日志捕获脚本 | 选择 SDPA/override 子集：5 passed、1 skipped、17 deselected | `/tmp/nanochat_w1_pytest/20261015T063208609407Z/` |
| 合成 tokenizer 与真实 loader | 生成隔离数据；6 组文本往返通过；实际词表 419；train/val 各 `[1,32]` 通过 | `/tmp/nanochat-w2-check/roundtrip.json` |
| 随机 Engine CPU 基准接口 | B1/B4 各 3 次预热、5 次正式记录，8 行全部 ok | `/tmp/nanochat-w2-final-validation/p8_g4_b1.jsonl` 与 `p8_g4_b4.jsonl` |
| 汇总与追溯 | 两组各 5 个有效样本，状态 CPU_OR_RANDOM_CONTRACT；有效 token、逐步时间、median/P95 与吞吐独立复算一致 | 同目录 `summary.csv`、`audit.jsonl` |
| 随机权重序列化后加载 | 生产保存/加载接口成功，双生成路径 4-token 前缀一致；这不是训练产物 | `/tmp/nanochat-w2-check/random_checkpoint_probe.json` |
| checkpoint 基准加载分支 | 随机序列化权重的加载、文件哈希和 3+5 记录成功，仅验证接口 | `/tmp/nanochat-w2-check/random-load-only/p8_g4_b1.jsonl` |
| 矩阵和材料工具 | 默认仅生成计划，长度约束与拒绝覆盖可验证；材料为待填写模板 | `/tmp/nanochat-w3-check/plan/`、`materials/` |
| 统计失败边界 | 不足预热、预热失败、未知 schema、无有效样本与显存聚合检查通过 | `/tmp/nanochat-w3-edge/`；其中 GPU 行是测试夹具，非 GPU 测量 |

<abbr title="P95 / 95th percentile · 第95百分位数。本工具按 nearest-rank 取排序后第 ceil(0.95×n) 个值；n=5时就是最大值，不能视为稳定的尾延迟估计。">P95</abbr>采用 nearest-rank；逐次 TPOT 为首 token 之后 G−1 个输出间隔的均值。汇总再计算每次运行指标的中位数，而不是混合所有步或手抄平均数。CPU 显存字段为空；吞吐为有效输出 token 总数除首 token 到最后输出的计时时间。

## 明确未验证的部分

没有执行两步或二十步真实训练，没有真实训练 checkpoint 的性能结果，没有 CUDA/FA3 实测、GPU 显存结果或优化加速比。训练包装器只验证参数计划与当前训练 CLI 的一致性；随机权重保存后加载不替代训练门槛。

全局 Markdown Preview Enhanced 复制按钮资源执行 `prepare_global_preview.py --check`，返回 `already-prepared`；未修改用户设置，未在真实 VS Code 窗口验收复制按钮或术语悬停效果。文档源文本结构检查不等于阅读器渲染验收。

`/tmp` 是本轮短期验证位置，可能被系统清理；以上表格保存本轮实际检查结论，复查时按教程重新运行并保留新的证据。原日历只是计划槽位，不能据本交付记录声称用户已经完成 W1–W3 所有实验。
