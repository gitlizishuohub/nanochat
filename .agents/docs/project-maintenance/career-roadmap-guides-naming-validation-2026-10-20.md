# W1–W3 命名与每日清单核验（2026-10-20）

21 篇每日教程使用全程 `d01_`–`d21_` 前缀，保留主题名和 `Wn Dn` 标题；13 个脚本按首次使用日命名并保留在对应周的 `scripts/`。不增加周 README。入口为 [W1 D1](../career_roadmap/w1/d01_starting_point.md)、[W2 D1](../career_roadmap/w2/d08_data_prerequisites.md)、[W3 D1](../career_roadmap/w3/d15_resume_draft.md)。

每日正文列出脚本用途、阶段、输入与条件，以及产物完整项目相对路径、生成者和生成方式。人工报告、材料骨架与完善后的正文分别描述；默认仅计划的脚本不宣称产生测量结果。终端检查由教程中的 Bash 日志捕获命令保存，退出码仍需按正文人工登记。

跨日复用保留首次归属，例如第 13 日首次调用的统计工具仍存于 W3 的 `scripts/d13_aggregate.py`；W3 继续更新第 12 日的 `experiments/inference_infra/d12_protocol.md`。矩阵通过必填 `--artifact-day` 为新一轮结果标注首次生成日。统计工具兼容带日编号和历史未编号的原始记录，历史文件不改名。上游源码、测试以及模型加载器要求的 tokenizer/checkpoint 固定格式保留，教程明确它们位于仓库外的数据根目录。

## 本轮真实验证

解释器：`/home/lizishuo/miniconda3/envs/nanochat/bin/python`；CPU、FP32。临时数据使用 `/tmp/nanochat-dnn-validation-8eyp7jyq/`，不写入正式实验目录。

| 检查 | 结果 |
|---|---|
| 21 篇教程 | 全程编号连续、脚本/产物清单存在、本地链接有效、代码围栏及术语标签闭合 |
| 74 个 Bash 代码块 | `bash -n` 语法检查通过；不代表所有命令已真实执行 |
| 13 个脚本 | Python 语法和 `--help` 通过 |
| 合成数据与真实 loader | 新建隔离 tokenizer/Parquet，文本往返和 train/val 批次检查成功 |
| 2/20 步训练包装器 | 默认标签与 d10/d11 计划命名正确；仅计划、状态 PENDING，没有训练日志或训练权重 |
| CPU 随机模型 B1/B4 | 各 3 次预热、5 次有效正式记录，带日编号原始文件可被统计脚本读取 |
| 自动统计 | 两组各 5 个有效样本、预热状态 OK；独立重算 TTFT 中位数一致 |
| 矩阵默认模式 | 12 行计划，sequence_len=512 时 4 行超上下文；未生成 raw/log；重复目标拒绝覆盖 |
| 矩阵显式执行失败分支 | 缺失 checkpoint 的子进程失败，生成 d17 对应日志及真实 error JSONL，矩阵保留非零子进程退出码 |
| 材料工具 | resume/entry/briefing 三分支按 d15/d20/d21 生成正文骨架及证据清单，拒绝覆盖 |
| 推理相关测试捕获 | `tests/test_engine.py` 和 `tests/test_attention_fallback.py`：13 passed、10 skipped；日志及退出码捕获成功 |
| 记忆与 diff | 记忆索引及严格结构检查通过；`git diff --check` 通过 |

<abbr title="TTFT / Time to First Token · 本次基准从处理已编码输入到首个输出列准备好的计时，包含缓存分配及同步等包装开销，不包括模型加载。">TTFT</abbr> 的统计检查只验证计算与追溯契约。测试日志位于 `/tmp/nanochat-dnn-pytest/20261020T110505013827Z/`；临时检查脚本为 `/tmp/check_nanochat_dnn.py`、`/tmp/validate_nanochat_dnn_runtime.py`。这些临时文件可能被系统清理，本记录不要求它们永久存在。

## 未验证范围

没有执行真实 2 步或 20 步训练，也没有产生真实训练 checkpoint 的性能结果；没有 CUDA/FA3 或 GPU 显存测量。未在阅读器中验收术语悬停和完整页面渲染。脚本测试和文档核验不表示用户已完成每日实验；原计划日期不变成完成日期。

已有 `experiments/`、`.vscode/` 和未跟踪的总计划保留；本轮不把已有实验或外部 MiniMind 历史证据重命名。此前的 [2026-10-15 验证记录](career-roadmap-guides-validation-2026-10-15.md) 仅维护现行导航，保留当时的结果与临时证据路径。
