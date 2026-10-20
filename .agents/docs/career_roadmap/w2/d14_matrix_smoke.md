---
source_files:
  - nanochat/common.py
  - nanochat/tokenizer.py
  - nanochat/dataset.py
  - nanochat/dataloader.py
  - scripts/tok_train.py
  - scripts/tok_eval.py
  - scripts/base_train.py
  - nanochat/checkpoint_manager.py
  - nanochat/gpt.py
  - nanochat/engine.py
  - scripts/infer_bench.py
  - nanochat/flash_attention.py
doc_type: tutorial
knowledge_scope: W2第7天实操与验收
---

# W2 D7：小矩阵试跑与 W2 验收

前一天：[返回前一天](d13_repeated_benchmark.md)；后一天：[W3第一天](../w3/d15_resume_draft.md)；[原始总计划](../nanochat_daily_execution_2026-10-01.md)。

## 开始前：固定 shell 与结果位置

所有命令从仓库根目录执行。首次打开终端先运行以下准备；`conda activate` 若已可用，可省略 `source` 一行。这里使用本项目的 Conda 环境；不要混用系统 Python。CPU实验建议先 `export NANOCHAT_DTYPE=float32`；脚本只在变量未设置时选择FP32，已有变量会遵从并记录，不能假定CPU必定FP32。GPU实验另按协议显式选dtype。

```bash
source /home/lizishuo/miniconda3/etc/profile.d/conda.sh
conda activate nanochat
cd /home/lizishuo/_program_/Project/nanochat
python -c "import sys; print(sys.executable); print(sys.version)"
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
exec > >(tee -a experiments/inference_infra/reports/d14_session.log) 2>&1
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。若环境不存在或第三方依赖缺失，先解决环境，不把 `--help` 成功当作模型实验通过。后面写入的 `experiments/` 文件是你实际执行教程时产生的正式成果；本指导不预先生成成绩。大数据、tokenizer、权重放在 `NANOCHAT_BASE_DIR`，不要加进 Git。表中的 `$NANOCHAT_BASE_DIR` 代表当前 shell 选定的真实绝对数据根；合成分支示例为 `/tmp/nanochat-w2-toy`，未设置时当前用户默认根为 `/home/lizishuo/.cache/nanochat`。在报告写出实际展开后的绝对路径；`*.parquet` 表示该目录内逐一列出的实际数据文件，旧数据回退目录为 `$NANOCHAT_BASE_DIR/base_data/`。

本 shell 后续终端输出同步保存在 `experiments/inference_infra/reports/d14_session.log`；命令返回后立即执行 `echo "exit=$?"`，避免丢失退出码。帮助、源码检索、解释器与 Git 检查只打印，不另有检查结果文件。日志是追加写入，重跑时在报告标明会话边界。

## 今日目标与时间盒

用 15 分钟检查门槛，30–60 分钟跑小矩阵，20 分钟复算，15 分钟写周结。前置是 [重复测量](d13_repeated_benchmark.md)；如果仍有错误，先修复最小P8/G4/B1，不直接加大矩阵。

## 第一步：给今天设置清晰的可完成门槛

小矩阵先固定 `P=8/32, G=4, B=1/4` 四组，用CPU/random验证接口；有真实checkpoint时，再选择相同小矩阵验证模型加载路径。今天不是W3完整性能矩阵，不要求所有P512组合。CPU慢时可以只跑两组B1/B4，并把其余标 `PENDING_CPU_TIME`。

## 第二步：执行CPU两种batch契约

两条命令都调用 [重复基准脚本](scripts/d12_benchmark.py)，仅改变B并使用不同raw路径：

```bash
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py --device cpu --random-model --prompt-tokens 8 --decode-tokens 4 --batch-size 1 --warmup 3 --repeats 5 --output experiments/inference_infra/results/raw/d14_p8_g4_b1_gate_cpu.jsonl
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py --device cpu --random-model --prompt-tokens 8 --decode-tokens 4 --batch-size 4 --warmup 3 --repeats 5 --output experiments/inference_infra/results/raw/d14_p8_g4_b4_gate_cpu.jsonl
```

B4每个有效run应 `generated_tokens=16`、`per_row_generated_tokens=[4,4,4,4]`。TPOT仍是三个输出列间隔的均值；它不应再除4。B4与B1的TTFT不同可能来自cache复制和分配，但随机CPU结果只用于检查代码，不做GPU“加速倍数”结论。

如果前两组稳定，再把P改成32，输出分别改为 `experiments/inference_infra/results/raw/d14_p32_g4_b1_gate_cpu.jsonl` 和 `experiments/inference_infra/results/raw/d14_p32_g4_b4_gate_cpu.jsonl`。对于真实checkpoint去掉 `--random-model` 并加 `--source base --model-tag d11_w2_20 --step 20`；必须保留合成训练来源或真实语料来源，不能用名字“base”掩盖来源区别。

## 第三步：逐行复算与自动汇总

打开 [W3统计脚本](../w3/scripts/d13_aggregate.py) 与 [统计复核指导](../w3/d19_statistics_audit.md)。先运行help确认参数，再用同一个统计入口处理这些raw。不要手抄五个TTFT到Excel后丢掉run_id。实际汇总使用以下命令：

```bash
python .agents/docs/career_roadmap/w3/scripts/d13_aggregate.py --raw-dir experiments/inference_infra/results/raw --output experiments/inference_infra/results/summary/d14_w2_gate.csv --audit experiments/inference_infra/results/summary/d14_w2_gate_audit.jsonl
```

该命令读取共享目录下全部匹配文件，也可能包含第 13 日的 session；输出文件名 d14 表示汇总文件首次生成日，不表示输入仅限第 14 日。按 `experiments/inference_infra/results/summary/d14_w2_gate_audit.jsonl` 的完整输入路径与 run_id 选出本日小矩阵的组，再检查以下关系；其它 session 保留为独立组，不计入本日验收：

| 检查项 | 每个ok/measure应满足 |
|---|---|
| 形状 | P+G≤sequence_len，无实际prompt静默变化 |
| 正式样本 | 每个session每组≥5行；warmup不计入 |
| 时间 | TTFT>0；elapsed>0；G>1时TPOT>0 |
| 步数 | actual_decode_steps=G，间隔个数G−1 |
| token | per_row长度B，每项G，sum等于generated_tokens |
| 吞吐 | generated_tokens×1000/elapsed_ms |
| 内存 | CPU必须null；CUDA记录bytes，不混成MiB |
| 身份 | 模型hash、tokenizer hash、精度与后端与协议一致 |

若汇总与手工选一行复算不一致，先检查单位，再看聚合顺序：median(tokens/s)与总token/总时长不是同一个统计口径。不要因数值接近就混写。

## 第四步：完成W2周结与W3交接

在 `experiments/inference_infra/reports/d14_w2_gate.md` 填写：

```text
完成物：库存 / roundtrip / 两步checkpoint / 二十步checkpoint / raw / summary
每项证据文件：
真实模型来源、tag、step、hash：
可用设备与精度：
正式成功组数与每组样本数：
失败组、异常原因和保留的raw路径：
最大阻塞：
PENDING_GPU项目：
下一周单一目标：冻结可运行基线矩阵并自动汇总
```

<abbr title="Gate · 阶段放行条件。本文要求能从命令生成原始记录再自动汇总，失败行可追溯；没有GPU时仅允许将CPU功能分支放行，不能把未测GPU性能标为完成。">W2 gate</abbr> 分层判断：文档/脚本已交付、CPU契约实际通过、真实checkpoint实际通过、CUDA性能实际通过分别记录。缺失真实checkpoint时不能写整个W2技术门槛全部通过，但仍可进入W3职业材料准备和GPU执行脚本设计。

## 失败分流与验收

- 多个配置在同一进程互相改变dtype：这里每条CLI是独立进程；不要用临时notebook改常量假装完成独立运行。
- raw里只有8行但都是warmup：按phase统计，不能只数总行数。
- 发现异常值：先保留，再记录背景进程、温度、调频与CPU线程数；只有有证据的运行污染才单独标记，不删除慢样本。
- 最终验收：至少一条从脚本→raw→自动summary的真实链；有一页周结能明确说明未验证范围；接着阅读 [W3简历初稿](../w3/d15_resume_draft.md)。

## 今日脚本与产物清单

每日编号使用全程 d01–d21；跨日复用保留首次实际执行日。以下项目相对路径均以仓库根目录为基准，脚本清单直接列出本日执行入口。

| 脚本完整项目路径 | 用途 | 执行阶段 | 输入 | 运行条件 |
|---|---|---|---|---|
| `.agents/docs/career_roadmap/w2/scripts/d12_benchmark.py` | 复用基准入口执行小矩阵 | 第二步 | 随机模型无需文件；真实模型读 `$NANOCHAT_BASE_DIR/base_checkpoints/d11_w2_20/model_000020.pt`、`$NANOCHAT_BASE_DIR/base_checkpoints/d11_w2_20/meta_000020.json`、`$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl`；P/G/B | 先最小组成功再扩展；可按资源留待执行 |
| `.agents/docs/career_roadmap/w3/scripts/d13_aggregate.py` | 复用自动统计 | 第三步 | experiments/inference_infra/results/raw 中本次 JSONL | 实际执行下面明确的汇总命令 |

| 产物完整项目相对路径 | 用途 | 生成者 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/results/raw/d14_p8_g4_b1_gate_cpu.jsonl` | B1 门槛记录 | d12_benchmark.py | 条件自动；实际运行 |
| `experiments/inference_infra/results/raw/d14_p8_g4_b4_gate_cpu.jsonl` | B4 门槛记录 | d12_benchmark.py | 条件自动；实际运行 |
| `experiments/inference_infra/results/raw/d14_p32_g4_b1_gate_cpu.jsonl` | 可选 P32/B1 记录 | d12_benchmark.py | 条件自动；扩展矩阵才运行 |
| `experiments/inference_infra/results/raw/d14_p32_g4_b4_gate_cpu.jsonl` | 可选 P32/B4 记录 | d12_benchmark.py | 条件自动；扩展矩阵才运行 |
| `experiments/inference_infra/results/summary/d14_w2_gate.csv` | 小矩阵统计 | d13_aggregate.py | 条件自动；第三步汇总命令 |
| `experiments/inference_infra/results/summary/d14_w2_gate_audit.jsonl` | 小矩阵审计 | d13_aggregate.py | 条件自动；第三步汇总命令 |
| `experiments/inference_infra/reports/d14_w2_gate.md` | 周结、分层验收和交接 | 第四步 | 人工编写 |
| `experiments/inference_infra/reports/d14_session.log` | 当天终端输出与手动记录的退出码 | 开场 shell 的 tee | 自动；仅捕获启用后的输出 |

原生 `tokenizer.pkl`、`token_bytes.pt`、`model_<step>.pt`、`meta_<step>.json` 与 `optim_<step>_rank0.pt` 是 loader 的格式契约，保留其名称；用带日编号的数据文件或模型标签归属日期。已有历史结果不原位重命名，新的命令使用本页命名。没有运行、条件不满足、人工未填写的项目均标 PENDING。

## 当天记录与离开前核对

把以下模板填写到当天报告末尾；运行命令原样保留，失败也写退出码。`PENDING` 表示没有执行；不要用空值代替失败事实。

```text
日期：
源码 commit / dirty 状态：
解释器 / 设备 / NANOCHAT_DTYPE：
NANOCHAT_BASE_DIR：
实际用时：
运行命令与退出码：
原始结果路径：
已验证的事实：
未验证或阻塞：
下一次恢复的第一条命令：
```

记录 commit 使用 `git rev-parse HEAD`，改动状态使用 `git status --short`。共享给别人前检查数据路径和日志是否包含私人信息。只提交本次自己的轻量报告和脚本，不提交权重、数据、环境目录或机器特有缓存。
