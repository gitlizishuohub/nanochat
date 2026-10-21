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
knowledge_scope: W2第3天实操与验收
---

# W2 D3：两步训练 smoke 与合法 checkpoint

前一天：[返回前一天](d09_token_roundtrip.md)；后一天：[继续下一天](d11_checkpoint_generation.md)；[原始总计划](../nanochat_daily_execution_2026-10-01.md)。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口：**[6.3 logits、目标对齐与有效监督](../../llm-training-workflow/06-forward-pass-and-prediction-loss.md#reading-6-3)；[7.1 一次更新的真实次序](../../llm-training-workflow/07-backpropagation-and-parameter-updates.md#reading-7-1)；[7.4 一次更新的流程图](../../llm-training-workflow/07-backpropagation-and-parameter-updates.md#reading-7-4)。
- **今天重点：**把昨日目标对齐接到有效目标损失、反向、梯度累积和参数更新，结合两步训练区分 backward 与 optimizer.step。
- **暂时跳过：**7.2.1 的算法数值和 7.3 的完整预算公式以后按需深读；今天只看训练计划里的实际步数。

**读完能解释：**loss.backward 和 optimizer.step 各做什么？微批次数为什么不一定等于参数更新次数？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 开始前：固定 shell 与结果位置

所有命令从仓库根目录执行。首次打开终端先运行以下准备；`conda activate` 若已可用，可省略 `source` 一行。这里使用本项目的 Conda 环境；不要混用系统 Python。CPU实验建议先 `export NANOCHAT_DTYPE=float32`；脚本只在变量未设置时选择FP32，已有变量会遵从并记录，不能假定CPU必定FP32。GPU实验另按协议显式选dtype。

```bash
source /home/lizishuo/miniconda3/etc/profile.d/conda.sh
conda activate nanochat
cd /home/lizishuo/_program_/Project/nanochat
python -c "import sys; print(sys.executable); print(sys.version)"
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
exec > >(tee -a experiments/inference_infra/reports/d10_session.log) 2>&1
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。若环境不存在或第三方依赖缺失，先解决环境，不把 `--help` 成功当作模型实验通过。后面写入的 `experiments/` 文件是你实际执行教程时产生的正式成果；本指导不预先生成成绩。大数据、tokenizer、权重放在 `NANOCHAT_BASE_DIR`，不要加进 Git。表中的 `$NANOCHAT_BASE_DIR` 代表当前 shell 选定的真实绝对数据根；合成分支示例为 `/tmp/nanochat-w2-toy`，未设置时当前用户默认根为 `/home/lizishuo/.cache/nanochat`。交流时给出实际展开后的绝对路径；`*.parquet` 表示该目录内逐一列出的实际数据文件，旧数据回退目录为 `$NANOCHAT_BASE_DIR/base_data/`。

本 shell 后续终端输出同步保存在 `experiments/inference_infra/reports/d10_session.log`；命令返回后立即执行 `echo "exit=$?"`，避免丢失退出码。帮助、源码检索、解释器与 Git 检查只打印，不另有检查结果文件。日志是追加写入，重跑时交流中注明会话边界或日志中的起止位置。

## 今日目标与时间盒

今天先计划、再运行两次优化步，不追求模型效果。用 15 分钟核对输入、10 分钟审查命令、30–60 分钟执行（CPU 视硬件增加）、15 分钟加载验收。前置是 [token 与批次检查](d09_token_roundtrip.md) 中真实数据断言成功。若还没成功，今天先修它。

<abbr title="Smoke test · 用最小工作量检查一条真实路径是否能执行到结束。本文必须经过真实训练、保存、重新加载；两步成功不能证明收敛、质量或稳定性能。">smoke</abbr> checkpoint 必须由 nanochat 的模型、tokenizer 和保存格式共同产生。MiniMind 权重不能进入这条路径。

## 第一步：核对运行输入与不会覆盖的模型标签

```bash
printf '%s\n' "$NANOCHAT_BASE_DIR"
python .agents/docs/career_roadmap/w2/scripts/d09_roundtrip.py --check-data --output experiments/inference_infra/reports/d10_pretrain_input_check.json
```

若变量为空，先按昨天的分支设置它。`/tmp/nanochat-w2-toy` 是合成 smoke 选择，不是隐含默认值。确认剩余磁盘能装模型、优化器和日志；即便两步训练也会写 optimizer shard。标签 `d10_w2_smoke` 专门保留两步证据，后续二十步用另一标签。

## 第二步：只生成可审查的训练计划

打开 [有界训练脚本](scripts/d10_train_smoke.py)：

```bash
python .agents/docs/career_roadmap/w2/scripts/d10_train_smoke.py --help
python .agents/docs/career_roadmap/w2/scripts/d10_train_smoke.py --device cpu --steps 2 --model-tag d10_w2_smoke --output-dir experiments/inference_infra/reports/d10_train_smoke_2
```

此时只写 `experiments/inference_infra/reports/d10_train_smoke_2/d10_plan.json`，状态为 `PENDING`，并打印将要调用的原始训练命令；没有 `--execute` 就不训练。逐项检查参数含义：

| 配置 | 为什么这样选 |
|---|---|
| depth=4, aspect-ratio 默认64, head-dim 默认128 | 得到 C=256、2个query/KV头，规模小且满足整除 |
| max-seq-len=512 | 后续长输入需遵守 P+G≤512 |
| device-batch-size=1, total-batch-size=512 | 单设备每步512 token，累积次数为1 |
| num-iterations=2 | 明确覆盖默认按参数比例计算步数的逻辑 |
| window-pattern=L | 此阶段固定全上下文，减少短窗变量 |
| run=dummy | 使用 DummyWandb，不要求外部实验账户 |
| eval-every=-1, core-metric-every=-1, sample-every=-1 | 不在极短训练中额外跑评估和采样；eval-tokens=512仅保留显式小值 |
| warmup-steps=0 | 两步实验不使用默认40步学习率预热；仍不构成训练效果实验 |
| save-every 默认-1 | 最后一步仍保存，结束条件核对 base_train 的 last_step 分支 |

注意脚本没有禁止原生产训练代码的其他初始化行为；`base_train.py` 在训练前导入评估模块，并读取 token_bytes，因此关评估也不等于可以缺文件或缺依赖。

## 第三步：执行真实两步并检查日志

```bash
python .agents/docs/career_roadmap/w2/scripts/d10_train_smoke.py --device cpu --steps 2 --model-tag d10_w2_smoke --output-dir experiments/inference_infra/reports/d10_train_smoke_2 --execute
```

该命令将日志写入 `experiments/inference_infra/reports/d10_train_smoke_2/d10_train.log`，终端可能暂时没有逐步输出。另一终端可以 `tail -n 30 experiments/inference_infra/reports/d10_train_smoke_2/d10_train.log` 查看进度。训练结束后打开 `experiments/inference_infra/reports/d10_train_smoke_2/d10_plan.json` 看 `returncode`。若标签目录已存在，脚本主动拒绝；选择新的标签并将后续命令一起改成该标签，不删除旧证据。

实际保存应在 `$NANOCHAT_BASE_DIR/base_checkpoints/d10_w2_smoke/`，包含 `model_000002.pt`、`meta_000002.json` 与 `optim_000002_rank0.pt`。前两者用于推理；恢复优化还需要 optimizer shard 与元数据状态。不能用文件名存在替代成功加载。

## 第四步：立即重新加载并运行双路径 smoke

使用 [checkpoint 生成检查脚本](scripts/d10_checkpoint_probe.py)：

```bash
python .agents/docs/career_roadmap/w2/scripts/d10_checkpoint_probe.py --device cpu --model-tag d10_w2_smoke --step 2 --max-tokens 4 --output experiments/inference_infra/reports/d10_checkpoint_smoke_2.json
```

脚本调用官方 `load_model('base', device, phase='eval', model_tag=..., step=...)`。它验证词表大小并严格加载权重，然后对同一 prompt 运行无缓存 `GPT.generate` 与 `Engine.generate` 的 greedy 前缀。输出可能无意义，属于两步小模型预期；不要用生成文本是否通顺决定保存格式是否合法。

## 失败分流、产物与验收

- CPU 过慢：先保留日志，确认是模型构建、torch 编译还是数据 iterator；不要后台开多个训练进程。长于今日时间盒就记录 `TRAINING_PENDING`，下次续查。
- OOM：停止同机重复进程，核对当前 batch 与 dtype；不能把失败改成“默认已通过”。GPU 模式必须使用对应 Conda 的 GPU PyTorch；本教程不替你切换安装。
- loss 非有限值：保存日志与配置，停止往后扩步；先检查 tokenizer ID、学习率和精度，别只换 checkpoint 标签反复碰运气。
- 验收：计划的实际退出码0、三个保存文件对应 step2、probe `status=ok`；直接在聊天中说明路径、配置及本日问题的证据；`experiments/inference_infra/reports/d10_training_smoke.md` 仅作可选整理。若任一失败，写明门槛未过，二十步训练暂缓。

## 今日脚本与产物清单

每日编号使用全程 d01–d21；跨日复用保留首次实际执行日。以下项目相对路径均以仓库根目录为基准，脚本清单直接列出本日执行入口。

| 脚本完整项目路径 | 用途 | 执行阶段 | 输入 | 运行条件 |
|---|---|---|---|---|
| `.agents/docs/career_roadmap/w2/scripts/d09_roundtrip.py` | 复用输入校验 | 训练前 | `$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl`、`$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt`、`$NANOCHAT_BASE_DIR/base_data_climbmix/*.parquet` | --check-data 成功后再训练 |
| `.agents/docs/career_roadmap/w2/scripts/d10_train_smoke.py` | 生成计划并调用原生训练 | 第二、三步 | `$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl`、`$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt`、`$NANOCHAT_BASE_DIR/base_data_climbmix/*.parquet` 与命令参数 | --execute 才训练；标签目录不得存在 |
| `.agents/docs/career_roadmap/w2/scripts/d10_checkpoint_probe.py` | 重新加载与双路径生成 | 第四步 | `$NANOCHAT_BASE_DIR/base_checkpoints/d10_w2_smoke/model_000002.pt`、`$NANOCHAT_BASE_DIR/base_checkpoints/d10_w2_smoke/meta_000002.json`、`$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl` | 训练成功且 checkpoint 完整 |

训练包装器内部调用原生入口 `scripts/base_train.py`（`python -m scripts.base_train`），只在 `--execute` 时执行；输入为同一数据根下的 tokenizer 与训练 split，保留原生脚本名。

| 产物完整项目相对路径 | 用途 | 生成者 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/reports/d10_pretrain_input_check.json` | 训练前输入断言 | d09_roundtrip.py | 自动；记录真实状态 |
| `experiments/inference_infra/reports/d10_train_smoke_2/d10_plan.json` | 两步命令、状态与退出码 | d10_train_smoke.py | 自动；仅计划时为 PENDING |
| `experiments/inference_infra/reports/d10_train_smoke_2/d10_train.log` | 原生训练 stdout/stderr | d10_train_smoke.py | 条件自动；--execute 且标签检查通过后创建 |
| `experiments/inference_infra/reports/d10_checkpoint_smoke_2.json` | step 2 加载与双路径结果 | d10_checkpoint_probe.py | 条件自动；执行 probe 后写出，失败也记录 |
| `experiments/inference_infra/reports/d10_training_smoke.md` | 两步训练验收与阻塞 | 验收步骤 | 可选人工整理；可直接在聊天中回答本日问题 |
| `experiments/inference_infra/reports/d10_session.log` | 当天终端输出与手动记录的退出码 | 开场 shell 的 tee | 自动；仅捕获启用后的输出 |

权重位于仓库外；以下是当前 `NANOCHAT_BASE_DIR=/tmp/nanochat-w2-toy` 时的完整绝对路径。使用自定义数据根时在交流中逐项提供真实绝对路径。仅真实训练保存阶段执行后生成；训练中断时不可宣称齐全：

| 完整绝对路径（示例数据根） | 用途 | 生成者与方式 |
|---|---|---|
| `/tmp/nanochat-w2-toy/base_checkpoints/d10_w2_smoke/model_000002.pt` | 模型参数 | scripts/base_train.py → nanochat/checkpoint_manager.py；条件自动 |
| `/tmp/nanochat-w2-toy/base_checkpoints/d10_w2_smoke/meta_000002.json` | 配置与训练状态 | scripts/base_train.py → nanochat/checkpoint_manager.py；条件自动 |
| `/tmp/nanochat-w2-toy/base_checkpoints/d10_w2_smoke/optim_000002_rank0.pt` | 单进程优化器状态 | scripts/base_train.py → nanochat/checkpoint_manager.py；条件自动 |

原生 `tokenizer.pkl`、`token_bytes.pt`、`model_<step>.pt`、`meta_<step>.json` 与 `optim_<step>_rank0.pt` 是 loader 的格式契约，保留其名称；用带日编号的数据文件或模型标签归属日期。已有历史结果不原位重命名，新的命令使用本页命名。没有执行或条件不满足的必需检查均标 PENDING；可选 Markdown 未整理不影响验收。

## 今日自验证与交流问题

自行验证后，直接在聊天中逐题简答即可，不要求正式报告、排版或长文。每个结论引用相应日志、自动结果中的字段或源码位置；没有执行的检查明确标为 `PENDING`，遇到失败给出真实退出码与错误。需要本地留存时再选用上表的可选 Markdown。

1. 仅生成计划和加 `--execute` 的运行有什么区别？从 `d10_plan.json` 指出状态、真实命令、步数、batch、上下文长度与退出码，解释为什么两步实验要设置 warmup=0。

2. 训练实际执行了几次优化步，日志里的 loss 是否有限？引用 `d10_train.log` 的位置与关键输出；没有完成时说明卡在初始化、数据读取还是训练阶段。

3. 实际数据根、模型标签和 step2 的三个保存文件是什么？推理与恢复优化分别需要哪些文件，为什么文件存在还不足以验收？

4. 重新加载后的 probe 是否 `status=ok`？引用 `d10_checkpoint_smoke_2.json`，解释双路径输出可读性与 checkpoint 格式合法性有什么关系。

5. 两步训练、保存、重新加载三个门槛哪些已经通过？哪些仍未验证（如收敛、质量、GPU性能），能否进入二十步实验，下一步依据是什么？

离开前保留本日会话日志与自动结果；若追加重跑，注明会话边界。交流时可附 `git rev-parse HEAD`、`git status --short` 及实际设备/精度以定位证据，不必另填通用表。下一天可直接复用本聊天中已确认的结论与证据路径。共享前检查数据路径和日志是否包含私人信息；若提交成果，只纳入自己的轻量记录与脚本，不提交权重、数据、环境目录或缓存。
