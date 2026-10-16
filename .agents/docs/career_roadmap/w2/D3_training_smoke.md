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

前一天：[返回前一天](D2_token_roundtrip.md)；后一天：[继续下一天](D4_checkpoint_generation.md)；[原始总计划](../nanochat_daily_execution_2026-10-01.md)。

## 开始前：固定 shell 与结果位置

所有命令从仓库根目录执行。首次打开终端先运行以下准备；`conda activate` 若已可用，可省略 `source` 一行。这里使用本项目的 Conda 环境；不要混用系统 Python。CPU实验建议先 `export NANOCHAT_DTYPE=float32`；脚本只在变量未设置时选择FP32，已有变量会遵从并记录，不能假定CPU必定FP32。GPU实验另按协议显式选dtype。

```bash
source /home/lizishuo/miniconda3/etc/profile.d/conda.sh
conda activate nanochat
cd /home/lizishuo/_program_/Project/nanochat
python -c "import sys; print(sys.executable); print(sys.version)"
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。若环境不存在或第三方依赖缺失，先解决环境，不把 `--help` 成功当作模型实验通过。后面写入的 `experiments/` 文件是你实际执行教程时产生的正式成果；本指导不预先生成成绩。大数据、tokenizer、权重放在 `NANOCHAT_BASE_DIR`，不要加进 Git。

## 今日目标与时间盒

今天先计划、再运行两次优化步，不追求模型效果。用 15 分钟核对输入、10 分钟审查命令、30–60 分钟执行（CPU 视硬件增加）、15 分钟加载验收。前置是 [token 与批次检查](D2_token_roundtrip.md) 中真实数据断言成功。若还没成功，今天先修它。

<abbr title="Smoke test · 用最小工作量检查一条真实路径是否能执行到结束。本文必须经过真实训练、保存、重新加载；两步成功不能证明收敛、质量或稳定性能。">smoke</abbr> checkpoint 必须由 nanochat 的模型、tokenizer 和保存格式共同产生。MiniMind 权重不能进入这条路径。

## 第一步：核对运行输入与不会覆盖的模型标签

```bash
printf '%s\n' "$NANOCHAT_BASE_DIR"
python .agents/docs/career_roadmap/w2/scripts/roundtrip.py --check-data --output experiments/inference_infra/reports/pretrain_input_check.json
```

若变量为空，先按昨天的分支设置它。`/tmp/nanochat-w2-toy` 是合成 smoke 选择，不是隐含默认值。确认剩余磁盘能装模型、优化器和日志；即便两步训练也会写 optimizer shard。标签 `d4-w2-smoke` 专门保留两步证据，后续二十步用另一标签。

## 第二步：只生成可审查的训练计划

打开 [有界训练脚本](scripts/train_smoke.py)：

```bash
python .agents/docs/career_roadmap/w2/scripts/train_smoke.py --help
python .agents/docs/career_roadmap/w2/scripts/train_smoke.py --device cpu --steps 2 --model-tag d4-w2-smoke --output-dir experiments/inference_infra/reports/train_smoke_2
```

此时只写 `plan.json`，状态为 `PENDING`，并打印将要调用的原始训练命令；没有 `--execute` 就不训练。逐项检查参数含义：

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
python .agents/docs/career_roadmap/w2/scripts/train_smoke.py --device cpu --steps 2 --model-tag d4-w2-smoke --output-dir experiments/inference_infra/reports/train_smoke_2 --execute
```

该命令将日志写入 `train_smoke_2/train.log`，终端可能暂时没有逐步输出。另一终端可以 `tail -n 30 experiments/inference_infra/reports/train_smoke_2/train.log` 查看进度。训练结束后打开 `plan.json` 看 `returncode`。若标签目录已存在，脚本主动拒绝；选择新的标签并将后续命令一起改成该标签，不删除旧证据。

实际保存应在 `$NANOCHAT_BASE_DIR/base_checkpoints/d4-w2-smoke/`，包含 `model_000002.pt`、`meta_000002.json` 与 `optim_000002_rank0.pt`。前两者用于推理；恢复优化还需要 optimizer shard 与元数据状态。不能用文件名存在替代成功加载。

## 第四步：立即重新加载并运行双路径 smoke

使用 [checkpoint 生成检查脚本](scripts/checkpoint_probe.py)：

```bash
python .agents/docs/career_roadmap/w2/scripts/checkpoint_probe.py --device cpu --model-tag d4-w2-smoke --step 2 --max-tokens 4 --output experiments/inference_infra/reports/checkpoint_smoke_2.json
```

脚本调用官方 `load_model('base', device, phase='eval', model_tag=..., step=...)`。它验证词表大小并严格加载权重，然后对同一 prompt 运行无缓存 `GPT.generate` 与 `Engine.generate` 的 greedy 前缀。输出可能无意义，属于两步小模型预期；不要用生成文本是否通顺决定保存格式是否合法。

## 失败分流、产物与验收

- CPU 过慢：先保留日志，确认是模型构建、torch 编译还是数据 iterator；不要后台开多个训练进程。长于今日时间盒就记录 `TRAINING_PENDING`，下次续查。
- OOM：停止同机重复进程，核对当前 batch 与 dtype；不能把失败改成“默认已通过”。GPU 模式必须使用对应 Conda 的 GPU PyTorch；本教程不替你切换安装。
- loss 非有限值：保存日志与配置，停止往后扩步；先检查 tokenizer ID、学习率和精度，别只换 checkpoint 标签反复碰运气。
- 验收：计划的实际退出码0、三个保存文件对应 step2、probe `status=ok`；在 `reports/training_smoke.md` 记录路径和配置。若任一失败，写明门槛未过，二十步训练暂缓。

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
