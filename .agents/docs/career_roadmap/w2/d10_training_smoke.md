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

- **阅读入口**：[6.3 logits、目标对齐与有效监督](../../llm-training-workflow/06-forward-pass-and-prediction-loss.md#reading-6-3)；[7.1 一次更新的真实次序](../../llm-training-workflow/07-backpropagation-and-parameter-updates.md#reading-7-1)；[7.4 一次更新的流程图](../../llm-training-workflow/07-backpropagation-and-parameter-updates.md#reading-7-4)。
- **今天重点**：把昨日目标对齐接到有效目标损失、反向、梯度累积和参数更新，结合两步训练区分 backward 与 optimizer.step。
- **暂时跳过**：7.2.1 的算法数值和 7.3 的完整预算公式以后按需深读；今天只看训练计划里的实际步数。

**读完能解释**：loss.backward 和 optimizer.step 各做什么？微批次数为什么不一定等于参数更新次数？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 开始前：固定 shell 与结果位置

所有命令从仓库根目录执行。当前终端会自动激活 nanochat 环境，无需重复激活；先运行以下检查。这里使用本项目的 Conda 环境；不要混用系统 Python。CPU实验建议先 `export NANOCHAT_DTYPE=float32`；脚本只在变量未设置时选择FP32，已有变量会遵从并记录，不能假定CPU必定FP32。GPU实验另按协议显式选dtype。

日志只在 `{ ...; }` 命令组执行期间记录；<abbr title="Bash process substitution · 进程替换。这里把命令组的输出交给 tee，同时显示并追加到日志；重定向在命令组结束后恢复。">进程替换</abbr>不会持续接管交互终端，因此提示符和 VS Code 终端集成控制字符不会混入日志。命令组在当前 shell 执行，变量设置会保留；代码块返回最后一条命令的退出码，需核对单条命令时立即记录 `$?`。已有的显式文件重定向仍写入各自文件。复制时包含外层花括号和末尾重定向。若之前执行过旧版全局日志命令，先关闭旧终端并新开终端；仅修改文档不会停止旧终端的记录。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
{
python -c "import sys; print(sys.executable); print(sys.version)"
} > >(tee -a experiments/inference_infra/reports/d10_session.log) 2>&1
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。若环境不存在或第三方依赖缺失，先解决环境，不把 `--help` 成功当作模型实验通过。后面写入的 `experiments/` 文件是你实际执行教程时产生的正式成果；本指导不预先生成成绩。大数据、tokenizer、权重放在 `NANOCHAT_BASE_DIR`，不要加进 Git。表中的 `$NANOCHAT_BASE_DIR` 代表当前 shell 选定的真实绝对数据根；合成分支示例为 `/tmp/nanochat-w2-toy`，未设置时当前用户默认根为 `/home/lizishuo/.cache/nanochat`。交流时给出实际展开后的绝对路径；`*.parquet` 表示该目录内逐一列出的实际数据文件，旧数据回退目录为 `$NANOCHAT_BASE_DIR/base_data/`。

本页各 Bash 代码块内的输出追加保存在 `experiments/inference_infra/reports/d10_session.log`；命令返回后立即执行 `echo "exit=$?"`，避免丢失退出码。帮助、源码检索、解释器与 Git 检查只打印，不另有检查结果文件。日志是追加写入，重跑时交流中注明会话边界或日志中的起止位置。

## 今日目标与时间盒

今天先计划、再运行两次优化步，不追求模型效果。用 15 分钟核对输入、10 分钟审查命令、30–60 分钟执行（CPU 视硬件增加）、15 分钟加载验收。前置是 [token 与批次检查](d09_token_roundtrip.md) 中真实数据断言成功。若还没成功，今天先修它。

<abbr title="Smoke test · 用最小工作量检查一条真实路径是否能执行到结束。本文必须经过真实训练、保存、重新加载；两步成功不能证明收敛、质量或稳定性能。">smoke</abbr> checkpoint 必须由 nanochat 的模型、tokenizer 和保存格式共同产生。MiniMind 权重不能进入这条路径。

## 第一步：核对运行输入与不会覆盖的模型标签

```bash
{
printf '%s\n' "$NANOCHAT_BASE_DIR"
python .agents/docs/career_roadmap/w2/scripts/d09_roundtrip.py --check-data --output experiments/inference_infra/reports/d10_pretrain_input_check.json
} > >(tee -a experiments/inference_infra/reports/d10_session.log) 2>&1
```

若变量为空，先按昨天的分支设置它。`/tmp/nanochat-w2-toy` 是合成 smoke 选择，不是隐含默认值。确认剩余磁盘能装模型、优化器和日志；即便两步训练也会写 optimizer shard。标签 `d10_w2_smoke` 专门保留两步证据，后续二十步用另一标签。

## 第二步：只生成可审查的训练计划

打开 [有界训练脚本](scripts/d10_train_smoke.py)：

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d10_train_smoke.py --help
python .agents/docs/career_roadmap/w2/scripts/d10_train_smoke.py --device cpu --steps 2 --model-tag d10_w2_smoke --output-dir experiments/inference_infra/reports/d10_train_smoke_2
} > >(tee -a experiments/inference_infra/reports/d10_session.log) 2>&1
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
{
python .agents/docs/career_roadmap/w2/scripts/d10_train_smoke.py --device cpu --steps 2 --model-tag d10_w2_smoke --output-dir experiments/inference_infra/reports/d10_train_smoke_2 --execute
} > >(tee -a experiments/inference_infra/reports/d10_session.log) 2>&1
```

该命令将日志写入 `experiments/inference_infra/reports/d10_train_smoke_2/d10_train.log`，终端可能暂时没有逐步输出。另一终端可以 `tail -n 30 experiments/inference_infra/reports/d10_train_smoke_2/d10_train.log` 查看进度。训练结束后打开 `experiments/inference_infra/reports/d10_train_smoke_2/d10_plan.json` 看 `returncode`。若标签目录已存在，脚本主动拒绝；选择新的标签并将后续命令一起改成该标签，不删除旧证据。

实际保存应在 `$NANOCHAT_BASE_DIR/base_checkpoints/d10_w2_smoke/`，包含 `model_000002.pt`、`meta_000002.json` 与 `optim_000002_rank0.pt`。前两者用于推理；恢复优化还需要 optimizer shard 与元数据状态。不能用文件名存在替代成功加载。

## 第四步：立即重新加载并运行双路径 smoke

使用 [checkpoint 生成检查脚本](scripts/d10_checkpoint_probe.py)：

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d10_checkpoint_probe.py --device cpu --model-tag d10_w2_smoke --step 2 --max-tokens 4 --output experiments/inference_infra/reports/d10_checkpoint_smoke_2.json
} > >(tee -a experiments/inference_infra/reports/d10_session.log) 2>&1
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
| `experiments/inference_infra/reports/d10_session.log` | 当天终端输出与手动记录的退出码 | 各 Bash 代码块的 tee | 自动；仅追加各代码块内的输出 |

权重位于仓库外；以下是当前 `NANOCHAT_BASE_DIR=/tmp/nanochat-w2-toy` 时的完整绝对路径。使用自定义数据根时在交流中逐项提供真实绝对路径。仅真实训练保存阶段执行后生成；训练中断时不可宣称齐全：

| 完整绝对路径（示例数据根） | 用途 | 生成者与方式 |
|---|---|---|
| `/tmp/nanochat-w2-toy/base_checkpoints/d10_w2_smoke/model_000002.pt` | 模型参数 | scripts/base_train.py → nanochat/checkpoint_manager.py；条件自动 |
| `/tmp/nanochat-w2-toy/base_checkpoints/d10_w2_smoke/meta_000002.json` | 配置与训练状态 | scripts/base_train.py → nanochat/checkpoint_manager.py；条件自动 |
| `/tmp/nanochat-w2-toy/base_checkpoints/d10_w2_smoke/optim_000002_rank0.pt` | 单进程优化器状态 | scripts/base_train.py → nanochat/checkpoint_manager.py；条件自动 |

原生 `tokenizer.pkl`、`token_bytes.pt`、`model_<step>.pt`、`meta_<step>.json` 与 `optim_<step>_rank0.pt` 是 loader 的格式契约，保留其名称；用带日编号的数据文件或模型标签归属日期。已有历史结果不原位重命名，新的命令使用本页命名。没有执行或条件不满足的必需检查均标 PENDING；可选 Markdown 未整理不影响验收。

## 今日自验证与交流问题

可直接在聊天中按题号回答，不要求手写报告。每题给出结论和可定位依据（源码位置、命令与退出码、日志片段或自动结果字段）；同一份证据可复用。概念推演须注明假设，实际运行须区分本次与历史结果。未执行写 `PENDING`，失败写真实错误和影响范围，不能据此勾选实验通过。

1. D09 的哪份证据允许今天启动训练？本次解释器、数据根、设备与实际精度分别是什么？

   **通过标准**：引用 `d10_pretrain_input_check.json.status`、两个 batch 检查及命令退出码，核对本次环境与日志；CPU 包装器使用 `setdefault`，已有 `NANOCHAT_DTYPE` 不会被强制改为 FP32，实际精度以训练日志为准，不能只引用推荐配置。

2. 仅生成计划与添加 `--execute` 分别产生什么？如何分辨训练未执行、成功和失败？

   **通过标准**：从 `d10_plan.json` 指出 `command`、`cwd`、`status`、`steps`、`model_tag`、`base_dir`；解释未执行为 `PENDING` 且没有 `returncode`，执行后才有 `returncode` 与 `checkpoint_dir` 及训练日志；包装器计划成功退出 0 不代表训练已经运行。

3. depth=4、aspect-ratio=64、head-dim=128 怎样得到模型宽度和头数？为什么最大上下文是 512？

   **通过标准**：对照 `base_train.py:build_model_meta` 计算 C=256、query/KV 各 2 头并满足整除；从计划命令的 `--max-seq-len=512` 定位配置，说明后续 prompt 的 BOS 也占 P、生成需遵守 P+G≤512，不能拿词表大小代替上下文长度。

4. 单设备 B=1、T=512、total-batch-size=512 时，一次参数更新包含几个微批次？假设总批量改为 1024 会怎样？

   **通过标准**：引用 `base_train.py` 的 `world_tokens_per_fwdbwd`、`grad_accum_steps` 与整除断言，计算原配置累积 1 次、假设改为 1024 累积 2 次；解释 backward 累加梯度、loss 除以累积次数，而 optimizer.step 在累积完成后更新一次，区分 token 数与句子数。

5. 一个训练微批次如何从 `[B,T]` 的 x/y 得到 loss，再完成一次更新？

   **通过标准**：按 `gpt.py:GPT.forward` 与 `base_train.py` 说明输出 `[B,T,V]`、去词表 padding、FP32 softcap 后交叉熵、目标为下一 token、有效目标归约，随后 backward、学习率/优化器更新及清梯度；可复用 D09 的错位例子，不能把 logits 直接当预测 ID。

6. 为什么显式限制两步、warmup=0、关闭评估/采样仍需要 token_bytes 和相关依赖？

   **通过标准**：引用计划中 `--num-iterations=2`、`--warmup-steps=0`、三个频率 -1 及 `base_train.py` 的初始化和调度分支；解释两步不适用默认 40 步预热，显式步数覆盖自动预算，评估关闭不跳过模块导入和 `get_token_bytes`，不声称因此能无数据训练。

7. 实际执行了几次优化更新？日志 loss 是否有限，step 2 保存是否意味着进行了第三次更新？

   **通过标准**：引用 `d10_train.log` 的 step 0/1 训练行、末尾保存行及真实退出码；依据 `base_train.py` 的 `last_step`、保存后 break 解释两次更新后在 step=2 保存，没有第三次 optimizer.step；未完成则报告停在哪一阶段及已观察范围。

8. 当前标签对应哪三个原生 checkpoint 文件？推理与恢复优化各需要什么，如何避免覆盖两步证据？

   **通过标准**：列出真实数据根下 `model_000002.pt`、`meta_000002.json`、`optim_000002_rank0.pt`；依据 `checkpoint_manager.py` 区分参数/配置与优化器、训练状态，说明包装器拒绝已存在标签，换新标签并同步后续命令，文件存在不能替代重新加载。

9. 重新加载后的双路径检查通过了吗？greedy 前缀一致具体比较什么，又没有比较什么？

   **通过标准**：引用 `d10_checkpoint_smoke_2.json` 的 `status`、`reference_ids`、`engine_ids`、`common_prefix_match`、`model_config` 及退出码；解释严格加载、词表大小检查、temperature=0、包含终止 ID 的共同前缀，Engine 可提前结束，脚本没有证明逐层 logits 等价或生成文本质量。

10. 训练、保存、重载三个门槛是否全部通过，是否能进入 D11？如果 OOM、非有限 loss 或训练未完成，影响范围是什么？

    **通过标准**：把计划 `returncode`、日志、三个文件和 probe 结果逐一对应；缺一则说明真实错误或 `PENDING` 并暂缓二十步，不以无意义文本判定格式失败，也不以两步成功宣称收敛、泛化、GPU 性能或长期稳定。

离开前保留本日会话日志与自动结果；若追加重跑，注明会话边界。交流时可附 `git rev-parse HEAD`、`git status --short` 及实际设备/精度以定位证据，不必另填通用表。下一天可直接复用本聊天中已确认的结论与证据路径。共享前检查数据路径和日志是否包含私人信息；若提交成果，只纳入自己的轻量记录与脚本，不提交权重、数据、环境目录或缓存。
