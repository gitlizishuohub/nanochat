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

今天回答一个问题：**昨天的输入与目标，怎样使模型下一次预测发生变化？** 前半页带你从一个预测错误推到参数更新，再把这条线对应到项目代码；这一部分只需阅读、手算，无需模型权重或 GPU。

后半页另有明确标出的 **CPU 实际操作**：用 D9 的合成小词表与数据，从随机初始化开始训练一个小模型，两次更新后保存并重载。它确实是训练，会更新权重；不要求你以前训练过模型。只阅读或生成计划时，没有执行训练；未做的 CPU 操作按实际记录即可。GPU 运行属于自选扩展，不是本日要求，也不需要记成待补欠账。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[6.3 logits、目标对齐与有效监督](../../llm-training-workflow/06-forward-pass-and-prediction-loss.md#reading-6-3)；[7.1 一次更新的真实次序](../../llm-training-workflow/07-backpropagation-and-parameter-updates.md#reading-7-1)；[7.4 一次更新的流程图](../../llm-training-workflow/07-backpropagation-and-parameter-updates.md#reading-7-4)。
- **今天重点**：把昨日目标对齐接到有效目标损失、反向、梯度累积和参数更新，结合两步训练区分 backward 与 optimizer.step。
- **暂时跳过**：7.2.1 的算法数值和 7.3 的完整预算公式以后按需深读；今天只看训练计划里的实际步数。

**读完能解释**：loss.backward 和 optimizer.step 各做什么？微批次数为什么不一定等于参数更新次数？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 1. 概念推演：一个错误预测怎样成为训练信号

本节例子是假设值，不是运行结果。沿用 D9 的错位目标，假设一段文本编码为 `[10, 20, 30, 40]`，取 `T=3`，得到：

| 位置 | 模型本位置输入 x | 本位置应预测的 y | 该预测能使用的上下文 |
|---|---|---|---|
| 0 | 10 | 20 | `[10]` |
| 1 | 20 | 30 | `[10, 20]` |
| 2 | 30 | 40 | `[10, 20, 30]` |

训练时三个位置可以一并计算，但因果注意力不允许位置 0 偷看输入中的 20、30。目标 `y` 是比较预测用的答案，不是额外喂给模型的未来上下文。

### 1.1 前向给出分数，loss 衡量目标概率

模型先计算 <abbr title="Forward pass · 前向计算。把输入通过当前模型参数变成预测分数；传入目标时，nanochat 的 GPT.forward 还会计算并返回损失。">前向</abbr>。每个位置输出对整个词表的分数，称为 <abbr title="Logits · 未归一化预测分数。每个位置有 V 个分数，V 是词表大小；交叉熵内部使用这些连续分数计算目标类别的负对数概率。">logits</abbr>。若批量为 `B=1`、长度为 `T=3`、词表大小为 `V`，输入与目标均是 `[1,3]`，logits 是 `[1,3,V]`，不是三个已经选好的 token。

假设三个位置对正确答案 20、30、40 给出的概率分别为 `0.5、0.25、0.5`。<abbr title="Cross-Entropy · 交叉熵。此处对每个有效位置的目标概率取负自然对数，再取平均；预测正确目标的概率越小，损失越大。">交叉熵</abbr>的平均损失是：

```text
loss = (-ln(0.5) - ln(0.25) - ln(0.5)) / 3 ≈ 0.9242
```

如果中间位置给正确答案 30 的概率从 `0.25` 升到 `0.5`，另外两个不变，平均 loss 就降为约 `0.6931`。这说明 loss 如何表达预测改善；并不是说实际训练每一步都保证所有位置改善。

训练需要保留整个连续分数计算过程。若先取最大分数的 token ID，再只比较“对/错”，就丢失了目标概率及其变化信息；argmax 的离散选择也无法提供这里需要的梯度。生成时选 token，训练时根据分数算 loss，是两种不同用途。

### 1.2 backward 找方向，step 才改变参数

<abbr title="Gradient · 梯度。损失对模型参数的局部变化率；backward 将计算结果累加到参数的 .grad 中，优化器再据此更新参数。">梯度</abbr>回答的是：“某个参数略微增大，loss 在当前位置会怎样变化？”`loss.backward()` 沿前向建立的依赖关系计算这些变化率。此时参数通常还没有改变，改变的是参数旁边记录的 `.grad`。

用一个只为理解方向的标量例子：若参数 `w=1`，梯度为 `+0.2`，学习率为 `0.1`，最简单的梯度下降会执行 `w = 1 - 0.1×0.2 = 0.98`。正梯度意味着局部增大 w 会增大 loss，所以沿相反方向走一小步。**这是假设的普通梯度下降，不是 nanochat 优化器的实际数值公式。** 本项目的 [GPT.setup_optimizer](../../../../nanochat/gpt.py) 将 Transformer 块矩阵交给 <abbr title="Muon · 本项目用于 Transformer 块矩阵参数的优化算法；它使用动量和矩阵变换等步骤处理梯度，不等同于直接减去学习率乘原始梯度。">Muon</abbr>，将嵌入、输出头和标量等交给 <abbr title="AdamW · 使用梯度的一阶、二阶历史统计并采用解耦权重衰减的优化算法；因此需要随参数保存用于恢复的优化器状态。">AdamW</abbr>；它们还使用历史状态等信息，不能套用这个算式预测真实权重。

`optimizer.step()` 使用梯度与优化器状态更新参数，随后 `model.zero_grad(set_to_none=True)` 清理本轮梯度，防止下一次更新混入上一轮的值。下一批输入的前向由更新后的参数计算。这条循环才是训练；只运行生成过程没有这两项反向与更新操作。

## 2. 源码带读：把一个训练步拼起来

先打开 [GPT.forward](../../../../nanochat/gpt.py)，直接定位末尾 `if targets is not None`。省略中间网络层后，这几行保留了今天需要的关系：

```python
logits = self.lm_head(x)
logits = logits[..., :self.config.vocab_size]
logits = logits.float()
logits = softcap * torch.tanh(logits / softcap)
if targets is not None:
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                           targets.view(-1), ignore_index=-1,
                           reduction=loss_reduction)
    return loss
else:
    return logits
```

`lm_head` 把每个位置的隐藏向量投影为词表分数；切片去掉实现中为了对齐补出的额外词表行，再转 FP32 并平滑限制分数幅度。`view` 把 `[B,T,V]` 展平为 `[B×T,V]`、把目标展平为 `[B×T]`，所以每一行分数仍对应原来的一个目标位置。默认 `loss_reduction='mean'`，目标等于 `-1` 的位置不参与损失；本日普通预训练批次来自 D9 的有效 token 错位，不应把所有位置都当成被忽略。

再打开 [scripts/base_train.py](../../../../scripts/base_train.py)，找到 `for micro_step in range(grad_accum_steps)`。下面摘取的是本日 CPU FP32、没有 loss scaler 的分支；源码还会取下一批输入、调整学习率并记录日志：

```python
for micro_step in range(grad_accum_steps):
    loss = model(x, y)
    loss = loss / grad_accum_steps
    loss.backward()
    # 原代码在这里提前取下一批 x/y
optimizer.step()
model.zero_grad(set_to_none=True)
```

第一行不是生成文本：`y` 被传进 `GPT.forward`，因此返回的是用于反向的标量 loss。梯度在每个小批次的 backward 中累加，到循环外才统一更新参数。这个小批次称为 <abbr title="Micro-batch · 微批次。一次前向和反向处理的小批量；若设备放不下一个更新所需的总 token 数，可以多次计算梯度后统一更新。">微批次</abbr>。

本日单进程、`B=1、T=512`，一次前向/反向处理 `512` 个目标位置，`total-batch-size=512`，所以累积次数为 `512/512=1`。两次训练更新各做一次前向、一次反向、一次 step，总共使用 `1024` 个目标位置；这是处理量，不保证有 1024 个互不相同的 token。若把总批量设为 `1024`，累积次数变为 2，每个小批次的 loss 先除以 2，使累加梯度对应两个等大小小批次的平均损失。把 `optimizer.step()` 移进循环，就会在两份数据之间改变参数，变成两次不同的更新。

### 2.1 为什么训练两步，却有 step=2 的保存循环

`base_train.py` 的循环先判断是否最后一步并保存，再决定是否进入更新段。初始 `step=0` 时做第 1 次更新，接着 `step=1` 做第 2 次；下一轮到 `step=2`，`last_step=True`，保存后 `break`，不再反向或更新。因此日志里的训练行通常是 `step 00000/00002`、`step 00001/00002`，保存名却是 `model_000002.pt`。不能只按打印行号猜更新次数。

训练日志的 `loss:` 是经指数移动平均平滑后的值；启用多微批次时，当前实现取最后一个微批次的 loss 参与日志平滑，并非另行算出的整步精确平均。`tok/sec` 是训练步处理 token 的速率，包含前向、反向和更新等工作；它也不是 D7 的生成吞吐。CPU 下 `bf16_mfu` 字段不具备 GPU 利用率意义。

## 3. CPU 实际操作前：固定 shell 与结果位置

以下才是终端操作。先选明本次范围：只运行不带 `--execute` 的计划命令，不会训练；运行第三步的 `--execute` 命令则会在 CPU 上真正训练小模型。所有命令从仓库根目录执行。当前终端会自动激活 nanochat 环境，无需重复激活。这里使用本项目的 Conda 环境；不要混用系统 Python。下面显式设置 `NANOCHAT_DTYPE=float32`；脚本内部仅在变量未设置时选 FP32，不能替你覆盖旧值。

日志只在 `{ ...; }` 命令组执行期间记录；<abbr title="Bash process substitution · 进程替换。这里把命令组的输出交给 tee，同时显示并追加到日志；重定向在命令组结束后恢复。">进程替换</abbr>不会持续接管交互终端，因此提示符和 VS Code 终端集成控制字符不会混入日志。命令组在当前 shell 执行，变量设置会保留；代码块返回最后一条命令的退出码，需核对单条命令时立即记录 `$?`。已有的显式文件重定向仍写入各自文件。复制时包含外层花括号和末尾重定向。若之前执行过旧版全局日志命令，先关闭旧终端并新开终端；仅修改文档不会停止旧终端的记录。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
{
export NANOCHAT_DTYPE=float32
python -c "import sys; print(sys.executable); print(sys.version)"
} > >(tee -a experiments/inference_infra/reports/d10_session.log) 2>&1
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。若环境不存在或第三方依赖缺失，先解决环境，不把 `--help` 成功当作模型实验通过。后面写入的 `experiments/` 文件是你实际执行教程时产生的正式成果；本指导不预先生成成绩。大数据、tokenizer、权重放在 `NANOCHAT_BASE_DIR`，不要加进 Git。表中的 `$NANOCHAT_BASE_DIR` 代表当前 shell 选定的真实绝对数据根；合成分支示例为 `/tmp/nanochat-w2-toy`，未设置时当前用户默认根为 `/home/lizishuo/.cache/nanochat`。交流时给出实际展开后的绝对路径；`*.parquet` 表示该目录内逐一列出的实际数据文件，旧数据回退目录为 `$NANOCHAT_BASE_DIR/base_data/`。

本页各 Bash 代码块内的输出追加保存在 `experiments/inference_infra/reports/d10_session.log`；命令返回后立即执行 `echo "exit=$?"`，避免丢失退出码。帮助、源码检索、解释器与 Git 检查只打印，不另有检查结果文件。日志是追加写入，重跑时交流中注明会话边界或日志中的起止位置。

### 本次小训练需要什么，不需要什么

使用 [D9](d09_token_roundtrip.md) 中已准备并校验成功的合成 tokenizer 和数据，或者你主动选择的真实语料分支；不需要已有模型 checkpoint，训练入口会创建并随机初始化模型。合成分支只检验路径，不代表真实语料训练效果。若 D9 未通过，仍可学完前两节；实际训练停在输入准备处，不能假定文件已存在。

实际分支先审查计划，再进行两次更新与重载。可给执行留 30–60 分钟的观察时间，但这不是完成时长保证：CPU、词表规模和首次 `torch.compile` 开销都影响耗时；已有大词表也会使“小深度”模型变大。无需为了阅读本页先跑完训练。

本日的 <abbr title="Smoke test · 用少量真实计算检查训练、保存和重载这条路径能否走通。两步成功不证明模型收敛、语言质量、长训稳定性或 GPU 性能。">smoke</abbr> 只验证这条短路径：nanochat 初始化 → 两次参数更新 → 保存 → 重新加载 → 少量推理。权重、配置与 tokenizer 必须配套；MiniMind 权重不能直接进入这条加载路径。

## 第一步：核对运行输入与不会覆盖的模型标签

```bash
{
printf '%s\n' "$NANOCHAT_BASE_DIR"
python .agents/docs/career_roadmap/w2/scripts/d09_roundtrip.py --check-data --output experiments/inference_infra/reports/d10_pretrain_input_check.json
} > >(tee -a experiments/inference_infra/reports/d10_session.log) 2>&1
```

若变量为空，先按昨天的分支设置它。`/tmp/nanochat-w2-toy` 是合成 smoke 选择，不是隐含默认值。确认剩余磁盘能装模型、优化器和日志；即便两步训练也会写 optimizer shard。标签 `d10_w2_smoke` 专门保留两步证据；若以后另做训练实验，应使用新的标签。本周不要求额外训练二十步。

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

模型宽度的来历可在 `base_train.py` 的 `build_model_meta` 中核对：先算 `depth×aspect_ratio`，再向上取整为 `head_dim` 的整数倍，最后用宽度除以 `head_dim` 得到头数。本日为 `4×64=256`，无需再取整，头数为 `256/128=2`；不要把这个头数与词表大小混淆。

注意脚本没有禁止原生产训练代码的其他初始化行为；`base_train.py` 在训练前导入评估模块，并读取 token_bytes，因此关评估也不等于可以缺文件或缺依赖。

## 第三步：选择执行 CPU 两步小模型训练

**运行下面这条命令就是训练，会计算梯度、更新参数并写 checkpoint。** 如果本次只学习概念和审查计划，到第二步即可；第三、四步可等你选择实际训练时再做。以下不是要求下载或训练一个可用聊天模型。

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d10_train_smoke.py --device cpu --steps 2 --model-tag d10_w2_smoke --output-dir experiments/inference_infra/reports/d10_train_smoke_2 --execute
} > >(tee -a experiments/inference_infra/reports/d10_session.log) 2>&1
```

该命令将日志写入 `experiments/inference_infra/reports/d10_train_smoke_2/d10_train.log`，终端可能暂时没有逐步输出。另一终端可以 `tail -n 30 experiments/inference_infra/reports/d10_train_smoke_2/d10_train.log` 查看进度。训练结束后打开 `experiments/inference_infra/reports/d10_train_smoke_2/d10_plan.json` 看 `returncode`。若标签目录已存在，脚本主动拒绝；选择新的标签并将后续命令一起改成该标签，不删除旧证据。

实际保存应在 `$NANOCHAT_BASE_DIR/base_checkpoints/d10_w2_smoke/`，包含 `model_000002.pt`、`meta_000002.json` 与 `optim_000002_rank0.pt`。前两者用于推理；恢复优化还需要 optimizer shard 与元数据状态。不能用文件名存在替代成功加载。

按下面的因果顺序读输出，而不是只找一个“成功”字样：

| 观察结果 | 可以说明什么 | 还不能说明什么 |
|---|---|---|
| `d10_plan.json` 的 `status=PENDING`，没有 `returncode` | 包装器写出了训练计划 | 没有证明训练启动；包装器此时退出 0 也只是计划成功 |
| `d10_train.log` 出现模型配置、`gradient accumulation steps: 1` | 初始化与批量设置已运行到相应位置 | 不代表两次更新已经完成 |
| 两条训练行、step 2 的保存记录，计划 `returncode=0` | 运行走到了两次更新后的保存与正常退出 | 包装器不会另行检查每个 loss 的数值健康；仍应检查是否有 NaN/Inf |
| probe `status=ok` | 保存资产可加载，指定提示下两条生成路径的前缀匹配 | 不证明语言质量、logits 完全一致或训练恢复成功 |

两步 loss 上下波动并不单独判定失败：两步接收的输入可以不同，日志又经过平滑。反过来，两步 loss 下降也不足以证明泛化能力。本日检查的是数值是否有限、路径是否完整；模型能力需要另设数据和评估。

## 第四步：立即重新加载并运行双路径 smoke

使用 [checkpoint 生成检查脚本](scripts/d10_checkpoint_probe.py)：

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d10_checkpoint_probe.py --device cpu --model-tag d10_w2_smoke --step 2 --max-tokens 4 --output experiments/inference_infra/reports/d10_checkpoint_smoke_2.json
} > >(tee -a experiments/inference_infra/reports/d10_session.log) 2>&1
```

脚本调用官方 `load_model('base', device, phase='eval', model_tag=..., step=...)`。它先按 metadata 重建模型并严格加载权重，再检查 tokenizer 词表大小，然后对同一 prompt 比较无缓存 `GPT.generate` 与 `Engine.generate` 的 <abbr title="Greedy decoding · 贪心解码。此处 temperature=0，按当前预测分数选择最大者，用于减少随机采样对双路径比较的干扰。">greedy</abbr> 输出前缀。JSON 中 `prompt_ids` 是输入，`reference_ids` 和 `engine_ids` 是输出，`common_prefix_match` 是比较结果。Engine 遇终止 ID 时可能先结束，所以比较共同前缀；这不是完整 logits 等价测试。

严格加载可检查参数键及形状是否匹配；词表大小相等仍不能证明同样 ID 对应同样文本，因此要一直使用 D9 配套 tokenizer。此 probe 没有加载 optimizer 状态，也未续训，不能验收断点恢复。输出可能无意义，属于两步小模型预期；不要用生成文本是否通顺决定保存格式是否合法。

## 失败分流、产物与验收

- CPU 过慢：先保留日志，确认是模型构建、torch 编译还是数据 iterator；不要后台开多个训练进程。可以停止本次操作，记录实际停留阶段；概念学习不因此失败，也不要求改用 GPU。
- OOM：停止同机重复进程，核对当前 batch 与 dtype；不能把失败改成“默认已通过”。GPU 模式必须使用对应 Conda 的 GPU PyTorch；本教程不替你切换安装。
- loss 非有限值：保存日志与配置，停止往后扩步；先检查 tokenizer ID、学习率和精度，别只换 checkpoint 标签反复碰运气。
- 选择了 CPU 训练时的运行验收：真实训练退出码 0、loss 有限、三个保存文件对应 step 2、probe `status=ok`。任一未通过就说明实际停留阶段，在修复前不扩大训练；这与前两节概念学习是否完成分别判断。`experiments/inference_infra/reports/d10_training_smoke.md` 仅作可选整理，可直接在聊天中交流。

**可选 GPU 扩展**：仅当你主动选择且已有可用 CUDA 环境时，才将两个脚本的 `--device` 改为 `cuda`，为该次运行另选模型标签和结果目录，并显式记录精度。它用于观察同一路径在另一设备上的行为，不是本日验收要求；没有 GPU 无需执行或标 PENDING。本页也不要求为此切换安装。

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

原生 `tokenizer.pkl`、`token_bytes.pt`、`model_<step>.pt`、`meta_<step>.json` 与 `optim_<step>_rank0.pt` 是 loader 的格式契约，保留其名称；用带日编号的数据文件或模型标签归属日期。已有历史结果不原位重命名，新的命令使用本页命名。仅计划分支的 JSON 按脚本实际写为 PENDING；交流时直接说明“只审查计划，未执行训练”。选择实际训练后才逐项判断其运行结果；可选 Markdown 与 GPU 扩展未做均不影响概念学习。

## 今日自验证与交流问题

按题号在聊天中回答即可。前 3 题只需正文与手推，后 2 题只在你选择实际训练时结合证据回答；未执行可直接说明本次只阅读或只生成计划。实测时开头一次注明模型标签、数据根、设备/实际精度与证据路径，同一份证据可复用；历史结果注明日期与版本。不要求另写报告或为答题重跑训练。

1. 对输入 `[10,20,30]` 和目标 `[20,30,40]`，解释位置 1 应预测什么、能看到什么。若目标概率由 `0.25` 变为 `0.5`，这一位置的交叉熵怎样变化？为什么不能先选 token ID 再用它代替 logits 算今天的 loss？

   **核对要点**：对齐本位置目标与因果上下文，使用负自然对数说明变化；区分连续预测分数和离散选择，不要求推导整个 Transformer。

2. 一次 `loss.backward()` 后，什么被计算并保存了？`optimizer.step()` 与 `zero_grad` 又分别做什么？如果跳过 step，却生成了几段文本，为什么不能据此说模型已经训练？

   **核对要点**：分清参数、梯度与优化器状态的作用；用本日执行次序说明更新发生在哪里，不把标量梯度下降示例当作 Muon/AdamW 的实际公式。

3. 单设备 B=1、T=512，若 total-batch-size 从 512 改为 1024，一次更新的前向/反向次数怎样变化？为何要在 backward 前缩放 loss，而 optimizer.step 不能放进每个微批次里？

   **核对要点**：利用全局每次前向/反向 token 数推导累积次数与整除条件；解释梯度累加、平均和一次参数更新的关系。假设不涉及 loss scaler 跳步，不将 token 数当句子数。

4. 结合你的计划与日志，说明实际停在“只生成计划、训练失败、训练执行成功”中的哪一阶段。若完成两次更新，为什么训练行是 step 0、1，保存却是 step 2？日志中还要检查哪一项数值问题？

   **核对要点**：区分 `--execute`、计划状态、真实 `returncode` 和训练日志；按保存与 `last_step` 的顺序说明为何没有第三次更新。检查有限 loss；已有标签被拒绝时保留原证据并另选标签。

5. 要判断这份 checkpoint 已能用于后续实验，你为何还需要重新加载和一次生成检查？从本次 probe 选择支持结论的结果，并说明它不能证明的一项模型能力。

   **核对要点**：将同一 tag/step 的模型、metadata 与 probe 对上；区分推理加载所需资产与恢复优化所需状态。解释严格权重加载、词表大小检查和 greedy 前缀检查分别排除什么；不能据两步文本不通顺判格式失败，也不能由 smoke 成功宣称收敛或性能通过。

离开前保留本日会话日志与自动结果；若追加重跑，注明会话边界。交流时可附 `git rev-parse HEAD`、`git status --short` 及实际设备/精度以定位证据，不必另填通用表。下一天可直接复用本聊天中已确认的结论与证据路径。共享前检查数据路径和日志是否包含私人信息；若提交成果，只纳入自己的轻量记录与脚本，不提交权重、数据、环境目录或缓存。
