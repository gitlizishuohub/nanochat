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
knowledge_scope: W2第4天实操与验收
---

# W2 D4：扩展到二十步并核对两条生成路径

前一天：[返回前一天](d10_training_smoke.md)；后一天：[继续下一天](d12_benchmark_schema.md)；[原始总计划](../nanochat_daily_execution_2026-10-01.md)。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[4.3 从配置重建模型并加载参数](../../llm-training-workflow/04-model-construction-and-weight-loading.md#reading-4-3)；[10.2 文件职责与保存时点](../../llm-training-workflow/10-logging-checkpoints-and-resume.md#reading-10-2)；[10.3 真实恢复与跨阶段加载](../../llm-training-workflow/10-logging-checkpoints-and-resume.md#reading-10-3)。
- **今天重点**：对照本次保存与重新加载的真实文件，分清模型配置、参数、优化器状态和恢复状态。注意推理加载与继续训练的要求不同。
- **暂时跳过**：不实际运行恢复示例；不能因为能加载模型，就认为已验证精确恢复。

**读完能解释**：你今天重新加载需要哪些文件？如果要恢复原训练，还缺哪些状态和一致性条件？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 开始前：固定 shell 与结果位置

所有命令从仓库根目录执行。当前终端会自动激活 nanochat 环境，无需重复激活；先运行以下检查。这里使用本项目的 Conda 环境；不要混用系统 Python。CPU实验建议先 `export NANOCHAT_DTYPE=float32`；脚本只在变量未设置时选择FP32，已有变量会遵从并记录，不能假定CPU必定FP32。GPU实验另按协议显式选dtype。

日志只在 `{ ...; }` 命令组执行期间记录；<abbr title="Bash process substitution · 进程替换。这里把命令组的输出交给 tee，同时显示并追加到日志；重定向在命令组结束后恢复。">进程替换</abbr>不会持续接管交互终端，因此提示符和 VS Code 终端集成控制字符不会混入日志。命令组在当前 shell 执行，变量设置会保留；代码块返回最后一条命令的退出码，需核对单条命令时立即记录 `$?`。已有的显式文件重定向仍写入各自文件。复制时包含外层花括号和末尾重定向。若之前执行过旧版全局日志命令，先关闭旧终端并新开终端；仅修改文档不会停止旧终端的记录。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
{
python -c "import sys; print(sys.executable); print(sys.version)"
} > >(tee -a experiments/inference_infra/reports/d11_session.log) 2>&1
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。若环境不存在或第三方依赖缺失，先解决环境，不把 `--help` 成功当作模型实验通过。后面写入的 `experiments/` 文件是你实际执行教程时产生的正式成果；本指导不预先生成成绩。大数据、tokenizer、权重放在 `NANOCHAT_BASE_DIR`，不要加进 Git。表中的 `$NANOCHAT_BASE_DIR` 代表当前 shell 选定的真实绝对数据根；合成分支示例为 `/tmp/nanochat-w2-toy`，未设置时当前用户默认根为 `/home/lizishuo/.cache/nanochat`。交流时给出实际展开后的绝对路径；`*.parquet` 表示该目录内逐一列出的实际数据文件，旧数据回退目录为 `$NANOCHAT_BASE_DIR/base_data/`。

本页各 Bash 代码块内的输出追加保存在 `experiments/inference_infra/reports/d11_session.log`；命令返回后立即执行 `echo "exit=$?"`，避免丢失退出码。帮助、源码检索、解释器与 Git 检查只打印，不另有检查结果文件。日志是追加写入，重跑时交流中注明会话边界或日志中的起止位置。

## 今日目标与时间盒

用 15 分钟复核两步证据，30–90 分钟扩到二十步，20 分钟检查两条生成路径，15 分钟冻结模型身份。前置是 [两步 smoke](d10_training_smoke.md)。今天建立可重复加载的系统实验模型，不要求语言质量。

## 第一步：新标签训练二十步

这是一次从头训练；不要把它描述为“从 step2 恢复”。真实恢复需要 optimizer 与数据状态，当前教学脚本没有提供 resume 参数。

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d10_train_smoke.py --device cpu --steps 20 --model-tag d11_w2_20 --output-dir experiments/inference_infra/reports/d11_train_smoke_20
python .agents/docs/career_roadmap/w2/scripts/d10_train_smoke.py --device cpu --steps 20 --model-tag d11_w2_20 --output-dir experiments/inference_infra/reports/d11_train_smoke_20 --execute
} > >(tee -a experiments/inference_infra/reports/d11_session.log) 2>&1
```

先审查计划再执行。模型参数、上下文长度与昨天一致，只改变训练步数和标签。路径仍取决于当前 shell 的 `NANOCHAT_BASE_DIR`；开新终端必须重设或核对它。

## 第二步：用同一个 prompt 运行两条路径

打开 [checkpoint 与双路径脚本](scripts/d10_checkpoint_probe.py)，执行：

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d10_checkpoint_probe.py --device cpu --model-tag d11_w2_20 --step 20 --prompt 'The chemical formula of water is' --max-tokens 8 --output experiments/inference_infra/reports/d11_checkpoint_generation_20.json
} > >(tee -a experiments/inference_infra/reports/d11_session.log) 2>&1
```

先看 `model_config` 的 `sequence_len/vocab_size/n_layer/n_head/n_kv_head/n_embd/window_pattern`，再看 `prompt_ids/reference_ids/engine_ids/common_prefix_match`。<abbr title="Greedy decoding · 每步选择当前 logits 最大的 token。本文通过 temperature=0 使用确定的选择规则；相同 token 不证明逐步 logits 数值完全相等。">greedy</abbr> 消除采样随机性，但设备与算子数值差异仍可能使近似并列的 token 改变。

两条路径的实际区别：

1. `GPT.generate` 每次把完整前缀送入 `forward`，没有 KV Cache，batch 只支持1；它不按 Engine 的终止 token 提前停。
2. `Engine.generate` 先对单个 prompt 做 prefill，然后复制 cache 给 `num_samples` 行，逐步 decode；它会处理 assistant_end/BOS 和工具状态。
3. 脚本仅比较 Engine 实际产生序列与 reference 的同长前缀，包含末尾终止 ID；不能无条件比较两个完整列表长度。
4. 工具调用场景会强制插入输出；这里的自然语言 prompt 没有声明要测工具，若进入该路径，记录并用纯模型逐步 logits 测试进一步定位。

<abbr title="KV Cache / Key-Value Cache · 保存过去位置各层注意力的键和值，使逐步生成不再重复计算完整前缀。nanochat 还保存 prev_embedding 以维护 smear 状态；它会增加持久缓存内存。">KV Cache</abbr> 不只是一个 K/V 张量列表，`prev_embedding` 也属于模型历史状态，因此“token 一样”只是这一页的 smoke，不是完整 cache 正确性验收。

## 第三步：冻结身份而不是只记 d4

模型身份由下面的自动日志与生成 JSON 提供，直接在聊天中回答文末问题即可。`experiments/inference_infra/reports/d11_checkpoint_manifest.md` 可选整理；不需要另填模板。读取 metadata 可用标准库，命令里的路径需与真实标签一致：

```bash
{
python -m json.tool "$NANOCHAT_BASE_DIR/base_checkpoints/d11_w2_20/meta_000020.json" > experiments/inference_infra/reports/d11_checkpoint_identity.log 2>&1
sha256sum "$NANOCHAT_BASE_DIR/base_checkpoints/d11_w2_20/model_000020.pt" "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" >> experiments/inference_infra/reports/d11_checkpoint_identity.log 2>&1
} > >(tee -a experiments/inference_infra/reports/d11_session.log) 2>&1
```

交流时核对来源是 synthetic-smoke 还是 real-corpus，并把标签/step、路径/hash、配置、训练退出码与双路径结果对应起来。模型质量、GPU性能、逐步logits误差和所有cache状态正确性仍需分别验证。

脚本和生产 loader 只检查 vocab_size 一致，不能证明 ID 到文本语义一致；因此 tokenizer hash 必须与训练时记录配对。后续运行 benchmark 将自动记录权重、metadata、tokenizer 三项 hash。

## 失败分流与验收

- 找不到 `meta_000020.json`：检查是否执行了20步、标签是否相同、变量是否指向另一个缓存根。
- vocab mismatch：停止换 tokenizer；找回训练时那份，不能修改 metadata 的 vocab_size 硬凑。
- prefix mismatch：保存两个 ID 序列与首个差异位置，检查 dtype、后端、工具/终止行为；不要通过删掉不一样的 token 让测试变绿。
- 无可用训练数据：仍可在后天使用 `d12_benchmark.py --random-model` 验证接口；交流中将模型身份状态标为 `PENDING_CHECKPOINT`，不能说已有训练模型。
- 验收：二十步退出码0、checkpoint可加载、相同 prompt 的两条路径记录完整、来源/hash/config可复查。

## 今日脚本与产物清单

每日编号使用全程 d01–d21；跨日复用保留首次实际执行日。以下项目相对路径均以仓库根目录为基准，脚本清单直接列出本日执行入口。

| 脚本完整项目路径 | 用途 | 执行阶段 | 输入 | 运行条件 |
|---|---|---|---|---|
| `.agents/docs/career_roadmap/w2/scripts/d10_train_smoke.py` | 复用训练包装器，从头训练 20 步 | 第一步 | `$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl`、`$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt`、`$NANOCHAT_BASE_DIR/base_data_climbmix/*.parquet`；--steps 20 | 两步门槛通过；新标签；--execute 才训练 |
| `.agents/docs/career_roadmap/w2/scripts/d10_checkpoint_probe.py` | 复用双路径检查 | 第二步 | `$NANOCHAT_BASE_DIR/base_checkpoints/d11_w2_20/model_000020.pt`、`$NANOCHAT_BASE_DIR/base_checkpoints/d11_w2_20/meta_000020.json`、`$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl` | 20 步成功且 P+G≤512 |

训练包装器内部调用原生入口 `scripts/base_train.py`（`python -m scripts.base_train`），只在 `--execute` 时执行；输入为同一数据根下的 tokenizer 与训练 split，保留原生脚本名。

| 产物完整项目相对路径 | 用途 | 生成者 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/reports/d11_train_smoke_20/d11_plan.json` | 二十步计划与退出码 | d10_train_smoke.py --steps 20 | 自动；不加 --execute 仅计划 |
| `experiments/inference_infra/reports/d11_train_smoke_20/d11_train.log` | 二十步训练 stdout/stderr | d10_train_smoke.py --steps 20 | 条件自动；执行训练后创建 |
| `experiments/inference_infra/reports/d11_checkpoint_generation_20.json` | step 20 双路径结果 | d10_checkpoint_probe.py | 条件自动；执行 probe 后写出 |
| `experiments/inference_infra/reports/d11_checkpoint_identity.log` | metadata 展开与 SHA256 输出 | 第三步重定向命令 | 条件自动；运行命令时保存，读取失败也留日志 |
| `experiments/inference_infra/reports/d11_checkpoint_manifest.md` | 冻结模型与 tokenizer 身份 | 第三步 | 可选人工整理；可直接在聊天中回答本日问题 |
| `experiments/inference_infra/reports/d11_session.log` | 当天终端输出与手动记录的退出码 | 各 Bash 代码块的 tee | 自动；仅追加各代码块内的输出 |

权重位于仓库外；以下是当前 `NANOCHAT_BASE_DIR=/tmp/nanochat-w2-toy` 时的完整绝对路径。使用自定义数据根时在交流中逐项提供真实绝对路径。仅真实训练保存阶段执行后生成；训练中断时不可宣称齐全：

| 完整绝对路径（示例数据根） | 用途 | 生成者与方式 |
|---|---|---|
| `/tmp/nanochat-w2-toy/base_checkpoints/d11_w2_20/model_000020.pt` | 模型参数 | scripts/base_train.py → nanochat/checkpoint_manager.py；条件自动 |
| `/tmp/nanochat-w2-toy/base_checkpoints/d11_w2_20/meta_000020.json` | 配置与训练状态 | scripts/base_train.py → nanochat/checkpoint_manager.py；条件自动 |
| `/tmp/nanochat-w2-toy/base_checkpoints/d11_w2_20/optim_000020_rank0.pt` | 单进程优化器状态 | scripts/base_train.py → nanochat/checkpoint_manager.py；条件自动 |

原生 `tokenizer.pkl`、`token_bytes.pt`、`model_<step>.pt`、`meta_<step>.json` 与 `optim_<step>_rank0.pt` 是 loader 的格式契约，保留其名称；用带日编号的数据文件或模型标签归属日期。已有历史结果不原位重命名，新的命令使用本页命名。没有执行或条件不满足的必需检查均标 PENDING；可选 Markdown 未整理不影响验收。

## 今日自验证与交流问题

可直接在聊天中按题号回答，不要求手写报告。每题给出结论和可定位依据（源码位置、命令与退出码、日志片段或自动结果字段）；同一份证据可复用。概念推演须注明假设，实际运行须区分本次与历史结果。未执行写 `PENDING`，失败写真实错误和影响范围，不能据此勾选实验通过。

1. **二十步的前置**：D10 的输入检查、两步训练、保存与重载中哪些已经有证据？本次解释器、数据根、设备和精度是否与这些证据一致？

   **通过标准**：关联实际日志与 JSON 的 status/退出码，列出缺项；CPU 的 setdefault 不会覆盖已设置的精度，不能把推荐 FP32 当作实测。

2. **从头训练与恢复**：二十步计划实际调用什么入口、标签和参数？为什么不是从 step2 接着训练，真正恢复还需要哪些状态？

   **通过标准**：引用 `d10_train_smoke.py` 的 command 及无 resume 参数事实；区分模型参数、优化器、数据位置/随机状态及调度一致性，不把能加载权重等同精确恢复。

3. **计划与实际更新**：本次是否加了 --execute，真实返回码与训练日志是什么？二十步记录对应几次参数更新，step=20 的保存是否多做一次更新？

   **通过标准**：对照 `d11_plan.json` 的 status/returncode 和 base_train 循环的 last_step/break；未执行计划为 PENDING，实际成功须有训练日志与保存证据。

4. **配置与来源**：实际 `model_config` 的 sequence_len、vocab_size、层数、头数、宽度、窗口是什么？与两步实验相比，除了步数和标签还改过什么？

   **通过标准**：引用 meta 与生成 JSON 的实际值，区分 synthetic-smoke 和 real-corpus；若配置或数据变了列明变化，不再称为只延长步数的同条件实验。

5. **加载资产与模型身份**：step20 的权重、metadata、optimizer 文件实际在哪里？本次加载选择了什么 tag/step，未显式固定时 loader 会怎样选？

   **通过标准**：列出展开后的绝对路径，指向 `load_model_from_dir/find_last_step`；说明推理不读取 optimizer，恢复训练另有要求，不用目录存在替代加载成功。

6. **词表配对与哈希**：权重与 tokenizer 的 SHA256 是什么、从哪份日志取得？为什么只有 vocab_size 一致不能保证两份 tokenizer 与同一模型兼容？

   **通过标准**：引用 `d11_checkpoint_identity.log`、生成记录和 loader 断言，说明每个 ID 的语义与特殊 token 协议必须配对；哈希只确认文件身份，不能独自证明质量或训练过程。

7. **同一 prompt 的比较对象**：实际 prompt_ids、reference_ids、engine_ids 各是什么？common_prefix_match 怎样计算，若 Engine 提前终止应比较哪段？

   **通过标准**：引用本次 `d11_checkpoint_generation_20.json`，说明两者使用同一加载模型和输入、temperature=0；比较 Engine 实际长度对应的 reference 前缀并包含终止 ID，不要求完整列表等长。

8. **两条生成路径的差异**：GPT.generate 和 Engine.generate 分别怎样处理历史、缓存和终止/工具状态？prev_embedding 在其中负责什么？

   **通过标准**：引用生成器实现，区分全前缀重算与 prefill 后单 token decode，以及 sampled/forced token；说明共同 token 前缀一致不能代替逐步 logits 与全部缓存状态对照。

9. **失败的分层定位**：若文件缺失、词表不匹配、P+G 超界或前缀不一致，分别先查什么？若真实不一致，首个差异在哪个 ID 位置？

   **通过标准**：依据真实 error/序列与加载日志定位阶段；没有发生故障可按源码推演，不制造错误，不修改 vocab_size 或删除 token 迎合结果，缺模型标 PENDING_CHECKPOINT。

10. **进入基准前的放行**：训练成功、资产齐备、重新加载与双路径检查是否分别通过？哪些结论可以交给 D12，哪些语言能力/GPU/数值等价结论仍未验证？

    **通过标准**：每个状态有本次或明确注明日期的历史证据；只把格式、加载与前缀 smoke 作为已验证范围，给出当前阻塞与最小下一步，不要求另写模型报告。

离开前保留本日会话日志与自动结果；若追加重跑，注明会话边界。交流时可附 `git rev-parse HEAD`、`git status --short` 及实际设备/精度以定位证据，不必另填通用表。下一天可直接复用本聊天中已确认的结论与证据路径。共享前检查数据路径和日志是否包含私人信息；若提交成果，只纳入自己的轻量记录与脚本，不提交权重、数据、环境目录或缓存。
