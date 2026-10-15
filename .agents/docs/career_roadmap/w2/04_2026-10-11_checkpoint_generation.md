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

# W2 · 2026-10-11：扩展到二十步并核对两条生成路径

前一天：[返回前一天](03_2026-10-10_training_smoke.md)；后一天：[继续下一天](05_2026-10-12_benchmark_schema.md)；[原始总计划](../nanochat_daily_execution_2026-10-01.md)。

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

用 15 分钟复核两步证据，30–90 分钟扩到二十步，20 分钟检查两条生成路径，15 分钟冻结模型身份。前置是 [两步 smoke](03_2026-10-10_training_smoke.md)。今天建立可重复加载的系统实验模型，不要求语言质量。

## 第一步：新标签训练二十步

这是一次从头训练；不要把它描述为“从 step2 恢复”。真实恢复需要 optimizer 与数据状态，当前教学脚本没有提供 resume 参数。

```bash
python .agents/docs/career_roadmap/w2/scripts/train_smoke.py --device cpu --steps 20 --model-tag d4-w2-20 --output-dir experiments/inference_infra/reports/train_smoke_20
python .agents/docs/career_roadmap/w2/scripts/train_smoke.py --device cpu --steps 20 --model-tag d4-w2-20 --output-dir experiments/inference_infra/reports/train_smoke_20 --execute
```

先审查计划再执行。模型参数、上下文长度与昨天一致，只改变训练步数和标签。路径仍取决于当前 shell 的 `NANOCHAT_BASE_DIR`；开新终端必须重设或核对它。

## 第二步：用同一个 prompt 运行两条路径

打开 [checkpoint 与双路径脚本](scripts/checkpoint_probe.py)，执行：

```bash
python .agents/docs/career_roadmap/w2/scripts/checkpoint_probe.py --device cpu --model-tag d4-w2-20 --step 20 --prompt 'The chemical formula of water is' --max-tokens 8 --output experiments/inference_infra/reports/checkpoint_generation_20.json
```

先看 `model_config` 的 `sequence_len/vocab_size/n_layer/n_head/n_kv_head/n_embd/window_pattern`，再看 `prompt_ids/reference_ids/engine_ids/common_prefix_match`。<abbr title="Greedy decoding · 每步选择当前 logits 最大的 token。本文通过 temperature=0 使用确定的选择规则；相同 token 不证明逐步 logits 数值完全相等。">greedy</abbr> 消除采样随机性，但设备与算子数值差异仍可能使近似并列的 token 改变。

两条路径的实际区别：

1. `GPT.generate` 每次把完整前缀送入 `forward`，没有 KV Cache，batch 只支持1；它不按 Engine 的终止 token 提前停。
2. `Engine.generate` 先对单个 prompt 做 prefill，然后复制 cache 给 `num_samples` 行，逐步 decode；它会处理 assistant_end/BOS 和工具状态。
3. 脚本仅比较 Engine 实际产生序列与 reference 的同长前缀，包含末尾终止 ID；不能无条件比较两个完整列表长度。
4. 工具调用场景会强制插入输出；这里的自然语言 prompt 没有声明要测工具，若进入该路径，记录并用纯模型逐步 logits 测试进一步定位。

<abbr title="KV Cache / Key-Value Cache · 保存过去位置各层注意力的键和值，使逐步生成不再重复计算完整前缀。nanochat 还保存 prev_embedding 以维护 smear 状态；它会增加持久缓存内存。">KV Cache</abbr> 不只是一个 K/V 张量列表，`prev_embedding` 也属于模型历史状态，因此“token 一样”只是这一页的 smoke，不是完整 cache 正确性验收。

## 第三步：冻结身份而不是只记 d4

在 `experiments/inference_infra/reports/checkpoint_manifest.md` 按下面模板记录。读取 metadata 可用标准库，命令里的路径需与真实标签一致：

```bash
python -m json.tool "$NANOCHAT_BASE_DIR/base_checkpoints/d4-w2-20/meta_000020.json"
sha256sum "$NANOCHAT_BASE_DIR/base_checkpoints/d4-w2-20/model_000020.pt" "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
```

```text
模型来源：nanochat base_train；synthetic-smoke / real-corpus（据实选）
标签与step：d4-w2-20 / 20
模型绝对路径与SHA256：
tokenizer绝对路径与SHA256：
model_config：
训练命令、退出码、日志：
双路径JSON、共同前缀是否通过：
未证明：模型质量、GPU性能、逐步logits误差、所有cache状态正确性
```

脚本和生产 loader 只检查 vocab_size 一致，不能证明 ID 到文本语义一致；因此 tokenizer hash 必须与训练时记录配对。后续运行 benchmark 将自动记录权重、metadata、tokenizer 三项 hash。

## 失败分流与验收

- 找不到 `meta_000020.json`：检查是否执行了20步、标签是否相同、变量是否指向另一个缓存根。
- vocab mismatch：停止换 tokenizer；找回训练时那份，不能修改 metadata 的 vocab_size 硬凑。
- prefix mismatch：保存两个 ID 序列与首个差异位置，检查 dtype、后端、工具/终止行为；不要通过删掉不一样的 token 让测试变绿。
- 无可用训练数据：仍可在后天使用 `benchmark.py --random-model` 验证接口；manifest 继续 `PENDING_CHECKPOINT`，不能说已有训练模型。
- 验收：二十步退出码0、checkpoint可加载、相同 prompt 的两条路径记录完整、来源/hash/config可复查。

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
