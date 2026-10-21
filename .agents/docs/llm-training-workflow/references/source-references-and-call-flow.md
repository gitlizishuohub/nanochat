---
doc_role: training-workflow
source_revision: fbc049f379926ecdb49071a129586278245d71c2
source_files:
  - nanochat/gpt.py
  - nanochat/tokenizer.py
  - nanochat/dataset.py
  - nanochat/dataloader.py
  - nanochat/common.py
  - nanochat/optim.py
  - nanochat/fp8.py
  - nanochat/checkpoint_manager.py
  - nanochat/loss_eval.py
  - nanochat/core_eval.py
  - nanochat/engine.py
  - scripts/tok_train.py
  - scripts/tok_eval.py
  - scripts/base_train.py
  - scripts/chat_sft.py
  - scripts/chat_rl.py
  - scripts/base_eval.py
  - scripts/chat_eval.py
  - scripts/chat_cli.py
  - tasks/common.py
  - tasks/smoltalk.py
  - tasks/mmlu.py
  - tasks/gsm8k.py
  - tasks/arc.py
  - tasks/humaneval.py
source_symbols:
  - GPT.forward
  - GPT.setup_optimizer
  - RustBPETokenizer.render_conversation
  - tokenizing_distributed_data_loader_with_state_bos_bestfit
  - MuonAdamW.step
  - Engine.generate_batch
---

# 源码导航与调用关系

[返回栏目](../README.md)

本表是原有调用关系快照，用于按主题查源码；不是正文第一课，也不表示本轮重新审阅了全部源文件。

<a id="sec-1"></a>

本项目的 <a id="ref-llm-001"></a><abbr title="LLM / Large Language Model｜大语言模型&#10;定义：根据上下文产生词元概率的参数化模型，语言能力存储于数值权重，而不是在文本数据库中直接搜索整句答案。&#10;作用：统一承担续写、对话和任务回答，同一模型可以在不同数据与目标下训练或只读推理。&#10;本项目：GPT是自回归、仅解码器Transformer；输入整数[B,T]，有targets返回损失，无targets返回[B,T,V]预测分数。&#10;注意：GPT名称不等于加载外部GPT-2；当前没有视觉输入、编码器解码器或MoE结构。模型权重与分词器需配套。&#10;位置：nanochat/gpt.py:156 GPT；nanochat/gpt.py:459 GPT.forward">LLM</abbr>（[释义](../glossary/01-training-stages.md#term-llm)） 是自己实现的<a id="ref-transformer-001"></a><abbr title="Transformer / Decoder Block｜Transformer与解码器块&#10;定义：以注意力聚合历史上下文，再以逐位置前馈变换特征，并通过残差传递信息的网络结构。&#10;作用：使每个token表示包含过去内容和非线性组合，支撑条件生成。&#10;本项目：Block采用pre-norm：x+Attention(norm(x))再加MLP(norm(x))，外层另有残差标量、初始嵌入混合/Backout。&#10;注意：仅因果self-attention，无编码器cross-attention；不是原始GPT-2复刻，也没有dropout与Linear bias。&#10;位置：nanochat/gpt.py:144 Block；nanochat/gpt.py:499 Transformer trunk">仅解码器 Transformer</abbr>（[释义](../glossary/04-model-and-forward-pass.md#term-transformer)），不是外部 GPT-2 的加载包装器。

| 职责 | 源码与实际调用关系 | 实现边界 |
| --- | --- | --- |
| 环境与参考流程 | [README.md](../../../../README.md)、[pyproject.toml](/home/lizishuo/_program_/Project/nanochat/pyproject.toml)、[runs/speedrun.sh](/home/lizishuo/_program_/Project/nanochat/runs/speedrun.sh)、[runs/runcpu.sh](/home/lizishuo/_program_/Project/nanochat/runs/runcpu.sh) | 组合脚本配置可能覆盖入口默认。 |
| 基础语料与装载 | [nanochat/dataset.py](/home/lizishuo/_program_/Project/nanochat/nanochat/dataset.py) → [nanochat/dataloader.py](/home/lizishuo/_program_/Project/nanochat/nanochat/dataloader.py) | 当前 ClimbMix，旧 Fineweb 只是目录回退。 |
| 分词与协议 | [scripts/tok_train.py](/home/lizishuo/_program_/Project/nanochat/scripts/tok_train.py) → [nanochat/tokenizer.py](/home/lizishuo/_program_/Project/nanochat/nanochat/tokenizer.py)，[scripts/tok_eval.py](/home/lizishuo/_program_/Project/nanochat/scripts/tok_eval.py) | 自训练词表与序列化协议。 |
| 模型与内核 | [nanochat/gpt.py](/home/lizishuo/_program_/Project/nanochat/nanochat/gpt.py) → [nanochat/flash_attention.py](/home/lizishuo/_program_/Project/nanochat/nanochat/flash_attention.py)，可选 [nanochat/fp8.py](/home/lizishuo/_program_/Project/nanochat/nanochat/fp8.py) | 有值嵌入、Smear、Backout；不照抄其它架构。 |
| 预训练 | [scripts/base_train.py](/home/lizishuo/_program_/Project/nanochat/scripts/base_train.py) → 装载器 → GPT.forward → GPT.setup_optimizer → MuonAdamW.step | 梯度同步由优化器负责。 |
| SFT | [scripts/chat_sft.py](/home/lizishuo/_program_/Project/nanochat/scripts/chat_sft.py) → load_model("base") → TaskMixture → render_conversation → SFT 装箱生成器 → 同一 GPT 与优化器 | 全参数监督微调。 |
| 可选 RL | [scripts/chat_rl.py](/home/lizishuo/_program_/Project/nanochat/scripts/chat_rl.py) → load_model("sft") → Engine.generate_batch → GSM8K.reward → GPT.forward(none) → optimizer.step | 仅数学奖励与简化策略梯度。 |
| 公共训练与保存 | [nanochat/common.py](/home/lizishuo/_program_/Project/nanochat/nanochat/common.py)、[nanochat/optim.py](/home/lizishuo/_program_/Project/nanochat/nanochat/optim.py)、[nanochat/checkpoint_manager.py](/home/lizishuo/_program_/Project/nanochat/nanochat/checkpoint_manager.py) | 权重、配置与每卡优化器状态不同。 |
| 评估与交付 | [nanochat/loss_eval.py](/home/lizishuo/_program_/Project/nanochat/nanochat/loss_eval.py)、[scripts/base_eval.py](/home/lizishuo/_program_/Project/nanochat/scripts/base_eval.py)、[nanochat/core_eval.py](/home/lizishuo/_program_/Project/nanochat/nanochat/core_eval.py)、[scripts/chat_eval.py](/home/lizishuo/_program_/Project/nanochat/scripts/chat_eval.py)、[scripts/chat_cli.py](/home/lizishuo/_program_/Project/nanochat/scripts/chat_cli.py)、[nanochat/engine.py](/home/lizishuo/_program_/Project/nanochat/nanochat/engine.py) | 验证损失、任务能力和生成展示不同。 |
| 数据适配与判题 | [tasks/common.py](/home/lizishuo/_program_/Project/nanochat/tasks/common.py)、[tasks/smoltalk.py](/home/lizishuo/_program_/Project/nanochat/tasks/smoltalk.py)、[tasks/mmlu.py](/home/lizishuo/_program_/Project/nanochat/tasks/mmlu.py)、[tasks/gsm8k.py](/home/lizishuo/_program_/Project/nanochat/tasks/gsm8k.py)、[tasks/arc.py](/home/lizishuo/_program_/Project/nanochat/tasks/arc.py)、[tasks/humaneval.py](/home/lizishuo/_program_/Project/nanochat/tasks/humaneval.py) | 训练任务与评估任务不完全相同。 |
