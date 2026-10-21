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

[术语目录](README.md) · [学习主线](../README.md)

# 奖励与生成轨迹

| 英文或缩写 | 中文名称 | 详细解释 | 项目对应文件或符号 | 返回正文链接 |
| --- | --- | --- | --- | --- |
| <a id="term-reward"></a>Reward / Advantage | 奖励与优势 | **定义：**奖励表示结果好坏，优势表示相对基准收益，决定提高或降低哪些回答的生成概率。<br>**作用：**用组内相对信息减小绝对奖励噪声，让好于平均回答获得正向更新。<br>**本项目：**reward为0/1，advantages=rewards−mean；[1,0,0,1]变[0.5,−0.5,−0.5,0.5]，广播至有效token。<br>**注意：**不除标准差；组全对/全错优势0，但历史动量/decay仍可能改参数。数学评分为字符串比较，无数值容差。<br>**位置：**tasks/gsm8k.py:109 reward；scripts/chat_rl.py:144 advantages；scripts/chat_rl.py:266 pg_obj | [reward（tasks/gsm8k.py:109）](/home/lizishuo/_program_/Project/nanochat/tasks/gsm8k.py:109)<br>[advantages（scripts/chat_rl.py:144）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_rl.py:144)<br>[pg_obj（scripts/chat_rl.py:266）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_rl.py:266) | [返回：14.1 提示、生成、奖励与有效目标·奖励/优势](../14-gsm8k-reinforcement-learning.md#ref-reward-001) |
| <a id="term-rollout"></a>Rollout / On-policy | 生成轨迹与当前策略采样 | **定义：**rollout是当前模型在提示后生成的回答轨迹，连同工具注入、有效mask和结果奖励。<br>**作用：**让RL训练围绕模型当前行为发生，而不是仅重用固定示范或离线旧样本。<br>**本项目：**每step对分配题目现场采样，padding/shift后重算logp，多题反向后统一更新；没有回放池。<br>**注意：**温度/top-k改变采样概率，训练logp未相同重归一化，不能把on-policy说明直接当标准无偏策略梯度证明。<br>**位置：**scripts/chat_rl.py:86 get_batch；scripts/chat_rl.py:248 rollout更新；nanochat/engine.py:176 generate | [get_batch（scripts/chat_rl.py:86）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_rl.py:86)<br>[rollout更新（scripts/chat_rl.py:248）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_rl.py:248)<br>[generate（nanochat/engine.py:176）](/home/lizishuo/_program_/Project/nanochat/nanochat/engine.py:176) | [返回：14.1 提示、生成、奖励与有效目标·rollout](../14-gsm8k-reinforcement-learning.md#ref-rollout-001) |
