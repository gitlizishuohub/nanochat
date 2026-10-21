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

[栏目目录](README.md) · [上一篇](08-precision-and-training-acceleration.md) · [下一篇](10-logging-checkpoints-and-resume.md) · [术语查阅](glossary/README.md)

> 原技术正文沿用原有源码核对记录；后续调整了结构。新增 SFT 串联说明做了局部静态核对，未执行训练。

# 9. 多卡分工与优化器通信

单卡学习可先理解本章的职责划分，通信细节按需查阅。

<a id="reading-9-1"></a>
## 9.1 数据分工与梯度累积

数据如何分配给各 <abbr title="rank · 分布式进程的编号；每个进程负责一部分数据和优化器状态。">rank</abbr>、有效 batch 如何由每卡输入和累积次数组成，统一见[批量与每卡分配](05-data-encoding-and-batching.md)。本章只负责梯度和优化器状态的通信。

<a id="reading-9-2"></a>
## 9.2 优化器中的通信与分片

模型不包DDP，<a id="ref-zero-001"></a><abbr title="ZeRO-2 Style / Reduce-scatter / All-gather｜优化器状态分片与归约收集&#10;定义：归约梯度并分配给负责进程，各rank更新其参数片段，再收集更新参数以恢复一致副本。&#10;作用：减少每卡优化器状态与重复更新计算，用异步操作重叠通信和计算。&#10;本项目：大AdamW沿首维分片，小参数all_reduce；Muon同形矩阵堆叠分块，reduce_scatter平均后更新、all_gather。&#10;注意：仓库自定义ZeRO-2风格，无DeepSpeed；模型权重/反向梯度仍有副本，不是完整模型分片或FSDP。&#10;位置：nanochat/optim.py:274 _reduce_adamw；nanochat/optim.py:295 _reduce_muon；nanochat/optim.py:419 _finish_gathers">ZeRO-2风格通信</abbr>（[释义](glossary/03-batching-and-distributed-training.md#term-zero)）在optimizer.step：异步归约→负责rank更新→all_gather。大AdamW参数沿首维切片且须整除world_size；小于1024元素用all_reduce；Muon同形矩阵堆叠按矩阵块分配并补通信空位。每卡保存其状态，模型权重仍复制；不是DeepSpeed/FSDP。

## 9.3 分片保存的后果

每卡优化器状态需要分别保存；恢复训练与改变卡数的限制见[检查点与断点恢复](10-logging-checkpoints-and-resume.md)。通信分片不等于模型权重也按层分片。
