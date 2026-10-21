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

# 批量与分布式

| 英文或缩写 | 中文名称 | 详细解释 | 项目对应文件或符号 | 返回正文链接 |
| --- | --- | --- | --- | --- |
| <a id="term-loader"></a>DataLoader / Generator | 数据装载器与生成器 | **定义：**把原始样本编码、装箱、构造目标并搬到设备的迭代接口，不要求使用某个同名框架类。<br>**作用：**连接磁盘数据和模型循环，让每次前向获得形状/监督语义一致的输入。<br>**本项目：**预训练为无限Python生成器并复用缓冲；SFT装载器定义在脚本内；RL get_batch还现场生成、判奖励。<br>**注意：**未使用torch.utils.data.DataLoader、DistributedSampler或统一collate_fn；复用预训练缓冲时需长期持有旧batch应复制。<br>**位置：**nanochat/dataloader.py:74 主生成器；scripts/chat_sft.py:180 SFT生成器；scripts/chat_rl.py:86 get_batch | [主生成器（nanochat/dataloader.py:74）](/home/lizishuo/_program_/Project/nanochat/nanochat/dataloader.py:74)<br>[SFT生成器（scripts/chat_sft.py:180）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_sft.py:180)<br>[get_batch（scripts/chat_rl.py:86）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_rl.py:86) | [返回：5.1 预训练：装满 T+1 然后 shift·数据生成器](../05-data-encoding-and-batching.md#ref-loader-001) |
| <a id="term-batch"></a>Batch / Micro-batch / Effective Batch | 批次、微批与有效批次 | **定义：**微批是一次每卡前向/反向处理的序列集合，有效批次是一次优化更新累计处理的总量。<br>**作用：**将显存容得下的小批次累计为目标规模，区分序列数、题数与token数。<br>**本项目：**预训练/SFT总token容量=B×T×world_size×累积次数；device-batch-size是每卡行数，total-batch-size是全局token。<br>**注意：**SFT容量含padding/提示，不等于监督token；RL examples-per-step为题数、num-samples为每题回答，不能混套单位。<br>**位置：**scripts/base_train.py:410 累积计算；scripts/chat_sft.py:121 批量计算；scripts/chat_rl.py:217 每卡题数 | [累积计算（scripts/base_train.py:410）](/home/lizishuo/_program_/Project/nanochat/scripts/base_train.py:410)<br>[批量计算（scripts/chat_sft.py:121）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_sft.py:121)<br>[每卡题数（scripts/chat_rl.py:217）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_rl.py:217) | [返回：5.3 batch、epoch 和每卡分配·有效 batch](../05-data-encoding-and-batching.md#ref-batch-001) |
| <a id="term-epoch"></a>Epoch / Step | 轮次与更新步 | **定义：**epoch通常指遍历某一数据集合，step通常是优化更新计数，但各入口实际计数实现需分别核实。<br>**作用：**描述预算与进度，只有明确数据集合和计数单位后才能比较实验。<br>**本项目：**预训练epoch从1起、无限循环文件，按num_iterations停；SFT默认按混合数据消费停；RL步数为题数整除后乘epochs。<br>**注意：**SFT num_iterations按生成微批计数且一轮条件仍可先触发，预取使进度领先，不能据帮助承诺精确更新N步。<br>**位置：**nanochat/dataloader.py:25 _document_batches；scripts/chat_sft.py:266 停止条件；scripts/chat_rl.py:82 num_steps | [_document_batches（nanochat/dataloader.py:25）](/home/lizishuo/_program_/Project/nanochat/nanochat/dataloader.py:25)<br>[停止条件（scripts/chat_sft.py:266）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_sft.py:266)<br>[num_steps（scripts/chat_rl.py:82）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_rl.py:82) | [返回：5.3 batch、epoch 和每卡分配·epoch/step](../05-data-encoding-and-batching.md#ref-epoch-001) |
| <a id="term-distributed"></a>Rank / World Size / torchrun / NCCL | 分布式进程与启动 | **定义：**rank为全局进程编号，world size为总进程数，local rank为节点内设备编号，torchrun设置这些环境变量。<br>**作用：**分配样本与梯度计算到多卡，同时通过通信保持各卡模型数值一致。<br>**本项目：**compute_init只在CUDA建NCCL组；预训练按row-group stride，SFT/RL按样本索引stride分配。<br>**注意：**变量ddp表示分布式请求，不代表模型包了DistributedDataParallel；优化器同步梯度。CPU/MPS不可照搬多rank训练命令。<br>**位置：**nanochat/common.py:151 get_dist_info；nanochat/common.py:174 compute_init；nanochat/optim.py:429 step | [get_dist_info（nanochat/common.py:151）](/home/lizishuo/_program_/Project/nanochat/nanochat/common.py:151)<br>[compute_init（nanochat/common.py:174）](/home/lizishuo/_program_/Project/nanochat/nanochat/common.py:174)<br>[step（nanochat/optim.py:429）](/home/lizishuo/_program_/Project/nanochat/nanochat/optim.py:429) | [返回：CPU/MPS 环境改选：uv sync --extra cpu·torchrun](../02-environment-and-data-preparation.md#ref-distributed-001)<br>[返回：5.3 batch、epoch 和每卡分配·数据并行分配](../05-data-encoding-and-batching.md#ref-distributed-002) |
| <a id="term-zero"></a>ZeRO-2 Style / Reduce-scatter / All-gather | 优化器状态分片与归约收集 | **定义：**归约梯度并分配给负责进程，各rank更新其参数片段，再收集更新参数以恢复一致副本。<br>**作用：**减少每卡优化器状态与重复更新计算，用异步操作重叠通信和计算。<br>**本项目：**大AdamW沿首维分片，小参数all_reduce；Muon同形矩阵堆叠分块，reduce_scatter平均后更新、all_gather。<br>**注意：**仓库自定义ZeRO-2风格，无DeepSpeed；模型权重/反向梯度仍有副本，不是完整模型分片或FSDP。<br>**位置：**nanochat/optim.py:274 _reduce_adamw；nanochat/optim.py:295 _reduce_muon；nanochat/optim.py:419 _finish_gathers | [_reduce_adamw（nanochat/optim.py:274）](/home/lizishuo/_program_/Project/nanochat/nanochat/optim.py:274)<br>[_reduce_muon（nanochat/optim.py:295）](/home/lizishuo/_program_/Project/nanochat/nanochat/optim.py:295)<br>[_finish_gathers（nanochat/optim.py:419）](/home/lizishuo/_program_/Project/nanochat/nanochat/optim.py:419) | [返回：9.2 优化器中的通信与分片·ZeRO-2风格通信](../09-distributed-optimizer-and-communication.md#ref-zero-001) |
