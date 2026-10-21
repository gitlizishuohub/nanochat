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

# 日志、保存与恢复

| 英文或缩写 | 中文名称 | 详细解释 | 项目对应文件或符号 | 返回正文链接 |
| --- | --- | --- | --- | --- |
| <a id="term-logging"></a>Logging / W&amp;B / EMA / MFU | 日志、实验记录、平滑与算力利用率 | **定义：**日志保存过程指标，EMA平滑曲线，MFU用估算FLOPs吞吐除以设备理论峰值。<br>**作用：**观察收敛/吞吐变化并比较实验，必须明确指标统计范围和预算。<br>**本项目：**run=dummy用DummyWandb，base每100/SFT每10步记录；loss末微批再EMA，训练计时排除前10步与评估。<br>**注意：**非端到端耗时，SFTtoken吞吐含提示/补齐；MFU硬编码峰值/结构估算，CPU或未知设备0不意味着无计算。<br>**位置：**nanochat/common.py:216 DummyWandb；scripts/base_train.py:548 日志；scripts/chat_sft.py:461 日志 | [DummyWandb（nanochat/common.py:216）](/home/lizishuo/_program_/Project/nanochat/nanochat/common.py:216)<br>[日志（scripts/base_train.py:548）](/home/lizishuo/_program_/Project/nanochat/scripts/base_train.py:548)<br>[日志（scripts/chat_sft.py:461）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_sft.py:461) | [返回：10.1 日志口径·日志/W&amp;B/EMA/MFU](../10-logging-checkpoints-and-resume.md#ref-logging-001) |
| <a id="term-checkpoint"></a>Checkpoint | 检查点 | **定义：**保存某个训练时刻的权重与辅助状态，让模型能加载、评估、阶段接续或部分恢复。<br>**作用：**保留昂贵训练成果，区分推理所需资产与接续优化所需状态。<br>**本项目：**rank0存model_XXXXXX.pt/meta.json，base/SFT各rank存optim分片，RL仅模型与model_config。<br>**注意：**词表独立目录；同tag/step文件需配套。保存非原子事务，最大step不等于最优；RL零基更新后步号与其它阶段不同。<br>**位置：**nanochat/checkpoint_manager.py:41 save_checkpoint；scripts/base_train.py:478 保存；scripts/chat_rl.py:314 保存 | [save_checkpoint（nanochat/checkpoint_manager.py:41）](/home/lizishuo/_program_/Project/nanochat/nanochat/checkpoint_manager.py:41)<br>[保存（scripts/base_train.py:478）](/home/lizishuo/_program_/Project/nanochat/scripts/base_train.py:478)<br>[保存（scripts/chat_rl.py:314）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_rl.py:314) | [返回：10.2 文件职责与保存时点·检查点](../10-logging-checkpoints-and-resume.md#ref-checkpoint-001) |
| <a id="term-resume"></a>Resume from Checkpoint | 断点恢复 | **定义：**读取被中断阶段保存的模型、优化器、循环/数据状态以接续同一训练过程。<br>**作用：**尽量维持中断前更新轨迹，区别于把旧权重用于新任务。<br>**本项目：**base提供resume-from-step，加载每卡optimizer与loop_state，数据记pq_idx/rg_idx/epoch后从后一个行组继续。<br>**注意：**未存装箱缓冲、预取batch、RNG、GradScaler，属于近似恢复；SFT/RL无同阶段恢复接口，改world size无状态迁移。<br>**位置：**scripts/base_train.py:157 resuming；scripts/base_train.py:330 loader state；nanochat/dataloader.py:53 恢复 | [resuming（scripts/base_train.py:157）](/home/lizishuo/_program_/Project/nanochat/scripts/base_train.py:157)<br>[loader state（scripts/base_train.py:330）](/home/lizishuo/_program_/Project/nanochat/scripts/base_train.py:330)<br>[恢复（nanochat/dataloader.py:53）](/home/lizishuo/_program_/Project/nanochat/nanochat/dataloader.py:53) | [返回：1.2 实际可选训练分支图·断点恢复](../01-training-pipeline-and-optional-branches.md#ref-resume-001)<br>[返回：10.3 真实恢复与跨阶段加载·预训练恢复](../10-logging-checkpoints-and-resume.md#ref-resume-002) |
| <a id="term-warmstart"></a>Warm Start / Weight Loading | 热启动与权重加载 | **定义：**用已有权重作为新阶段起点，可导入优化器统计，但数据与循环通常重新开始。<br>**作用：**让SFT继承基础能力、RL继承对话/工具能力，减少从零学习成本。<br>**本项目：**load_model映射base/sft/rl，默认最深tag/最新step；SFT默认读base完整optimizer后只重置lr，RL只读sft模型。<br>**注意：**不是断点恢复；state_dict还覆盖beta/decay，日志“仅momentum”不准确。最好显式tag/step避免自动挑错权重。<br>**位置：**nanochat/checkpoint_manager.py:148 load_model_from_dir；scripts/chat_sft.py:134 热启动；scripts/chat_rl.py:74 加载 | [load_model_from_dir（nanochat/checkpoint_manager.py:148）](/home/lizishuo/_program_/Project/nanochat/nanochat/checkpoint_manager.py:148)<br>[热启动（scripts/chat_sft.py:134）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_sft.py:134)<br>[加载（scripts/chat_rl.py:74）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_rl.py:74) | [返回：4.3 从配置重建模型并加载参数·加载已有权重](../04-model-construction-and-weight-loading.md#ref-warmstart-001)<br>[返回：10.3 真实恢复与跨阶段加载·新阶段热启动](../10-logging-checkpoints-and-resume.md#ref-warmstart-002) |
