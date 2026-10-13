---
doc_role: training-workflow
source_revision: fbc049f379926ecdb49071a129586278245d71c2
review_date: 2026-10-12
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

[栏目目录](README.md) · [上一篇](12-glossary-index.md) · [下一篇](12-2-data-and-conversation-protocol.md)

> 拆分自原文；源码核对日期为 2026-10-12。本次仅整理文档结构，未重新核验源码或运行训练。

### 12.1 训练阶段与实现边界

| 英文或缩写 | 中文名称 | 详细解释 | 项目对应文件或符号 | 返回正文链接 |
| --- | --- | --- | --- | --- |
| <a id="term-llm"></a>LLM / Large Language Model | 大语言模型 | **定义：**根据上下文产生词元概率的参数化模型，语言能力存储于数值权重，而不是在文本数据库中直接搜索整句答案。<br>**作用：**统一承担续写、对话和任务回答，同一模型可以在不同数据与目标下训练或只读推理。<br>**本项目：**GPT是自回归、仅解码器Transformer；输入整数[B,T]，有targets返回损失，无targets返回[B,T,V]预测分数。<br>**注意：**GPT名称不等于加载外部GPT-2；当前没有视觉输入、编码器解码器或MoE结构。模型权重与分词器需配套。<br>**位置：**nanochat/gpt.py:156 GPT；nanochat/gpt.py:459 GPT.forward | [GPT（nanochat/gpt.py:156）](/home/lizishuo/_program_/Project/nanochat/nanochat/gpt.py:156)<br>[GPT.forward（nanochat/gpt.py:459）](/home/lizishuo/_program_/Project/nanochat/nanochat/gpt.py:459) | [返回第1节：1. 源码依据与调用关系·LLM](01-source-references-and-call-flow.md#ref-llm-001) |
| <a id="term-pretrain"></a>Pretraining | 预训练 | **定义：**在普通文档上学习下一词元预测的阶段，本项目从随机初始化权重开始，不要求每篇文档是问答示范。<br>**作用：**建立一般语言统计与知识基础，产出的基础权重可以直接续写，也可交给对话微调。<br>**本项目：**base_train调用在线分词装箱器，用T+1行错位成x/y，对全部T目标计算交叉熵；默认数据与统计参数比12。<br>**注意：**比例采用transformer_matrices+lm_head，不等于总参数或epoch数；普通文档中的BOS也可成为训练目标。<br>**位置：**scripts/base_train.py:129 build_model_meta；scripts/base_train.py:263 get_scaling_params | [build_model_meta（scripts/base_train.py:129）](/home/lizishuo/_program_/Project/nanochat/scripts/base_train.py:129)<br>[get_scaling_params（scripts/base_train.py:263）](/home/lizishuo/_program_/Project/nanochat/scripts/base_train.py:263) | [返回第2节：2. 必需主线与实际可选分支·预训练](02-training-pipeline-and-optional-branches.md#ref-pretrain-001) |
| <a id="term-sft"></a>SFT / Supervised Fine-Tuning | 监督微调 | **定义：**以示范回答的目标词元训练模型，前向使用数据中的真实前缀，而不是先自由生成再判定对错。<br>**作用：**把基础续写模型调整为遵守角色与工具协议、回答选择题和数学题的对话模型。<br>**本项目：**chat_sft从base加载完整GPT，混合SmolTalk、MMLU、GSM8K，监督助手正文、调用及assistant_end，忽略用户/工具结果。<br>**注意：**全参数更新，没有LoRA；默认加载base优化器是跨阶段热启动，不是恢复中断的SFT循环。<br>**位置：**scripts/chat_sft.py:94 base加载；scripts/chat_sft.py:180 sft_data_generator_bos_bestfit | [base加载（scripts/chat_sft.py:94）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_sft.py:94)<br>[sft_data_generator_bos_bestfit（scripts/chat_sft.py:180）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_sft.py:180) | [返回第2节：2. 必需主线与实际可选分支·SFT](02-training-pipeline-and-optional-branches.md#ref-sft-001) |
| <a id="term-rl"></a>RL / Reinforcement Learning | 强化学习 | **定义：**利用模型生成结果的奖励调整行为；训练信号是结果好坏，不是每个位置应写哪一个示范词元。<br>**作用：**在已有SFT对话能力上提高数学答案正确率，是主线外的任务专项可选阶段。<br>**本项目：**chat_rl从sft加载，每题默认采16回答，GSM8K.reward给0/1，奖励减组均值形成优势，再对有效生成词元求策略梯度。<br>**注意：**没有奖励模型、价值网络、参考KL、PPO比率裁剪或fp16 GradScaler；训练后仍需真实评估。<br>**位置：**scripts/chat_rl.py:74 sft加载；scripts/chat_rl.py:86 get_batch；tasks/gsm8k.py:109 reward | [sft加载（scripts/chat_rl.py:74）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_rl.py:74)<br>[get_batch（scripts/chat_rl.py:86）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_rl.py:86)<br>[reward（tasks/gsm8k.py:109）](/home/lizishuo/_program_/Project/nanochat/tasks/gsm8k.py:109) | [返回第2节：2. 必需主线与实际可选分支·强化学习](02-training-pipeline-and-optional-branches.md#ref-rl-001)<br>[返回第10节：10.1 提示、生成、奖励与有效目标·数学RL](10-gsm8k-reinforcement-learning.md#ref-rl-002) |
| <a id="term-lora"></a>LoRA / Low-Rank Adaptation | 低秩适配 | **定义：**通常冻结已有矩阵，只训练低秩增量BA，通过W加缩放后的BA改变模型输出。<br>**作用：**在支持此机制的系统中减少可训练参数与优化器状态，这是解释性对照，不是当前训练阶段。<br>**本项目：**当前源码没有低秩层、适配器文件或LoRA入口；GPT.setup_optimizer直接收集原模型所有参数，SFT更新完整权重。<br>**注意：**存在微调不能证明存在LoRA，全参数SFT与低秩适配是不同实现选择；本项目无可直接用的LoRA命令。<br>**位置：**nanochat/gpt.py:419 GPT.setup_optimizer；scripts/chat_sft.py:132 优化器构造 | [GPT.setup_optimizer（nanochat/gpt.py:419）](/home/lizishuo/_program_/Project/nanochat/nanochat/gpt.py:419)<br>[优化器构造（scripts/chat_sft.py:132）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_sft.py:132) | [返回第2节：2. 必需主线与实际可选分支·LoRA](02-training-pipeline-and-optional-branches.md#ref-lora-001) |
| <a id="term-distill"></a>Knowledge Distillation | 知识蒸馏 | **定义：**学生学习教师给出的输出、概率或中间表示，以转移能力；需要明确教师来源、学生和训练目标。<br>**作用：**常用于压缩模型或迁移能力，但不能仅凭使用示范文本就断言有在线教师管线。<br>**本项目：**三个入口只构造一个被优化GPT，没有教师前向、软目标温度或蒸馏损失；当前没有蒸馏入口和对应权重衔接。<br>**注意：**数据可能含其它模型生成文本不证明仓库实现在线蒸馏；teacher forcing也不表示存在教师模型。<br>**位置：**scripts/base_train.py:146 模型构建；nanochat/gpt.py:520 损失分支 | [模型构建（scripts/base_train.py:146）](/home/lizishuo/_program_/Project/nanochat/scripts/base_train.py:146)<br>[损失分支（nanochat/gpt.py:520）](/home/lizishuo/_program_/Project/nanochat/nanochat/gpt.py:520) | [返回第2节：2. 必需主线与实际可选分支·蒸馏](02-training-pipeline-and-optional-branches.md#ref-distill-001) |
| <a id="term-dpo"></a>DPO / Direct Preference Optimization | 直接偏好优化 | **定义：**通常以同提示下偏好/非偏好回答构造相对概率目标，直接训练模型倾向偏好答案。<br>**作用：**在实现该算法的系统中利用成对偏好标签，不必先训练可学习奖励模型。<br>**本项目：**当前没有chosen/rejected适配、DPO损失或训练入口；SFT用示范，RL用每题多回答的数值正确性奖励。<br>**注意：**组内奖励比较不是DPO；不能把它画成SFT之后必须阶段，也没有已确认可执行命令。<br>**位置：**scripts/chat_sft.py:163 train_tasks；scripts/chat_rl.py:118 rewards；nanochat/gpt.py:520 损失 | [train_tasks（scripts/chat_sft.py:163）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_sft.py:163)<br>[rewards（scripts/chat_rl.py:118）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_rl.py:118)<br>[损失（nanochat/gpt.py:520）](/home/lizishuo/_program_/Project/nanochat/nanochat/gpt.py:520) | [返回第2节：2. 必需主线与实际可选分支·DPO 偏好优化](02-training-pipeline-and-optional-branches.md#ref-dpo-001) |
| <a id="term-grpo"></a>GRPO / PPO / REINFORCE | 组相对优化与策略梯度对照 | **定义：**REINFORCE以奖励加权生成logp；GRPO使用组内相对奖励，PPO通常还有新旧策略概率比和裁剪。<br>**作用：**辨明源码真正优化的目标，避免把脚本标题算法名当成完整标准实现。<br>**本项目：**chat_rl自称带引号GRPO，优势仅减均值不除标准差，逐分块按有效token数/分块数/每卡题数归一化。<br>**注意：**没有ratio+clip、KL或value网络；默认top-k采样与训练完整softmax概率不一致，不能据此声称标准无偏性已验证。<br>**位置：**scripts/chat_rl.py:1 算法说明；scripts/chat_rl.py:264 logp/pg_obj | [算法说明（scripts/chat_rl.py:1）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_rl.py:1)<br>[logp/pg_obj（scripts/chat_rl.py:264）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_rl.py:264) | [返回第2节：2. 必需主线与实际可选分支·GRPO/PPO](02-training-pipeline-and-optional-branches.md#ref-grpo-001)<br>[返回第10节：10.2 真实策略梯度目标·REINFORCE简化实现](10-gsm8k-reinforcement-learning.md#ref-grpo-002) |
