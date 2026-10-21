---
doc_role: training-workflow
source_files:
  - scripts/chat_sft.py
  - nanochat/tokenizer.py
  - nanochat/checkpoint_manager.py
source_symbols:
  - sft_data_generator_bos_bestfit
  - RustBPETokenizer.render_conversation
  - load_model
source_review:
  status: verified
  reviewed_at: 2026-10-21
  source_commit: ae0c2733222f0e4791285955df099bb9e9073d44
  scope: SFT 加载、协议截断、装箱与目标、优化器继承、停止、评估及保存；仅静态核对
---

[栏目目录](README.md) · [上一篇](10-logging-checkpoints-and-resume.md) · [下一篇](12-inference-and-generation.md) · [术语查阅](glossary/README.md)

> 2026-10-21 对照相关源码串联 SFT 流程；只做静态核对，未执行训练。

# 11. 监督微调：从基础模型到对话模型

<a id="sec-sft"></a>
本章把基础权重到对话模型的流程串起来。前置为分词与对话协议、目标对齐、参数更新和检查点；具体生成和评价机制在后两章展开。

<a id="reading-11-1"></a>
## 11.1 起点与训练目标

<abbr title="SFT / Supervised Fine-Tuning · 监督微调；用示范回答的目标词元更新已有模型，本项目更新完整模型参数。">SFT</abbr> 从 `load_model("base", ..., phase="train")` 加载基础模型、原分词器和元数据。训练使用真实对话前缀预测助手目标，不先自由生成再给回答打分；也不引入新的模型结构。模型重建与参数载入见[模型加载](04-model-construction-and-weight-loading.md#sec-5)，同阶段恢复与热启动的区别见[检查点](10-logging-checkpoints-and-resume.md#sec-9)。

<a id="reading-11-2"></a>
## 11.2 示范数据到输入和目标

训练混合为 SmolTalk(train)×1、MMLU(all,auxiliary_train)×3、GSM8K(main,train)×4；<a id="ref-task-002"></a><abbr title="Task / TaskMixture / HubDataset｜任务适配、混合与数据包装&#10;定义：Task把数据变为对话并提供评分，TaskMixture组合任务实例，HubDataset包装Arrow表与索引置换。&#10;作用：使SFT示范、RL奖励和任务评估共用协议，同时隔离不同源数据格式。&#10;本项目：load_hub_dataset缓存Hub Parquet；TaskMixture列所有索引并Random(42)洗牌，SFT以重复MMLU/GSM8K实例提高权重。&#10;注意：1/3/4是重复实例次数，不是精确抽样概率；无需datasets库，实际条数以len而非近似注释为准。&#10;位置：tasks/common.py:21 HubDataset；tasks/common.py:45 load_hub_dataset；tasks/common.py:129 TaskMixture">TaskMixture</abbr>（[释义](glossary/02-data-and-protocol.md#term-task)）展开样本索引再按种子42洗牌，不是精确按1:3:4概率在线抽样。验证为 SmolTalk(test)、MMLU(test,stop=5200)、GSM8K(test,stop=420)；正式评价要说明 test 子集已用于训练期监控。

数据链为：任务样本 → 角色与工具协议编码 → 对话截断 → 整段装箱与补齐 → 输入/目标错位 → 屏蔽非监督目标。协议中哪些位置接受监督，统一见[分词与对话协议](03-tokenizer-and-conversation-protocol.md#sec-4)；逐位置对齐示例和装箱限制统一见[数据编码](05-data-encoding-and-batching.md#sec-6)。用户和工具结果仍作为上下文，仅对应的目标损失被忽略。

## 11.3 更新与优化器继承

先按<abbr title="Cross-Entropy · 对正确目标词元的预测概率取负对数；此处只对有效的助手目标计算损失，再求梯度。">交叉熵</abbr>求梯度，再通过累积与优化器更新参数；SFT 特有的设置是加载基础权重后如何继承训练参数和优化器状态。

### 11.3.1 深入：继承值与调度边界

SFT默认继承base meta长度、每卡行数、总batch，以及base user_config中三个lr；缺失回退2048、32、524288、0.3、0.004、0.02。load_optimizer=1加载完整state_dict后**只显式重置lr**，再乘init_lr_frac=0.8；betas/decay等元数据也可能继承，日志“momentum buffers only”不符合完整行为。

新建Muon传decay=0，但默认加载状态可能恢复base最后实际更新的微小非零decay；AdamW本来还有固定decay，不能说SFT全部无衰减。load_optimizer=0才保留新建组元数据。SFT没有应用base_train全部预算/batch倍率重算。

默认warmup_ratio=0，后半消费进度降lr到0；Muon前300更新从0.85到0.95。progress来自预取/consumed，可能超1，调度不clamp，边界设置下不保证lr非负。SFT的user_config在继承值前复制，meta内可能保留None；末尾BPB即使eval_every≤0仍执行。

一次更新按有效目标交叉熵、微批梯度累积、优化器更新、清梯度推进。通用公式和微批均值的含义见[反向与参数更新](07-backpropagation-and-parameter-updates.md#sec-8)；本节只维护 SFT 特有的继承和调度规则。

## 11.4 停止条件与运行边界

**SFT --num-iterations 实现边界：**生成器内部 it 每生成一个微批加1并触发 last_step，而非外层优化更新计数；外层还预取下一批，consumed满一轮也可先结束，多个rank以MAX同步结束。不能承诺“精确更新N步”。

`render_conversation` 默认最多 2048 个 ID，SFT 调用没有传入 `args.max_seq_len`。修改训练行长度不会同步修改这一上游截断条件，也不会重建加载模型的窗口/RoPE 配置，却会将新长度写入保存元数据，可能导致再次加载时的行为与训练时不同。装箱与全忽略目标的边界见[数据章节](05-data-encoding-and-batching.md)。

<a id="reading-11-5"></a>
## 11.5 保存、评价与下一阶段

训练结束时保存至 `chatsft_checkpoints/<tag>/`：<abbr title="rank · 分布式训练进程的编号；rank 0 负责公共模型文件，各进程保存自己的优化器分片。">rank</abbr> 0 写模型与元数据，各 rank 写优化器状态。它可供 `load_model("sft", ...)` 评价和生成，但脚本没有恢复同一 SFT 循环的入口。文件职责与恢复限制统一见[检查点章节](10-logging-checkpoints-and-resume.md)。

训练期可监控 <abbr title="BPB / Bits Per Byte｜每字节比特数&#10;定义：将目标NLL从自然对数转比特，再除这些普通目标词元对应的原始字节总数。&#10;作用：在相同语料/评估构造下减小词表大小变化对平均token loss比较的干扰。&#10;本项目：token_bytes.pt记录原始字节，特殊ID0；evaluate_bpb排除y&lt;0与0字节目标，各rank求和后除ln2×总bytes。&#10;注意：不是平均loss除平均字数；装箱/裁切/SFT mask改变统计范围，不能保证不同评价任意可比；零字节结果inf。&#10;位置：scripts/tok_train.py:80 token_bytes；nanochat/loss_eval.py:9 evaluate_bpb">验证BPB</abbr> 与 <abbr title="ChatCORE / Categorical Evaluation｜对话能力汇总与分类评估&#10;定义：按对话协议运行多个任务，校正随机基线后平均；分类评价在候选答案字母中直接选最高分。&#10;作用：比较SFT/RL的回答能力，并区分有限选项识别与开放生成。&#10;本项目：ARC-Easy/Challenge、MMLU比较提示末字母logits，GSM8K/HumanEval用Engine生成判题，前三baseline0.25后二0。&#10;注意：训练期生成任务默认24题而非全量，字母须单token；分数可低于0，不能当作自由生成答案的等价准确率。&#10;位置：scripts/chat_eval.py:87 run_categorical_eval；scripts/chat_eval.py:156 run_chat_eval；scripts/chat_sft.py:377 centered_mean">对话ChatCORE</abbr>；结束时 BPB 仍会执行，ChatCORE 受其开关控制。保存了权重不表示模型质量已达标，生成与评估的解释见后续两章。<abbr title="RL / Reinforcement Learning · 强化学习；本项目使用生成回答的数学奖励调整已有 SFT 模型，为可选阶段。">RL</abbr> 是使用 SFT 权重的可选扩展，不是交付对话模型的必要条件。可复制的条件性命令统一放在[运行示例](15-running-examples-and-delivery.md)。

## 11.6 源码核对范围

2026-10-21 静态核对 [SFT 加载与参数继承](../../../scripts/chat_sft.py:94)、[装箱与目标对齐](../../../scripts/chat_sft.py:180)、[停止和训练循环](../../../scripts/chat_sft.py:263)、[保存分支](../../../scripts/chat_sft.py:391)及[来源映射](../../../nanochat/checkpoint_manager.py:167)；分词截断见[对话编码](../../../nanochat/tokenizer.py:140)。没有执行训练、数据下载或模型质量评价。
