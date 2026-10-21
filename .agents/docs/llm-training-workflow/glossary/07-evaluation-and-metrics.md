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

# 评估与能力指标

| 英文或缩写 | 中文名称 | 详细解释 | 项目对应文件或符号 | 返回正文链接 |
| --- | --- | --- | --- | --- |
| <a id="term-bpb"></a>BPB / Bits Per Byte | 每字节比特数 | **定义：**将目标NLL从自然对数转比特，再除这些普通目标词元对应的原始字节总数。<br>**作用：**在相同语料/评估构造下减小词表大小变化对平均token loss比较的干扰。<br>**本项目：**token_bytes.pt记录原始字节，特殊ID0；evaluate_bpb排除y&lt;0与0字节目标，各rank求和后除ln2×总bytes。<br>**注意：**不是平均loss除平均字数；装箱/裁切/SFT mask改变统计范围，不能保证不同评价任意可比；零字节结果inf。<br>**位置：**scripts/tok_train.py:80 token_bytes；nanochat/loss_eval.py:9 evaluate_bpb | [token_bytes（scripts/tok_train.py:80）](/home/lizishuo/_program_/Project/nanochat/scripts/tok_train.py:80)<br>[evaluate_bpb（nanochat/loss_eval.py:9）](/home/lizishuo/_program_/Project/nanochat/nanochat/loss_eval.py:9) | [返回：3.1 训练词表与配套产物·BPB](../03-tokenizer-and-conversation-protocol.md#ref-bpb-001)<br>[返回：13.1 四种评价出口·验证BPB](../13-evaluation-and-metrics.md#ref-bpb-002) |
| <a id="term-core"></a>CORE / ICL / Few-shot | 基础能力指标与上下文示例学习 | **定义：**ICL是在提示中加入示例不更新权重；CORE汇总若干基础任务的随机基线校正准确率。<br>**作用：**衡量基础续写模型完成任务的能力，作为基础阶段评估出口。<br>**本项目：**base_eval读eval_bundle的core.yaml/CSV；core_eval按多选、schema、续写评分，各任务校正再平均。<br>**注意：**任务列表/few-shot数来自外部bundle，本次未下载无法固定写死；多选平均NLL与chat字母logits评法不同，校正分可负。<br>**位置：**scripts/base_eval.py:57 evaluate_core；nanochat/core_eval.py:168 evaluate_example；nanochat/core_eval.py:244 evaluate_task | [evaluate_core（scripts/base_eval.py:57）](/home/lizishuo/_program_/Project/nanochat/scripts/base_eval.py:57)<br>[evaluate_example（nanochat/core_eval.py:168）](/home/lizishuo/_program_/Project/nanochat/nanochat/core_eval.py:168)<br>[evaluate_task（nanochat/core_eval.py:244）](/home/lizishuo/_program_/Project/nanochat/nanochat/core_eval.py:244) | [返回：13.1 四种评价出口·基础CORE/ICL](../13-evaluation-and-metrics.md#ref-core-001) |
| <a id="term-chatcore"></a>ChatCORE / Categorical Evaluation | 对话能力汇总与分类评估 | **定义：**按对话协议运行多个任务，校正随机基线后平均；分类评价在候选答案字母中直接选最高分。<br>**作用：**比较SFT/RL的回答能力，并区分有限选项识别与开放生成。<br>**本项目：**ARC-Easy/Challenge、MMLU比较提示末字母logits，GSM8K/HumanEval用Engine生成判题，前三baseline0.25后二0。<br>**注意：**训练期生成任务默认24题而非全量，字母须单token；分数可低于0，不能当作自由生成答案的等价准确率。<br>**位置：**scripts/chat_eval.py:87 run_categorical_eval；scripts/chat_eval.py:156 run_chat_eval；scripts/chat_sft.py:377 centered_mean | [run_categorical_eval（scripts/chat_eval.py:87）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_eval.py:87)<br>[run_chat_eval（scripts/chat_eval.py:156）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_eval.py:156)<br>[centered_mean（scripts/chat_sft.py:377）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_sft.py:377) | [返回：13.1 四种评价出口·对话ChatCORE](../13-evaluation-and-metrics.md#ref-chatcore-001) |
| <a id="term-benchmarks"></a>GSM8K / MMLU / ARC / HumanEval | 数学、知识、科学与代码任务 | **定义：**不同数据与评分接口，各检验一种能力，不是训练算法名称。<br>**作用：**可用于示范、结果奖励或独立任务评分，但用途随入口而不同。<br>**本项目：**SFT用MMLU/GSM8K，RL只GSM8K，ChatCORE再加ARC/HumanEval；数学抽####数字字符串、代码执行测试。<br>**注意：**SFT验证取部分test，需记录污染/使用范围；HumanEval执行器不是对抗性安全隔离，本次未运行；实际条数未下载确认。<br>**位置：**tasks/gsm8k.py:86 evaluate；tasks/mmlu.py:28 get_example；tasks/arc.py:23 get_example；tasks/humaneval.py:79 evaluate | [evaluate（tasks/gsm8k.py:86）](/home/lizishuo/_program_/Project/nanochat/tasks/gsm8k.py:86)<br>[get_example（tasks/mmlu.py:28）](/home/lizishuo/_program_/Project/nanochat/tasks/mmlu.py:28)<br>[get_example（tasks/arc.py:23）](/home/lizishuo/_program_/Project/nanochat/tasks/arc.py:23)<br>[evaluate（tasks/humaneval.py:79）](/home/lizishuo/_program_/Project/nanochat/tasks/humaneval.py:79) | [返回：14.1 提示、生成、奖励与有效目标·GSM8K](../14-gsm8k-reinforcement-learning.md#ref-benchmarks-001)<br>[返回：13.1 四种评价出口·HumanEval](../13-evaluation-and-metrics.md#ref-benchmarks-002) |
| <a id="term-passk"></a>pass@k | 多次采样通过率 | **定义：**每题前k个生成结果至少一个正确判通过，再计算题目层面的通过比例。<br>**作用：**衡量多尝试找到正确数学/代码答案的能力，区别于单次生成准确率。<br>**本项目：**RL每60步默认最多400题生成device_batch_size样本，对前k做any计数，多rank求和；chat_eval也可多样本any。<br>**注意：**是直接经验计数，未用组合数无偏估计；温度、top-k、样本数和题数改变结果，不能与单次贪心不加说明横比。<br>**位置：**scripts/chat_rl.py:150 run_gsm8k_eval；scripts/chat_rl.py:229 passk；scripts/chat_eval.py:54 outcomes | [run_gsm8k_eval（scripts/chat_rl.py:150）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_rl.py:150)<br>[passk（scripts/chat_rl.py:229）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_rl.py:229)<br>[outcomes（scripts/chat_eval.py:54）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_eval.py:54) | [返回：14.3 调度、评价、产物·pass@k](../14-gsm8k-reinforcement-learning.md#ref-passk-001) |
