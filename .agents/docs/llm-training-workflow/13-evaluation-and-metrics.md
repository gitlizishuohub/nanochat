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

[栏目目录](README.md) · [上一篇](12-inference-and-generation.md) · [下一篇](14-gsm8k-reinforcement-learning.md) · [术语查阅](glossary/README.md)

> 原技术正文沿用 2026-10-12 的源码核对记录；2026-10-21 调整结构。新增 SFT 串联说明做了局部静态核对，未执行训练。

# 13. 验证与能力评估

<a id="sec-11"></a>
先读[推理与生成](12-inference-and-generation.md)，再区分固定目标评分、候选分类与自由生成判题。

## 13.1 四种评价出口

| 评价 | 源码过程 | 解释边界 |
| --- | --- | --- |
| 分词压缩 | tok_eval round-trip与bytes/token | 不是模型能力；对照词表可能联网缓存。 |
| <a id="ref-bpb-002"></a><abbr title="BPB / Bits Per Byte｜每字节比特数&#10;定义：将目标NLL从自然对数转比特，再除这些普通目标词元对应的原始字节总数。&#10;作用：在相同语料/评估构造下减小词表大小变化对平均token loss比较的干扰。&#10;本项目：token_bytes.pt记录原始字节，特殊ID0；evaluate_bpb排除y&lt;0与0字节目标，各rank求和后除ln2×总bytes。&#10;注意：不是平均loss除平均字数；装箱/裁切/SFT mask改变统计范围，不能保证不同评价任意可比；零字节结果inf。&#10;位置：scripts/tok_train.py:80 token_bytes；nanochat/loss_eval.py:9 evaluate_bpb">验证BPB</abbr>（[释义](glossary/07-evaluation-and-metrics.md#term-bpb)） | 普通未忽略目标总NLL/总字节 | 不同装箱/监督范围不可随意比较，0有效字节返回inf。 |
| <a id="ref-core-001"></a><abbr title="CORE / ICL / Few-shot｜基础能力指标与上下文示例学习&#10;定义：ICL是在提示中加入示例不更新权重；CORE汇总若干基础任务的随机基线校正准确率。&#10;作用：衡量基础续写模型完成任务的能力，作为基础阶段评估出口。&#10;本项目：base_eval读eval_bundle的core.yaml/CSV；core_eval按多选、schema、续写评分，各任务校正再平均。&#10;注意：任务列表/few-shot数来自外部bundle，本次未下载无法固定写死；多选平均NLL与chat字母logits评法不同，校正分可负。&#10;位置：scripts/base_eval.py:57 evaluate_core；nanochat/core_eval.py:168 evaluate_example；nanochat/core_eval.py:244 evaluate_task">基础CORE/ICL</abbr>（[释义](glossary/07-evaluation-and-metrics.md#term-core)） | 外部bundle按task_type/few-shot评分再校正 | 未下载bundle，不能确认固定任务数/清单。 |
| <a id="ref-chatcore-001"></a><abbr title="ChatCORE / Categorical Evaluation｜对话能力汇总与分类评估&#10;定义：按对话协议运行多个任务，校正随机基线后平均；分类评价在候选答案字母中直接选最高分。&#10;作用：比较SFT/RL的回答能力，并区分有限选项识别与开放生成。&#10;本项目：ARC-Easy/Challenge、MMLU比较提示末字母logits，GSM8K/HumanEval用Engine生成判题，前三baseline0.25后二0。&#10;注意：训练期生成任务默认24题而非全量，字母须单token；分数可低于0，不能当作自由生成答案的等价准确率。&#10;位置：scripts/chat_eval.py:87 run_categorical_eval；scripts/chat_eval.py:156 run_chat_eval；scripts/chat_sft.py:377 centered_mean">对话ChatCORE</abbr>（[释义](glossary/07-evaluation-and-metrics.md#term-chatcore)） | 3类字母分类+2类生成判题 | 选择题限制候选字母，不是自由生成。 |

$$
\mathrm{BPB}=\frac{\sum_{y\ge0,\;\mathrm{bytes}(y)>0}\mathrm{NLL}(y)}
{\ln2\sum_{y\ge0}\mathrm{bytes}(y)}.
$$

训练CE可监督特殊token，但BPB把其字节置0并排除。CORE求mean((accuracy−baseline)/(1−baseline))，CSV基线以百分比读入；ChatCORE三选择任务基线0.25、数学/代码0，校正分可以为负。训练期每生成任务默认24题，并不是全量质量结论。

CORE多选按候选续写平均NLL最小选择；语言续写逐位置argmax全匹配，不采样。chat_eval分类取提示末位置、指定单token答案字母的最大logits；GSM8K/HumanEval用Engine生成再判定，num_samples>1时任一成功计该题通过。

<a id="ref-benchmarks-002"></a><abbr title="GSM8K / MMLU / ARC / HumanEval｜数学、知识、科学与代码任务&#10;定义：不同数据与评分接口，各检验一种能力，不是训练算法名称。&#10;作用：可用于示范、结果奖励或独立任务评分，但用途随入口而不同。&#10;本项目：SFT用MMLU/GSM8K，RL只GSM8K，ChatCORE再加ARC/HumanEval；数学抽####数字字符串、代码执行测试。&#10;注意：SFT验证取部分test，需记录污染/使用范围；HumanEval执行器不是对抗性安全隔离，本次未运行；实际条数未下载确认。&#10;位置：tasks/gsm8k.py:86 evaluate；tasks/mmlu.py:28 get_example；tasks/arc.py:23 get_example；tasks/humaneval.py:79 evaluate">HumanEval</abbr>（[释义](glossary/07-evaluation-and-metrics.md#term-benchmarks)）拼接生成代码与测试，由execute_code在新subprocess/临时目录限时限内存、禁部分函数，源码明确不是对抗性安全隔离。本次未运行代码评测。eval不自动关闭求导，评估函数显式no_grad，Engine用inference_mode。

## 13.2 能力与系统性能的边界

本章讨论分词压缩、预测损失和任务能力。推理延迟、吞吐、显存属于系统性能，实验方法见[每日路线 D12](../career_roadmap/w2/d12_benchmark_schema.md)和[D13](../career_roadmap/w2/d13_repeated_benchmark.md)；不能用回答质量推导加速比。
