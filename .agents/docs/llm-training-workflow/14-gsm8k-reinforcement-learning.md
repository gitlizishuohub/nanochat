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

[栏目目录](README.md) · [上一篇](13-evaluation-and-metrics.md) · [下一篇](15-running-examples-and-delivery.md) · [术语查阅](glossary/README.md)

> 原技术正文沿用 2026-10-12 的源码核对记录；2026-10-21 调整结构。新增 SFT 串联说明做了局部静态核对，未执行训练。

# 14. 可选 GSM8K 强化学习

<a id="sec-10"></a>

## 14.1 提示、生成、奖励与有效目标

<a id="ref-rl-002"></a><abbr title="RL / Reinforcement Learning｜强化学习&#10;定义：利用模型生成结果的奖励调整行为；训练信号是结果好坏，不是每个位置应写哪一个示范词元。&#10;作用：在已有SFT对话能力上提高数学答案正确率，是主线外的任务专项可选阶段。&#10;本项目：chat_rl从sft加载，每题默认采16回答，GSM8K.reward给0/1，奖励减组均值形成优势，再对有效生成词元求策略梯度。&#10;注意：没有奖励模型、价值网络、参考KL、PPO比率裁剪或fp16 GradScaler；训练后仍需真实评估。&#10;位置：scripts/chat_rl.py:74 sft加载；scripts/chat_rl.py:86 get_batch；tasks/gsm8k.py:109 reward">数学RL</abbr>（[释义](glossary/01-training-stages.md#term-rl)）加载sft。<a id="ref-benchmarks-001"></a><abbr title="GSM8K / MMLU / ARC / HumanEval｜数学、知识、科学与代码任务&#10;定义：不同数据与评分接口，各检验一种能力，不是训练算法名称。&#10;作用：可用于示范、结果奖励或独立任务评分，但用途随入口而不同。&#10;本项目：SFT用MMLU/GSM8K，RL只GSM8K，ChatCORE再加ARC/HumanEval；数学抽####数字字符串、代码执行测试。&#10;注意：SFT验证取部分test，需记录污染/使用范围；HumanEval执行器不是对抗性安全隔离，本次未运行；实际条数未下载确认。&#10;位置：tasks/gsm8k.py:86 evaluate；tasks/mmlu.py:28 get_example；tasks/arc.py:23 get_example；tasks/humaneval.py:79 evaluate">GSM8K</abbr>（[释义](glossary/07-evaluation-and-metrics.md#term-benchmarks)）完整对话最后回答供参考答案判定，render_for_completion删除它，仅保留用户题目并追加assistant_start，避免答案泄露。<a id="ref-rollout-001"></a><abbr title="Rollout / On-policy｜生成轨迹与当前策略采样&#10;定义：rollout是当前模型在提示后生成的回答轨迹，连同工具注入、有效mask和结果奖励。&#10;作用：让RL训练围绕模型当前行为发生，而不是仅重用固定示范或离线旧样本。&#10;本项目：每step对分配题目现场采样，padding/shift后重算logp，多题反向后统一更新；没有回放池。&#10;注意：温度/top-k改变采样概率，训练logp未相同重归一化，不能把on-policy说明直接当标准无偏策略梯度证明。&#10;位置：scripts/chat_rl.py:86 get_batch；scripts/chat_rl.py:248 rollout更新；nanochat/engine.py:176 generate">rollout</abbr>（[释义](glossary/08-rewards-and-rollouts.md#term-rollout)）包含模型现场生成和强制工具结果。

默认num_epochs1、examples_per_step16（全局题数）、num_samples16（每题回答）、device_batch_size8（每次前向最大行数）、max_new_tokens256、temperature1、top_k50。题数须整除world_size；采样组数为num_samples//device_batch_size，代码未补余数，应用正整数倍，否则样本数缩小或空采样失败。

<a id="ref-reward-001"></a><abbr title="Reward / Advantage｜奖励与优势&#10;定义：奖励表示结果好坏，优势表示相对基准收益，决定提高或降低哪些回答的生成概率。&#10;作用：用组内相对信息减小绝对奖励噪声，让好于平均回答获得正向更新。&#10;本项目：reward为0/1，advantages=rewards−mean；[1,0,0,1]变[0.5,−0.5,−0.5,0.5]，广播至有效token。&#10;注意：不除标准差；组全对/全错优势0，但历史动量/decay仍可能改参数。数学评分为字符串比较，无数值容差。&#10;位置：tasks/gsm8k.py:109 reward；scripts/chat_rl.py:144 advantages；scripts/chat_rl.py:266 pg_obj">奖励/优势</abbr>（[释义](glossary/08-rewards-and-rollouts.md#term-reward)）：提取####后数字字符串，去逗号，以相等给0/1；不是数值容差，“10”与“10.0”通常不同。参考和预测都解析不到时None==None还可能被判对，因此依赖有效参考格式，本次未核验数据。

每题多回答padding到最长，用assistant_end补位。Engine mask提示为0、模型采样为1、强制工具结果为0；padding也0，shift后设无效target为−1。generate_batch排除终止BOS/assistant_end，所以RL不直接监督被排除终止位置，区别于SFT的assistant_end监督。

## 14.2 真实策略梯度目标

组内优势：

$$
A_i=r_i-\frac1G\sum_{j=1}^G r_j.
$$

例如r=[1,0,0,1]，A=[0.5,−0.5,−0.5,0.5]；全对/全错优势全0。负NLL得到logp，按每个device分块求：

$$
\mathcal L_{q,p}=
-\frac{\sum_{i,t\in S_{q,p}}A_i\log p_\theta(y_{i,t}\mid x_{i,\le t})}
{\max(1,|S_{q,p}|)\,P\,E_{\rm rank}}.
$$

q是一题，p是其device分块，P是分块数，E_rank是本rank本步题数。分别按分块有效token归一化再累积，不是整步统一token均值。正优势增概率、负优势减概率；零策略梯度不保证历史动量/衰减完全不改权重。

更接近带组均值基线的<a id="ref-grpo-002"></a><abbr title="GRPO / PPO / REINFORCE｜组相对优化与策略梯度对照&#10;定义：REINFORCE以奖励加权生成logp；GRPO使用组内相对奖励，PPO通常还有新旧策略概率比和裁剪。&#10;作用：辨明源码真正优化的目标，避免把脚本标题算法名当成完整标准实现。&#10;本项目：chat_rl自称带引号GRPO，优势仅减均值不除标准差，逐分块按有效token数/分块数/每卡题数归一化。&#10;注意：没有ratio+clip、KL或value网络；默认top-k采样与训练完整softmax概率不一致，不能据此声称标准无偏性已验证。&#10;位置：scripts/chat_rl.py:1 算法说明；scripts/chat_rl.py:264 logp/pg_obj">REINFORCE简化实现</abbr>（[释义](glossary/01-training-stages.md#term-grpo)）：不除奖励标准差，没有PPO比率裁剪、参考KL或value loss。top-k/temperature使采样分布与训练重新前向的完整softmax概率不同，不能把脚本的on-policy说明当作无偏性验证。

## 14.3 调度、评价、产物

RL组学习率按embedding0.2、unembedding0.004、matrix0.02构造，再全乘init_lr_frac0.05；每步乘1−step/num_steps，最后更新索引不到num_steps，所以实际末更新lr不精确为0。--weight-decay帮助写embedding/unembedding，实参传给setup_optimizer影响Muon；AdamW仍固定组decay。

每60步评最多400题的<a id="ref-passk-001"></a><abbr title="pass@k｜多次采样通过率&#10;定义：每题前k个生成结果至少一个正确判通过，再计算题目层面的通过比例。&#10;作用：衡量多尝试找到正确数学/代码答案的能力，区别于单次生成准确率。&#10;本项目：RL每60步默认最多400题生成device_batch_size样本，对前k做any计数，多rank求和；chat_eval也可多样本any。&#10;注意：是直接经验计数，未用组合数无偏估计；温度、top-k、样本数和题数改变结果，不能与单次贪心不加说明横比。&#10;位置：scripts/chat_rl.py:150 run_gsm8k_eval；scripts/chat_rl.py:229 passk；scripts/chat_eval.py:54 outcomes">pass@k</abbr>（[释义](glossary/07-evaluation-and-metrics.md#term-passk)），每题采device_batch_size个回答，前k任一正确就计通过，各rank求和。最后更新后不保证额外最终评估，应独立chat_eval。RL不存optimizer、不编译模型、不用GradScaler或梯度裁剪。

~~~bash
# 前提：兼容d20 SFT权重与原词表；本次未执行。
python -m scripts.chat_rl --model-tag=d20 \
  --num-samples=16 --device-batch-size=8 \
  --examples-per-step=16 --run=dummy
python -m scripts.chat_eval -i rl -g d20 -a GSM8K
~~~
