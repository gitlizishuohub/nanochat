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

[栏目目录](README.md) · [上一篇](10-gsm8k-reinforcement-learning.md) · [下一篇](12-glossary-index.md)

> 拆分自原文；源码核对日期为 2026-10-12。本次仅整理文档结构，未重新核验源码或运行训练。

<a id="sec-11"></a>
## 11. 验证、评估、生成与交付

### 11.1 四种评价出口

| 评价 | 源码过程 | 解释边界 |
| --- | --- | --- |
| 分词压缩 | tok_eval round-trip与bytes/token | 不是模型能力；对照词表可能联网缓存。 |
| <a id="ref-bpb-002"></a>[验证BPB](12-6-checkpoints-evaluation-and-delivery.md#term-bpb "BPB / Bits Per Byte｜每字节比特数&#10;定义：将目标NLL从自然对数转比特，再除这些普通目标词元对应的原始字节总数。&#10;作用：在相同语料/评估构造下减小词表大小变化对平均token loss比较的干扰。&#10;本项目：token_bytes.pt记录原始字节，特殊ID0；evaluate_bpb排除y&lt;0与0字节目标，各rank求和后除ln2×总bytes。&#10;注意：不是平均loss除平均字数；装箱/裁切/SFT mask改变统计范围，不能保证不同评价任意可比；零字节结果inf。&#10;位置：scripts/tok_train.py:80 token_bytes；nanochat/loss_eval.py:9 evaluate_bpb") | 普通未忽略目标总NLL/总字节 | 不同装箱/监督范围不可随意比较，0有效字节返回inf。 |
| <a id="ref-core-001"></a>[基础CORE/ICL](12-6-checkpoints-evaluation-and-delivery.md#term-core "CORE / ICL / Few-shot｜基础能力指标与上下文示例学习&#10;定义：ICL是在提示中加入示例不更新权重；CORE汇总若干基础任务的随机基线校正准确率。&#10;作用：衡量基础续写模型完成任务的能力，作为基础阶段评估出口。&#10;本项目：base_eval读eval_bundle的core.yaml/CSV；core_eval按多选、schema、续写评分，各任务校正再平均。&#10;注意：任务列表/few-shot数来自外部bundle，本次未下载无法固定写死；多选平均NLL与chat字母logits评法不同，校正分可负。&#10;位置：scripts/base_eval.py:57 evaluate_core；nanochat/core_eval.py:168 evaluate_example；nanochat/core_eval.py:244 evaluate_task") | 外部bundle按task_type/few-shot评分再校正 | 未下载bundle，不能确认固定任务数/清单。 |
| <a id="ref-chatcore-001"></a>[对话ChatCORE](12-6-checkpoints-evaluation-and-delivery.md#term-chatcore "ChatCORE / Categorical Evaluation｜对话能力汇总与分类评估&#10;定义：按对话协议运行多个任务，校正随机基线后平均；分类评价在候选答案字母中直接选最高分。&#10;作用：比较SFT/RL的回答能力，并区分有限选项识别与开放生成。&#10;本项目：ARC-Easy/Challenge、MMLU比较提示末字母logits，GSM8K/HumanEval用Engine生成判题，前三baseline0.25后二0。&#10;注意：训练期生成任务默认24题而非全量，字母须单token；分数可低于0，不能当作自由生成答案的等价准确率。&#10;位置：scripts/chat_eval.py:87 run_categorical_eval；scripts/chat_eval.py:156 run_chat_eval；scripts/chat_sft.py:377 centered_mean") | 3类字母分类+2类生成判题 | 选择题限制候选字母，不是自由生成。 |

$$
\mathrm{BPB}=\frac{\sum_{y\ge0,\;\mathrm{bytes}(y)>0}\mathrm{NLL}(y)}
{\ln2\sum_{y\ge0}\mathrm{bytes}(y)}.
$$

训练CE可监督特殊token，但BPB把其字节置0并排除。CORE求mean((accuracy−baseline)/(1−baseline))，CSV基线以百分比读入；ChatCORE三选择任务基线0.25、数学/代码0，校正分可以为负。训练期每生成任务默认24题，并不是全量质量结论。

CORE多选按候选续写平均NLL最小选择；语言续写逐位置argmax全匹配，不采样。chat_eval分类取提示末位置、指定单token答案字母的最大logits；GSM8K/HumanEval用Engine生成再判定，num_samples>1时任一成功计该题通过。

<a id="ref-benchmarks-002"></a>[HumanEval](12-6-checkpoints-evaluation-and-delivery.md#term-benchmarks "GSM8K / MMLU / ARC / HumanEval｜数学、知识、科学与代码任务&#10;定义：不同数据与评分接口，各检验一种能力，不是训练算法名称。&#10;作用：可用于示范、结果奖励或独立任务评分，但用途随入口而不同。&#10;本项目：SFT用MMLU/GSM8K，RL只GSM8K，ChatCORE再加ARC/HumanEval；数学抽####数字字符串、代码执行测试。&#10;注意：SFT验证取部分test，需记录污染/使用范围；HumanEval执行器不是对抗性安全隔离，本次未运行；实际条数未下载确认。&#10;位置：tasks/gsm8k.py:86 evaluate；tasks/mmlu.py:28 get_example；tasks/arc.py:23 get_example；tasks/humaneval.py:79 evaluate")拼接生成代码与测试，由execute_code在新subprocess/临时目录限时限内存、禁部分函数，源码明确不是对抗性安全隔离。本次未运行代码评测。eval不自动关闭求导，评估函数显式no_grad，Engine用inference_mode。

### 11.2 未来启动命令与默认参考

以下CPU极小模型示例需先准备第3节数据与词表。参数名、batch整除及源码路径已静态核对，未运行、不承诺训练质量或当前依赖可用。保留2048长度以减少SFT默认2048上限与短行冲突。

~~~bash
NANOCHAT_DTYPE=float32 python -m scripts.base_train \
  --device-type=cpu --depth=4 --head-dim=64 --window-pattern=L \
  --max-seq-len=2048 --device-batch-size=1 --total-batch-size=2048 \
  --num-iterations=20 --eval-tokens=2048 --eval-every=10 \
  --core-metric-every=-1 --sample-every=-1 --save-every=10 \
  --model-tag=teach-d4 --run=dummy

NANOCHAT_DTYPE=float32 python -m scripts.base_eval \
  --device-type=cpu --model-tag=teach-d4 --eval=bpb \
  --device-batch-size=1 --split-tokens=2048

# SFT默认消费混合数据一轮，CPU可能很久；
# --num-iterations不能据帮助文字承诺精确优化更新步数，此处不使用。
NANOCHAT_DTYPE=float32 python -m scripts.chat_sft \
  --device-type=cpu --model-tag=teach-d4 --load-optimizer=0 \
  --eval-tokens=2048 --eval-every=200 --chatcore-every=-1 --run=dummy

NANOCHAT_DTYPE=float32 python -m scripts.chat_eval \
  --device-type=cpu -i sft -g teach-d4 -a ARC-Easy -x 16 -b 1
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli \
  --device-type=cpu -i sft -g teach-d4 -p "Why is the sky blue?"
~~~

20步例大半位于warmup，不是充分收敛。SFT仍有截断、全mask和CPU编译等边界，不能因为命令通过静态检查就说运行通过。GPU参考来自当前runs/speedrun.sh主体：

~~~bash
# 已准备依赖、170训练分片、原词表及兼容多GPU；本次未执行。
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=24 --target-param-data-ratio=8 --device-batch-size=16 \
  --fp8 --fp8-recipe=tensorwise --run=dummy
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
  --device-batch-size=16
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=dummy
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
python -m scripts.chat_cli -i sft -g d24
~~~

脚本当前d24/ratio8/batch16/FP8，与入口d20/ratio12/batch32及旧d20注释不同。整个bash脚本会安装、下载、训练，本次未执行。显存不足可减device-batch-size，同时保持总量整除，不能宣称当前显卡可承载。

### 11.3 生成、缓存与返回格式

<a id="ref-generate-001"></a>[生成](12-6-checkpoints-evaluation-and-delivery.md#term-generate "Generation / Prefill / Decode｜生成、提示预填充与逐词解码&#10;定义：生成重复预测下一ID，prefill一次处理完整提示，decode随后每次只处理新token并读历史缓存。&#10;作用：把训练概率模型用于实际回答，减少每步重复计算旧前缀的成本。&#10;本项目：Engine先B1预填充再复制缓存到多样本行，后续[B,1]解码；generate_batch返回含提示序列与mask，去终止ID。&#10;注意：生成不backward；评分前切prefix_length。流式CLI可能显示终止标记并补assistant_end，不能与批量契约混用。&#10;位置：nanochat/engine.py:176 Engine.generate；nanochat/engine.py:277 generate_batch；scripts/chat_cli.py:86 生成")从渲染好的提示ID开始：B=1的<a id="ref-generate-002"></a>[prefill](12-6-checkpoints-evaluation-and-delivery.md#term-generate "Generation / Prefill / Decode｜生成、提示预填充与逐词解码&#10;定义：生成重复预测下一ID，prefill一次处理完整提示，decode随后每次只处理新token并读历史缓存。&#10;作用：把训练概率模型用于实际回答，减少每步重复计算旧前缀的成本。&#10;本项目：Engine先B1预填充再复制缓存到多样本行，后续[B,1]解码；generate_batch返回含提示序列与mask，去终止ID。&#10;注意：生成不backward；评分前切prefix_length。流式CLI可能显示终止标记并补assistant_end，不能与批量契约混用。&#10;位置：nanochat/engine.py:176 Engine.generate；nanochat/engine.py:277 generate_batch；scripts/chat_cli.py:86 生成")完整前向，复制提示缓存至多样本行，<a id="ref-generate-003"></a>[decode](12-6-checkpoints-evaluation-and-delivery.md#term-generate "Generation / Prefill / Decode｜生成、提示预填充与逐词解码&#10;定义：生成重复预测下一ID，prefill一次处理完整提示，decode随后每次只处理新token并读历史缓存。&#10;作用：把训练概率模型用于实际回答，减少每步重复计算旧前缀的成本。&#10;本项目：Engine先B1预填充再复制缓存到多样本行，后续[B,1]解码；generate_batch返回含提示序列与mask，去终止ID。&#10;注意：生成不backward；评分前切prefix_length。流式CLI可能显示终止标记并补assistant_end，不能与批量契约混用。&#10;位置：nanochat/engine.py:176 Engine.generate；nanochat/engine.py:277 generate_batch；scripts/chat_cli.py:86 生成")每步输入[B,1]，从末位置[B,V]分数按<a id="ref-sampling-001"></a>[温度/top-k](12-6-checkpoints-evaluation-and-delivery.md#term-sampling "Temperature / Top-k / Greedy｜温度、截断采样与贪心&#10;定义：温度改变分数差异，top-k限制最高分候选，greedy直接取最高分ID。&#10;作用：调节生成随机性与候选范围，是推理配置而非学习率或可学习权重。&#10;本项目：sample_next_token在温度0先argmax；正温度top_k&gt;0取k分数除温度softmax采样，CLI0.6/50，RL1/50。&#10;注意：温度0时top-k分支不运行，增加温度不保证质量；评估需固定设置，采样分布可能不同于RL训练全词表logp。&#10;位置：nanochat/engine.py:141 sample_next_token；scripts/chat_cli.py:19 配置；scripts/chat_rl.py:47 生成配置")选下个ID。temperature=0直接argmax；CLI默认0.6/top-k50、每答最大256。

<a id="ref-kv-001"></a>[KVCache](12-6-checkpoints-evaluation-and-delivery.md#term-kv "KV Cache｜键值缓存&#10;定义：保存过去位置K/V激活，下一步只投影新位置并读取已算历史，不保存Q。&#10;作用：减少自回归生成重复计算，缓存大小由层数、长度、KV头数与单头维决定。&#10;本项目：K/V分别[L,B,Tmax,Hkv,D]，int32[B] cache_seqlens，另存Smear prev_embedding，最后层advance。&#10;注意：不是可学习权重、optimizer或checkpoint；默认MHA无共享头节省，容量/RoPE范围有限，不能承诺无限对话。&#10;位置：nanochat/engine.py:82 KVCache；nanochat/gpt.py:113 缓存路径；nanochat/gpt.py:123 advance")存K/V为[L,B,Tmax,Hkv,D]，cache_seqlens为int32[B]，另存Smear的prev_embedding；最后一层才advance。它不是可学习权重/optimizer状态/训练checkpoint，不缓存Q。默认MHA没有GQA共享头收益。缓存与RoPE都有有限余量，CLI没有自动历史摘要或截断，不能承诺无限多轮；clear会重置对话。

generate_batch返回含原提示的序列与mask，评分先去prefix_length，终止BOS/assistant_end不在结果里；强制工具词元mask0。流式CLI逐词显示，最后保证补assistant_end，不能混用两个返回契约。

### 11.4 模型交付与缺失的部署能力

交付选定model.pt、同一步meta.json、原tokenizer.pkl/token_bytes.pt、仓库版本、启动配置与实际评估范围/结果；恢复base另需匹配rank优化器分片，RL没有该产物。能力与性能结果必须真实执行后填写。

现有形态是PyTorch state_dict+JSON+CLI/Engine；没有ONNX/GGUF/Safetensors导出、量化部署、Web服务或发布管线。[scripts/infer_bench.py](/home/lizishuo/_program_/Project/nanochat/scripts/infer_bench.py)是单CUDA卡性能入口，不是训练，打印依赖meta['step']；RL meta没有step，直接测RL可能失败。本次没有运行，不编造缺失功能命令。
