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

[栏目目录](README.md) · [上一篇](11-supervised-fine-tuning.md) · [下一篇](13-evaluation-and-metrics.md) · [术语查阅](glossary/README.md)

> 原技术正文沿用 2026-10-12 的源码核对记录；2026-10-21 调整结构。新增 SFT 串联说明做了局部静态核对，未执行训练。

# 12. 推理、采样与 KV Cache

<a id="sec-inference"></a>
先读模型前向与分词协议；本章解释如何把预测分数变成回答，是后续生成评估和 RL 的前置知识。

## 12.1 无缓存生成与缓存生成

`GPT.generate` 每步重新输入完整前缀；`Engine.generate` 先处理提示，再缓存历史 <abbr title="Key / Value · 注意力中的键和值；保存历史位置的计算结果，供后续生成步骤复用。">K/V</abbr>，后续只输入新增 token。两条路径都调用同一个模型前向；是否训练、是否有缓存是不同维度。

## 12.2 生成控制流与返回格式

<a id="ref-generate-001"></a><abbr title="Generation / Prefill / Decode｜生成、提示预填充与逐词解码&#10;定义：生成重复预测下一ID，prefill一次处理完整提示，decode随后每次只处理新token并读历史缓存。&#10;作用：把训练概率模型用于实际回答，减少每步重复计算旧前缀的成本。&#10;本项目：Engine先B1预填充再复制缓存到多样本行，后续[B,1]解码；generate_batch返回含提示序列与mask，去终止ID。&#10;注意：生成不backward；评分前切prefix_length。流式CLI可能显示终止标记并补assistant_end，不能与批量契约混用。&#10;位置：nanochat/engine.py:176 Engine.generate；nanochat/engine.py:277 generate_batch；scripts/chat_cli.py:86 生成">生成</abbr>（[释义](glossary/09-inference-and-sampling.md#term-generate)）从渲染好的提示ID开始：B=1的<a id="ref-generate-002"></a><abbr title="Generation / Prefill / Decode｜生成、提示预填充与逐词解码&#10;定义：生成重复预测下一ID，prefill一次处理完整提示，decode随后每次只处理新token并读历史缓存。&#10;作用：把训练概率模型用于实际回答，减少每步重复计算旧前缀的成本。&#10;本项目：Engine先B1预填充再复制缓存到多样本行，后续[B,1]解码；generate_batch返回含提示序列与mask，去终止ID。&#10;注意：生成不backward；评分前切prefix_length。流式CLI可能显示终止标记并补assistant_end，不能与批量契约混用。&#10;位置：nanochat/engine.py:176 Engine.generate；nanochat/engine.py:277 generate_batch；scripts/chat_cli.py:86 生成">prefill</abbr>（[释义](glossary/09-inference-and-sampling.md#term-generate)）完整前向，复制提示缓存至多样本行，<a id="ref-generate-003"></a><abbr title="Generation / Prefill / Decode｜生成、提示预填充与逐词解码&#10;定义：生成重复预测下一ID，prefill一次处理完整提示，decode随后每次只处理新token并读历史缓存。&#10;作用：把训练概率模型用于实际回答，减少每步重复计算旧前缀的成本。&#10;本项目：Engine先B1预填充再复制缓存到多样本行，后续[B,1]解码；generate_batch返回含提示序列与mask，去终止ID。&#10;注意：生成不backward；评分前切prefix_length。流式CLI可能显示终止标记并补assistant_end，不能与批量契约混用。&#10;位置：nanochat/engine.py:176 Engine.generate；nanochat/engine.py:277 generate_batch；scripts/chat_cli.py:86 生成">decode</abbr>（[释义](glossary/09-inference-and-sampling.md#term-generate)）每步输入[B,1]，从末位置[B,V]分数按<a id="ref-sampling-001"></a><abbr title="Temperature / Top-k / Greedy｜温度、截断采样与贪心&#10;定义：温度改变分数差异，top-k限制最高分候选，greedy直接取最高分ID。&#10;作用：调节生成随机性与候选范围，是推理配置而非学习率或可学习权重。&#10;本项目：sample_next_token在温度0先argmax；正温度top_k&gt;0取k分数除温度softmax采样，CLI0.6/50，RL1/50。&#10;注意：温度0时top-k分支不运行，增加温度不保证质量；评估需固定设置，采样分布可能不同于RL训练全词表logp。&#10;位置：nanochat/engine.py:141 sample_next_token；scripts/chat_cli.py:19 配置；scripts/chat_rl.py:47 生成配置">温度/top-k</abbr>（[释义](glossary/09-inference-and-sampling.md#term-sampling)）选下个ID。temperature=0直接argmax；CLI默认0.6/top-k50、每答最大256。

<a id="ref-kv-001"></a><abbr title="KV Cache｜键值缓存&#10;定义：保存过去位置K/V激活，下一步只投影新位置并读取已算历史，不保存Q。&#10;作用：减少自回归生成重复计算，缓存大小由层数、长度、KV头数与单头维决定。&#10;本项目：K/V分别[L,B,Tmax,Hkv,D]，int32[B] cache_seqlens，另存Smear prev_embedding，最后层advance。&#10;注意：不是可学习权重、optimizer或checkpoint；默认MHA无共享头节省，容量/RoPE范围有限，不能承诺无限对话。&#10;位置：nanochat/engine.py:82 KVCache；nanochat/gpt.py:113 缓存路径；nanochat/gpt.py:123 advance">KVCache</abbr>（[释义](glossary/09-inference-and-sampling.md#term-kv)）存K/V为[L,B,Tmax,Hkv,D]，cache_seqlens为int32[B]，另存Smear的prev_embedding；最后一层才advance。它不是可学习权重/optimizer状态/训练checkpoint，不缓存Q。默认MHA没有GQA共享头收益。缓存与RoPE都有有限余量，CLI没有自动历史摘要或截断，不能承诺无限多轮；clear会重置对话。

generate_batch返回含原提示的序列与mask，评分先去prefix_length，终止BOS/assistant_end不在结果里；强制工具词元mask0。流式CLI逐词显示，最后保证补assistant_end，不能混用两个返回契约。
