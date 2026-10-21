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

[术语目录](README.md) · [学习主线](../README.md)

# 推理与采样

| 英文或缩写 | 中文名称 | 详细解释 | 项目对应文件或符号 | 返回正文链接 |
| --- | --- | --- | --- | --- |
| <a id="term-generate"></a>Generation / Prefill / Decode | 生成、提示预填充与逐词解码 | **定义：**生成重复预测下一ID，prefill一次处理完整提示，decode随后每次只处理新token并读历史缓存。<br>**作用：**把训练概率模型用于实际回答，减少每步重复计算旧前缀的成本。<br>**本项目：**Engine先B1预填充再复制缓存到多样本行，后续[B,1]解码；generate_batch返回含提示序列与mask，去终止ID。<br>**注意：**生成不backward；评分前切prefix_length。流式CLI可能显示终止标记并补assistant_end，不能与批量契约混用。<br>**位置：**nanochat/engine.py:176 Engine.generate；nanochat/engine.py:277 generate_batch；scripts/chat_cli.py:86 生成 | [Engine.generate（nanochat/engine.py:176）](/home/lizishuo/_program_/Project/nanochat/nanochat/engine.py:176)<br>[generate_batch（nanochat/engine.py:277）](/home/lizishuo/_program_/Project/nanochat/nanochat/engine.py:277)<br>[生成（scripts/chat_cli.py:86）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_cli.py:86) | [返回：12.2 生成控制流与返回格式·生成](../12-inference-and-generation.md#ref-generate-001)<br>[返回：12.2 生成控制流与返回格式·prefill](../12-inference-and-generation.md#ref-generate-002)<br>[返回：12.2 生成控制流与返回格式·decode](../12-inference-and-generation.md#ref-generate-003) |
| <a id="term-kv"></a>KV Cache | 键值缓存 | **定义：**保存过去位置K/V激活，下一步只投影新位置并读取已算历史，不保存Q。<br>**作用：**减少自回归生成重复计算，缓存大小由层数、长度、KV头数与单头维决定。<br>**本项目：**K/V分别[L,B,Tmax,Hkv,D]，int32[B] cache_seqlens，另存Smear prev_embedding，最后层advance。<br>**注意：**不是可学习权重、optimizer或checkpoint；默认MHA无共享头节省，容量/RoPE范围有限，不能承诺无限对话。<br>**位置：**nanochat/engine.py:82 KVCache；nanochat/gpt.py:113 缓存路径；nanochat/gpt.py:123 advance | [KVCache（nanochat/engine.py:82）](/home/lizishuo/_program_/Project/nanochat/nanochat/engine.py:82)<br>[缓存路径（nanochat/gpt.py:113）](/home/lizishuo/_program_/Project/nanochat/nanochat/gpt.py:113)<br>[advance（nanochat/gpt.py:123）](/home/lizishuo/_program_/Project/nanochat/nanochat/gpt.py:123) | [返回：12.2 生成控制流与返回格式·KVCache](../12-inference-and-generation.md#ref-kv-001) |
| <a id="term-sampling"></a>Temperature / Top-k / Greedy | 温度、截断采样与贪心 | **定义：**温度改变分数差异，top-k限制最高分候选，greedy直接取最高分ID。<br>**作用：**调节生成随机性与候选范围，是推理配置而非学习率或可学习权重。<br>**本项目：**sample_next_token在温度0先argmax；正温度top_k&gt;0取k分数除温度softmax采样，CLI0.6/50，RL1/50。<br>**注意：**温度0时top-k分支不运行，增加温度不保证质量；评估需固定设置，采样分布可能不同于RL训练全词表logp。<br>**位置：**nanochat/engine.py:141 sample_next_token；scripts/chat_cli.py:19 配置；scripts/chat_rl.py:47 生成配置 | [sample_next_token（nanochat/engine.py:141）](/home/lizishuo/_program_/Project/nanochat/nanochat/engine.py:141)<br>[配置（scripts/chat_cli.py:19）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_cli.py:19)<br>[生成配置（scripts/chat_rl.py:47）](/home/lizishuo/_program_/Project/nanochat/scripts/chat_rl.py:47) | [返回：12.2 生成控制流与返回格式·温度/top-k](../12-inference-and-generation.md#ref-sampling-001) |
