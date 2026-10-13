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

[栏目目录](README.md) · [上一篇](11-evaluation-generation-and-delivery.md) · [下一篇](12-1-training-stages-and-implementation-boundaries.md)

> 拆分自原文；源码核对日期为 2026-10-12。本次仅整理文档结构，未重新核验源码或运行训练。

<a id="sec-12"></a>
## 12. 专有名词对照表

每项以定义、作用、本项目、注意、位置五部分解释，正文title由完全同一字段生成。每处引用有独立ref锚点；返回标签指明章节、小节与出现词。表项不再递归添加术语链接。


- [12.1 训练阶段与实现边界](12-1-training-stages-and-implementation-boundaries.md)
- [12.2 数据与对话协议](12-2-data-and-conversation-protocol.md)
- [12.3 批量与分布式](12-3-batching-and-distributed-training.md)
- [12.4 构建与模型前向](12-4-model-construction-and-forward-pass.md)
- [12.5 损失与参数更新](12-5-loss-and-parameter-updates.md)
- [12.6 保存、评估与交付](12-6-checkpoints-evaluation-and-delivery.md)
