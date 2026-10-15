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

[栏目目录](README.md) · [上一篇](03-environment-and-data-preparation.md) · [下一篇](05-model-construction-and-weight-loading.md)

> 拆分自原文；源码核对日期为 2026-10-12。本次仅整理文档结构，未重新核验源码或运行训练。

<a id="sec-4"></a>
# 4. tokenizer 与对话协议

## 4.1 训练词表与配套产物

<a id="ref-bpe-001"></a><abbr title="BPE / Byte Pair Encoding｜字节对编码&#10;定义：从字节片段出发反复合并常见相邻对，形成可复用的编码单元，再把文本变为这些单元的ID序列。&#10;作用：在控制词表数量时压缩序列，兼顾多语言与未见文本的表示，减少模型所需处理位置。&#10;本项目：先按SPLIT_PATTERN切分，数字采用1至2位；总词表先减9个特殊token容量再训练普通排名，默认总量32768。&#10;注意：普通文本走encode_ordinary，不把用户中的协议字符串自动当特殊ID；tokenizer字符预算不等于模型训练token预算。&#10;位置：nanochat/tokenizer.py:26 SPLIT_PATTERN；nanochat/tokenizer.py:42 train_from_iterator">字节级 BPE</abbr>（[释义](12-2-data-and-conversation-protocol.md#term-bpe)）先按 SPLIT_PATTERN 预切分、再学合并排名。数字按 1–2 位预切分，区别于代码注释对照的 GPT-4 1–3 位。默认 max_chars=2,000,000,000，doc_cap=10000 字符，<a id="ref-vocab-001"></a><abbr title="Vocabulary / Vocabulary Size｜词表与词表大小&#10;定义：所有可编码ID的集合及数量，定义输出分类空间与输入查表范围。&#10;作用：决定输入输出层尺寸，也影响token压缩率和[B,T,V]分数张量的显存。&#10;本项目：默认总32768包括9特殊ID；GPT内部把词表向上补到64倍数，forward裁掉超出真实V的输出列。&#10;注意：词表维度补齐不同于序列padding；默认32768已整除64，没有额外列，补齐列不代表新增可生成token。&#10;位置：scripts/tok_train.py:19 vocab_size；nanochat/gpt.py:170 padded_vocab_size；nanochat/gpt.py:513 裁列">vocab_size</abbr>（[释义](12-2-data-and-conversation-protocol.md#term-vocab)）=32768（包括 9 个特殊 ID）；达到字符阈值后可能略超预算。这不是模型训练 token 数。

保存 tokenizer/tokenizer.pkl 和 token_bytes.pt；后者记录每个<a id="ref-token-001"></a><abbr title="Token / Token ID｜词元与词元编号&#10;定义：词元是字节片段或特殊标记，ID是词表对应整数索引；ID序列是模型实际接收和预测的对象。&#10;作用：作为嵌入表查询和下一步分类目标，最终需解码才能变为可读文本。&#10;本项目：encode返回整数列表；预训练输入/目标为long[B,T]，SFT输入int32、目标int64，最后预测维度V为真实词表。&#10;注意：ID不是概率也不是字符位置，中文字符数和token数不一一对应；示例字母仅解释shift，不声称实际编码如此。&#10;位置：nanochat/tokenizer.py:96 encode；nanochat/dataloader.py:115 缓冲区；scripts/chat_sft.py:281 tensors">token ID</abbr>（[释义](12-2-data-and-conversation-protocol.md#term-token)）的原始字节长度，特殊 ID 为 0，供<a id="ref-bpb-001"></a><abbr title="BPB / Bits Per Byte｜每字节比特数&#10;定义：将目标NLL从自然对数转比特，再除这些普通目标词元对应的原始字节总数。&#10;作用：在相同语料/评估构造下减小词表大小变化对平均token loss比较的干扰。&#10;本项目：token_bytes.pt记录原始字节，特殊ID0；evaluate_bpb排除y&lt;0与0字节目标，各rank求和后除ln2×总bytes。&#10;注意：不是平均loss除平均字数；装箱/裁切/SFT mask改变统计范围，不能保证不同评价任意可比；零字节结果inf。&#10;位置：scripts/tok_train.py:80 token_bytes；nanochat/loss_eval.py:9 evaluate_bpb">BPB</abbr>（[释义](12-6-checkpoints-evaluation-and-delivery.md#term-bpb)）用。不要先把单 ID 解码为 UTF-8 字符串再算长度，部分字节片段无法单独构成合法 UTF-8。tok_train 有 round-trip 断言，tok_eval 比较 bytes/token；这些是代码已有检查，本次未执行，未记为通过。

所有阶段必须保留原 ID 映射。load_model 只检查词表大小相等，同样 32768 的另一个词表也可能错配，不可重训后随意替换。

## 4.2 角色、工具与监督位置

<a id="ref-protocol-001"></a><abbr title="Conversation Protocol｜对话协议&#10;定义：把messages的角色、正文和工具分段序列化为带边界的token序列，使训练与生成遵循统一结构。&#10;作用：标明谁在发言、何时调用工具、何时结束回答，避免阶段之间格式错位。&#10;本项目：9个特殊ID；render_conversation要求user/assistant交替，首system并进下一user；assistant支持text/python/python_output。&#10;注意：没有独立system ID或通用Jinja chat_template；render_for_completion删最后示范答案并追加assistant_start，不能把标准答案留进提示。&#10;位置：nanochat/tokenizer.py:140 render_conversation；nanochat/tokenizer.py:241 render_for_completion">对话协议</abbr>（[释义](12-2-data-and-conversation-protocol.md#term-protocol)）包含 9 个特殊 token，没有独立 system/PAD token：

~~~text
<|bos|>
<|user_start|> <|user_end|>
<|assistant_start|> <|assistant_end|>
<|python_start|> <|python_end|>
<|output_start|> <|output_end|>

示例协议序列（不是实际 token ID）：
<|bos|><|user_start|>1+1等于几？<|user_end|>
<|assistant_start|>2<|assistant_end|>
~~~

首条 system 若存在，和下一 user 内容以两个换行合并；其它消息严格 user/assistant 交替。encode 普通文本用 encode_ordinary，边界通过 encode_special 插入，用户字符串形似特殊标记并不会自动变为协议 ID。

<a id="ref-tool-001"></a><abbr title="Tool Use / Calculator｜工具使用与计算器&#10;定义：模型生成调用表达式，外部程序执行并强制插回结果，结果不需要模型自己预测。&#10;作用：为算术或允许的字符串计数提供实际计算值，并让后续生成读到外部信息。&#10;本项目：GSM8K拆&lt;&lt;表达式=结果&gt;&gt;；Engine收集python_start/end中的表达式，受限计算器返回值，以output标记注入。&#10;注意：SFT监督调用不监督结果；RL采样调用mask1，强制结果mask0。不是通用Python REPL，也不是HumanEval代码执行器。&#10;位置：tasks/gsm8k.py:51 get_example；nanochat/engine.py:46 use_calculator；nanochat/engine.py:252 工具状态机">工具调用</abbr>（[释义](12-2-data-and-conversation-protocol.md#term-tool)）在 assistant 分段 content 中区分 text/python/python_output。GSM8K 的 <<表达式=结果>> 拆成调用与输出，Engine 推理时运行受限计算器并强制插回结果，不是通用 Python REPL。

| 原序列位置 | render_conversation 的 mask | 作用 |
| --- | --- | --- |
| BOS、用户边界/正文、assistant_start | 0 | 输入条件，目标不监督。 |
| 助手正文 | 1 | 学示范回答。 |
| python_start、表达式、python_end | 1 | 学会调用工具。 |
| output_start、工具结果、output_end | 0 | 外部计算器强制提供。 |
| assistant_end | 1 | 学会结束。 |

<a id="ref-lossmask-001"></a><abbr title="Loss Mask / Ignore Index｜监督掩码与忽略标签&#10;定义：标记哪些目标位置有效的数组/哨兵，用来决定计算损失的范围。&#10;作用：只监督期望模型生成的行为，同时保留用户提示和外部结果作上下文。&#10;本项目：render_conversation给0/1，SFT shift后0位置targets=−1；cross_entropy(ignore_index=−1)，RL用Engine采样mask同样过滤。&#10;注意：不是attention mask；assistant_start不监督、SFT assistant_end监督。全−1均值CE可能NaN，入口没有样本级过滤保证。&#10;位置：nanochat/tokenizer.py:140 render_conversation；scripts/chat_sft.py:290 mask_targets；nanochat/gpt.py:520 cross_entropy">监督 mask</abbr>（[释义](12-2-data-and-conversation-protocol.md#term-lossmask)）需在目标 shift 后对齐。<a id="ref-attnmask-001"></a><abbr title="Causal / Attention Mask｜因果与注意力掩码&#10;定义：约束query位置能读取哪些key位置，基本因果条件是key位置不大于query位置。&#10;作用：阻止未来目标泄露，并可进一步限定左侧可读窗口，保持自回归条件预测。&#10;本项目：注意力调用causal=True；SDPA用is_causal或显式允许布尔mask，缓存场景以Tk−Tq偏移对齐位置。&#10;注意：不是loss mask，不按BOS文档边界切断；窗口是过去位置数并含当前项，SFT监督mask不传进注意力。&#10;位置：nanochat/gpt.py:110 attention调用；nanochat/flash_attention.py:77 _sdpa_attention">注意力 mask</abbr>（[释义](12-2-data-and-conversation-protocol.md#term-attnmask)）控制可读上下文，监督 mask 控制损失；两者不同。<a id="ref-bos-001"></a><abbr title="BOS / Beginning of Sequence｜序列开始标记&#10;定义：表示文档或对话开始的专用ID，本项目字符串写作&lt;|bos|&gt;。&#10;作用：提供边界线索，使每个装箱行从一个明确文档/对话起点开始，也被复用作部分路径占位。&#10;本项目：预训练prepend BOS，SFT每段先加BOS；SFT/分类评估可用BOS补齐，Engine采到BOS时结束该行。&#10;注意：不自动重置注意力，也不禁止后文看前文；真实预训练BOS目标可计CE，BPB则因特殊字节长度0而排除。&#10;位置：nanochat/tokenizer.py:9 SPECIAL_TOKENS；nanochat/tokenizer.py:176 BOS加入；nanochat/engine.py:249 结束条件">BOS</abbr>（[释义](12-2-data-and-conversation-protocol.md#term-bos)）只是边界 ID，不会自动重置注意力。
