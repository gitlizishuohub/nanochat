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

[栏目目录](README.md) · [上一篇](04-model-construction-and-weight-loading.md) · [下一篇](06-forward-pass-and-prediction-loss.md) · [术语查阅](glossary/README.md)

> 原技术正文沿用原有源码核对记录；后续调整了结构。新增 SFT 串联说明做了局部静态核对，未执行训练。

# 5. 编码、目标对齐与批量组织

<a id="sec-6"></a>

<a id="reading-5-1"></a>
## 5.1 预训练：装满 T+1 然后 shift

自定义<a id="ref-loader-001"></a><abbr title="DataLoader / Generator｜数据装载器与生成器&#10;定义：把原始样本编码、装箱、构造目标并搬到设备的迭代接口，不要求使用某个同名框架类。&#10;作用：连接磁盘数据和模型循环，让每次前向获得形状/监督语义一致的输入。&#10;本项目：预训练为无限Python生成器并复用缓冲；SFT装载器定义在脚本内；RL get_batch还现场生成、判奖励。&#10;注意：未使用torch.utils.data.DataLoader、DistributedSampler或统一collate_fn；复用预训练缓冲时需长期持有旧batch应复制。&#10;位置：nanochat/dataloader.py:74 主生成器；scripts/chat_sft.py:180 SFT生成器；scripts/chat_rl.py:86 get_batch">数据生成器</abbr>（[释义](glossary/03-batching-and-distributed-training.md#term-loader)）读取 text 行组，在线分词并 prepend BOS。<a id="ref-packing-001"></a><abbr title="Packing / Best Fit｜拼接装箱与最佳适配&#10;定义：把变长文档/对话放进固定容量行，优先挑能完整装入剩余空间的最长样本。&#10;作用：减少空位浪费，复用固定形状前向，并使每行从完整文档/对话边界开始。&#10;本项目：预训练缓冲1000，装不下裁最短文档；SFT缓冲100，装不下补BOS/0mask，行容量T+1再shift。&#10;注意：没有按文档分块的attention mask；SFT装箱不裁剪不代表上游render_conversation没截断，后文仍可看前文。&#10;位置：nanochat/dataloader.py:74 主生成器；scripts/chat_sft.py:180 SFT生成器">best-fit packing</abbr>（[释义](glossary/02-data-and-protocol.md#term-packing)）从默认 1000 文档缓冲选择能装入剩余空间的最长文档；都装不下时裁最短文档前缀填满，弃尾。行容量 T+1，不进行序列 padding。

<a id="ref-labels-001"></a><abbr title="Targets / Labels / Shift｜目标标签与错位对齐&#10;定义：标签是每个输入位置希望预测的下一个ID，不是输入位置本身的ID。&#10;作用：把自回归文本预测变为逐位置分类，避免直接复制当前输入造成虚假的低损失。&#10;本项目：z长T+1，x=z[:-1]、y=z[1:]；SFT同时取mask[1:]设无效y为−1；GPT不二次shift。&#10;注意：[BOS,A,B,C]变为x=[BOS,A,B]和y=[A,B,C]；这里targets契约不同于某些框架自动错位labels，不能套用。&#10;位置：nanochat/dataloader.py:154 inputs/targets；scripts/chat_sft.py:281 tensors；nanochat/gpt.py:520 targets">输入与目标错位</abbr>（[释义](glossary/02-data-and-protocol.md#term-labels)）在装载器完成，GPT 不再 shift：

~~~text
示意 z=[BOS,A,B,C,D]，T=4：
x=[BOS,A,B,C]
y=[A,  B,C,D]
位置0预测A，不是BOS。
拼接可为 [BOS, 文档甲..., BOS, 文档乙...]
后一个BOS可能成为预测目标；未按文档隔离注意力。
~~~

CPU 行缓冲 [B,T+1]，inputs/targets 均 long [B,T]；设备缓冲复用，搬运 2×B×T 整数。源码 use_cuda=(device=="cuda")，而常见调用传 torch.device("cuda",rank)，类型比较未匹配，不能保证 pinned/non_blocking 优化在此路径生效。本次未测量注释“约35%数据被裁掉”的统计。

<a id="ref-truncate-001"></a><abbr title="Truncation / Cropping｜截断与裁切&#10;定义：长度预算不足时只保留序列前缀或指定片段，丢弃超出范围的数据。&#10;作用：控制上下文和显存，但可能丢失回答、停止标记或监督位置，需定位发生层次。&#10;本项目：tok_train每文档最多10000字符；预训练装箱裁尾；render_conversation默认前2048个ID和mask，SFT调用没传新上限。&#10;注意：字符限制不同于token限制；改变SFT行长不改变2048裁切，过短行可能装不进缓冲长对话，不能笼统说SFT不丢数据。&#10;位置：scripts/tok_train.py:28 text_iterator；nanochat/tokenizer.py:221 截断；scripts/chat_sft.py:208 调用">裁切</abbr>（[释义](glossary/02-data-and-protocol.md#term-truncate)）丢弃的是数据尾部。所有目标参与训练 CE，包括内部 BOS；BPB 才过滤特殊目标。后一个文档可看见前一个文档的过去 token，没有 block-diagonal attention mask。

<a id="reading-5-2"></a>
## 5.2 SFT：先截对话、再装箱补齐

任务混合、验证集来源与消费结束条件统一见[SFT 全流程](11-supervised-fine-tuning.md)。本节只解释编码后的对话如何装箱、补齐并与目标对齐。

render_conversation 默认裁到2048个 ID，SFT 调用**未传 max_tokens**。再装箱到 T+1；装不下的剩余空间使用<a id="ref-padding-001"></a><abbr title="Padding｜补齐&#10;定义：用占位值把不同长度序列补为同长矩阵，便于形成batch并执行统一形状计算。&#10;作用：固定训练/评估形状，用无效标签使占位目标不贡献误差。&#10;本项目：预训练填满T+1不补位；SFT用BOS和0mask；RL用assistant_end补本题最长序列、mask补0。&#10;注意：没有独立PAD ID；忽略目标损失不等于屏蔽输入注意力。右侧padding配合因果方向不影响更早真实位置。&#10;位置：scripts/chat_sft.py:250 补位；scripts/chat_rl.py:130 补位；scripts/chat_eval.py:109 padded_prompt_ids">BOS padding</abbr>（[释义](glossary/02-data-and-protocol.md#term-padding)）与0 mask。“装箱不裁剪”不等于数据链路未截断。修改 --max-seq-len 不同步改变上游2048限制，过短行可能装不进长对话，极端时全补齐行、targets全−1；脚本未显式过滤，mean CE 可能 NaN。

~~~text
假设每段正文仅一个token，US/UE/AS/AE仅示意：
z    =[BOS,US,问,UE,AS,答,AE,PAD]
mask =[ 0, 0, 0, 0, 0, 1, 1, 0]
x    =[BOS,US,问,UE,AS,答,AE]
y原始=[US, 问,UE,AS,答,AE,PAD]
y最终=[-1,-1,-1,-1,答,AE,-1]
实际PAD占位使用BOS，并没有独立PAD ID。
~~~

模型在 AS 位置预测“答”、在“答”位置预测 AE。<a id="ref-lossmask-002"></a><abbr title="Loss Mask / Ignore Index｜监督掩码与忽略标签&#10;定义：标记哪些目标位置有效的数组/哨兵，用来决定计算损失的范围。&#10;作用：只监督期望模型生成的行为，同时保留用户提示和外部结果作上下文。&#10;本项目：render_conversation给0/1，SFT shift后0位置targets=−1；cross_entropy(ignore_index=−1)，RL用Engine采样mask同样过滤。&#10;注意：不是attention mask；assistant_start不监督、SFT assistant_end监督。全−1均值CE可能NaN，入口没有样本级过滤保证。&#10;位置：nanochat/tokenizer.py:140 render_conversation；scripts/chat_sft.py:290 mask_targets；nanochat/gpt.py:520 cross_entropy">mask_targets</abbr>（[释义](glossary/02-data-and-protocol.md#term-lossmask)）=mask[:,1:]，不能取 mask[:,:-1]。SFT inputs 为 int32、targets 为 int64；补齐边界从 content_len−1 起另设−1。用户/工具结果仍参与上下文计算，仅它们的目标损失被忽略。

<a id="reading-5-3"></a>
## 5.3 batch、epoch 和每卡分配

<a id="ref-batch-001"></a><abbr title="Batch / Micro-batch / Effective Batch｜批次、微批与有效批次&#10;定义：微批是一次每卡前向/反向处理的序列集合，有效批次是一次优化更新累计处理的总量。&#10;作用：将显存容得下的小批次累计为目标规模，区分序列数、题数与token数。&#10;本项目：预训练/SFT总token容量=B×T×world_size×累积次数；device-batch-size是每卡行数，total-batch-size是全局token。&#10;注意：SFT容量含padding/提示，不等于监督token；RL examples-per-step为题数、num-samples为每题回答，不能混套单位。&#10;位置：scripts/base_train.py:410 累积计算；scripts/chat_sft.py:121 批量计算；scripts/chat_rl.py:217 每卡题数">有效 batch</abbr>（[释义](glossary/03-batching-and-distributed-training.md#term-batch)）的单位在预训练/SFT为全局词元容量：

$$
N_{\rm step}=B_{\rm device}TWA,\qquad
A=\frac{\text{total_batch_size}}{B_{\rm device}TW}\in\mathbb Z_{>0}.
$$

例如 B=16、T=2048、W=8、总量1048576，则 A=4；单卡同配置 A=32。减每卡行数降低激活显存，入口自动增累积次数，前提是整除。SFT容量含提示和padding，并非监督token数；RL用每步题数与每题样本数。

<a id="ref-distributed-002"></a><abbr title="Rank / World Size / torchrun / NCCL｜分布式进程与启动&#10;定义：rank为全局进程编号，world size为总进程数，local rank为节点内设备编号，torchrun设置这些环境变量。&#10;作用：分配样本与梯度计算到多卡，同时通过通信保持各卡模型数值一致。&#10;本项目：compute_init只在CUDA建NCCL组；预训练按row-group stride，SFT/RL按样本索引stride分配。&#10;注意：变量ddp表示分布式请求，不代表模型包了DistributedDataParallel；优化器同步梯度。CPU/MPS不可照搬多rank训练命令。&#10;位置：nanochat/common.py:151 get_dist_info；nanochat/common.py:174 compute_init；nanochat/optim.py:429 step">数据并行分配</abbr>（[释义](glossary/03-batching-and-distributed-training.md#term-distributed)）：预训练每文件内读取 rank、rank+W 的行组；SFT cursor 从rank起每次加W；RL循环rank-stride题目索引。没有 DataLoader num_workers、DistributedSampler 或 DDP模型包装，梯度同步在优化器中。

<a id="ref-epoch-001"></a><abbr title="Epoch / Step｜轮次与更新步&#10;定义：epoch通常指遍历某一数据集合，step通常是优化更新计数，但各入口实际计数实现需分别核实。&#10;作用：描述预算与进度，只有明确数据集合和计数单位后才能比较实验。&#10;本项目：预训练epoch从1起、无限循环文件，按num_iterations停；SFT默认按混合数据消费停；RL步数为题数整除后乘epochs。&#10;注意：SFT num_iterations按生成微批计数且一轮条件仍可先触发，预取使进度领先，不能据帮助承诺精确更新N步。&#10;位置：nanochat/dataloader.py:25 _document_batches；scripts/chat_sft.py:266 停止条件；scripts/chat_rl.py:82 num_steps">epoch/step</abbr>（[释义](glossary/03-batching-and-distributed-training.md#term-epoch)）：预训练无限循环数据、epoch从1开始，停止看优化更新数；SFT按 consumed 达混合数据规模停止；RL num_steps=(len(train)//examples_per_step)×num_epochs，整除余数不保证严格覆盖一轮。SFT每轮不重新洗牌，缓冲预取使epoch/消费进度与模型已用数据不同。

SFT 的停止计数与上下文配置边界统一见[SFT 停止条件](11-supervised-fine-tuning.md)。
