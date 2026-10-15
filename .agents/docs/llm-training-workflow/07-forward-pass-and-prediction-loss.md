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

[栏目目录](README.md) · [上一篇](06-data-encoding-and-batching.md) · [下一篇](08-backpropagation-and-parameter-updates.md)

> 拆分自原文；源码核对日期为 2026-10-12。本次仅整理文档结构，未重新核验源码或运行训练。

<a id="sec-7"></a>
# 7. 内部前向、模块作用、张量与预测损失

## 7.1 一次前向的形状线

默认入口 L=20、C=1280、Hq=Hkv=10、D=128、V=32768，<a id="ref-shape-001"></a><abbr title="Tensor Shape / B,T,C,H,D,V｜张量形状记号&#10;定义：B为本卡行数，T为输入长度，C为隐藏宽度，H为头数，D为单头宽度，V为词表大小。&#10;作用：用于追踪轴含义、检查投影乘法和目标对齐，避免把层数或全局batch误当轴尺寸。&#10;本项目：默认C1280/Hq=Hkv10/D128，Q为[B,T,Hq,D]，K/V为[B,T,Hkv,D]，logits[B,T,32768]。&#10;注意：SDPA暂转[B,H,T,D]，FA3原始布局不转；模型这里B不是total-batch-size，V在QKV语境还可指值激活。&#10;位置：scripts/base_train.py:129 build_model_meta；nanochat/gpt.py:84 CausalSelfAttention.forward">B/T/C/H/D/V</abbr>（[释义](12-4-model-construction-and-forward-pass.md#term-shape)）分别是每卡行数、输入长度、宽度、头数、单头维、词表大小。

| 顺序 | 实际计算与作用 | 形状 |
| --- | --- | --- |
| 1 | <a id="ref-embedding-001"></a><abbr title="Embedding / Unembedding / LM Head｜输入嵌入与输出词表投影&#10;定义：输入嵌入按整数ID查C维向量；输出投影把隐藏向量变为V个词元类别分数。&#10;作用：连接离散ID与连续特征，让模型可以利用上下文并分类下一词元。&#10;本项目：wte权重[Vpad,C]，查[B,T]得[B,T,C]；独立lm_head同尺寸把隐藏映到[B,T,Vpad]再裁V。&#10;注意：二者权重不绑定；虽然都二维却使用AdamW，不能仅以二维判为Muon。词表补列不等于序列padding。&#10;位置：nanochat/gpt.py:173 transformer；nanochat/gpt.py:177 lm_head；nanochat/gpt.py:419 setup_optimizer">wte查表</abbr>（[释义](12-4-model-construction-and-forward-pass.md#term-embedding)）、转compute dtype | [B,T]→[B,T,1280] |
| 2 | <a id="ref-rms-001"></a><abbr title="RMSNorm / QK Norm｜均方根归一化与查询键归一化&#10;定义：按向量末维均方根缩放，QK Norm是在每个query/key的单头维度做此操作。&#10;作用：控制激活尺度和打分幅度，与减均值的LayerNorm不同，帮助稳定表示。&#10;本项目：norm调用F.rms_norm，无learnable scale/bias，用于输入、块前、最终输出与旋转后Q/K，再各乘1.2。&#10;注意：没有gamma参数；不是残差可学习标量、注意力位置限制或梯度裁剪，不能因为名字含norm就混为同一操作。&#10;位置：nanochat/gpt.py:42 norm；nanochat/gpt.py:102 QK norm；nanochat/gpt.py:150 Block.forward">无参数RMSNorm</abbr>（[释义](12-4-model-construction-and-forward-pass.md#term-rms)）、<a id="ref-smear-001"></a><abbr title="Smear｜相邻嵌入混入&#10;定义：以小门控把前一个位置的归一化初始嵌入混到当前位置，加入近邻二元组信息。&#10;作用：在Transformer主干前提供廉价局部顺序线索，减少仅靠注意力提取近邻特征的负担。&#10;本项目：门控读当前嵌入前24通道并乘smear_lambda，初始lambda0；整序列切片，单步解码读prev_embedding。&#10;注意：不是batch间混合或数据增强，不在BOS自动分段；门控/标量可学习，缓存还须随生成样本复制与重置。&#10;位置：nanochat/gpt.py:186 参数；nanochat/gpt.py:477 smear；nanochat/engine.py:104 prev_embedding">Smear</abbr>（[释义](12-4-model-construction-and-forward-pass.md#term-smear)）混入前一个初始嵌入 | [B,T,1280] |
| 3 | 保存x0，每层<a id="ref-residual-001"></a><abbr title="Residual Stream / resid_lambdas / x0_lambdas｜残差流与层标量&#10;定义：残差流是贯穿网络的[B,T,C]向量，残差连接保留旧表示；层标量学习当前与初始输入的混合比例。&#10;作用：提供跨层信息/梯度通路，并让不同深度自行调节特征保留量。&#10;本项目：每层先resid_lambdas[i]×x+x0_lambdas[i]×x0；前者真实初始化1.15→1.05，后者0.20→0.05。&#10;注意：它们是可学习权重不是CLI配置，使用不同AdamW组；meta构造或旧注释1/0不是实际init_weights起点。&#10;位置：nanochat/gpt.py:183 层参数；nanochat/gpt.py:238 初始化；nanochat/gpt.py:500 层混合">残差标量混合</abbr>（[释义](12-4-model-construction-and-forward-pass.md#term-residual)） | [B,T,1280] |
| 4 | 注意力前norm，投影<a id="ref-qkv-001"></a><abbr title="Q/K/V / Query, Key, Value｜查询、键和值&#10;定义：Q是当前位置查询，K是各历史位置供匹配特征，V是被注意力权重聚合的内容向量。&#10;作用：通过QK相关性选择历史信息，再把V加权形成当前位置上下文表示。&#10;本项目：c_q/c_k/c_v投影[B,T,C]为多头布局，Q/K旋转归一化后各乘1.2；可选值嵌入只混入V。&#10;注意：Q/K/V是中间激活，投影矩阵才是权重；训练不使用KVCache，不能把缓存当一个额外可训练模块。&#10;位置：nanochat/gpt.py:67 CausalSelfAttention；nanochat/gpt.py:84 forward">Q/K/V</abbr>（[释义](12-4-model-construction-and-forward-pass.md#term-qkv)） | Q/K/V默认各[B,T,10,128] |
| 5 | 选定层的<a id="ref-ve-001"></a><abbr title="Value Embedding / Value Residual｜值嵌入与值残差&#10;定义：另查一个词元值表，通过输入相关门控加到注意力V，而不是替换查询或位置编码。&#10;作用：给注意力读取提供直接词元信息通道，是本项目实验性结构增强。&#10;本项目：has_ve选与末层同奇偶层、末层总包含；值表宽Hkv×D，门控读前12通道并乘3×sigmoid。&#10;注意：不是复制第一层V，也不是KV缓存；默认20层用奇数索引，值表用AdamW，块内ve_gate用Muon。&#10;位置：nanochat/gpt.py:53 has_ve；nanochat/gpt.py:93 value residual；nanochat/gpt.py:192 value_embeds">值嵌入</abbr>（[释义](12-4-model-construction-and-forward-pass.md#term-ve)）经门控加到V | 表查出[B,T,1280]再reshape |
| 6 | Q/K经<a id="ref-rope-001"></a><abbr title="RoPE / Rotary Position Embedding｜旋转位置编码&#10;定义：按位置相关角度旋转Q/K通道，使点积能反映相对顺序，不新增绝对位置查表。&#10;作用：提供词序信号，区分相同内容在不同位置的关系。&#10;本项目：base100000，cos/sin[1,10×sequence_len,1,D/2]；apply_rotary_emb按末轴两半配对，采用负角度旋转约定。&#10;注意：无可学习参数，persistent=False；10倍缓存是预计算余量，不证明10倍上下文质量或无限窗口，生成还受容量限制。&#10;位置：nanochat/gpt.py:57 apply_rotary_emb；nanochat/gpt.py:270 _precompute_rotary_embeddings">RoPE</abbr>（[释义](12-4-model-construction-and-forward-pass.md#term-rope)）、QK norm、各乘1.2 | cos/sin切片[1,T,1,64] |
| 7 | <a id="ref-flash-001"></a><abbr title="FA3 / FlashAttention；SDPA｜高效注意力内核与回退&#10;定义：FA3是融合高效注意力接口，SDPA是PyTorch缩放点积注意力，在FA3不适用时提供回退。&#10;作用：降低注意力开销，并适配不同设备或计算精度，保持相同轴/窗口语义。&#10;本项目：自动检测内核与dtype，bf16且有内核选FA3，否则转[B,H,T,D]用SDPA，支持因果/滑窗/GQA/缓存。&#10;注意：训练入口旧警告SDPA不支持滑窗已过时，现用显式mask实现；功能支持不代表同样效率，本次未测内核可用性。&#10;位置：nanochat/flash_attention.py:23 内核加载；nanochat/flash_attention.py:56 选择；nanochat/flash_attention.py:115 flash_attn_func">FA3/SDPA</abbr>（[释义](12-4-model-construction-and-forward-pass.md#term-flash)）因果滑窗聚合V | [B,T,10,128] |
| 8 | 合并头、c_proj、残差相加 | [B,T,1280] |
| 9 | norm→<a id="ref-mlp-001"></a><abbr title="MLP / FFN / ReLU²｜前馈网络与平方整流激活&#10;定义：逐位置两层线性变换之间插非线性，不直接跨时间轴混合信息。&#10;作用：把注意力形成的上下文向量变为更丰富的非线性特征，提高表达能力。&#10;本项目：C→4C→C，F.relu(x).square()；默认C1280中间宽5120，两个Linear无bias，c_proj初始零。&#10;注意：不是SwiGLU门控双分支、GELU或MoE；输出初始零不表示模块永远无梯度，它与残差组成正常训练计算图。&#10;位置：nanochat/gpt.py:131 MLP；nanochat/gpt.py:137 forward；nanochat/gpt.py:231 初始化">ReLU² MLP</abbr>（[释义](12-4-model-construction-and-forward-pass.md#term-mlp)）→残差相加 | 1280→5120→1280 |
| 10 | <a id="ref-backout-001"></a><abbr title="Backout｜中层残差扣除&#10;定义：最终输出前从深层残差中减去按可学习比例缩放的中层输出。&#10;作用：调节中低层特征在最终预测中的占比，是项目实验性结构而非通用必需模块。&#10;本项目：缓存零基循环索引L//2处x，最终x减backout_lambda×缓存，再norm；lambda初值0.2。&#10;注意：20层取索引10输出，不是恰好前10层；不是训练回滚、反向错误回退或删层，所有层仍前向执行。&#10;位置：nanochat/gpt.py:188 参数；nanochat/gpt.py:497 backout_layer；nanochat/gpt.py:507 扣除">Backout</abbr>（[释义](12-4-model-construction-and-forward-pass.md#term-backout)）扣中层特征，最终norm | [B,T,1280] |
| 11 | 独立lm_head、裁词表、fp32软限幅 | [B,T,32768] |

Block主干可理解为：

$$
x'=x+\operatorname{Attention}(\operatorname{RMSNorm}(x)),\quad
x''=x'+W_2\operatorname{ReLU}(W_1\operatorname{RMSNorm}(x'))^2.
$$

所有Linear无bias，RMSNorm无可学习gamma，输入输出不绑权重，没有dropout。没有门控SwiGLU、MoE或编码器cross-attention。模型外层还先用 resid_lambdas×x+x0_lambdas×x0，每个Value Embedding层把3×sigmoid门控后的值表加到V；Smear读前24通道，值门控读前12通道。Backout缓存零基索引L//2层的输出，最终减可学习比例，不是回滚训练。

## 7.2 注意力方向、窗口与RoPE

单头概念式（Q/K已旋转、归一化并各乘1.2）：

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt D}+M\right),\qquad Y=AV.
$$

M在不可访问位置等效负无穷，SDPA布尔mask代表允许的位置。概念分数矩阵为[B,Hq,T,T]，高效内核可避免显式存储全矩阵。Q/K各乘1.2使点积额外放大1.44倍。

<a id="ref-window-001"></a><abbr title="Sliding Window / SSSL｜滑动窗口与层模式&#10;定义：限定每个query能读取的过去位置数量，层模式决定哪些层用短窗或长窗。&#10;作用：减少部分层长序列注意力成本，同时保留长层对更远上下文的信息路径。&#10;本项目：SSSL循环、末层强制L；L左窗sequence_len，S由−(−sequence_len//4//128)×128算，2048时512。&#10;注意：旧注释half/768不符合代码；左窗512另含当前，最多513个key，不按文档隔离，也不等于缓存仅保留这长度。&#10;位置：nanochat/gpt.py:287 _compute_window_sizes；nanochat/flash_attention.py:77 _sdpa_attention">SSSL窗口</abbr>（[释义](12-4-model-construction-and-forward-pass.md#term-window)）：S源码表达式−(−sequence_len//4//128)×128，2048时是512；L左窗口2048，最后一层强制L。左窗数是可读过去位置，另含当前位置，最多513个key。注释“half”或“2048→768”不符合实际式子。SDPA现已实现滑窗mask，base_train警告“SDPA不支持滑窗”是旧文字；语义支持不代表与FA3一样快。本次未测内核等价或性能。

RoPE base=100000，完整cos/sin缓存[1,10×sequence_len,1,D/2]，使用末轴前后两半配对的负角度约定，不新增绝对位置表。10倍预计算余量不证明10倍上下文质量，窗口仍受配置限制。

## 7.3 logits、目标对齐与有效监督

<a id="ref-logits-001"></a><abbr title="Logits / Softmax / Softcap｜预测分数、概率归一化与软限幅&#10;定义：logits是未归一化类别分数，softmax将其转为概率，softcap光滑限制数值幅度。&#10;作用：为训练分类损失与生成采样提供共同输出，并避免过大的预测分数。&#10;本项目：lm_head产生[B,T,Vpad]、裁V、转fp32，15×tanh(z/15)；CE内部归一化，生成另按温度/top-k softmax。&#10;注意：logits不是概率或文本，softcap不是硬clip/梯度裁剪；无targets时返回分数，生成函数负责选ID。&#10;位置：nanochat/gpt.py:511 logits；nanochat/engine.py:141 sample_next_token">logits</abbr>（[释义](12-4-model-construction-and-forward-pass.md#term-logits)）转fp32，按15×tanh(z/15)软限幅，再供损失和采样：

$$
\tilde z=15\tanh(z/15),\qquad
p(v\mid x_{\le t})=\frac{e^{\tilde z_v}}{\sum_u e^{\tilde z_u}}.
$$

<a id="ref-ce-001"></a><abbr title="Cross-Entropy / NLL｜交叉熵与负对数似然&#10;定义：对正确目标ID概率取负对数，越不相信真实目标则误差越大，多位置通常再求平均。&#10;作用：将可读文本预测转为可微目标，为反向传播提供数值误差信号。&#10;本项目：GPT对[B×T,V]分数和[B×T]targets调用cross_entropy(ignore_index=−1)；默认mean，none返回展平逐位置值。&#10;注意：不二次shift；训练可含有效特殊目标，BPB再排除它们。所有target忽略时mean无有效分母，可能NaN。&#10;位置：nanochat/gpt.py:520 损失；nanochat/loss_eval.py:33 逐位置；scripts/chat_rl.py:264 logp">交叉熵</abbr>（[释义](12-5-loss-and-parameter-updates.md#term-ce)）对有效目标S求平均：

$$
\mathcal L=-\frac1{|S|}\sum_{(b,t)\in S}\log p(y_{b,t}\mid x_{b,\le t}),
\quad S=\{(b,t):y_{b,t}\ne-1\}.
$$

预训练S为全部B×T，SFT为助手有效目标，RL先取逐位置NLL再奖励加权。数据已shift，GPT不二次shift。loss_reduction='none'返回展平B×T向量，RL再view_as(inputs)，不能按注释假设直接[B,T]。

<a id="ref-teacher-001"></a><abbr title="Teacher Forcing｜真实前缀条件训练&#10;定义：在每个训练预测位置使用数据中真实前序token，而不是用模型自己上一时刻采样结果作为前缀。&#10;作用：让所有位置可并行计算并对齐已知标签，形成稳定的监督下一词元目标。&#10;本项目：装载器将已知整行错位为x/y，因果注意力确保位置t仅看真实x≤t；SFT助手示范直接放在输入前缀里。&#10;注意：这里teacher指真实序列，不表示教师模型或知识蒸馏；推理则只能利用已生成词元，不能预先放入标准答案。&#10;位置：nanochat/dataloader.py:154 shift；scripts/chat_sft.py:281 inputs/targets；nanochat/gpt.py:459 forward">Teacher forcing</abbr>（[释义](12-5-loss-and-parameter-updates.md#term-teacher)）指训练时使用真实前序token；“teacher”不是另一个模型，不能把它误认为蒸馏。无targets时只返回logits，生成器才挑ID与解码文本。
