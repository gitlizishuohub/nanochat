---
source_files:
  - nanochat/gpt.py
  - nanochat/flash_attention.py
  - .agents/docs/career_roadmap/w1/scripts/d04_probe_model_and_cache.py
source_symbols:
  - GPTConfig
  - CausalSelfAttention.forward
  - GPT.forward
  - GPT.generate
doc_type: explanation
knowledge_scope: W1 D04 模型形状与生成路径交流核对
---

# W1 D04：模型形状与生成路径交流记录

- 交流日期：2026-10-26（Asia/Shanghai）。
- 教程：[D04 从 token ID 走到 logits](../../../.agents/docs/career_roadmap/w1/d04_model_shapes.md)。
- 本次操作：核对用户回答、当前源码和已有实验文件，补全“略”与“不知道”的部分；未重新运行模型实验。
- 当前源码核对 commit：`95473804336aef6e0bdaca25982c4877f472bd9a`。这不是已确认的实验运行版本；已有 JSON 未记录运行 commit、日期、精确命令或退出码。

## 1. 配置、完整形状链与 K/V 宽度

用户回答：`1；8；128；4；2；4；128；2`。其中 C 和 Dh 需要纠正，其余数字正确。

| 符号 | 含义 | 本次值 |
|---|---|---|
| B | 批量大小 | 1 |
| T | 本次输入的 token 数 | 8 |
| C | 每个 token 的隐藏向量宽度，即 n_embd | 64 |
| Hq | Query 头数 | 4 |
| Hkv | Key/Value 头数 | 2 |
| Dh | 每个头的维度，C/Hq | 16 |
| V | 实际词表大小 | 128 |
| L | Transformer 层数 | 2 |

<abbr title="token ID · 模型输入的离散单位在词表中的整数编号；查表后得到连续向量，不等同于字符编号。">token ID</abbr> 经 embedding 查表，从整数变成隐藏向量；最后通过 lm_head 得到每个词表候选的 <abbr title="logits · 模型给各候选 token 的未归一化分数，尚不是概率。本项目返回前还会执行 softcap。">logits</abbr>：

```text
整数 ID [1,8]
→ embedding → compute dtype → norm [1,8,64]
→ smear [1,8,64]
→ 两层 Transformer [1,8,64]
→ backout → 最终 norm [1,8,64]
→ lm_head → 裁回实际词表 → FP32 → softcap [1,8,128]
```

注意力内部，Q 线性输出的最后一维为 `Hq×Dh=4×16=64`；K/V 各为 `Hkv×Dh=2×16=32`。这是 <abbr title="GQA / Grouped-Query Attention · 分组查询注意力。多个 Query 头共享较少的 Key/Value 头；这里每两个 Q 头共享一个 K/V 头。">GQA</abbr>，因此 K/V 比 Q 窄。不要把隐藏宽度 C=64 与词表大小 V=128 混淆。

## 2. 从线性输出到多头布局

用户回答“不知道”，本节补答。

源码先调用线性层，再用 view 拆分最后一维：

```python
k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
```

因此 K/V 均从 `[1,8,32]` 变成 `[1,8,2,16]`，因为 `32=2×16`。view 不增加元素，也不重新计算 K/V，只改变张量的组织方式；这里的四个轴依次是批次、位置、头、头内通道。

<abbr title="forward hook · 注册在模块上的前向观察函数。这里记录指定模块输入与输出的尺寸；线性模块结束后调用，因此尚未观察到调用方随后执行的 view。">hook</abbr> 挂在 c_k/c_v 线性模块上，所以 JSON 直接记录的是 `[1,8,32]`。头数 2、每头维度 16 和拆头后的布局，需要根据配置与 CausalSelfAttention.forward 推导。Q 同理从 `[1,8,64]` 变成 `[1,8,4,16]`。

## 3. 归一化、位置处理与保持形状的操作

用户回答“略；不知道”，本节补答。

| 操作 | 链路位置 | 作用及形状 |
|---|---|---|
| RMSNorm | embedding 后、各 Block 的 attention/MLP 前、最终 lm_head 前 | 沿最后一维归一化，保持输入形状 |
| RoPE | attention 内 Q/K 投影并拆头后 | 旋转 Q/K，利用位置信息；Q `[1,8,4,16]`、K `[1,8,2,16]` 不变 |
| Q/K norm | RoPE 后、attention 计算前 | 分别对 Q/K 的每头通道归一化，随后各乘 1.2；形状不变 |
| smear | embedding 首次 norm 后、进入各层前 | 将前一位置的 embedding 按 gate 混入当前位置；保持 `[1,8,64]` |
| backout | 所有 Block 结束后、最终 norm 前 | 减去缩放后的中间层残差；保持 `[1,8,64]` |
| softcap | lm_head、裁回 V、转 FP32 后 | 执行 `15*tanh(logits/15)`，平滑限制分数幅度；保持 `[1,8,128]` |

<abbr title="RMSNorm / Root Mean Square Normalization · 按最后一维的均方根缩放输入。本项目 norm 没有可学习的归一化权重；不改变张量形状。">RMSNorm</abbr> 不会改变通道数。<abbr title="RoPE / Rotary Position Embedding · 旋转位置编码。对 Q/K 通道对进行随位置变化的旋转，使注意力利用相对位置信息；不是向 token embedding 加一个位置向量。">RoPE</abbr> 同样保持形状。

无缓存路径的 smear 分支显式断言 `T>1`，并通过 `x[:,1:]` 与 `x[:,:-1]` 混合相邻位置，因此当前实现不接受无缓存 `T=1`。这是本版本实现约束，不是所有 Transformer 的共同限制。缓存路径可以通过 prev_embedding 保存前一个 embedding，支持单 token decode。

## 4. 生成、采样与 targets 返回值

用户已答对“取最后位置”“cat 拼接”的主干；“softmax 后选最大分数”需要按 temperature 分支纠正。

GPT.generate 的实际流程：

1. 将完整前缀 ids 传给 forward，得到 `[1,T,V]`。
2. 取最后一个位置的整组词表分数 `logits[:,-1,:]`，得到 `[1,V]`。该位置用于预测下一个 token。
3. 若启用 top_k，将低于第 k 大分数的候选设为负无穷；边界并列分数可能被一起保留。
4. 若 `temperature>0`，先用 logits 除以 temperature，再 softmax 得到概率，通过 `torch.multinomial` 按概率抽取一个 token；并非固定选择最大概率。
5. 若 `temperature<=0`，直接 argmax，选择最大分数对应的 token。
6. 用 `torch.cat((ids,next_ids),dim=1)` 拼回前缀，再用 `yield token` 交出该 token 并暂停；调用者继续迭代时恢复执行。

<abbr title="temperature · 温度。正值时通过缩放 logits 改变采样分布；本项目小于等于零时走 argmax 贪心分支。">temperature</abbr> 为正时，<abbr title="softmax · 将一组分数转换为总和为 1 的概率。得到概率后，仍需由抽样或取最大值等规则决定输出类别。">softmax</abbr> 本身不负责选 token；multinomial 才执行概率抽样。<abbr title="argmax · 返回最大值的位置。这里该位置对应词表中的 token ID，属于贪心选择。">argmax</abbr> 则直接选择最高分候选。

下一轮输入从 T 变为 T+1，是因为 GPT.generate 没有使用 KV Cache，每次都重新处理拼接后的完整前缀。

传入 targets 时，GPT.forward 计算并返回 <abbr title="交叉熵损失 / Cross-Entropy Loss · 衡量模型预测与目标 token 的差异。这里展平位置后计算，忽略 target=-1 的位置；默认归约为平均损失标量。">交叉熵损失</abbr>，默认返回标量，不再返回 `[B,T,V]` logits；其他 loss_reduction 设置会影响损失形状。

## 5. 已有实验结果、设计原因与验证边界

用户回答“略”，本节依据已有文件补答。

- 结构化结果：[d04_shapes_probe.json](../results/w1/d04_shapes_probe.json)。
- 终端日志：[d04_checks.log](d04_checks.log)。
- 脚本：[d04_probe_model_and_cache.py](../../../.agents/docs/career_roadmap/w1/scripts/d04_probe_model_and_cache.py)。

| 项目 | 已有文件及源码支持的结论 |
|---|---|
| 模式 | shapes |
| logits shape | `[1,8,128]` |
| logits dtype | `torch.float32`，即 FP32 |
| PyTorch | `2.9.1+cpu` |
| seed | 42 |
| 注意力后端 | `USE_FA3=false`；结合 CPU 设置及后端源码，对应 SDPA 回退路径 |
| 有限值 | JSON 未单列 finite 字段；当前脚本在写文件前断言 `torch.isfinite(logits).all()`。已有结果与断言通过后写出的结构一致，本轮未重新计算 |

<abbr title="SDPA / Scaled Dot Product Attention · 缩放点积注意力。这里指项目在未使用 FA3 时调用的 PyTorch 注意力回退实现。">SDPA</abbr> 是此次记录对应的注意力路径。

以下是教程中的复现调用，不将其冒充为日志已记录的精确历史命令：

```bash
python .agents/docs/career_roadmap/w1/scripts/d04_probe_model_and_cache.py --mode shapes
```

已有证据没有完整记录运行日期、解释器绝对路径、精确命令、退出码和运行时 commit，因此这些信息仍待补充；本报告日期和源码核对 commit 不替代它们。

脚本给 attention 与 MLP 的输出投影填入非零随机权重，并将 smear_lambda 设为 0.15。默认初始化中的零输出投影可能掩盖注意力或缓存错误；非零 smear 则让相邻 embedding 混合逻辑实际参与计算。该设置仍是随机功能实验，不是训练好的模型。

已有结果支持该小配置的形状链路记录；不能证明语言质量、训练效果、真实 checkpoint 正确性或真实推理性能。shapes 模式的 steps 为空，也不能据此声称缓存数值等价已通过。

## 6. 交流结论与下一步

用户已掌握配置中的多数数字与生成主干；本次主要纠正 C/Dh、补充线性输出拆头，并区分概率抽样和取最大值。补答已记录，但助手给出答案不等于用户已经独立掌握。

建议先自行口述 `32=Hkv×Dh=2×16`，以及 temperature 为正与为零时分别如何选择 token，再进入 [D05 缓存生命周期](../../../.agents/docs/career_roadmap/w1/d05_cache_lifecycle.md)。实验命令、退出码和运行版本的证据缺项仍保留为待补充，不倒填。
