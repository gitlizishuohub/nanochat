---
source_files:
  - nanochat/engine.py
  - nanochat/gpt.py
  - nanochat/flash_attention.py
  - tests/test_engine.py
source_symbols:
  - KVCache
  - KVCache.prefill
  - Engine.generate
  - GPT.forward
  - CausalSelfAttention.forward
doc_type: tutorial
knowledge_scope: W1 D5 预分配缓存生命周期与状态探查
---

# W1 D5 · 2026-10-05：跟踪 prefill、批量复制和逐步 decode

今天不重写缓存。你要理解已有缓存的生命周期，并实际检查位置推进与 smear 状态。预计 90 分钟：20 分钟容器，25 分钟生成器，20 分钟探查，15 分钟报告，10 分钟验收。没有 GPU 和 checkpoint 都可用 CPU 随机模型完成。

前置：[D4 模型与形状](04_2026-10-04_model_shapes.md)。D4 的无缓存路径是对照；今天必须使用相同权重与相同前缀比较，不能比较两个独立随机模型。

## 1. 先读缓存容器，不先读采样循环

```bash
cd /home/lizishuo/_program_/Project/nanochat
source /home/lizishuo/miniconda3/etc/profile.d/conda.sh
conda activate nanochat
sed -n '80,155p' nanochat/engine.py
rg -n 'kv_cache.advance|prev_embedding|T0 =|flash_attn_with_kvcache' nanochat/gpt.py
```

<abbr title="KV Cache / Key-Value Cache · 每层保存已处理位置的注意力 Key/Value。nanochat 预先分配容量，用位置向量区分有效前缀和未使用区域；分配容量不等于有效长度。">KV Cache</abbr> 的两块本体形状均为 `[L,B,Tmax,Hkv,Dh]`。逐项确认：

- `get_layer_cache(i)` 返回去掉层轴的视图 `[B,Tmax,Hkv,Dh]`，不是复制一份新缓存。
- `cache_seqlens` 为每行一个 int32 位置；当前 `get_pos()` 只读第一行，因此一般使用假设是批内位置一致。
- `advance(T)` 只更新位置，不负责写 K/V。
- K/V 写入发生在 attention 后端；位置只在最后一个模型层处理后推进一次，不能每层都加 T。
- `reset()` 清空位置、把 `prev_embedding` 设为 None，**没有把 K/V 存储全部置零**。旧数值仍可能存在，但正确实现不应读有效长度外的区域。

<abbr title="prev_embedding · 缓存的前一 token 归一化 embedding，形状通常为 [B,1,C]。它供下一步 smear 使用，不是 Key 或 Value 的别名。仅拷贝 K/V 而漏掉它，会破坏当前模型的增量推理状态。">prev_embedding</abbr> 是今天最容易漏掉的状态。报告中必须单独画出来。

## 2. 对照 Engine.generate 的两次分配

```bash
sed -n '195,310p' nanochat/engine.py
sed -n '470,535p' nanochat/gpt.py
```

在纸上写一条 `P=8,B=4,G=4` 的时间线：

| 阶段 | 输入与缓存 | 位置与状态 |
|---|---|---|
| 起始 | token ID 列表长度 P | 尚无缓存 |
| prefill 分配 | `[L,1,P,Hkv,Dh]` | 位置 0，prev 为 None |
| prefill forward | `[1,P]` | 所有层写入；位置变 P；prev 为 `[1,1,C]` |
| decode 分配 | `[L,B,P+G,Hkv,Dh]` | 新缓存位置 0 |
| `prefill(other)` | 广播复制有效 K/V 前缀 | 每行位置 P；prev expand 后 clone |
| 第一次采样/yield | prefill 最后 logits 扩到 B 行 | 第一 token 的分数来自 prefill |
| 下次恢复生成器 | `[B,1]` forward | 写入位置 P；随后变 P+1 |

<abbr title="prefill · 把完整 prompt 一次送入模型，建立各层历史缓存；它提供第一生成 token 的分数。nanochat 先做 batch=1 prefill，再复制给多行样本。">prefill</abbr> 与 <abbr title="decode · 利用已经建立的历史状态，每次只输入一个新 token 并得到下一个位置的分数。与每步重算整个前缀的无缓存路径相比，模型输入长度通常为 1。">decode</abbr> 是输入长度不同的两段计算。

`yield` 会暂停函数：拿到第一列 token 时，下方下一次 forward 尚未发生。再次 `next()` 才恢复。当前循环在 `yield` 后先做下一次 forward，下一圈顶部才检查 token 上限；因此取完指定列后再尝试 `next()` 以发现结束，可能还执行一次未产出 token 的 forward。D7 审计必须讨论这个边界。

## 3. 用随机模型检查状态和数值

使用与 D4 同一份[模型与缓存探查脚本](scripts/probe_model_and_cache.py)：

```bash
python .agents/docs/career_roadmap/w1/scripts/probe_model_and_cache.py --mode cache --prompt-tokens 8 --decode-steps 4
```

默认写 `experiments/inference_infra/results/w1/cache_probe.json`。脚本直接调用 `GPT.forward(...,kv_cache=...)`，这样把缓存数值验证与 Engine 的结束 token/工具状态机分开。它做：

1. 以 P=8 prefill；比较缓存与无缓存最后位置 logits。
2. 根据同一 greedy token 共同扩展前缀，再比较 3 步单 token 输入。
3. 每步断言缓存位置等于已喂入的总前缀长度，并保存 before/after、误差和 greedy token。
4. 把已建立缓存复制到 batch=4；检查每行各层有效 K/V 与 `prev_embedding` 相同。
5. reset 后检查位置为 0、prev 为 None。

<abbr title="greedy decoding · 每步选取分数最大的 token。避免随机采样差异干扰缓存数值对照，但不能取代所有数值容差检查。">greedy</abbr> 相同与 `assert_close` 都必须通过。当前脚本容差为 `atol=1e-5,rtol=1e-4`，只用于这份 CPU/FP32 smoke；不能未经验证套用到 bf16 或 GPU。

结构预期如下，实际误差以运行 JSON 为准：

| step | 当前输入长度 | cache_before | cache_after |
|---:|---:|---:|---:|
| 0 | 8 | 0 | 8 |
| 1 | 1 | 8 | 9 |
| 2 | 1 | 9 | 10 |
| 3 | 1 | 10 | 11 |

这里“四步”包含 prefill 这一步，不等于四次 decode。复制后 K/V shape 为 `[2,4,64,2,16]`，单份 K 或 V 为 65,536 字节，两者合计 131,072 字节，不包含模型、位置向量、prev 和临时张量。

编写时验证记录（2026-10-15）：Conda `nanochat` / torch 2.9.1+cpu 实测 4 步位置为 8、9、10、11；最大绝对 logits 差 `7.450580596923828e-09`，greedy token 一致；B=4 的 K/V 与 prev 复制、reset 均通过。这里只验证上述小配置，不外推 GPU、真实 checkpoint 或所有边界。

## 4. 写缓存报告与内存公式

新建 `experiments/inference_infra/reports/cache_path.md`，按“容器 → prefill → 复制 → decode → reset”顺序写。至少保存：

```markdown
# 缓存路径
- 实际日期 / commit / Python / dtype：
- 源码入口：KVCache、Engine.generate、GPT.forward
- K/V 本体 shape：
- cache_seqlens 类型与批内位置假设：
- prev_embedding 来源、更新和复制规则：
- 每层写入位置 / 最后一层推进规则：
- 随机模型探查 JSON：
- 实际最大误差与 greedy 是否一致：
- 复制与 reset 检查：
- 未覆盖：真实 checkpoint、GPU后端、长上下文边界等
```

预分配 K/V 理论字节数：`2 * L * B * Tmax * Hkv * Dh * 每元素字节数`。有效 token 数 P 不应替代容量 Tmax 估算分配量。不要声称 KV Cache 降低总显存；它用额外状态换取后续少算历史。

## 5. 失败分流与当天验收

- 依赖失败：回 D2；完成源码表也只能标静态阅读，运行仍 BLOCKED。
- 位置错误：先检查推进是不是只在最后一层，以及是否误把采样次数当成已喂入 token 数。
- step 0 对，step 1 起错：优先核对 RoPE 的偏移和 `prev_embedding`；不要只放宽容差。
- 单行对，复制后错：检查批量广播、有效前缀和 prev 的复制。
- 误差为零：可以是真结果，但还要看是否用了非零投影和 smear；脚本已避免默认零投影让测试失去区分力。

验收：生命周期表完整；掌握容量/有效长度差别；能解释 prev；保存真实 JSON 或阻塞原因；不把这份小探查当作 W9 全覆盖回归。追加真实日期、commit、用时、命令、结果路径、状态、下一步。

上一篇：[D4 模型形状](04_2026-10-04_model_shapes.md)。下一篇：[D6 后端路由](06_2026-10-06_attention_backends.md)。
