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

# W1 D5：用同一段前缀看懂缓存怎样替代重复计算

今天解决一个问题：**模型每次只输入一个新 token，为什么仍能根据整个前缀预测下一个 token？** 答案不只是“有 KV Cache”，还涉及缓存了什么、位置怎样对齐，以及本项目额外保存的 embedding 状态。

前置：[D4 模型与形状](d04_model_shapes.md)。第 1–3 节是纸面推演，无需运行模型。第 4 节才在 CPU 上初始化一个随机小模型，执行真实前向与数值比较；不下载权重、不训练、不评价文字质量。GPU 性能和已有 checkpoint 都不是本日学习条件。可按“概念与例子 30 分钟、源码带读 25 分钟、CPU 探查与交流 25 分钟”安排。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[12.1 无缓存生成与缓存生成](../../llm-training-workflow/12-inference-and-generation.md#reading-12-1)；[12.2 生成控制流与返回格式](../../llm-training-workflow/12-inference-and-generation.md#reading-12-2)。
- **今天重点**：对照今天的缓存探查，跟踪提示预填充、缓存复制、新词元解码和位置推进；区分缓存状态与可学习权重。
- **暂时跳过**：温度、top-k 和工具状态机先认识用途；今天优先看缓存和返回格式。

**读完能解释**：处理完整提示和随后每次新增一个词元时，输入形状及缓存状态分别怎样变化？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 1. 缓存究竟替代了哪部分计算

设提示包含 8 个 token，位置编号为 0–7。先不考虑提前结束与工具调用，假设之后依次生成 a、b、c；这只是用于追踪位置的符号，不是要求你运行生成。

无缓存时，预测 a 需要处理位置 0–7；预测 b 时重新输入“提示+a”，处理位置 0–8；预测 c 时再处理位置 0–9。因果注意力不允许旧位置看未来，因此固定模型与输入后，新增 a 不会改变旧位置各层已经算出的 K/V。可以保存这些历史结果，下次只计算新位置。

<abbr title="KV Cache / Key-Value Cache · 为每一层保存已处理位置的注意力 Key 和 Value。它是本次请求的运行状态，与可学习的模型参数不同。">KV Cache</abbr> 中，K 用于和新 query 计算匹配分数，V 是按注意力权重汇总的内容。每一步的新 Q 仍须计算；新位置仍会执行各层注意力与 MLP，且通常仍需读取可见历史 K/V。缓存减少历史位置的重复计算，不意味着下一步只做一次查表，也不意味着整次请求的内存更少。

| 状态 | 本项目放在哪里 | 下次有什么用 |
|---|---|---|
| 各层已处理位置的 K/V | `KVCache.k_cache`、`v_cache` | 参与新位置的注意力计算 |
| 每行已经处理的长度 | `cache_seqlens` | 决定新 K/V 的写入位置和有效范围 |
| 前一位置的归一化词嵌入 | `prev_embedding` | 衔接本项目的 smear 计算，见第 3 节 |
| token ID 历史 | Engine 的 `RowState.current_tokens` | 记录输出与工具状态；不在 K/V 张量中 |
| 模型权重 | `GPT` 的参数 | 各请求共享；缓存更新不会训练这些参数 |

缓存不保存全部旧 Q、全部 logits 或整个注意力分数矩阵。新请求必须建立自己的有效历史，不能直接把上一个请求的有效长度沿用过去。

### 1.1 先分清“分配了多少”与“已经写入多少”

沿用小模型 `L=2、Hq=4、Hkv=2、C=64`，每个头的维度 `Dh=C/Hq=16`。K 和 V 各有一个 `[L,B,Tmax,Hkv,Dh]` 张量：层、样本行、容量、K/V 头、头内维度。这里按 Hkv 分配，不能用 Q 的头数 Hq 代替。

假设容量 Tmax=12，位置长度为 8，则 `[0:8]` 是有效历史，`[8:12]` 只是预留空间。Python 切片右端不包含在内，`[0:8]` 对应位置 0–7。这解释了为什么“cache shape 没变”和“已处理长度增加”可以同时成立。

## 2. 用 P=8、B=4、G=3 推演 Engine 的两次分配

P 是提示 token 数，B 是同一提示的采样行数，G 是最多产出的 token 列数。B=4 不是四个不同用户的请求；当前 Engine 先只处理一次共同提示，再复制出四条生成分支。

<abbr title="prefill · 一次输入完整提示，建立历史 K/V，并从最后位置的 logits 得到第一个生成 token 的分数。">prefill</abbr> 和 <abbr title="decode · 复用历史状态，输入新 token，计算下一 token 的分数。这里使用每步一个 token 的增量路径。">decode</abbr> 的区别首先是输入与历史状态不同。跟着 [Engine.generate](../../../../nanochat/engine.py) 的这段摘录走；省略了分配参数与采样设置：

```python
logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
logits = logits[:, -1, :].expand(num_samples, -1)
# ... 创建更大的 kv_cache_decode ...
kv_cache_decode.prefill(kv_cache_prefill)
```

第一行的 `ids` 为 `[1,8]`。前向处理提示的八个位置，输出 `[1,8,Vocab]`；第二行只取最后位置的预测分数，再扩成 `[4,Vocab]`。四行此时共享同一分布，随机采样后可以得到不同 token；扩展 logits 不等于把一个已经采好的 token 复制四次。

| 时刻 | 本次模型输入 | K/V 本体形状（K、V 各一份） | 有效位置与下一次写入点 |
|---|---|---|---|
| 创建提示缓存 | 尚无 | `[2,1,8,2,16]` | 长度 0 |
| 提示前向结束 | `[1,8]` | `[2,1,8,2,16]` | 长度 8；有效 0–7 |
| 创建生成缓存并复制 | 无新增前向 | `[2,4,11,2,16]` | 四行长度均为 8；复制有效 0–7 |
| 产出第一列 a | 只采样并 `yield` | 同上 | 仍为 8；a 尚未送入模型 |
| 恢复后处理 a，产出 b | `[4,1]` | 同上 | a 写在位置 8，长度变 9 |
| 恢复后处理 b，产出 c | `[4,1]` | 同上 | b 写在位置 9，长度变 10 |

表中 a/b/c 表示各行对应的输出，四行具体 ID 可以不同。生成缓存容量是 `P+G=11`，产出 c 时只有 10 个位置已处理。**“已经返回给调用者”与“已经输入模型并写入缓存”是两件事。**

`KVCache.prefill(other)` 名字相同，但它只做状态复制；真正的提示计算在上面的 `model.forward`。复制时 `[L,1,P,Hkv,Dh]` 会广播到 B 行，并复制长度与 `prev_embedding`。目标缓存须为空、容量须至少容纳源缓存容量，相关维度也必须匹配。`get_layer_cache(i)` 返回本层的视图 `[B,Tmax,Hkv,Dh]`，后端向视图写入就是更新原缓存。

生成器在 `yield` 暂停；再次 `next()` 才继续。当前实现恢复后先处理刚产出的 token，随后在下一圈顶部检查上限。因此若调用者继续迭代直到自然结束，产出 c 后还会处理 c、把长度推进到 11，却不再产出第四列；若调用者取到三列就停止恢复，则不会发生这次前向。D7 会用这一边界解释计时差异，今天只需把它与缓存位置对应起来。

## 3. 沿着一个新位置，带读源码里的三种状态更新

### 3.1 RoPE 读位置，attention 写 K/V，最后一层推进长度

现在缓存长度为 8，输入 a 的形状为 `[4,1]`。[GPT.forward](../../../../nanochat/gpt.py) 先读取位置，再切取 <abbr title="RoPE / Rotary Position Embedding · 旋转位置编码。模型按绝对位置选取旋转参数，作用到 Q/K，使匹配分数包含相对位置信息。缓存解码必须接着历史位置编号。">RoPE</abbr> 参数：

```python
T0 = 0 if kv_cache is None else kv_cache.get_pos()
cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
```

此时 T0=8、T=1，所以使用 `[8:9]`，不能再使用位置 0。随后每一层根据本层输入计算 Q/K/V；K 已经过旋转、归一化和缩放，V 也可能混入 value embedding，再交给后端缓存。缓存保存的是这些已加工的本层 K/V，不是原始 token embedding。

[SDPA 缓存分支](../../../../nanochat/flash_attention.py) 将新 K/V 写到 `[:,8:9]`，注意力读取有效范围 `[:,0:9]`。它只负责读写，不推进全局长度。[CausalSelfAttention.forward](../../../../nanochat/gpt.py) 最后才执行：

```python
if self.layer_idx == kv_cache.n_layers - 1:
    kv_cache.advance(T)
```

两层都必须处理同一个位置 8。如果第一层提前推进，第二层便可能把同一个 a 当成位置 9，形成跨层错位。因此一次模型前向只推进一次，结束后四行长度均为 9。当前 `get_pos()` 和 SDPA 路径都读第一行长度，前提是批内位置一致，不能据此声称已支持各行不同长度的任意请求批处理。

### 3.2 为什么除了 K/V 还需要 prev_embedding

本项目有 <abbr title="smear · 将前一 token 的归一化词嵌入按门控系数混入当前位置，让当前输入带有邻近 token 信息。其参数属于模型，prev_embedding 是请求状态。">smear</abbr>：记归一化词嵌入为 e，位置 i 的输入会混入 `gate(e_i) * e_(i-1)`。完整前缀一次输入时，相邻两份 embedding 都在张量里；单 token 解码时，上一位置已不在当前输入里，需要额外保存。

在 `GPT.forward` 中，先执行下面两行，然后才对 x 做 smear：

```python
x_pre_smear = kv_cache.prev_embedding
kv_cache.prev_embedding = x[:, -1:, :]
```

提示前向结束后，`prev_embedding` 保存位置 7 的**归一化、尚未 smear 的词嵌入**，形状 `[1,1,64]`；复制后为 `[4,1,64]`。处理位置 8 的 a 时，先取旧状态 e7，再保存 e8，随后用 e7 补足跨调用的邻接关系。它不是最后一层隐状态，也不是 K/V 的别名。

如果只复制 K/V 而漏掉这份状态，位置 8 的 smear 会缺少 e7，后续 logits 就可能偏离完整前缀计算。若 `smear_lambda=0`，错误可能暂时不影响输出；探查脚本将它设为非零，避免这种错误被默认值掩盖。

### 3.3 reset 的语义是“历史无效”，不是“内存清零”

`KVCache.reset()` 只把长度归零、`prev_embedding` 设为 None，旧 K/V 数值仍在原内存中。下一个请求必须从位置 0 重建有效前缀，读历史时不能越过有效范围。reset 能复用分配过的空间，但“reset 字段正确”本身还不证明第二个请求的数值正确，后者需要专门的复用前后数值测试。

## 4. CPU 实操：用完整前缀作为数值对照

这里实际执行模型前向；使用随机初始化的小模型，开启推理模式，没有反向传播或优化器更新。脚本主动隐藏 CUDA、指定 FP32，不调用 Engine 的 tokenizer 或下载 checkpoint。运行前先预测：四次前向的输入长度应为 8、1、1、1，缓存长度应依次到 8、9、10、11。

复用 [D4 模型与缓存探查脚本](scripts/d04_probe_model_and_cache.py)。终端沿用已激活的 `nanochat` 环境。花括号内的输出局部追加到日志，不接管终端提示符；旧版全局日志重定向仍在运行时需先新开终端。退出码在命令后立即保存：

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
python -c 'import sys; print(sys.executable)'
python .agents/docs/career_roadmap/w1/scripts/d04_probe_model_and_cache.py --mode cache --prompt-tokens 8 --decode-steps 4
d05_status=$?
printf 'cache_probe_exit=%s\n' "$d05_status"
(exit "$d05_status")
} > >(tee -a experiments/inference_infra/reports/d05_checks.log) 2>&1
```

脚本的缓存与对照路径共享**同一份模型权重和相同前缀**：缓存路径只输入新增 token，对照路径每次输入完整前缀；两者都取最后位置的 <abbr title="logits · 模型为词表每个 token 输出的预测分数。比较整组 logits 比只比较分数最大的位置更严格；此处仍只检验每次前向的最后位置。">logits</abbr> 比较。第 0 步输入提示，随后从缓存结果取分数最大的 ID，同时加到两条路径的前缀上。若使用两个独立随机模型或让两条路径自行随机采样不同 ID，就不再是缓存等价性对照。

| JSON 字段/输出 | 默认参数下应怎样理解 |
|---|---|
| `steps[0]` | `input_tokens=8`、`cache_before=0`、`cache_after=8`；这是 prefill |
| `steps[1:4]` | 输入长度均为 1；位置为 8→9、9→10、10→11；共三次 decode |
| `cache_seqlens` | 当前只有一行；应与 `cache_after` 相同 |
| `prev_embedding_shape` | `[1,1,64]`，用于下一次跨调用的 smear |
| `max_abs_logit_diff` | 与完整前缀路径的最大绝对分数差；还须结合 assert_close 的相对容差判断 |
| `replication_position` | 11；本脚本在四步结束后才复制，不是 Engine 在提示结束的 8 处复制 |
| `replication_batch`、K/V shape | 4、`[2,4,64,2,16]`；脚本固定容量 64，不使用上例的 Engine 容量 11 |
| `reset_verified` | 长度归零且 prev 为 None；不表示 K/V 已清零或复用请求已完成数值验证 |

脚本用 `torch.testing.assert_close(atol=1e-5,rtol=1e-4)` 检查末位置全部 logits，并另查 <abbr title="greedy · 每步选取 logits 最大的 token。两个路径选出同一 ID 只说明最大分数位置一致，不能证明所有分数或缓存状态都正确。">greedy</abbr> ID 一致。同一个最大 ID 可以对应差别很大的分数向量，所以只展示 token 一致不足以证明缓存正确。脚本还逐行比较复制后的有效 K/V 和 prev；这些是补充的状态证据，并未逐位置验证缓存与独立参考 K/V 完全相同。

成功时退出 0，并在全部断言后写 JSON。若有断言失败，不会写本次成功结果；旧同名 JSON 可能还在，须根据日志与时间辨认。step 0 起错先检查输入/权重/精度；step 0 对、step 1 起错优先看位置与 prev；只在复制后错再查广播和状态复制。不要靠放宽容差掩盖最早分歧。

### 4.1 从 shape 算存储，而不是从有效长度猜内存

脚本目标缓存的两份 K/V 共占 `2×L×B×Tmax×Hkv×Dh×元素字节数`。代入 FP32 每元素 4 字节：`2×2×4×64×2×16×4 = 131072 字节 = 128 KiB`。对应 JSON 的 `storage_bytes_kv`，不包括模型权重、prev、位置向量或临时张量。

把有效长度从 11 reset 到 0 不改变这个预分配量。把容量从 64 改成 32 才会使相同配置的 K/V 分配量减半；把 Q 头数代进去则会高估当前 GQA 缓存。CPU 这里讨论的是内存，不应称为实测显存或 GPU 性能结论。

## 5. 当天自验证与交流问题

可直接在聊天按题号回答，概念题用推演，实操题引用本次 JSON 和日志即可。命令、解释器、退出码和证据路径只附一次；人工报告可选。不选择 GPU 扩展不形成欠账；CPU 未运行时如实说“尚未执行”，不能引用旧结果冒充本次结果。

1. **改变生成列数**：上面的 Engine 例子改为 G=4。提示结束时、产出第四列时、继续迭代到自然结束时，缓存容量与有效长度分别怎样？为什么输出列数不等于已输入的新 token 数？

   **核对要点**：从 yield/恢复顺序推导，不把产出的 token 立即视为缓存的一部分。

2. **跟踪位置 8**：输入 a 时，RoPE 读哪个切片、每层 K/V 写哪个切片、注意力读到哪里、什么时候 advance？若每层各推进一次，第二层会错在哪里？

   **核对要点**：给出半开区间并把读位置、写缓存、推进长度串成顺序。

3. **丢失额外状态**：K/V 与长度复制正确，prev 却为 None，为什么仍可能使增量路径出错？smear 系数为零时测试通过又说明什么？

   **核对要点**：说明 prev 的表示来源与读取/保存顺序，区分未触发错误和状态已正确。

4. **解释本次证据**：选择 JSON 中一个 decode 步，说明两条路径输入的内容和长度为何不同但历史相同；用位置、logits 容差与 token 检查解释结果。再指出脚本的复制发生在何时，与 Engine 有什么差别。

   **核对要点**：引用实际字段；仅 greedy 相同不够；脚本四步包含一次 prefill。

5. **复用与容量**：reset 后字节数为什么不变？若要进一步声称“第二个请求没有读到旧请求的状态”，当前 reset 断言还缺什么验证？

   **核对要点**：按分配量公式计算，提出与新缓存/完整前缀的数值对照，不要求当天再编写测试。

## 6. 脚本、产物与回查入口

以下均以项目根目录为基准；清单描述生成条件，不表示已经执行或生成。

| 脚本及阶段 | 完整输入路径/来源 | 条件与作用 |
|---|---|---|
| [`.agents/docs/career_roadmap/w1/scripts/d04_probe_model_and_cache.py`](scripts/d04_probe_model_and_cache.py)，第 4 节 `--mode cache` | `nanochat/gpt.py`、`nanochat/engine.py`、`nanochat/flash_attention.py`；脚本内部随机前缀和同一随机模型 | CPU/dev 依赖齐备；不需 tokenizer、checkpoint 或 GPU；首次编号 d04 保留 |

| 完整项目相对路径 | 用途 | 生成者与条件 |
|---|---|---|
| `experiments/inference_infra/results/w1/d05_cache_probe.json` | 位置、末位置 logits 差、复制/reset 和存储量 | 第 4 节脚本导入与全部断言成功后自动写入；同名文件会覆盖 |
| `experiments/inference_infra/reports/d05_checks.log` | 本次终端输出、退出码和错误 | 第 4 节命令组 `tee` 自动追加；仅打印的解释器信息也在此保存 |
| `experiments/inference_infra/reports/d05_cache_path.md` | 自己整理缓存生命周期与容量公式 | 人工可选编写；聊天回答即可完成交流 |

源码回查：[缓存容器与 Engine](../../../../nanochat/engine.py)、[GPT 的位置/smear/最后层推进](../../../../nanochat/gpt.py)、[后端 K/V 写入](../../../../nanochat/flash_attention.py)、[缓存容器与 Mock Engine 测试](../../../../tests/test_engine.py)。MockModel 只模拟 logits 和推进，不计算真实 K/V；不要把它的生成测试当作本日随机 GPT 的数值对照。

本日主线的完成依据是能解释状态推进并解读 CPU 对照结果。小配置通过不证明所有窗口、chunk、reset 后第二请求、真实权重或 GPU 路径均正确，也不要求为这些边界追加硬件实验。

上一篇：[D4 模型形状](d04_model_shapes.md)。下一篇：[D6 后端路由](d06_attention_backends.md)。
