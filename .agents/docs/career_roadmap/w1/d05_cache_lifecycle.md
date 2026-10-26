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

# W1 D5：跟踪 prefill、批量复制和逐步 decode

今天不重写缓存。你要理解已有缓存的生命周期，并实际检查位置推进与 smear 状态。预计 90 分钟：20 分钟容器，25 分钟生成器，20 分钟探查，15 分钟核验问题，10 分钟验收。没有 GPU 和 checkpoint 都可用 CPU 随机模型完成。

前置：[D4 模型与形状](d04_model_shapes.md)。D4 的无缓存路径是对照；今天必须使用相同权重与相同前缀比较，不能比较两个独立随机模型。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[12.1 无缓存生成与缓存生成](../../llm-training-workflow/12-inference-and-generation.md#reading-12-1)；[12.2 生成控制流与返回格式](../../llm-training-workflow/12-inference-and-generation.md#reading-12-2)。
- **今天重点**：对照今天的缓存探查，跟踪提示预填充、缓存复制、新词元解码和位置推进；区分缓存状态与可学习权重。
- **暂时跳过**：温度、top-k 和工具状态机先认识用途；今天优先看缓存和返回格式。

**读完能解释**：处理完整提示和随后每次新增一个词元时，输入形状及缓存状态分别怎样变化？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 当日脚本与产物清单

所有项目内路径均以 nanochat 项目根目录为基准；外部 MiniMind 只读历史证据不改名。`dNN_` 表示全程首次使用或产生日，跨日复用不改前缀；上游源码和测试保留原生名称。

| 角色与脚本 | 用途与执行阶段 | 输入路径或来源 | 运行条件 |
|---|---|---|---|
| 跨日复用：[`.agents/docs/career_roadmap/w1/scripts/d04_probe_model_and_cache.py`](scripts/d04_probe_model_and_cache.py) | 第 3 节 `--mode cache` 验证位置、数值、复制和 reset | `nanochat/gpt.py`、`nanochat/engine.py`、`nanochat/flash_attention.py`；内部使用同一随机模型与前缀 | 与 D4 相同；首次使用日为 d04，保留脚本名前缀 |

| 完整项目相对路径 | 用途 | 生成者 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/results/w1/d05_cache_probe.json` | 缓存位置、误差、复制/reset 检查 | `d04_probe_model_and_cache.py --mode cache` | 条件生成：导入和全部断言成功后自动写入；产物首次属于 d05 |
| `experiments/inference_infra/reports/d05_cache_path.md` | 缓存生命周期与容量公式 | 第 4 节问答可选整理 | 可选人工整理；直接聊天回答即可，不影响验收 |
| `experiments/inference_infra/reports/d05_checks.log` | 本日终端检查输出与错误 | 各 Bash 代码块的 tee | 自动追加各代码块内的输出；命令及退出码在交流时补充 |

清单是执行后的预期，不表示文件已经存在。已有未加前缀的历史实验文件保留，新执行按上述路径写入。

正文的 `sed`、`rg`、`head`、解释器检查等只向终端打印，不会自动生成独立结果文件；各 Bash 代码块内的输出追加保存到 `experiments/inference_infra/reports/d05_checks.log`；交流时附所用命令、退出码和关键结果，自动结果已有的字段无需手抄。 盘点或探查的结构化结果保存为 <abbr title="JSON / JavaScript Object Notation · 用键值和数组组织数据的文本格式；这里保存脚本实际采集的事实或断言通过后的结果，可供程序读取。">JSON</abbr>。 未执行或失败时按真实状态记录，不能把旧同名文件当作本次成功。

## 1. 先读缓存容器，不先读采样循环

日志只在 `{ ...; }` 命令组执行期间记录；<abbr title="Bash process substitution · 进程替换。这里把命令组的输出交给 tee，同时显示并追加到日志；重定向在命令组结束后恢复。">进程替换</abbr>不会持续接管交互终端，因此提示符和 VS Code 终端集成控制字符不会混入日志。命令组在当前 shell 执行，变量设置会保留；代码块返回最后一条命令的退出码，需核对单条命令时立即记录 `$?`。已有的显式文件重定向仍写入各自文件。复制时包含外层花括号和末尾重定向。若之前执行过旧版全局日志命令，先关闭旧终端并新开终端；仅修改文档不会停止旧终端的记录。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
sed -n '80,155p' nanochat/engine.py
rg -n 'kv_cache.advance|prev_embedding|T0 =|flash_attn_with_kvcache' nanochat/gpt.py
} > >(tee -a experiments/inference_infra/reports/d05_checks.log) 2>&1
```

<abbr title="KV Cache / Key-Value Cache · 每层保存已处理位置的注意力 Key/Value。nanochat 预先分配容量，用位置向量区分有效前缀和未使用区域；分配容量不等于有效长度。">KV Cache</abbr> 的两块本体形状均为 `[L,B,Tmax,Hkv,Dh]`。逐项确认：

- `get_layer_cache(i)` 返回去掉层轴的视图 `[B,Tmax,Hkv,Dh]`，不是复制一份新缓存。
- `cache_seqlens` 为每行一个 int32 位置；当前 `get_pos()` 只读第一行，因此一般使用假设是批内位置一致。
- `advance(T)` 只更新位置，不负责写 K/V。
- K/V 写入发生在 attention 后端；位置只在最后一个模型层处理后推进一次，不能每层都加 T。
- `reset()` 清空位置、把 `prev_embedding` 设为 None，**没有把 K/V 存储全部置零**。旧数值仍可能存在，但正确实现不应读有效长度外的区域。

<abbr title="prev_embedding · 缓存的前一 token 归一化 embedding，形状通常为 [B,1,C]。它供下一步 smear 使用，不是 Key 或 Value 的别名。仅拷贝 K/V 而漏掉它，会破坏当前模型的增量推理状态。">prev_embedding</abbr> 是今天最容易漏掉的状态。交流时需单独说明其来源、更新与复制。

## 2. 对照 Engine.generate 的两次分配

```bash
{
sed -n '195,310p' nanochat/engine.py
sed -n '470,535p' nanochat/gpt.py
} > >(tee -a experiments/inference_infra/reports/d05_checks.log) 2>&1
```

对照下面 `P=8,B=4,G=4` 的时间线，确认每步状态，可直接口述：

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

使用与 D4 同一份[模型与缓存探查脚本](scripts/d04_probe_model_and_cache.py)：

```bash
{
python .agents/docs/career_roadmap/w1/scripts/d04_probe_model_and_cache.py --mode cache --prompt-tokens 8 --decode-steps 4
} > >(tee -a experiments/inference_infra/reports/d05_checks.log) 2>&1
```

默认写 `experiments/inference_infra/results/w1/d05_cache_probe.json`。脚本直接调用 `GPT.forward(...,kv_cache=...)`，这样把缓存数值验证与 Engine 的结束 token/工具状态机分开。它做：

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

编写时验证记录：Conda `nanochat` / torch 2.9.1+cpu 实测 4 步位置为 8、9、10、11；最大绝对 logits 差 `7.450580596923828e-09`，greedy token 一致；B=4 的 K/V 与 prev 复制、reset 均通过。这里只验证上述小配置，不外推 GPU、真实 checkpoint 或所有边界。

## 4. 当天自验证与交流问题及内存公式

按题号在聊天中回答即可，无需写报告。概念题说明理由或推导；实测题引用当天日志或 JSON 中支撑结论的片段，运行命令、退出码和证据路径提供一次即可。未执行的实测项标 `PENDING`；引用历史结果时注明日期与版本，不必为回答问题额外重跑实验。

1. **容量与有效长度**：对 P=8、B=4、G=4，Engine 的 prefill cache 和 decode cache 各有多大、哪些位置有效？解释层视图中的 B/Tmax/Hkv/Dh 分别代表什么。

   **核对要点**：沿两次分配与复制给出具体 shape 和位置；区分当前输入长度、已处理长度与预分配容量，head 数采用 K/V 头数。

2. **首 token 与恢复执行**：首次 yield 的 token 来自哪次 forward，此时它是否已经写入 cache？再次 next 时输入什么，位置怎样变化？

   **核对要点**：按生成器执行顺序解释采样、yield 与下一次单 token forward；区分已经输出和已经喂入模型的 token。

3. **位置必须一致**：cache 位置为8时输入一个新 token，RoPE 使用哪段位置参数、K/V 写到哪里、何时推进长度？为什么每层都 advance 会出错？

   **核对要点**：把 GPT.forward、attention 写入和最后一层推进连成同一条位置链；给出具体区间，而非只说“位置加一”。

4. **K/V 之外的历史状态**：prev_embedding 保存的是什么时刻的表示，下一步怎样使用？若只复制 K/V 而漏掉它，会破坏什么计算？

   **核对要点**：依据 smear 的读取/赋值顺序说明状态来源和作用，并解释为何非零 smear 的探查更容易暴露该问题。

5. **用真实探查检验等价**：选本次 JSON 中 prefill 后的一步，解释缓存与无缓存路径如何保持同一前缀，并用位置和 logits 差异判断结果。“四步”为什么不等于四次 decode？

   **核对要点**：引用所选 step 的 input_tokens、before/after 与误差，结合 assert_close 和 greedy 检查解释结论；失败则定位最早分歧，不只放宽容差。

6. **复制与重新使用的语义**：复制到 B=4 与 reset 分别改变哪些状态？为什么 reset 后位置为0并不意味着 K/V 内存已清零，也不代表“重用后的第二个请求”已经测过？

   **核对要点**：结合 prefill/reset 代码和 replication/reset 断言，区分有效前缀复制、prev 状态与存储内容；明确当前探查的覆盖边界。

7. **计算实际缓存代价**：按复制后 L=2、B=4、Tmax=64、Hkv=2、Dh=16、FP32，推导两份 K/V 的字节数，并与 storage_bytes_kv 对照。若只减少有效长度而不重新分配，存储量会变吗？

   **核对要点**：写清每个因子的来源和单位，区分理论分配量、有效前缀、模型/临时张量与总内存；结论限定为本次配置。

预分配 K/V 理论字节数：`2 * L * B * Tmax * Hkv * Dh * 每元素字节数`。有效 token 数 P 不应替代容量 Tmax 估算分配量。不要声称 KV Cache 降低总显存；它用额外状态换取后续少算历史。

## 5. 失败分流与当天验收

- 依赖失败：回 D2；完成源码表也只能标静态阅读，运行仍 BLOCKED。
- 位置错误：先检查推进是不是只在最后一层，以及是否误把采样次数当成已喂入 token 数。
- step 0 对，step 1 起错：优先核对 RoPE 的偏移和 `prev_embedding`；不要只放宽容差。
- 单行对，复制后错：检查批量广播、有效前缀和 prev 的复制。
- 误差为零：可以是真结果，但还要看是否用了非零投影和 smear；脚本已避免默认零投影让测试失去区分力。

验收：能讲清生命周期各阶段；掌握容量/有效长度差别；能解释 prev；保存真实 JSON 或阻塞原因；不把这份小探查当作 W9 全覆盖回归。交流附真实日期、commit、命令、结果路径、状态与下一步；用时可选。

上一篇：[D4 模型形状](d04_model_shapes.md)。下一篇：[D6 后端路由](d06_attention_backends.md)。
