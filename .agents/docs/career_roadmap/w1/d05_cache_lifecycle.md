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

可直接在聊天中按题号回答，不要求手写报告。每题给出结论和可定位依据（源码位置、命令与退出码、日志片段或自动结果字段）；同一份证据可复用。概念推演须注明假设，实际运行须区分本次与历史结果。未执行写 `PENDING`，失败写真实错误和影响范围，不能据此勾选实验通过。

1. **四类缓存状态**：K、V、`cache_seqlens` 与 `prev_embedding` 分别存什么？用 L=2、B=4、Tmax=64、Hkv=2、Dh=16、C=64 写出各自 shape，并说明各自 dtype 的来源。

   **通过标准**：对照 `KVCache.__init__`、`GPT.forward` 与本次配置；区分 K/V 本体、int32 有效长度及 smear 用的归一化 embedding，不能把 prev 当 K/V 别名。

2. **视图、写入与推进职责**：`get_layer_cache(i)`、注意力后端写 K/V、`advance(T)` 各负责什么？若两层模型每层都推进 P=8，会使下一层从什么错误位置读写？

   **通过标准**：指出层视图共享存储、写入区间和最后一层才推进的源码条件；用具体位置解释不能每层加 T，不把 advance 说成计算或写入 K/V。

3. **位置编码的偏移**：prefill 后位置为 8，下一步输入一个 token 时，`T0`、RoPE 切片和 K/V 写入区间分别是什么？为什么不能仍使用位置 0 的旋转参数？

   **通过标准**：沿 `GPT.forward` 到 `flash_attn_with_kvcache` 对齐同一个绝对位置；说明位置推进发生在处理完成后，引用源码而非仅凭最后文本猜测。

4. **Engine 两次分配的时间线**：对 P=8、B=4、G=4，从生成器创建、B=1 prefill、分配 decode cache、复制到首次 yield，分别列出输入 shape、缓存容量、有效长度和 batch。

   **通过标准**：对照 `Engine.generate` 区分 prefill 容量 P 与 decode 容量 P+G，说明第一列分数来自 prefill 最后位置并 expand；复制四份状态不等于四个独立 prompt。

5. **yield 暂停与生成计数**：首次 yield 时首个采样 token 是否已经写入 cache？再取第二列时输入什么、位置如何变化？只取完 G 列就停止与再 next 一次探测结束有什么差别？

   **通过标准**：按 yield 后的代码顺序推导；区分已产出的 token 和已喂入模型的 token，指出探测 StopIteration 可能触发额外 forward；直接 forward 探查不验证此生成器边界。

6. **探查四步的真实含义**：本次 `steps` 各行的 `input_tokens/cache_before/cache_after/cache_seqlens/prev_embedding_shape` 是什么？`--decode-steps 4` 为何只包含三次单 token decode？

   **通过标准**：逐行引用本次 JSON，默认应核对 prefill 及之后三步；不是机械复述历史 8/9/10/11，未运行或参数不同时给真实结果与差异。

7. **同一前缀的数值对照**：脚本的缓存与无缓存分支是否使用同一份权重、前缀和下一 token？每步比较哪个位置，`max_abs_logit_diff`、greedy 相同与 assert_close 各证明什么？

   **通过标准**：引用循环中 reference、cached、argmax 和前缀追加代码；报告实际误差，说明 atol=1e-5/rtol=1e-4 分别为绝对与相对容差，按 `atol + rtol*abs(reference)` 组合判定，不能只比较最大绝对差或编造 JSON 中不存在的 greedy_match 字段。

8. **smear 状态的来源与时机**：`prev_embedding` 保存的是 smear 前还是之后的表示？prefill、单 token decode 时分别如何读取和更新；漏拷贝它为何可能从第一步 decode 开始出错？

   **通过标准**：指出 `x_pre_smear`、赋值与 gate 使用顺序，说明 prev 的 shape 及来源；结合脚本非零 smear 和输出投影解释验证的区分能力，不靠放宽容差处理状态错误。

9. **prefill 计算与容器复制**：`KVCache.prefill(other)` 是否执行模型前向？它复制有效前缀还是全部容量，除了 K/V 还复制什么，为什么 prev 使用 expand 后 clone？

   **通过标准**：区分两个 prefill 含义，指出有效区间、位置 fill 和 prev 独立存储；引用本次 replication 字段与各行断言，说明脚本并未在复制后继续做 B=4 数值对照或专门验证修改隔离。

10. **复制与批内位置的前提**：目标非空、层/head/dim 不匹配或目标容量小于源容量时，当前复制接口有什么限制？`get_pos()` 与 SDPA 读取第一行意味着能否直接支持不同长度请求？

    **通过标准**：逐项对应 prefill 的真实断言，注意容量检查比较 max_seq_len 而非只有 other_pos；说明当前均匀位置假设，不把一般的批内变长能力当作已支持，不要求实际制造失败。

11. **reset 的准确含义**：本次 reset 后位置和 prev 是什么，K/V 中旧数值是否必然为零？为什么有效长度外的数据在正确路径中不应被读取？

    **通过标准**：引用 `reset_verified` 及源码，区分 reset 断言与“重用后数值正确”的额外验证；不将位置重置等同于清空内存或已经测过第二个请求。

12. **容量公式与具体字节数**：代入第 1 题参数和 FP32，每份 K、两份 K/V 合计各多少字节？若只把 Tmax 从 64 改为 Engine 示例的 P+G=12，理论分配量如何变化？

    **通过标准**：展示 `2*L*B*Tmax*Hkv*Dh*每元素字节数` 和代入过程，与 `storage_bytes_kv` 对照；区分容量与有效长度、字节与 KiB，另列 prev/位置/权重/临时张量未计入。

13. **按症状定位而非猜测**：若 step 0 一致而 step 1 开始不一致、单行一致但复制检查失败、位置增加成两倍，分别优先核对哪段代码和哪些状态？

    **通过标准**：明确这是故障推演，分别落到 RoPE/prev、复制有效区间/广播、最后层推进等可检查位置；有真实失败则引用首个错误，不伪造故障日志，也不只回答“放宽容差”。

14. **本日通过与未覆盖范围**：本次命令、退出码、JSON 路径、随机配置、精度和后端是什么？容量耗尽、长上下文、不同批内位置、真实 checkpoint、GPU 与工具/结束状态机是否都测过？

    **通过标准**：将源码推导、探查已执行断言和未执行分支分开；JSON 只在全部断言成功后写入，旧文件存在不代表重跑成功；给出下一项最相关验证，不宣称全覆盖或 KV 降低总显存。

预分配 K/V 理论字节数：`2 * L * B * Tmax * Hkv * Dh * 每元素字节数`。有效 token 数 P 不应替代容量 Tmax 估算分配量。不要声称 KV Cache 降低总显存；它用额外状态换取后续少算历史。

## 5. 失败分流与当天验收

- 依赖失败：回 D2；完成源码表也只能标静态阅读，运行仍 BLOCKED。
- 位置错误：先检查推进是不是只在最后一层，以及是否误把采样次数当成已喂入 token 数。
- step 0 对，step 1 起错：优先核对 RoPE 的偏移和 `prev_embedding`；不要只放宽容差。
- 单行对，复制后错：检查批量广播、有效前缀和 prev 的复制。
- 误差为零：可以是真结果，但还要看是否用了非零投影和 smear；脚本已避免默认零投影让测试失去区分力。

验收：能讲清生命周期各阶段；掌握容量/有效长度差别；能解释 prev；保存真实 JSON 或阻塞原因；不把这份小探查当作 W9 全覆盖回归。交流附真实日期、commit、命令、结果路径、状态与下一步；用时可选。

上一篇：[D4 模型形状](d04_model_shapes.md)。下一篇：[D6 后端路由](d06_attention_backends.md)。
