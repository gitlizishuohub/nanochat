---
source_files:
  - nanochat/flash_attention.py
  - nanochat/common.py
  - tests/test_attention_fallback.py
source_symbols:
  - _load_flash_attention_3
  - _resolve_use_fa3
  - _sdpa_attention
  - flash_attn_func
  - flash_attn_with_kvcache
  - TestSDPAOnly
doc_type: tutorial
knowledge_scope: W1 D6 注意力后端条件与CPU覆盖边界
---

# W1 D6：同一个注意力接口，实际走了哪种实现

今天解决两个相连的问题：**模型写着 `flash_attn`，为什么 CPU 也能运行？换了底层实现，怎样保证“能看哪些历史位置”没有变？** 我们先理解统一接口，再判断后端选择，最后把 D5 的缓存位置转成注意力可见范围。

前置：[D5 缓存生命周期](d05_cache_lifecycle.md)。第 1–3 节是源码带读与纸面例子，无需操作。第 4 节使用已有脚本运行 CPU 测试，不需要 GPU、tokenizer 或训练权重，不下载内核。测试中有对随机 Q/K/V 求梯度的用例，但没有训练 GPT、没有优化器更新。GPU 上比较两个实现是可选扩展，不是本日验收的欠账。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[6.2 注意力方向、窗口与RoPE](../../llm-training-workflow/06-forward-pass-and-prediction-loss.md#reading-6-2)；[8.1 显式混合精度与梯度缩放](../../llm-training-workflow/08-precision-and-training-acceleration.md#reading-8-1)。
- **今天重点**：带着昨天的缓存路径回读因果掩码和滑动窗口；精度小节只区分参数、计算及损失的数值类型。结合当天源码确认后端条件。
- **暂时跳过**：<abbr title="FP8 · 8 位浮点计算；本项目的可选训练加速方式，需要兼容硬件。">FP8</abbr> 和缩放器的实现细节暂不展开，8.2 留到 D17 按设备条件选读。

**读完能解释**：为什么 CPU 上的注意力回退测试通过，仍不能证明 GPU 内核通过？精度会影响哪些选择条件？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 1. 先理解“接口相同，实现可以不同”

D4 中每层从输入表示投影出 Q、K、V。可以把注意力理解为：用 Q 与可见的 K 算匹配分数，缩放并归一化成权重，再按权重汇总 V。底层实现可以改变计算与存储安排，但对于本项目调用的因果注意力，应保持输出的含义、形状以及可见位置一致。

[模型注意力层](../../../../nanochat/gpt.py) 不直接判断 GPU 型号，而是调用 [统一封装](../../../../nanochat/flash_attention.py) 的两个入口：

```python
# 无缓存：完整输入的位置一起计算
flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
# 有缓存：把新 K/V 写进已有空间，再利用有效历史
flash_attn.flash_attn_with_kvcache(
    q, k_cache, v_cache, k=k, v=v,
    cache_seqlens=kv_cache.cache_seqlens,
    causal=True, window_size=window_size,
)
```

第一种也用于 D4 的无缓存推理对照；即使源码注释写着 Training，也不能把“调用无缓存入口”当成“发生训练”。是否训练还要看损失、反向传播和参数更新。

这里的 `flash_attn` 是模块末尾导出的 `SimpleNamespace`，里面放了这两个封装函数，不等于已经载入 GPU 内核。函数内部依据 `USE_FA3` 分支：

- <abbr title="FA3 / Flash Attention 3 · 本项目尝试通过 kernels 包加载的 GPU 注意力实现。接口存在、实现可加载和本次最终选用是不同事实。">FA3</abbr> 分支调用载入接口的函数。
- <abbr title="SDPA / Scaled Dot Product Attention · PyTorch 的缩放点积注意力接口。本项目用它构成回退分支，并自行处理窗口、矩形因果对齐和缓存写入。它的名字不指定 PyTorch 内部一定使用哪种内核。">SDPA</abbr> 分支转为 PyTorch 接口，自己补齐缓存与掩码逻辑。

本课讨论的是 GPT 实际传入 `causal=True`、右窗口 0 的调用。SDPA 辅助函数按因果规则实现，没有据公开参数的 `causal=False` 切换非因果行为，也不实现任意右窗口；因此不要从“统一接口”外推为所有参数组合都已等价。

### 1.1 用 D4 的数值把布局转换走一遍

取 `B=1、T=8、Hq=4、Hkv=2、Dh=16`。外部接口和缓存都把序列轴放在头轴前；SDPA 计算接口把头轴放在序列轴前。

| 阶段 | Q | K/V | 输出 |
|---|---|---|---|
| 外部注意力接口 | `[1,8,4,16]` | `[1,8,2,16]` | 返回 `[1,8,4,16]` |
| `transpose(1,2)` 后 | `[1,4,8,16]` | `[1,2,8,16]` | SDPA 产生 `[1,4,8,16]` |
| 输出再 transpose | — | — | 恢复 `[1,8,4,16]` |

这里没有改变 token 所属的位置，只交换轴的解释。封装用 `q.size(1) != k.size(1)` 得到 `enable_gqa=True`。<abbr title="GQA / Grouped-Query Attention · 分组查询注意力。多个 Q 头共享一组 K/V 头，减少 K/V 的表示和缓存规模；输出仍按 Q 头组织。">GQA</abbr> 在这个例子中让四个 Q 头使用两个 K/V 头；不是因为 K/V 少两头就把输出也改成两头。缓存张量仍是 `[B,Tmax,Hkv,Dh]`，不能直接把 SDPA 的 `[B,H,T,D]` 当成缓存存储格式。

## 2. 后端选择要经过“可加载”和“最终使用”两步

### 2.1 跟着源码的选择顺序走

在 [flash_attention.py](../../../../nanochat/flash_attention.py) 导入时，先调用 `_load_flash_attention_3()`：无 CUDA 立即返回 None；有 CUDA 时才尝试加载 kernel 接口。设备 capability 主版本为 9 时尝试一个指定包，其余情况走另一个包的可用性检查；加载异常也返回 None。这里可能涉及外部内核获取，所以本日脚本主动隐藏 CUDA，避免把学习依赖变成下载任务。

随后 `_fa3 is not None` 决定 `HAS_FA3`。`_resolve_use_fa3()` 再决定 `USE_FA3`，其逻辑可概括为：

```python
if _override_impl == 'fa3':
    assert HAS_FA3
    return True
if _override_impl == 'sdpa':
    return False
# 自动模式
return HAS_FA3 and COMPUTE_DTYPE == torch.bfloat16
```

这是便于理解的等价摘录；完整源码还包括计算精度的导入。**HAS 回答“载入了吗”，USE 回答“封装最终会调用它吗”。** 自动模式多了一道计算精度检查，二者不能画等号。

[common.py](../../../../nanochat/common.py) 的 `COMPUTE_DTYPE` 也在导入时确定。`NANOCHAT_DTYPE` 显式设置优先；无 CUDA 时默认 <abbr title="FP32 / float32 · 32 位浮点格式，每元素占 4 字节。这里是 CPU 学习主线的计算类型，与指定 bf16 的自动路由情形区分。">FP32</abbr>，有 CUDA 且 capability≥8.0 时默认 <abbr title="bf16 / bfloat16 · 16 位浮点格式。本项目自动选择 FA3 时要求计算精度为 bf16，但仅这个条件不能证明内核可用。模型参数存储类型与计算类型也不必相同。">bf16</abbr>，更早 CUDA 设备默认 FP32。环境变量应在导入前设置，不能在同一进程中改完环境变量就假定已导入模块的选择随之更新。

### 2.2 预测六种配置的分支

下表是依据当前源码做的条件推导，除 CPU 主线外都不表示你拥有或已测试相应硬件：

| 给定条件 | HAS_FA3 | USE_FA3 | 解释 |
|---|---:|---:|---|
| 当前进程无 CUDA、自动精度、自动模式 | false | false | 不进入内核加载；计算默认 FP32，本页 CPU 脚本采用此条件 |
| CUDA 可用但内核加载失败，计算 bf16 | false | false | 精度合适不能补足接口缺失 |
| 已载入接口，计算 FP32，自动模式 | true | false | 自动路由回退 SDPA |
| 已载入接口，计算 bf16，自动模式 | true | true | 当前自动条件全部满足 |
| 已载入接口，bf16，但 override=sdpa | true | false | 测试强制选择优先 |
| 接口未载入，却 override=fa3 | false | 不产生正常选择 | 断言失败，不能强制创造一个内核 |

如果接口已载入，override=fa3 会直接选 FA3，绕过自动精度检查；这不是“任意 dtype 均安全”的保证。override 是测试机制，日常学习不需要修改模块私有变量。[测试辅助函数 `set_impl`](../../../../tests/test_attention_fallback.py) 会重新调用解析器并赋值 `USE_FA3`，生产调用不会按每个输入 tensor 的 device/dtype 自动重选后端。

因此，当前导入环境、选择变量和实际 tensor 应配套。不能根据源码中的硬件注释声称某张卡已经通过验证，也不能根据“这次在 GPU 上运行”就断言使用了 FA3。

## 3. 同样重要的契约：当前 query 能看到哪些 K

<abbr title="causal mask · 因果掩码。当前位置只能使用自己及之前位置的信息。缓存中的 query 可能位于完整历史的末端，需要按绝对位置而非局部行号决定可见范围。">因果掩码</abbr> 的目的不是机械地取一个矩阵三角形，而是限制“不能看未来”。<abbr title="sliding window · 滑动窗口。本文 W 表示最多向左看的位置数，还可以看自己，所以可见 K 最多 W+1 个；这是当前后端左窗口参数的含义。">滑动窗口</abbr> 再限制能向左看多远。看 [_sdpa_attention](../../../../nanochat/flash_attention.py) 的三个分支，用下列例子理解为什么不能统一写一个 `is_causal=True`。

### 3.1 完整提示：Tq=Tk=8

Q 和 K 都是位置 0–7，且窗口覆盖整个上下文。第 0 行只能看 key 0；第 7 行可以看 key 0–7。两者从同一原点对齐，代码直接交给 SDPA 的 `is_causal=True`。

### 3.2 单 token 解码：旧历史 9 个，新位置为 9

D5 的缓存分支先把新 K/V 写入位置 9；此时 `Tq=1、Tk=10`，query 对应的是绝对位置 9，K 对应位置 0–9。所有有效 K 都不在未来，所以无需再加三角掩码，代码使用 `is_causal=False`。

若左窗口 W=2，当前 query 只能看位置 7、8、9，代码先从 `start=Tk-(W+1)=7` 截取 K/V，再调用 SDPA。False 在这里不意味着放开未来位置，而是未来和过远历史都已经不在传入的 K/V 中。把 W 当成总数量会漏掉当前位置，产生差一错误。

### 3.3 一次输入两个新 token：旧历史 8 个，新位置为 8、9

这叫 chunk 推理，Tq=2、Tk=10。两行 query 的局部索引是 0、1，但对应绝对位置 8、9。代码因此构造：

```python
row_idx = (Tk - Tq) + torch.arange(Tq).unsqueeze(1)
col_idx = torch.arange(Tk).unsqueeze(0)
mask = col_idx <= row_idx
# 有限左窗口还要求 row_idx - col_idx <= W
```

摘录省略 device 参数。`Tk-Tq=8` 是两条新 query 在历史中的起点。W=2 时，mask 中 True 表示允许参与注意力：

| query 绝对位置 | 可见 key | 为什么 |
|---:|---|---|
| 8 | 6、7、8 | 9 是未来；0–5 超过向左两格的窗口 |
| 9 | 7、8、9 | 可以看同一 chunk 中更早的位置 8，也可以看自己 |

若误把 query 当成绝对位置 0、1，只会得到 key 0 和 key 0–1，历史对齐就完全错了。这正是矩形输入不能随意套用从零起算的方阵因果掩码的原因。

### 3.4 把 mask 与 D5 的缓存写入接起来

`flash_attn_with_kvcache` 的 SDPA 分支读取 `pos=cache_seqlens[0]`，把新 K/V 写入 `[pos:pos+T_new]`，再只取 `[:pos+T_new]` 参与注意力。上例 pos=8、T_new=2，写入 `[8:10]`，传给 mask 的历史长度为 10。

后端不修改 `cache_seqlens`；模型最后一层才 advance。预留容量即使为 64，也不能把 64 当成 Tk，把未写区域加入注意力。本路径假设各行位置一致；读第一行长度并不能支持“批内每行任意不同位置”的结论。

## 4. CPU 实操：分别观察选择与测试断言

本节复用 [D3 测试捕获脚本](scripts/d03_run_inference_tests.py)。先核验当前解释器和 CPU 路由，再运行相关用例。命令的环境设置发生在进程导入前，隐藏 CUDA、指定 FP32；只检查本地实现。`tee` 只捕获花括号内输出并追加日志，退出后恢复终端；旧版全局重定向仍在运行时先新开终端。每条关键命令后立即取退出码：

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
CUDA_VISIBLE_DEVICES='' NANOCHAT_DTYPE=float32 python -c 'import sys; from nanochat import flash_attention as fa; from nanochat.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON; print(sys.executable); print(COMPUTE_DTYPE, COMPUTE_DTYPE_REASON); print("HAS_FA3=", fa.HAS_FA3, "USE_FA3=", fa.USE_FA3)'
d06_status=$?
printf 'route_check_exit=%s\n' "$d06_status"
(exit "$d06_status")
} > >(tee -a experiments/inference_infra/reports/d06_checks.log) 2>&1
```

预期看到当前 Python 路径、`torch.float32`、`HAS_FA3=False`、`USE_FA3=False`。这证明这个新进程的路由选择，不等于已经算过一次 attention。真正计算由下面的测试完成；脚本的子进程也主动固定 CPU/FP32：

```bash
{
python .agents/docs/career_roadmap/w1/scripts/d03_run_inference_tests.py --select 'SDPAOnly or OverrideMechanism'
d06_status=$?
printf 'attention_tests_exit=%s\n' "$d06_status"
(exit "$d06_status")
} > >(tee -a experiments/inference_infra/reports/d06_checks.log) 2>&1
```

脚本打印 `Logs:` 后的真实时间戳目录和 pytest 摘要。打开该目录的 `d03_run.json`，看 `argv` 是否含本次筛选表达式、`exit_code` 是否为 0，以及固定环境字段；再读 `d03_stdout.log` 中通过、跳过、未选中的数量。它复用 d03 名称但生成新时间戳，不覆盖 D3 结果。

把测试名与 [真实断言](../../../../tests/test_attention_fallback.py) 对上，比背通过数量更有意义：

| 测试 | 当前断言能说明什么 | 不能据此声称什么 |
|---|---|---|
| `TestSDPAOnly.test_basic_forward` | 因果前向输出 shape 正确且没有 NaN | 没有与独立数值参考逐元素比较，也未检查所有非有限值 |
| `TestSDPAOnly.test_backward` | Q/K/V 梯度存在，Q 梯度没有 NaN | 没有断言所有梯度均有限、没有训练或检查参数更新 |
| `TestSDPAOnly.test_kvcache` | prefill 与单 token 输出 shape 正确；调用者 advance 后长度为 16、17 | 未做缓存 logits 等价对照；当前位置是测试主动推进，不是后端自己推进 |
| `test_override_sdpa`、`test_override_auto` | 此 CPU 环境的强制回退与自动选择通过 | auto 测试的 `USE_FA3==HAS_FA3` 断言不能概括有 FA3 且计算 FP32 的配置 |
| `test_override_fa3` | 无 FA3 时按条件跳过 | skip 不是 FA3 通过 |

本次筛选不执行 `TestFA3VsSDPA`；它们是未选中的用例，不能统称为“全部因缺 GPU 跳过”。CPU 主线的三个 SDPAOnly 用例也没有逐数值检验滑动窗口、矩形 chunk 或 GQA。第 3 节是在源码上推导其实现，不应写成这些 CPU 边界都已实测通过。D5 的小 GPT 对照补充了一个全上下文 GQA 缓存场景，仍不覆盖全部 mask 分支。

若自己选择未来做 GPU 对照，要固定 Q/K/V、精度和窗口，分别比较输出、缓存或梯度，并使用合适容差；不能从 CPU 通过推断 FA3 正确或更快。本日无需准备这组硬件实验。

## 5. 当天自验证与交流问题

直接在聊天中作答即可，命令、退出码、解释器与证据路径只附一次。概念题不要求运行；CPU 实测题引用自己的日志，尚未执行就明确说明。人工 Markdown 报告可选。

1. **配置选择**：同一台可载入 FA3 的机器，自动模式下分别设计算精度 FP32 与 bf16，会选什么？如果计算仍为 bf16 但内核加载失败，又会怎样？

   **核对要点**：依据 HAS、dtype、override 顺序判断；区分条件推导和自己的 CPU 实测。

2. **布局与接口**：把例子改为 B=2、T=3，其余头数与维度不变。给出外部 Q/K/V、SDPA 内部 Q/K/V 与返回输出的 shape，并解释 GQA 为何打开。

   **核对要点**：transpose 交换正确的轴；输出按 Q 头组织；不把缓存容量轴当成本次 T。

3. **迁移到新的位置**：旧历史 5 个，一次输入 2 个新 token，W=1。两个 query 的绝对位置和可见 key 分别是什么？第二行为什么能看到同一 chunk 的第一行？

   **核对要点**：从 Tk−Tq 推导并同时满足因果、左窗口；不能把局部行号当绝对位置。

4. **单 token 特例**：Tk=10、Tq=1、W=0 时会截取哪些 K/V？为什么此时 `is_causal=False` 仍符合因果要求？

   **核对要点**：把 W 解释为向左位置数，说明有效 K 中是否存在未来。

5. **按证据报告覆盖**：选择本次一个已通过测试说明真实断言，再各给一个“源码已推导但这次 CPU 测试未数值验证”的行为和“需要其他硬件才能比较”的行为。

   **核对要点**：区分 shape、NaN、位置推进、数值对照；不把 deselected 当 skipped，不把 skipped 当 passed。

导入失败先回 D2 核对依赖和解释器；CPU 用例失败保存最早断言、输入 shape 与日志；不要为了让测试通过改成另一个后端或声称 GPU 待补即可忽略 CPU 失败。

## 6. 脚本、产物与回查入口

路径均相对于项目根目录。清单是实际运行时的生成约定，不表示已经生成，也不要求新写报告。

| 脚本/阶段 | 完整输入路径或来源 | 运行条件 |
|---|---|---|
| [`.agents/docs/career_roadmap/w1/scripts/d03_run_inference_tests.py`](scripts/d03_run_inference_tests.py)，第 4 节定向测试 | `tests/test_engine.py`、`tests/test_attention_fallback.py`；`--select 'SDPAOnly or OverrideMechanism'` 传入 pytest `-k` | 本地项目/dev 依赖齐备；固定 CPU/FP32，无 GPU、checkpoint 或内核下载 |
| 第 4 节单行 Python 路由检查 | `nanochat/flash_attention.py`、`nanochat/common.py` 和当前解释器 | 仅打印，输出保存在下列 checks 日志，不单独写 JSON |

| 完整项目相对路径 | 用途 | 生成者与条件 |
|---|---|---|
| `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_stdout.log` | 本次 pytest 标准输出 | 捕获脚本自动创建，子进程输出写入 |
| `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_stderr.log` | 本次 pytest 标准错误 | 同上，可为空 |
| `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_run.json` | 实际 argv、环境、退出码与耗时 | 子进程正常返回后自动写入，测试失败也写 FAIL |
| `experiments/inference_infra/reports/d06_checks.log` | 两个命令组的选择信息、测试输出与即时退出码 | 第 4 节局部 tee 自动追加 |
| `experiments/inference_infra/reports/d06_attention_backends.md` | 自己整理配置选择和掩码推导 | 人工可选编写；聊天回答即可 |

`<UTC时间戳>` 格式为 `YYYYmmddTHHMMSSffffffZ`，按 `Logs:` 打印的真实路径查找。脚本不能启动子进程等异常情况下可能只留下部分文件，不能只凭目录存在判定成功。

源码回查：[后端检测、路由与 SDPA 掩码](../../../../nanochat/flash_attention.py)、[计算精度](../../../../nanochat/common.py)、[模型调用与推进位置](../../../../nanochat/gpt.py)、[CPU 与 FA3 对照测试](../../../../tests/test_attention_fallback.py)。

上一篇：[D5 缓存生命周期](d05_cache_lifecycle.md)。下一篇：[D7 基准审计](d07_benchmark_audit.md)。
