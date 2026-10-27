---
source_files:
  - nanochat/flash_attention.py
  - nanochat/common.py
  - tests/test_attention_fallback.py
source_symbols:
  - _resolve_use_fa3
  - _sdpa_attention
  - flash_attn_func
  - flash_attn_with_kvcache
  - TestSDPAOnly
  - TestOverrideMechanism
doc_type: explanation
knowledge_scope: W1 D06 注意力后端选择、掩码与实测覆盖交流核对
---

# W1 D06：注意力后端交流验收记录

- 交流及报告日期：2026-10-27（Asia/Shanghai）。
- 教程：[D06 注意力后端](../../../.agents/docs/career_roadmap/w1/d06_attention_backends.md)。
- 本次操作：核对用户五题回答，补全掩码推导与测试覆盖边界，并归档已有运行证据；未重新运行测试或 GPU 实验。
- 当前源码核对 commit：`152e1fbe56a55c3f3ed5dcb01481dd943566c588`。历史测试日志未记录运行 commit，不能用此版本倒填。

## 1. 后端可加载性与实际选择

用户回答的选择逻辑基本正确，需补充强制分支的断言、精度判定对象与重新计算时机。

<abbr title="FA3 / FlashAttention-3 · 本项目尝试通过 kernels 包加载的 GPU 注意力实现。加载成功只表示获得接口，最终是否使用还取决于选择逻辑。">FA3</abbr> 的 `HAS_FA3` 表示接口是否加载成功，`USE_FA3` 才决定调用哪个分支。<abbr title="override · 显式覆盖自动选择的测试开关。本项目使用模块私有变量 _override_impl，测试通过 set_impl 重新计算 USE_FA3。">override</abbr> 优先于自动模式：

| 模式 | 条件与结果 |
|---|---|
| `_override_impl='fa3'` | 先断言 `HAS_FA3=true`，成功后返回 true；不可用时触发断言，不是静默回退；此分支不再检查计算精度 |
| `_override_impl='sdpa'` | 无论 `HAS_FA3` 如何，均返回 false |
| `_override_impl=None` | 仅当 `HAS_FA3=true` 且 `COMPUTE_DTYPE==torch.bfloat16` 时返回 true，否则为 false |

这里检查的是 `nanochat.common.COMPUTE_DTYPE`，即模块的计算精度配置，不是在每次调用时读取 Q 的 dtype。<abbr title="bf16 / bfloat16 · 16 位浮点格式。本项目自动模式只在 FA3 可加载且计算精度为 bf16 时选择 FA3；不能据此推断强制模式下任意精度都安全。">bf16</abbr> 是自动选择的必要条件之一。

`USE_FA3` 在模块导入时计算一次；测试的 `set_impl()` 会在修改 override 后显式调用 `_resolve_use_fa3()`。仅修改 `_override_impl` 不会自动更新 `USE_FA3`。

本次引用的 CPU/FP32 定向测试覆盖 SDPA 及相应 override 行为；FA3 的实际 GPU 运行仍为 **PENDING_GPU**。可加载但计算精度为 bf16、FP16 或 FP32 时的不同自动选择结果，本次仅按源码静态推导。

源码：[后端选择实现](../../../nanochat/flash_attention.py)、[计算精度配置](../../../nanochat/common.py)。

## 2. 张量布局转换与 GQA

用户回答正确；补全 D04 的具体头数与轴含义。

<abbr title="SDPA / Scaled Dot Product Attention · 缩放点积注意力。本文指项目回退分支调用的 PyTorch 接口，内部按批次、头、位置、头内维度组织张量。">SDPA</abbr> 回退先将 Q、K、V 从 `[B,T,H,D]` 转为 `[B,H,T,D]`。`transpose(1,2)` 交换从 0 开始编号的第 1、2 轴，不改变元素数量或头数：

```text
Q：   [B,T,4,D] → [B,4,T,D]
K/V： [B,T,2,D] → [B,2,T,D]
输出：[B,4,T,D] → [B,T,4,D]
```

转换后，`q.size(1)=4`、`k.size(1)=2`，因此 `enable_gqa=True`，在调用 SDPA 时启用 <abbr title="GQA / Grouped-Query Attention · 分组查询注意力。多个 Q 头共享较少的 K/V 头；此处每两个 Q 头共享一个 K/V 头，输出头数仍为 4。">GQA</abbr>。输出再经 `transpose(1,2)` 恢复外部布局。

KV 缓存的存储布局仍为 `[B,T_max,Hkv,D]`；缓存分支先取有效前缀，再为 SDPA 转置，不能将存储布局与运算布局混为一谈。

源码：[flash_attn_func 与缓存接口](../../../nanochat/flash_attention.py)。

## 3. chunk 掩码：位置偏移与两个可见集合

用户答对 query 位置 8、9，以及位置 8 可访问 6、7、8；漏答位置 9 的集合，并表示不清楚错误对齐会造成什么影响。本节补全。

<abbr title="chunk inference · 一次处理多个新 token 的增量推理。这里 Q 仅包含末尾两个新位置，K/V 则包含历史与这两个新位置，因此二者的起点不同。">chunk inference</abbr> 中，`Tk=10`、`Tq=2`，query 是有效 K/V 序列末尾的两个位置：

```text
query 绝对位置 = (Tk − Tq) + query 行号 = 8 + 行号
```

左窗口 `W=2` 时，允许的 key 同时满足 `key<=query` 和 `query-key<=2`。前者是 <abbr title="causal mask · 因果掩码。禁止 query 访问晚于自身的 key；缓存推理必须使用 query 在有效序列中的真实位置进行判断。">因果约束</abbr>，后者限制向左的距离，窗口还包含当前位置。

| query 行号 | 绝对位置 | 可访问 key |
|---|---|---|
| 0 | 8 | 6、7、8 |
| 1 | 9 | 7、8、9 |

若错误地把 query 当成位置 0、1，而 key 仍使用真实编号 0–9，同样的条件将分别产生 `{0}` 和 `{0,1}`。这会把注意力错误地对齐到序列开头，漏掉真正需要的近期上下文。`Tk−Tq` 的偏移就是为了对齐两者的位置。

位置 8 不能访问 key 9，而位置 9 可以访问它，因此两个 query 需要不同的可见集合，不能仅把矩形注意力矩阵当作从零开始的方阵三角掩码。

源码：[_sdpa_attention](../../../nanochat/flash_attention.py) 的 `row_idx`、`col_idx` 与 bool mask。

## 4. 单 token 特例：先裁窗口，再允许访问全部剩余 key

用户给出的 7、8、9 正确；“代码实现了，无需再掩码”需要补上安全依据。

当 `Tk=10`、`Tq=1`、`W=2` 时，唯一 query 位于位置 9。代码计算：

```text
start = max(0, Tk − (W+1)) = 7
k、v 取位置 [7:10]，即 7、8、9
```

这三个位置包含左侧两个历史位置和当前位置，全部不晚于 query 9，所以裁剪后可直接使用 `is_causal=False`。调用前已经只取缓存有效前缀，尚未使用的缓存容量也没有进入计算。

安全依据是：唯一 query 位于有效序列末端，而且输入 K/V 已满足因果与窗口约束。`is_causal=False` 在此表示允许访问全部已筛选的 key，不能理解为允许访问未来。

与上一题相比，chunk 中 key 9 对 query 8 是未来、对 query 9 是当前位置，需要逐行显式掩码；单 token 场景只需为唯一的 query 裁好 K/V。

源码：[_sdpa_attention 单 token 分支及缓存有效前缀](../../../nanochat/flash_attention.py)。

## 5. 已通过断言与两后端数值对照的缺口

用户正确指出现有证据不能证明 FA3 与 SDPA 输出一致，但“只检查 shape”不准确。

例如，本次选中的 `TestSDPAOnly.test_basic_forward` 检查两项：

```python
assert y.shape == (B, T, H, D)
assert not torch.isnan(y).any(), "Output contains NaN"
```

它检查输出形状与是否含 <abbr title="NaN / Not a Number · 非数值浮点标记，可能由无效运算产生。没有 NaN 不代表全部有限，也不代表输出与参考实现一致。">NaN</abbr>，没有比较参考输出。其他选中测试还检查 Q/K/V 梯度是否存在、Q 梯度是否不含 NaN，以及 prefill 和单 token decode 后的输出形状与缓存位置。源码没有完整检查所有梯度的有限性，不将其扩大表述为全部梯度均有限。

若要证明 FA3 与 SDPA 在给定条件下数值接近，需要兼容 GPU 上的对照：固定相同输入、精度、参数及初始缓存，分别运行两实现，按明确的绝对/相对误差容限比较输出；需要验证反向时再比较梯度。文件中的 `TestFA3VsSDPA` 已提供相关对照测试，但此次筛选没有运行该类；本次 CPU 通过不能替代这部分证据，也不证明任意输入下严格相等。

源码：[tests/test_attention_fallback.py](../../../tests/test_attention_fallback.py)。

## 6. 历史运行证据与本次核对边界

已有目录时间戳 `20261026T124516758971Z` 对应 2026-10-26 12:45:16 UTC，即北京时间 2026-10-26 20:45:16；它是历史运行，不是本报告日期的新运行。

教程入口命令由检查日志记录，工作目录为项目根目录：

```bash
python .agents/docs/career_roadmap/w1/scripts/d03_run_inference_tests.py --select 'SDPAOnly or OverrideMechanism'
```

结构化结果记录的实际子进程命令为：

```bash
/home/lizishuo/miniconda3/envs/nanochat/bin/python -m pytest -q tests/test_engine.py tests/test_attention_fallback.py -p no:cacheprovider -k 'SDPAOnly or OverrideMechanism'
```

运行设置为 `NANOCHAT_DTYPE=float32`、`CUDA_VISIBLE_DEVICES=''`，对应 CPU/FP32。结果为 **5 passed、1 skipped、17 deselected**，子进程退出码 **0**，状态 `PASS`。pytest 用时 6.26 秒，脚本记录子进程用时约 8.29 秒，计时范围不同。

五项通过对应 SDPA 基础前向、反向、缓存及 SDPA/auto override；一项跳过对应需要 FA3 的强制 override。17 项 deselected 是未被选择，包含 `TestFA3VsSDPA` 数值对照，不能算作通过或本次跳过。

- [d03_run.json](../results/w1/d03_pytest/20261026T124516758971Z/d03_run.json)：精确参数、工作目录、环境设置、子进程退出码与用时。
- [d03_stdout.log](../results/w1/d03_pytest/20261026T124516758971Z/d03_stdout.log)：测试摘要。
- [d03_stderr.log](../results/w1/d03_pytest/20261026T124516758971Z/d03_stderr.log)：本次读取为空。
- [d06_checks.log](d06_checks.log)：已有源码检查、测试调用及结果；原始文件含 VS Code 终端控制字符，本次原样归档。

已有证据支持本次筛选测试通过；未记录历史运行 commit，未验证 FA3 的 GPU 计算、两后端数值接近或 GPU 性能。本次仅核对回答、源码和日志，不将手算掩码或静态阅读当成新增数值实验。

## 7. 交流结论与下一步

第 1、2 题主干正确；第 3 题补全第二个 query 的可见集合和错误对齐后果；第 4 题补全末端 query 与窗口裁剪的安全依据；第 5 题纠正测试覆盖表述。补答已记录，不代表用户已经独立掌握。

可先自行复述为什么同一个 key 9 对 query 8 不可见、对 query 9 可见，再进入 [D07 基准审计](../../../.agents/docs/career_roadmap/w1/d07_benchmark_audit.md)。FA3 对照保留为 **PENDING_GPU**。
