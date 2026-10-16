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

# W1 D6：画出 FA3 与 SDPA 的选择条件

今天要回答“实际用的是哪个后端，为什么”，而不是只看到文件名里有 Flash Attention 就认定正在调用 GPU 内核。预计 60–90 分钟：20 分钟路由，20 分钟掩码，15 分钟定向测试，20 分钟后端条件表，10 分钟口述。

前置：[D5 缓存生命周期](D5_cache_lifecycle.md)。不用下载内核；CPU 只验证 SDPA 与相关契约，FA3 留待 GPU 条件齐备。

## 1. 按源码分开“可加载”与“最终使用”

```bash
cd /home/lizishuo/_program_/Project/nanochat
source /home/lizishuo/miniconda3/etc/profile.d/conda.sh
conda activate nanochat
sed -n '1,100p' nanochat/flash_attention.py
sed -n '10,36p' nanochat/common.py
```

<abbr title="FA3 / Flash Attention 3 · 本项目通过 kernels 包加载的 GPU 注意力实现。能否加载由硬件、内核获取和运行依赖共同决定；模块里定义了接口并不说明本次正在使用它。">FA3</abbr> 的探测在模块导入时执行。无 CUDA 则返回 None；有 CUDA 时依据设备能力尝试不同 kernel 包，异常返回 None。不要把源码中的硬件注释当作你手里那张卡已验证的支持表。

`HAS_FA3` 表示加载到了接口；`USE_FA3` 是经过 override 和 compute dtype 判断后实际选择的分支。自动模式下，即使 HAS 为 true，也只有 `COMPUTE_DTYPE == torch.bfloat16` 才选 FA3。<abbr title="bf16 / bfloat16 · 16 位浮点格式。这里是自动选择 FA3 的条件之一；它不是 GPU 存在的充分条件，也不代表所有 dtype 分支都测过。">bf16</abbr> 要结合当前路由理解。

把下面表复制进报告，最后一列填真实证据状态：

| 条件 | HAS_FA3 | 自动 USE_FA3 | 本机证据 |
|---|---|---|---|
| 无 CUDA | false | false | CPU 进程可核验 |
| 有 CUDA，kernel 加载失败 | false | false | 未在 CPU 核验 |
| 可加载且 compute 为 bf16 | true | true | PENDING_GPU |
| 可加载且 compute 为 FP32/FP16 | true | false | 静态路由已核对；运行待 GPU |
| override 为 sdpa | 任意 | false | CPU 测试可核验 |
| override 为 fa3 | 必须 true | true | 只检查可用性，不代表任意 dtype 安全 |

override 是测试机制；不建议在日常实验中随意修改模块私有变量。测试文件的 `set_impl()` 会重新计算 USE_FA3；生产模块不是每个调用都自动重算。

## 2. 从外部布局读到 SDPA 内部布局

```bash
sed -n '98,230p' nanochat/flash_attention.py
```

外部统一接口用 `[B,T,H,D]`，回退先 transpose 到 `[B,H,T,D]`，调用 PyTorch，再转回来。<abbr title="SDPA / Scaled Dot Product Attention · PyTorch 的缩放点积注意力接口。nanochat 用它实现回退，并显式处理滑动窗口和缓存的因果对齐。">SDPA</abbr> 的内部布局与 K/V 缓存存储布局不同；不要为了“统一图”把它们混写。

`enable_gqa` 由 Q 与 K 的 head 数是否不同决定。找到它并用 D4 的 Hq=4/Hkv=2 解释为什么为 true。

## 3. 分三种输入解释掩码

<abbr title="causal mask · 因果掩码。只允许当前位置访问自己和之前的位置，防止看到尚未生成的未来 token。缓存场景的查询位置通常偏移到历史末尾，不能简单把不同长度矩阵当成从零对齐。">因果掩码</abbr> 与滑动窗口都在 `_sdpa_attention` 中决定：

1. `Tq==Tk` 且窗口覆盖本次上下文：直接使用 `is_causal=True`。
2. `Tq==1`：当前 query 位于全部 K/V 的末端，因此无未来 K；使用 `is_causal=False`。如左窗口为 W，只取最近 `W+1` 个 K/V，包括当前位置。
3. 其余情况，包括 chunk inference：构造 bool mask；query 行位置从 `Tk-Tq` 起算，条件为 `col<=row`；有限窗口还要求 `row-col<=W`。

手算例子：`Tk=10,Tq=2` 的 query 位置是 8、9。位置 8 不能看 key 9；位置 9 可以看。若 W=2，位置 8 可看 6、7、8。这一步能防止把矩形矩阵的上三角/下三角对齐想错。

`flash_attn_with_kvcache` 的 SDPA 分支读取 `cache_seqlens[0]`，假设批内一致位置；先把新 K/V 写到 `[pos:pos+T_new]`，再取有效前缀。它不推进长度，推进由模型最后一层做。

## 4. 只运行对应 CPU 测试并看跳过

使用[测试捕获脚本](scripts/run_inference_tests.py)：

```bash
python .agents/docs/career_roadmap/w1/scripts/run_inference_tests.py --select 'SDPAOnly or OverrideMechanism'
```

保留新时间戳目录，不能覆盖 D3。阅读测试时核对每个断言：

```bash
rg -n 'class TestSDPAOnly|class TestOverrideMechanism|def test_|skipif' tests/test_attention_fallback.py
```

基础前向检查 shape 与 NaN；反向检查梯度存在且有限；KV 用例检查 prefill 和单 token decode 的位置。没有任何一条 CPU 通过能证明 FA3 输出与 SDPA 一致，带 FA3 条件的对照用例仍需 GPU。

若前文 D4/D5 已执行，可引用 `shapes_probe.json` 中的 `USE_FA3=false` 佐证随机模型探查走回退。仅脚本 `--help` 成功不足以证明后端计算通过。

## 5. 写后端报告与边界

在 `experiments/inference_infra/reports/attention_backends.md` 保存：条件表、两种布局、三分支掩码、批内等长假设、测试原始日志、无法 CPU 验证的路径。

```markdown
# 后端路由
- 实际日期 / commit / Python / NANOCHAT_DTYPE：
- 本次选中后端与证据：
- HAS_FA3 与 USE_FA3 的差别：
- 外部布局 / SDPA 内部布局：
- Tq==Tk、Tq==1、chunk 三种处理：
- 选定测试摘要与日志：
- GPU专属未验证项：
- 后续GPU会话需补哪些对照：
```

失败时先判断：导入依赖缺失 → 环境；FA3 override 断言 → 当前后端不可用；dtype 不匹配 → 检查导入前环境；CPU 用例数值失败 → 保存最早失败和 shape，不换后端掩盖问题。

当天验收：能解释 HAS/USE；手算 chunk mask；指出 `Tq=1` 的 `W+1`；报告明确 CPU 已覆盖与 GPU 未覆盖。追加真实日期、commit、用时、命令、日志、状态及下一步。

上一篇：[D5 缓存生命周期](D5_cache_lifecycle.md)。下一篇：[D7 基准审计](D7_benchmark_audit.md)。
