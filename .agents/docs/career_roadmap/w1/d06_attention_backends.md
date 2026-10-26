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

# W1 D6：解释 FA3 与 SDPA 的选择条件

今天要回答“实际用的是哪个后端，为什么”，而不是只看到文件名里有 Flash Attention 就认定正在调用 GPU 内核。预计 60–90 分钟：20 分钟路由，20 分钟掩码，15 分钟定向测试，20 分钟核验后端条件，10 分钟口述。

前置：[D5 缓存生命周期](d05_cache_lifecycle.md)。不用下载内核；CPU 只验证 SDPA 与相关契约，FA3 留待 GPU 条件齐备。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[6.2 注意力方向、窗口与RoPE](../../llm-training-workflow/06-forward-pass-and-prediction-loss.md#reading-6-2)；[8.1 显式混合精度与梯度缩放](../../llm-training-workflow/08-precision-and-training-acceleration.md#reading-8-1)。
- **今天重点**：带着昨天的缓存路径回读因果掩码和滑动窗口；精度小节只区分参数、计算及损失的数值类型。结合当天源码确认后端条件。
- **暂时跳过**：<abbr title="FP8 · 8 位浮点计算；本项目的可选训练加速方式，需要兼容硬件。">FP8</abbr> 和缩放器的实现细节暂不展开，8.2 留到 D17 按设备条件选读。

**读完能解释**：为什么 CPU 上的注意力回退测试通过，仍不能证明 GPU 内核通过？精度会影响哪些选择条件？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 当日脚本与产物清单

所有项目内路径均以 nanochat 项目根目录为基准；外部 MiniMind 只读历史证据不改名。`dNN_` 表示全程首次使用或产生日，跨日复用不改前缀；上游源码和测试保留原生名称。

| 角色与脚本 | 用途与执行阶段 | 输入路径或来源 | 运行条件 |
|---|---|---|---|
| 跨日复用：[`.agents/docs/career_roadmap/w1/scripts/d03_run_inference_tests.py`](scripts/d03_run_inference_tests.py) | 第 4 节 `--select` 验证 SDPA 与 override | `tests/test_engine.py`、`tests/test_attention_fallback.py`；按表达式筛选 | 项目/dev 依赖齐备；CPU/FP32；不需 GPU；保留 d03 脚本名 |

| 完整项目相对路径 | 用途 | 生成者 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_stdout.log` | 本次筛选测试结果 | 复用 `d03_run_inference_tests.py` | 自动；使用新时间戳，保留首次日 d03 文件名 |
| `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_stderr.log` | 本次筛选测试诊断 | 同上 | 自动，可为空 |
| `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_run.json` | 本次筛选表达式、退出码与用时 | 同上 | 条件生成：子进程正常返回后写入 |
| `experiments/inference_infra/reports/d06_attention_backends.md` | 后端条件表、掩码和覆盖边界 | 第 5 节问答可选整理 | 可选人工整理；直接聊天回答即可，不影响验收 |
| `experiments/inference_infra/reports/d06_checks.log` | 本日终端检查输出与错误 | 各 Bash 代码块的 tee | 自动追加各代码块内的输出；命令及退出码在交流时补充 |

清单是执行后的预期，不表示文件已经存在。已有未加前缀的历史实验文件保留，新执行按上述路径写入。 `<UTC时间戳>` 由脚本按协调世界时生成，格式为 `YYYYmmddTHHMMSSffffffZ`；以终端 `Logs:` 打印的真实目录为准。

正文的 `sed`、`rg`、`head`、解释器检查等只向终端打印，不会自动生成独立结果文件；各 Bash 代码块内的输出追加保存到 `experiments/inference_infra/reports/d06_checks.log`；交流时附所用命令、退出码和关键结果，自动结果已有的字段无需手抄。 测试输出由清单中的 stdout/stderr 日志捕获。 未执行或失败时按真实状态记录，不能把旧同名文件当作本次成功。

## 1. 按源码分开“可加载”与“最终使用”

日志只在 `{ ...; }` 命令组执行期间记录；<abbr title="Bash process substitution · 进程替换。这里把命令组的输出交给 tee，同时显示并追加到日志；重定向在命令组结束后恢复。">进程替换</abbr>不会持续接管交互终端，因此提示符和 VS Code 终端集成控制字符不会混入日志。命令组在当前 shell 执行，变量设置会保留；代码块返回最后一条命令的退出码，需核对单条命令时立即记录 `$?`。已有的显式文件重定向仍写入各自文件。复制时包含外层花括号和末尾重定向。若之前执行过旧版全局日志命令，先关闭旧终端并新开终端；仅修改文档不会停止旧终端的记录。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
sed -n '1,100p' nanochat/flash_attention.py
sed -n '10,36p' nanochat/common.py
} > >(tee -a experiments/inference_infra/reports/d06_checks.log) 2>&1
```

<abbr title="FA3 / Flash Attention 3 · 本项目通过 kernels 包加载的 GPU 注意力实现。能否加载由硬件、内核获取和运行依赖共同决定；模块里定义了接口并不说明本次正在使用它。">FA3</abbr> 的探测在模块导入时执行。无 CUDA 则返回 None；有 CUDA 时依据设备能力尝试不同 kernel 包，异常返回 None。不要把源码中的硬件注释当作你手里那张卡已验证的支持表。

`HAS_FA3` 表示加载到了接口；`USE_FA3` 是经过 override 和 compute dtype 判断后实际选择的分支。自动模式下，即使 HAS 为 true，也只有 `COMPUTE_DTYPE == torch.bfloat16` 才选 FA3。<abbr title="bf16 / bfloat16 · 16 位浮点格式。这里是自动选择 FA3 的条件之一；它不是 GPU 存在的充分条件，也不代表所有 dtype 分支都测过。">bf16</abbr> 要结合当前路由理解。

对照下面条件表核验最后一列的真实证据状态，可直接在聊天中说明：

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
{
sed -n '98,230p' nanochat/flash_attention.py
} > >(tee -a experiments/inference_infra/reports/d06_checks.log) 2>&1
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

使用[测试捕获脚本](scripts/d03_run_inference_tests.py)：

```bash
{
python .agents/docs/career_roadmap/w1/scripts/d03_run_inference_tests.py --select 'SDPAOnly or OverrideMechanism'
} > >(tee -a experiments/inference_infra/reports/d06_checks.log) 2>&1
```

保留新时间戳目录，不能覆盖 D3。阅读测试时核对每个断言：

```bash
{
rg -n 'class TestSDPAOnly|class TestOverrideMechanism|def test_|skipif' tests/test_attention_fallback.py
} > >(tee -a experiments/inference_infra/reports/d06_checks.log) 2>&1
```

基础前向检查 shape 与 NaN；反向检查梯度存在且有限；KV 用例检查 prefill 和单 token decode 的位置。没有任何一条 CPU 通过能证明 FA3 输出与 SDPA 一致，带 FA3 条件的对照用例仍需 GPU。

若前文 D4/D5 已执行，可引用 `d04_shapes_probe.json` 中的 `USE_FA3=false` 佐证随机模型探查走回退。仅脚本 `--help` 成功不足以证明后端计算通过。

## 5. 当天自验证与交流问题及边界

按题号在聊天中回答即可，无需写报告。概念题说明理由或推导；实测题引用当天日志或 JSON 中支撑结论的片段，运行命令、退出码和证据路径提供一次即可。未执行的实测项标 `PENDING`；引用历史结果时注明日期与版本，不必为回答问题额外重跑实验。

1. **判断实际后端**：为什么 HAS_FA3=true 仍可能 USE_FA3=false？用自动模式和强制 override 的逻辑解释本次选择，哪些条件只做了静态推导？

   **核对要点**：根据可加载性、compute dtype 与 override 的优先级判断；区分模块导入时选择和测试中显式重算，不由文件名推断实际分支。

2. **对齐张量布局**：用 D4 的 Hq=4/Hkv=2 解释：外部注意力接口怎样转为 SDPA 内部布局，再怎样返回？在哪一步需要 GQA？

   **核对要点**：轴的含义与 transpose、头数判定对应；不把缓存存储布局和 SDPA 运算布局混写。

3. **手算 chunk 掩码**：Tk=10、Tq=2、左窗口 W=2，两个 query 的绝对位置和可访问 key 各是什么？若把 query 当作位置0、1，会算错什么？

   **核对要点**：给出两个可见集合及因果/窗口约束的推导，能解释 Tk−Tq 的偏移来源，不直接套用方阵三角掩码。

4. **解释单 token 特例**：Tk=10、Tq=1、W=2 时，代码截取哪些 K/V，为什么可以 is_causal=False？与上一题显式掩码相比，安全依据是什么？

   **核对要点**：说明当前 query 在有效历史末端，并算清窗口是否包含当前位置；不把 is_causal=False 理解为可以访问未来。

5. **辨认实测覆盖**：从本次定向测试选一个已通过断言，说明它检查了什么；若要声称 FA3 与 SDPA 等价，现有证据还缺什么？

   **核对要点**：引用实际结果与断言，区分 shape/NaN/缓存操作和两实现数值对照；skip 不计通过，未来对照应固定输入、精度与比较对象。

失败时先判断：导入依赖缺失 → 环境；FA3 override 断言 → 当前后端不可用；dtype 不匹配 → 检查导入前环境；CPU 用例数值失败 → 保存最早失败和 shape，不换后端掩盖问题。

当天验收：能解释 HAS/USE；手算 chunk mask；指出 `Tq=1` 的 `W+1`；回答明确 CPU 已覆盖与 GPU 未覆盖。交流附实际日期、commit、命令、日志路径、状态与下一步；用时可选。

上一篇：[D5 缓存生命周期](d05_cache_lifecycle.md)。下一篇：[D7 基准审计](d07_benchmark_audit.md)。
