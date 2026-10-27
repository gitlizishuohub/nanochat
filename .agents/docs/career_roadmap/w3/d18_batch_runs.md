---
source_files:
  - nanochat/engine.py
  - .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py
doc_type: tutorial
knowledge_role: supporting
knowledge_scope: W3第4日学习与实践
---

# W3 D4 · 批量生成与失败记录

[上一日](d17_single_batch_runs.md) · [下一日](d19_statistics_audit.md)。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[12.2 生成控制流与返回格式](../../llm-training-workflow/12-inference-and-generation.md#reading-12-2)；[5.3 batch、epoch 和每卡分配](../../llm-training-workflow/05-data-encoding-and-batching.md#reading-5-3)。
- **今天重点**：回看一次提示预填充后复制到多样本行的行为，结合有效输出数理解批量对总耗时、总吞吐及缓存容量的影响。
- **暂时跳过**：多卡通信和独立请求调度不属于今天的同提示批量实验；不由批量增大直接推断一定加速。

**读完能解释**：B=4 在这个 Engine 中代表什么？为什么不能直接称为四条独立请求的服务吞吐？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 今天学什么，做到哪里算完成

今天研究“总吞吐变高”和“每条生成更快”有什么区别，约 60–90 分钟。主线使用 CPU 随机模型，复用昨天 B1；已有 W2 D14 的 `experiments/inference_infra/results/raw/d14_p8_g4_b4_gate_cpu.jsonl` 也可直接核查，不改名、不重复测量。**不训练，不要求 checkpoint 或 GPU。**

关键任务是判断 B1/B4 是否可比较，并解释一条 B4 raw。失败也能提供边界证据，但不能当作成功性能样本；未选择的大矩阵与 GPU 分支不属于欠账。

## 1. 先手算一个吞吐反例

假设 B1 每行生成四个有效 token 用 40ms；B4 每行也四个，但整批用 80ms：

| B | 总有效 token | 整批耗时 | 总吞吐 | 一行拿齐四个输出的等待 |
|---|---|---|---|---|
| 1 | 4 | 40ms | `4/0.040=100 token/s` | 40ms |
| 4 | 16 | 80ms | `16/0.080=200 token/s` | 80ms |

总吞吐提高两倍，每行的完成等待也变长。<abbr title="Batch size / 批大小 · 此 Engine 中对应 num_samples：同一提示先预填一次，再复制缓存至 B 行生成。不是 B 个不同用户请求的调度实验。">B</abbr> 不会自动让单条更快；已经按 16 个有效 token 算出的吞吐不能再乘 4。以上是假设算例，不能写成实测加速。

## 2. 沿源码理解共享了什么、增加了什么

在 [Engine.generate](../../../../nanochat/engine.py) 中依次看：

```python
ids = torch.tensor([tokens], dtype=torch.long, device=device)
logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
logits = logits[:, -1, :].expand(num_samples, -1)
# 后续建立 batch_size=num_samples 的缓存，再调用：
# kv_cache_decode.prefill(kv_cache_prefill)
```

第一行只有一个提示行；`expand` 把首个预测分数扩成 B 行视图，随后缓存复制到多行，逐步前向才处理 B 个 token。因此 B4 首个等待可能包含更大的分配和复制，不能说“做了四次完整 prefill”。当前 temperature=0 且同一提示时，样本常常完全相同；这不是四种问题的回答能力测试。

缓存主项中 B 增大四倍，权重仍是同一套；总内存不按四倍增长。D16 的玩具配置中，复制阶段 B1 主项 20 KiB，B4 主项 56 KiB，也不是四倍。源码只说明潜在开销，单个 TTFT 无法告诉你分配/复制/计算各占多少，需要独立分段计时或性能 trace 才能归因。

## 3. 先检查可比条件，再决定要不要新测

保持 P=8/G=4、seed=42、fill ID=1、模型配置/初始化实现、CPU float32、实际后端、线程数与计时方法相同。原始 env 的 commit 不同需检查差异：纯文档提交不改变生成实现，但不能只因 tag 一样就忽略源码变更。session 不同应分别统计，比较不等于合并；背景负载与运行顺序仍可能干扰结果。

若已有 B1/B4 合格原始记录，本节运行可跳过；若 B1 环境不同，先说明不可直接比较，再选择匹配的一对，或以新 attempt 重测两组，不覆盖旧证据。以下命令只新测 B4，B1 命令见 D17：

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
NANOCHAT_DTYPE=float32 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py \
  --device cpu --random-model --prompt-tokens 8 --decode-tokens 4 \
  --batch-size 4 --warmup 3 --repeats 5 --seed 42 \
  --output experiments/inference_infra/results/raw/w3_cpu_attempt01/d18_p8_g4_b4_random_cpu.jsonl
D18_RC=$?
printf 'benchmark_exit=%s\n' "$D18_RC"
} > >(tee -a experiments/inference_infra/reports/d18_checks.log) 2>&1
```

使用已核对的 nanochat 解释器；当前终端自动激活时不重复激活。raw 追加写，已有目标改用新 attempt。命令只初始化小模型并前向生成，没有训练和下载。

## 4. 用每行计数判断是否真的做满工作量

从 [W2 包装器 measure](../w2/scripts/d12_benchmark.py) 的 `counts`、`completed`、`forced` 读起。每列返回 token 与 mask：普通采样 mask=1，工具返回等 forced token 的 mask=0。遇到 BOS/assistant_end 会标记完成；本包装器发现任一行完成就停止本次测量，以维持固定长度可比条件。Engine 自身的终止规则与包装器规则不能混为一谈。

完整 P8/G4/B4 raw 应满足：

```text
phase=measure, status=ok
actual_decode_steps=4
per_row_generated_tokens=[4,4,4,4]
generated_tokens=16
len(decode_step_ms)=3
```

假设实际每行有效数变为 `[4,4,2,4]`，总数最多为 14，不能拿名义 16 当分子。即使有四列，也不保证每列四个都属于有效输出；终止标记和 forced 输出有各自语义。本例不作为固定长度成功样本。

可以复用 D17 的只读打印代码，把路径换成 B4 原文件；再对照下面的短检查：

```bash
{
python - <<'PY'
import json
from pathlib import Path
p = Path('experiments/inference_infra/results/raw/w3_cpu_attempt01/d18_p8_g4_b4_random_cpu.jsonl')
for r in map(json.loads, p.read_text().splitlines()):
    if r.get('phase') != 'measure':
        continue
    print(r['run_id'], r['status'], r.get('per_row_generated_tokens'), r.get('generated_tokens'))
    if r['status'] == 'ok':
        print('TPOT=', r['tpot_ms'], 'total token/s=', r['tokens_per_s'])
PY
} > >(tee -a experiments/inference_infra/reports/d18_checks.log) 2>&1
```

TPOT 仍是一列输出的平均等待，不应除以 B 当作每行间隔。当前吞吐包含首个等待；同方法下比较 B1/B4，仍不能从几次 CPU 时间推出 GPU 或服务端收益。

## 5. 失败怎样提供有用信息

<abbr title="OOM / Out Of Memory · 内存不足。应从异常栈定位加载、缓存分配或计算阶段；无法定位时记录未知。失败不进入成功延迟统计。">OOM</abbr> 时先保留异常栈与 P/G/B，不凭“B4 比 B1 大”断言失败一定在 KV 分配。模型加载、临时激活也可能失败；CPU 的 CUDA allocated/reserved 字段为 null，不报告零显存。

`early_stop` 或 error 保留在 raw，D19 会记录排除原因；没有 raw 时查 stderr 的启动/导入/参数阶段。若探索 B2，另建文件且把真实 B=2 写入字段；不要覆盖 B4 的失败来制造成功。可选真实 checkpoint/GPU 试验需要先完成 D16 身份核对，与 CPU 主线独立，不要求本日执行。

## 本日脚本与产物清单

路径相对于项目根目录。下表只列实际选择分支的条件目标；复用 W2 原始文件保留旧编号。

| 脚本完整路径 | 用途与输入 | 条件 |
|---|---|---|
| [.agents/docs/career_roadmap/w2/scripts/d12_benchmark.py](../w2/scripts/d12_benchmark.py) | P8/G4/B4，逐次生成与计数 | 已核对 PyTorch 环境，CPU随机模型 |

| 产物完整路径 | 生成者 | 含义与条件 |
|---|---|---|
| `experiments/inference_infra/results/raw/w3_cpu_attempt01/d18_p8_g4_b4_random_cpu.jsonl` | d12_benchmark.py | 选择新测才生成；保留失败行，不代表全部成功 |
| `experiments/inference_infra/reports/d18_checks.log` | 局部命令组 tee | 输出、错误及立即捕获的退出码 |
| `experiments/inference_infra/reports/d18_baseline_batch.md` | 本人自愿整理 | 可选，聊天与自动结果可替代 |

## 本日自验证与交流

1. 从一对真实 B1/B4 记录判断是否可比；若不可比，指出一个决定性差异。

   **核对要点**：检查模型、P/G、输入、环境、精度/后端和计时；不能把 session 不同直接合成一组。

2. 用本日假设数字解释：吞吐提高两倍，为什么单行完成等待反而变长？

   **核对要点**：总产出与每行等待分别计算，不能把吞吐再次乘 B 或把 TPOT 除以 B。

3. “B4 首个等待较大，所以做了四次 prefill”，你如何用代码回应？还缺什么才知道各项耗时？

   **核对要点**：单行 prefill、扩展 logits、复制缓存；需要分段或 trace，不能只据总时间归因。

4. 一行提前结束时，包装器怎样处理？这能直接证明 cache 错误吗？

   **核对要点**：区分结束语义、有效计数和数值正确性；保存失败后另查原因，不补写固定长度成功。

## 完成核对

- [ ] 能核对一条 B4 raw，解释有效总数与步间隔。
- [ ] 能判断同条件比较，分别描述总吞吐与单行等待。
- [ ] 失败保留、可选扩展不欠账，没有预设批量加速。
