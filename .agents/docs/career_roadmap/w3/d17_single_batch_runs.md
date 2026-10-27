---
source_files:
  - nanochat/engine.py
  - .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py
doc_type: tutorial
knowledge_role: supporting
knowledge_scope: W3第3日学习与实践
---

# W3 D3 · 单样本生成与测量结果核查

[上一日](d16_baseline_matrix.md) · [下一日](d18_batch_runs.md)。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[8.1 显式混合精度与梯度缩放](../../llm-training-workflow/08-precision-and-training-acceleration.md#reading-8-1)；[8.3 编译与动态输入](../../llm-training-workflow/08-precision-and-training-acceleration.md#reading-8-3)。
- **今天重点**：结合今天的真实设备和 dtype，区分精度选择、编译准备与正式测量。仅在需要理解 <abbr title="FP8 · 8 位浮点计算；本项目的可选训练加速方式，需要兼容硬件。">FP8</abbr> 模型来源时选读 8.2 的实现限制。
- **暂时跳过**：没有对应 GPU 时不做 FP8 实测；本日不要求理解动态量化或优化内核。

**读完能解释**：编译或预热耗时与正式生成延迟怎样区分？CPU 实测能支持哪些结论，哪些仍需 GPU？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 今天学什么，做到哪里算完成

今天用一个 B=1 的小案例，学会从源码和 raw 判断“一次测量做了什么”，约 60–90 分钟。**CPU 随机小模型即可；初始化和前向生成不会训练模型，不需要 checkpoint 或外部分词器文件。**

已完成 W2 D13 的 `experiments/inference_infra/results/raw/d13_p8_g4_b1_random_cpu.jsonl`，或 D14 的 B1 gate 记录时，可以直接复用原文件，不改编号、不重复测量。若输入或环境不同，先按下文检查；没有 raw 再执行第 3 节。此前单测已经通过且源码未改，无需为本日再跑整套测试。

## 1. 先在纸面算清一条有效记录

固定 P=8/G=4/B=1，无提前结束或工具输出。模型先处理 8 个提示位置产生第一个预测；再分别输入前三个生成 token，得到后续三个预测。输出四列只有三个后续间隔，不能把 G 当成间隔数。

假设首个输出等 30ms，后续等 8、10、12ms：

| 量 | 计算 | 解释 |
|---|---|---|
| TTFT | 30ms | 第一次 next 前到首个 yield 返回 |
| TPOT | `(8+10+12)/3=10ms` | 后续输出间隔的算术平均 |
| elapsed | `30+8+10+12=60ms` | 包含首个等待的总时间 |
| tokens/s | `4/0.060≈66.67` | 有效总输出除以同一段总时间 |

这是教学算例，不是速度目标。<abbr title="TTFT / Time To First Token · 首 token 等待；此处包括 Engine 的分配、prefill、缓存复制和首个采样，不包含模型加载和文本编码。">TTFT</abbr> 不等于纯 prefill 算子时间；<abbr title="TPOT / Time Per Output Token · 本包装器每次运行取首个输出之后的间隔均值；G=1 时没有后续间隔，记空值。">TPOT</abbr> 在 G=1 时应为空，不能用 0 表示“瞬间生成”。

## 2. 带读真正决定边界的四处代码

对照 [d12_benchmark.py](../w2/scripts/d12_benchmark.py) 的 `load`、`measure`、`main`：

1. `load` 的 `--random-model` 分支建立 L=2、C=64、词表 512 的模型，调用 `init_weights()` 和 `eval()`；随后在 `torch.inference_mode()` 中测量，没有 loss/backward/optimizer.step。`ContractTokenizer` 是脚本内合成 ID 映射，不读取训练词表。
2. 输入构造为 `[BOS]+[1]×(P−1)`，控制 token 数和计算形状；不是中文文本的质量测试。模型加载、指纹和环境检查在计时外。
3. `measure` 在首次 `next(stream)` 前记起点，每次返回后记录时间；CPU 的 `sync` 是空操作，CUDA 才同步。计时含 Python 控制和逐步处理开销，极小模型上这些开销可能明显。
4. 只取至多 G 次 yield，最后 `close()`；不会为探测 StopIteration 再恢复生成器。结合 [Engine.generate](../../../../nanochat/engine.py) 的 `yield` 后前向位置，可以解释为何这能避免一次无用的最终前向。

因此本日核查的是一个明确包装器的生成与计时契约。`model_tag` 的旧默认字符串在随机分支也可能记录到 raw；判断模型来源应看 `model_kind=random` 与配置，不能据此声称加载了训练权重。

## 3. 有合格记录就复用，没有才运行

从项目根执行，核对当前自动激活的解释器。以下命令固定 CPU float32 和线程数，避免后台线程策略变成额外变量。若已有相同目标，换成新 attempt 目录，因 raw 是追加写，不能把两次 session 意外堆进一个“八行结果”。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
python -c "import sys,torch; print(sys.executable); print(torch.__version__)"
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py \
  --help
} > >(tee -a experiments/inference_infra/reports/d17_checks.log) 2>&1
```

`import torch` 失败就先回 D02 核对解释器，不误称为 CPU 验证通过。当前本机 nanochat 解释器应为 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`；换机按实际路径核对，不另建环境。

```bash
{
NANOCHAT_DTYPE=float32 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py \
  --device cpu --random-model --prompt-tokens 8 --decode-tokens 4 \
  --batch-size 1 --warmup 3 --repeats 5 --seed 42 \
  --output experiments/inference_infra/results/raw/w3_cpu_attempt01/d17_p8_g4_b1_random_cpu.jsonl
D17_RC=$?
printf 'benchmark_exit=%s\n' "$D17_RC"
} > >(tee -a experiments/inference_infra/reports/d17_checks.log) 2>&1
```

正常情况有 3 条 `warmup` 和 5 条 `measure`；预热不参与正式统计，用于减轻首次初始化等干扰，不能保证机器完全稳定。这里没有显式 `torch.compile`，不能把所有预热都称作“编译”。

## 4. 读结果并交叉验证

下面只读文件，不运行模型。复用 W2 结果时，把 `raw` 改为原路径；不要复制改名冒充 D17 新执行。

```bash
{
python - <<'PY'
import json
from collections import Counter
from pathlib import Path
raw = Path('experiments/inference_infra/results/raw/w3_cpu_attempt01/d17_p8_g4_b1_random_cpu.jsonl')
rows = [json.loads(s) for s in raw.read_text().splitlines() if s.strip()]
print(Counter((r.get('session_id'), r.get('phase'), r.get('status')) for r in rows))
row = next((r for r in rows if r.get('phase') == 'measure' and r.get('status') == 'ok'), None)
print(json.dumps(row if row is not None else rows[-1], ensure_ascii=False, indent=2))
PY
} > >(tee -a experiments/inference_infra/reports/d17_checks.log) 2>&1
```

从一条正式行核对：`actual_decode_steps=4`、`len(decode_step_ms)=3`、`per_row_generated_tokens=[4]`、`generated_tokens=4`；再复算 elapsed、TPOT 和吞吐，时间须为正且有限。核对 CPU、random、实际 precision/backend、model_config 和 env；CUDA 显存字段为 null 表示不适用，不是“零显存”。D19 会自动做更完整的验证。

若没有有效行，上面会打印最后一条以便看失败。`early_stop` 表示未满足固定长度契约；BOS/assistant_end、工具 forced token 等可能影响计数，须看原始状态，不能补写成四个。退出 2 可能是提前结束，也可能是 argparse 参数错误；后者可能没有 raw，须合看 stderr。加载或运行异常通常退出 1、保留 error 行；进程启动/导入失败也可能只留下日志。修复后用新路径，保留原记录。

## 可选扩展

已有匹配训练权重且希望测量时，沿 D16 的身份核对后再设计独立实验。GPU 未选择不登记待补。CPU 随机模型即使八行全部成功，也不证明语言能力、GPU 性能或缓存相对无缓存的加速比；本日没有执行那个对照。

## 本日脚本与产物清单

路径相对于项目根目录。脚本保留首次使用日编号，历史文件不改名；表中目标只在实际运行时生成。

| 脚本完整路径 | 用途与输入 | 条件 |
|---|---|---|
| [.agents/docs/career_roadmap/w2/scripts/d12_benchmark.py](../w2/scripts/d12_benchmark.py) | 固定 P8/G4/B1；随机小模型生成与 raw | 已有可导入 torch 的环境；CPU，无训练 |

| 产物完整路径 | 生成者 | 含义与条件 |
|---|---|---|
| `experiments/inference_infra/results/raw/w3_cpu_attempt01/d17_p8_g4_b1_random_cpu.jsonl` | d12_benchmark.py | 新测才生成；复用 W2 时保留 W2 原路径 |
| `experiments/inference_infra/reports/d17_checks.log` | 本日各命令组 tee | 局部 stdout/stderr、退出码；不持续接管终端 |
| `experiments/inference_infra/reports/d17_baseline_b1.md` | 本人自愿整理 | 可选，聊天与自动记录可替代 |

## 本日自验证与交流

用同一份真实 raw 回答；没有执行条件时概念题可标明假设，实际操作状态如实说明，不强制另写报告。

1. 从一条正式行复算三种指标。为什么 G=4 对应三个间隔？

   **核对要点**：首个预测来自 prefill；计数、单位和时间范围对应，实测不要求等于教学数值。

2. 你这份文件有几个 session、多少预热与有效正式行？是否达到本日方法约定？

   **核对要点**：按身份分开计数，不能把两次不足五行的 session 直接凑成一个；不能只看进程提示。

3. 若日志只有参数错误而没有 raw，应先重新训练模型吗？下一步最小检查是什么？

   **核对要点**：先查解释器、帮助与参数，区分发生阶段；无 raw 不等于缓存算法失败。

4. 现在能否把结果写为“实现了 GPU 推理加速”？请改成证据能支持的一句描述。

   **核对要点**：说明本人动作、CPU/random 范围及计时核查，不能越过设备与对照边界。

## 完成核对

- [ ] 能解释源码的起止边界，并交叉核对一条记录。
- [ ] 已有合格结果被复用；新测或失败有真实文件与退出码。
- [ ] 主线无训练/GPU前置，未选择扩展不留欠账。
