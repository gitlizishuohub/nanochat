---
source_files:
  - .agents/docs/career_roadmap/w3/scripts/d13_aggregate.py
  - .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py
doc_type: tutorial
knowledge_role: supporting
knowledge_scope: W3第5日学习与实践
---

# W3 D5 · 逐行审计和统计复算

[上一日](d18_batch_runs.md) · [下一日](d20_reproduction_entry.md)。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[13.1 四种评价出口](../../llm-training-workflow/13-evaluation-and-metrics.md#reading-13-1)；[13.2 能力与系统性能的边界](../../llm-training-workflow/13-evaluation-and-metrics.md#reading-13-2)。
- **今天重点**：完整认识四种能力评价出口，再回到当天统计复算，明确损失、任务正确率、推理延迟和吞吐各自的统计对象。
- **暂时跳过**：能力指标公式只需能解释分子、分母与范围；不必下载全部评测任务或运行能力评测。

**读完能解释**：<abbr title="BPB / Bits per Byte · 每字节比特数；用有效目标负对数似然和字节数衡量预测误差。">BPB</abbr>、任务准确率与 <abbr title="TPOT / Time per Output Token · 首词元之后的平均输出间隔；具体统计以当日协议为准。">TPOT</abbr> 分别评价什么？为什么一个指标变好不代表另外两个也变好？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 今天学什么，做到哪里算完成

今天从逐次 raw 得到可反查的统计结论，约 60–90 分钟。使用 D17/D18 或 W2 已有 CPU 随机模型记录，**只读数据，不训练、不要求 GPU 或 checkpoint**。聚合只需 Python 标准库。尚无 raw 时先完成下面的纸面统计，再返回 D17 的小实验；不能拿计划矩阵生成测量结果。

## 1. 先分清两个统计层级

一次运行 G=4，首个输出等 10ms，后续间隔 `[2,3,4]ms`：elapsed=19ms，TPOT=3ms，B1 整体吞吐约 `4/0.019=210.53 token/s`。先算每次运行的指标，再对同组正式运行统计。

假设五次运行各自的 TPOT 排序为 `[2,3,3,4,8]ms`，中位数为 3ms，min=2ms，max=8ms。<abbr title="P95 · 第95百分位；本脚本采用 nearest-rank：排序后取第 ceil(0.95×n) 个数，Python 下标还需减1。">P95</abbr> 取第 `ceil(0.95×5)=5` 个，即 8ms。五个样本的 P95 就是最大值，只描述这五次，不能当成稳定尾延迟或服务承诺。以上都是教学数值，不是本机成绩。

把所有运行的 decode 间隔倒在一起求中位数，是“间隔”的统计；每 run 先取均值再求中位数，是“运行平均间隔”的统计。即使偶然算得相同，也不是同一定义。当前脚本采用后者；也不通过“总有效数/总时间”代替逐 run 吞吐的中位数。

## 2. 带读：从一条 raw 到一个统计组

打开[聚合脚本](scripts/d13_aggregate.py)，按以下执行顺序读：

1. `glob` 只读取指定目录的 `dNN_p*_g*_b*.jsonl`，兼容旧 `p*_g*_b*.jsonl`，不递归；`d16_matrix.jsonl` 不匹配。每条 run_id 必须存在且全输入唯一，坏 JSON 或重复 ID 会中止，不能保证总有完整 audit 输出。
2. 组键 `KEYS` 包括 session、设备/模型类型、来源/tag/step、指纹、seed、输入、精度、后端、P/G/B，再加入完整 env 和 model_config。形状相同并不自动合并；不同 session 分开统计。
3. 预热行单独登记，不计入 `n_total` 与 audit；正式行交给 `metrics`。它检查 schema、状态、上下文、G 列、G−1 个间隔、B 行均有效输出 G 个 token，以及正且有限的时间。
4. `elapsed≈TTFT+sum(decode_step_ms)` 使用相对 3% 或绝对 0.5ms 的容差。通过后从间隔重算 TPOT、从有效数和 elapsed 重算吞吐，不盲信 raw 中已存的汇总字段。

```python
valid = [values for _, values, _ in entries if values is not None]
# 然后分别对每 run 的指标做 median、p95、min、max
# 非预热行：n_total = n_valid + n_excluded
```

这里是结构核验，不是对原始执行真实性的独立证明。脚本没有重新执行 Engine，也没有保存并核查全部 token/mask；不能把它描述成“证明任何 raw 都真实”。`timing_scope`、`batch_semantics` 等也没有全部加入组键，仍需人工核对，不能只看 CSV 标签。

## 3. 选择明确的输入范围再聚合

D17/D18 新测的默认目录是 `experiments/inference_infra/results/raw/w3_cpu_attempt01`。复用 W2 时，将下方 `--raw-dir` 改成真实目录；若为上级 raw，脚本会读该目录全部匹配文件而非只读两份，须核对所有分组。也可将明确选择的原始文件复制到全新隔离输入目录，保留原名与内容，登记原来源，避免重复副本；不要拼接整个目录树或改 run_id。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
python .agents/docs/career_roadmap/w3/scripts/d13_aggregate.py \
  --raw-dir experiments/inference_infra/results/raw/w3_cpu_attempt01 \
  --output experiments/inference_infra/results/summary/d19_baseline.csv \
  --audit experiments/inference_infra/results/summary/d19_baseline_audit.jsonl
D19_RC=$?
printf 'aggregate_exit=%s\n' "$D19_RC"
} > >(tee -a experiments/inference_infra/reports/d19_checks.log) 2>&1
```

它不改 raw，但会重写指定 CSV/audit；需要保留旧汇总就换新输出名。只有预热、无匹配文件等会报错。存在全失败组时可生成 `NO_VALID_SAMPLES` 行，这比悄悄丢弃失败更有用。CPU 的 CUDA 显存统计为空，不填 0。

## 4. 先读样本数，再读结论标签

| 字段或状态 | 应怎样解释 |
|---|---|
| n_total / n_valid / n_excluded | 非预热总数、有效数、排除数；核对相加关系 |
| warmup_status=OK | 同组有效正式样本所用 session 至少三次预热且全部状态/schema 合格；不是对每个预热完整复算 metrics |
| CPU_OR_RANDOM_CONTRACT | 有有效 CPU 或随机模型记录，不能升级为 GPU 权重基线；**标签本身不保证满五次或预热充分** |
| GPU_BASELINE | CUDA checkpoint 组满足至少五次有效正式记录及预热门槛；仍不证明能力或优化收益 |
| INSUFFICIENT_WARMUP / INSUFFICIENT_SAMPLES | CUDA checkpoint 组相应门槛不足；不是零延迟 |
| NO_VALID_SAMPLES | 没有有效正式样本，应查失败而非填写性能数字 |

主线约定至少 3 次成功预热、5 次有效正式运行，也要对 CPU 组显式检查这两个字段。不是所有脚本状态都替你执行了全部验收条件。

## 5. 从 CSV 反查一条真正的证据

下面读 CSV 的计数，再从 audit 选择一个纳入的行，按它的绝对路径和行号回读 raw；不运行模型：

```bash
{
python - <<'PY'
import csv, json
from pathlib import Path
summary = Path('experiments/inference_infra/results/summary/d19_baseline.csv')
for r in csv.DictReader(summary.open()):
    print({k:r[k] for k in ('session_id','batch_size','n_total','n_valid','n_excluded','warmup_status','conclusion_status')})
audit = Path('experiments/inference_infra/results/summary/d19_baseline_audit.jsonl')
items = [json.loads(s) for s in audit.read_text().splitlines()]
a = next((x for x in items if x['included']), None)
if a:
    r = json.loads(Path(a['file']).read_text().splitlines()[a['line']-1])
    assert r['run_id'] == a['run_id']
    print('provenance:', a)
    print('recomputed:', r['ttft_ms'], sum(r['decode_step_ms'])/len(r['decode_step_ms']) if r['decode_step_ms'] else None,
          r['generated_tokens']*1000/r['elapsed_ms'])
else:
    print('No included row; inspect exclusion reasons:', items)
PY
} > >(tee -a experiments/inference_infra/reports/d19_checks.log) 2>&1
```

这完成“结论 → 组 → audit → raw”的反查。audit 的路径依赖当前机器；交给别人时需要保留 raw 并说明原路径映射，或在新位置重新聚合。不要只交 CSV 后声称能独立重建。

重复 run_id 时查是否重复复制；坏 JSON 时保留原件和行号排查写入中断；不足五次就如实报告，不删失败、不改 ID 凑样本。无需为回答问题故意制造错误。

## 本日脚本与产物清单

路径相对于项目根目录，只有选择并成功运行相应命令才生成；失败组可以被保留在汇总内。

| 脚本完整路径 | 用途与输入 | 条件 |
|---|---|---|
| [.agents/docs/career_roadmap/w3/scripts/d13_aggregate.py](scripts/d13_aggregate.py) | 从匹配 raw 复算统计与审计 | Python 标准库，至少存在非预热记录；不读取矩阵 |

| 产物完整路径 | 生成者 | 含义与条件 |
|---|---|---|
| `experiments/inference_infra/results/summary/d19_baseline.csv` | 聚合脚本 --output | 有测量组时生成，可包含无有效样本组；会重写目标 |
| `experiments/inference_infra/results/summary/d19_baseline_audit.jsonl` | 聚合脚本 --audit | 非预热行纳入/排除及来源；预热仍查 raw |
| `experiments/inference_infra/reports/d19_checks.log` | 本日局部命令组 tee | 命令输出、错误与退出码 |
| `experiments/inference_infra/reports/d19_statistics_audit.md` | 本人自愿整理 | 可选复算笔记，聊天可替代 |

## 本日自验证与交流

1. 五次 TPOT `[2,3,3,4,8]ms` 的中位数/P95 是多少？把 8ms 写成“95% 服务保障”缺少什么？

   **核对要点**：给出排序与排名，说明样本很少、非服务请求分布，不把经验分位数当承诺。

2. 任选实际组，核对总数、有效数、排除数，并反查一个 run_id；为什么预热不在 audit 中？

   **核对要点**：沿脚本分支与原始行定位；全成功就查纳入原因，无需造失败。

3. 如果 CPU_OR_RANDOM_CONTRACT 组只有两次有效正式行，你会称其满足本周重复测量约定吗？

   **核对要点**：不会；标签说明证据范围，仍须检查 n_valid 和 warmup_status。

4. 两份 P/G/B 一样的文件为什么可能分成不同组？把文件搬到另一台机器，怎样让统计仍可追溯？

   **核对要点**：session/env/模型身份等；保留 raw 与来源，重新聚合或映射路径，不仅交 summary。

## 完成核对

- [ ] 能先算单 run，再算组中位数/P95。
- [ ] 实际输入范围明确，计数、纳入原因与原始行可反查。
- [ ] CPU 结论范围与样本充分性分开核查，没有虚构 GPU 数据。
