---
source_files:
  - .agents/docs/career_roadmap/nanochat_daily_execution_2026-10-01.md
  - ../minimind/experiments/archive/minimind_baseline-before-embedding-20261014/README.md
  - ../minimind/experiments/archive/minimind_baseline-before-embedding-20261014/reports/reproduction.md
  - ../minimind/experiments/archive/minimind_baseline-before-embedding-20261014/results/tables/generation_benchmark_raw.csv
source_symbols: []
doc_type: tutorial
knowledge_scope: W1 D1 起点证据复核
---

# W1 D1：把 MiniMind 已完成的事与 nanochat 待做的事分开

今天只做一件事：用能找到的文件说明你的项目起点。你不是从零开始，也不能把另一个项目的结果移植成 nanochat 的成绩。计划日期沿用 10 月 1 日；实际执行日期请写当天真实日期。

预计 60–90 分钟：10 分钟确认目录，20 分钟核查报告，20 分钟读原始表，20 分钟写起点报告，10 分钟验收。没有 GPU 不影响今天；MiniMind 旧文件缺失时只补对应证据，今天不重跑训练。

## 当日脚本与产物清单

所有项目内路径均以 nanochat 项目根目录为基准；外部 MiniMind 只读历史证据不改名。`dNN_` 表示全程首次使用或产生日，跨日复用不改前缀；上游源码和测试保留原生名称。

| 角色与脚本 | 用途与执行阶段 | 输入路径或来源 | 运行条件 |
|---|---|---|---|
| 入口：[`.agents/docs/career_roadmap/w1/scripts/d01_inventory_starting_point.py`](scripts/d01_inventory_starting_point.py) | 第 2 节只读盘点 MiniMind 历史证据 | 外部输入 `../minimind/experiments/minimind_baseline/`、`../minimind/experiments/archive/minimind_baseline*/`；可用 `--minimind-root` 指定实际根目录 | Python 标准库；证据缺失仍输出盘点 JSON，退出码为 2 |

| 完整项目相对路径 | 用途 | 生成者 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/results/w1/d01_starting_point_inventory.json` | 历史文件存在性、哈希与 CSV 摘要 | `d01_inventory_starting_point.py` | 自动：完成盘点即写入；无匹配证据时退出码 2 仍有 JSON |
| `experiments/inference_infra/reports/d01_starting_point.md` | 起点结论、证据与缺口 | 第 4 节在编辑器编写 | 人工编写；运行脚本不会生成报告 |
| `experiments/inference_infra/reports/d01_checks.log` | 本日终端检查输出与错误 | 首段 Bash 的 exec / tee | 自动捕获启用后的输出；命令及退出码人工补记，重复运行追加 |

清单是执行后的预期，不表示文件已经存在。已有未加前缀的历史实验文件保留，新执行按上述路径写入。

正文的 `sed`、`rg`、`head`、解释器检查等只向终端打印，不会自动生成独立结果文件；输出在执行日志捕获命令后保存到 `experiments/inference_infra/reports/d01_checks.log`；把所用命令、退出码和关键结果另记入当天报告。 盘点或探查的结构化结果保存为 <abbr title="JSON / JavaScript Object Notation · 用键值和数组组织数据的文本格式；这里保存脚本实际采集的事实或断言通过后的结果，可供程序读取。">JSON</abbr>。 未执行或失败时按真实状态记录，不能把旧同名文件当作本次成功。

## 1. 准备终端与正确解释器

所有命令从 nanochat 根目录执行。每次新开终端先运行：

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
exec > >(tee -a experiments/inference_infra/reports/d01_checks.log) 2>&1
source /home/lizishuo/miniconda3/etc/profile.d/conda.sh
conda activate nanochat
python -c 'import sys; print(sys.executable)'
git status --short
```

解释器应是 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。本轮编写教程时已创建这个 Conda 环境；创建环境不等于依赖安装完毕或模型测试通过，后两项在 D2、D3 留独立证据。Git 输出表示你开始前的已有改动，保留它们，不执行 `reset --hard`。

相邻项目默认位于 `../minimind/`。这里的 <abbr title="MiniMind · 相邻的教学语言模型项目。本文只把其历史日志作为已完成训练到推理闭环的证据，不把它的模型权重用于 nanochat。">MiniMind</abbr> 与 nanochat 的模型和词表不同，不能互换权重。

## 2. 先定位文件，再阅读结论

使用[起点证据盘点脚本](scripts/d01_inventory_starting_point.py)：

```bash
python .agents/docs/career_roadmap/w1/scripts/d01_inventory_starting_point.py --help
python .agents/docs/career_roadmap/w1/scripts/d01_inventory_starting_point.py
```

脚本默认扫描相邻 MiniMind 的当前实验目录及 `experiments/archive/minimind_baseline*`，读取文件、计算 SHA256、统计 CSV 行数，不导入模型、不训练、不写 MiniMind。输出为 `experiments/inference_infra/results/w1/d01_starting_point_inventory.json`。

如果 MiniMind 不在相邻位置，显式传 `--minimind-root /实际路径/minimind`。重复运行覆盖同名盘点 JSON；要保留快照可传 `--output experiments/inference_infra/results/w1/d01_starting_point_inventory_实际日期.json`。

本轮核查发现：旧 D1–D14 证据已经迁到 `../minimind/experiments/archive/minimind_baseline-before-embedding-20261014/`。原计划指向的当前实验目录还在，但不能再据此假定旧 README 仍在那里。看到 `exists: false` 时先检查归档候选，不立刻认定实验没做。

## 3. 对照报告、复现记录和原始表

先看摘要，再向下追证据：

```bash
sed -n '1,190p' ../minimind/experiments/archive/minimind_baseline-before-embedding-20261014/README.md
cat ../minimind/experiments/archive/minimind_baseline-before-embedding-20261014/reports/reproduction.md
head -n 5 ../minimind/experiments/archive/minimind_baseline-before-embedding-20261014/results/tables/d14_clean_kv_cache.csv
head -n 5 ../minimind/experiments/archive/minimind_baseline-before-embedding-20261014/results/tables/generation_benchmark_raw.csv
```

按下面顺序在编辑器做四个批注：

1. README 的环境栏：设备为 CPU、数值类型 FP32。<abbr title="FP32 / 32-bit Floating Point · 32 位浮点数。这里描述历史训练与正确性实验的数值格式，不代表使用了 GPU。">FP32</abbr> 不等于 CUDA；CPU 也使用它。
2. D14 表：这是干净 shell 复现，记录了显式解释器与退出码。它证明当时那台机器、那些依赖和数据下可复现；不证明换环境仍直接成功。
3. KV 表：`cache_token` 与 `no_cache_token` 应逐行一致，`max_abs_logit_diff` 记录同一步最后位置输出的最大绝对差。<abbr title="logits · 模型为词表中每个候选 token 输出的未归一化分数。比较同一前缀最后位置的 logits，可发现仅比较最终文本可能漏掉的数值差异。">logits</abbr> 误差小与生成 token 相同是两条证据，不能只保留后一条。
4. 原始基准表：查看 `device`、`dtype`、`warmup`、`repetition`。盘点 JSON 中每组应能看到五个正式重复编号。原 D13 有 60 行、12 组；D14 降级矩阵是 20 行、4 组。二者不是互相矛盾，而是矩阵大小不同。

历史报告记录 D12 最大差约 `2.38418579e-06`，小于它选定的 `1e-5` 容差。你的报告应写“历史 CSV 记录”，除非今天重新执行并产生了新日志，不能写“今天复测通过”。<abbr title="KV Cache / Key-Value Cache · 缓存过去 token 的注意力 Key 和 Value。避免每个生成步骤重新计算全部历史，但需要额外缓存存储；它本身不意味着总显存减少。">KV Cache</abbr> 是后续要迁移的理解方式，而不是直接复制实现。

## 4. 写正式起点报告

```bash
mkdir -p experiments/inference_infra/reports
```

在编辑器新建 `experiments/inference_infra/reports/d01_starting_point.md`，按此模板填写。模板里的 `PENDING` 是待完成状态，不能直接替换为 PASS：

```markdown
# 起点复核
- 实际执行日期：
- nanochat commit：
- MiniMind 证据目录：
- 本次操作：只读复核 / 另有复测（附命令）

| 声明 | 文件与具体字段 | 证据状态 | 不能推出的结论 |
|---|---|---|---|
| CPU 训练与恢复闭环 | reproduction.md 对应行 | 待核对 | GPU 吞吐、语言质量 |
| KV 与无缓存逐步一致 | d14_clean_kv_cache.csv | 待核对 | nanochat 缓存已正确 |
| 重复计时协议已有实践 | benchmark raw 的 repetition | 待核对 | nanochat 加速比 |

## nanochat 本期目标
可信重复基线 → trace → 受控优化 → 回归与复现。

## 现阶段缺口
- nanochat 环境运行验证：PENDING
- nanochat tokenizer / checkpoint：PENDING
- GPU 性能与算子实验：PENDING_GPU（若无卡）

## 下一步
完成环境盘点，不重新跑完整 MiniMind 训练。
```

`trace` 是带时间戳的运行事件记录，用来定位耗时来源；W1 不要求提前产出它。起点报告中的路径要能实际打开，不能只有“已完成”三个字。

## 5. 失败时如何继续

| 现象 | 检查与处理 | 今天仍可交付 |
|---|---|---|
| 所有候选不存在 | 核对 `--minimind-root`；确认是否换机 | 缺文件清单及阻塞原因 |
| 有 README，无 CSV | 检查归档表目录；缺口只标对应行 | 文档声称与原始证据分层 |
| CSV 行数不同 | 比较 D13/D14、矩阵参数与重复次数 | 分别列出两次实验范围 |
| 数字与摘要不一致 | 优先记录具体表和字段，不凭记忆修数据 | 不一致项与复核动作 |
| shell 没有 Conda 函数 | 先 source 指定 `conda.sh`，再 activate | 可先用完整解释器路径运行盘点 |

## 6. 当天验收与记录

- [ ] 找到旧证据的真实位置，或写出查找失败的真实原因。
- [ ] 至少一条结论从 README 追到 CSV 或原始日志。
- [ ] 起点报告明确 CPU 功能证据、模型质量空白、GPU 性能空白。
- [ ] 没有重训、混用权重或创造 nanochat 性能数字。

在报告末尾留当天记录：`实际日期 / commit / 真实用时 / 执行命令 / JSON 路径 / PASS、PARTIAL 或 BLOCKED / 下一步`。PARTIAL 表示完成了部分可核验工作，不是全面通过。

下一篇：[D2 环境盘点](d02_environment_inventory.md)。原路线：[逐日总计划](../nanochat_daily_execution_2026-10-01.md)。
