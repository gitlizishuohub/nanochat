---
source_files:
  - .agents/docs/career_roadmap/w3/scripts/d20_reproduce_cpu.py
  - .agents/docs/career_roadmap/w3/scripts/d15_prepare_materials.py
  - .agents/docs/career_roadmap/w3/scripts/d13_aggregate.py
  - .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py
doc_type: tutorial
knowledge_role: supporting
knowledge_scope: W3第6日学习与实践
---

# W3 D6 · 写出他人能执行的复现入口

[上一日](d19_statistics_audit.md) · [下一日](d21_project_briefing.md)。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[15.1 运行前的条件](../../llm-training-workflow/15-running-examples-and-delivery.md#reading-15-1)；[15.3 交付资产与部署边界](../../llm-training-workflow/15-running-examples-and-delivery.md#reading-15-3)。
- **今天重点**：把复现入口需要的环境、模型、词表、配置和结果范围列全，逐项对应本项目已有文件与仍未执行的条件。
- **暂时跳过**：15.2 只对照入口与前置条件，不复制运行整套训练；不存在的部署形态不写成已交付。

**读完能解释**：一个同学拿到你的复现入口，还必须拿到哪些资产，才能运行同一模型并复核同一结果？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 今天学什么，做到哪里算完成

今天学习写一份别人能按步骤执行的复现入口，约 60–90 分钟。主线提供两条短路径：**重新运行 CPU 随机小模型**，以及**仅从保留的 raw 重建统计**。不训练、不要求 GPU、checkpoint 或外部分词器资产。

本日实际交付是 `experiments/inference_infra/reports/w3_reproduction_v1/d20_reproduction_entry.md` 的完整正文；它供他人操作，需要保存。额外实验报告仍可省略。要写已核查的命令、输入范围与成功判据，不能只生成模板就说完成。

## 1. 为什么“执行成功”还不是“可以复现”

假设你只写“运行 aggregate 就能得到表格”。新读者还不知道输入目录、文件名规则、用哪个 Python、输出到哪里、什么算成功，更无法找到一条有效样本。将这句话展开为：

```text
起点：仓库版本 + 已有可导入 torch 的环境（统计重建只需标准库）
输入：明确的原始文件及模型/输入身份；或随机模型的完整参数
动作：从进入目录开始的完整命令
判据：退出码 + 实际记录计数/状态 + 可反查的有效样本
边界：本次只验证 CPU/random，未测模型能力与 GPU 性能
```

<abbr title="复现 · 在明确输入、环境和方法下重新执行核查。重新测量的耗时会波动；对同一 raw 重建统计则应保留相同分组和有效样本选择。">复现</abbr>不是要求每次毫秒数相同。重新测量产生新的 session/run_id；已有 raw 的统计重建才使用原来的身份与样本。两者需要分别命名，不能用新数据替换旧证据后称“复现了旧结果”。

## 2. 创建骨架，再填完整正文

以下标准库工具只生成模板与证据清单，拒绝覆盖已有稿件。已有 `w3_reproduction_v1` 就继续编辑，或选择 v2；不为生成模板重写旧实验。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
python .agents/docs/career_roadmap/w3/scripts/d15_prepare_materials.py \
  --kind entry --output-dir experiments/inference_infra/reports/w3_reproduction_v1
D20_TEMPLATE_RC=$?
printf 'template_exit=%s\n' "$D20_TEMPLATE_RC"
} > >(tee -a experiments/inference_infra/reports/d20_checks.log) 2>&1
```

有真实证据时可增加 `--evidence` 参数；没有就不填假路径。[工具](scripts/d15_prepare_materials.py)中的 `is_file` 与哈希只识别文件版本，不检查能力或作者身份。生成后需把下面两条路径、实际版本、日志位置和失败指引填入正文；方括号不是已验证结果。

## 3. 路径 A：从新终端重新运行 CPU 小模型

先核对环境：本机终端自动激活 nanochat 时不重复激活；`python` 当前应为 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。换机需替换仓库与解释器路径并参考 D02 准备依赖，不假定所有人都有这个绝对路径。此路径不使用旧终端里的 META、MODEL_TAG 或缓存根。

用[复现脚本](scripts/d20_reproduce_cpu.py)运行固定的 P8/G4、B1/B4、seed=42、三次预热和五次正式测量。它为子进程固定 CPU float32 和两个线程环境变量为 1，使用当前 Python；只初始化并前向，不训练。以下目标必须尚不存在；重试时同步改成 attempt02 等新目录，脚本会拒绝覆盖或追加旧结果。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
python -c "import sys,torch; print(sys.executable); print(torch.__version__)"
git rev-parse HEAD
python .agents/docs/career_roadmap/w3/scripts/d20_reproduce_cpu.py \
  --mode measure \
  --raw-dir experiments/inference_infra/results/raw/w3_d20_attempt01 \
  --summary-dir experiments/inference_infra/results/summary/w3_d20_attempt01
D20_RC=$?
printf 'reproduction_exit=%s\n' "$D20_RC"
} > >(tee -a experiments/inference_infra/reports/d20_checks.log) 2>&1
```

脚本依次调用 B1、B4 和聚合程序；每步 stdout/stderr 写独立日志，实际 argv、工作目录、退出码保存在 `d20_reproduction.json`。某个 benchmark 非零会停止后续步骤，保留已写 raw 和日志。`processes_ok` 只表示子进程均退出 0，仍须检查内容；中断时清单可能停在 started，不能读成成功。

成功判据：两个 benchmark 退出 0；每个 raw 一次 session、3 次预热和 5 次正式完整输出，B1 有效总数 4、B4 总数 16；聚合退出 0，通常两组，各 `n_valid=5`、`warmup_status=OK`、`conclusion_status=CPU_OR_RANDOM_CONTRACT`。按 D19 方法从 audit 反查一条原始记录。帮助能打开、目录已创建或只看最后一行输出都不够。

## 4. 路径 B：不运行模型，仅重建已有统计

已有 D17/D18 或 W2 raw 时，本日可以只验证这一条路径，明确写“统计重建已复查，新的测量命令尚未本次执行”。无需为写入口重复所有测量，尤其不要求 GPU 重测。

下面是独立调用，不依赖路径 A 的终端变量。把 `--raw-dir` 替换为真实输入目录，输出另选全新目录。脚本不递归；复用 W2 上级 raw 时会读取其全部匹配文件，正文应列清范围，避免读者以为只输入 B1/B4 两份。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
python .agents/docs/career_roadmap/w3/scripts/d20_reproduce_cpu.py \
  --mode rebuild \
  --raw-dir experiments/inference_infra/results/raw/w3_cpu_attempt01 \
  --summary-dir experiments/inference_infra/results/summary/w3_d20_rebuild01
D20_REBUILD_RC=$?
printf 'rebuild_exit=%s\n' "$D20_REBUILD_RC"
} > >(tee -a experiments/inference_infra/reports/d20_checks.log) 2>&1
```

核对原汇总与新汇总的组身份、run_ids、纳入数量和数值；同一原始内容应可重建。audit 的绝对来源路径可能因搬迁变化，不能把路径变化当统计数值不一致。搬迁后以新 audit 追溯实际 raw，不留下只在作者机器有效的隐含路径。

## 5. 带读依赖分支，解释为何无需训练资产

[W2 基准 load](../w2/scripts/d12_benchmark.py)的 `args.random_model` 分支创建 `GPTConfig` 与 `ContractTokenizer`，不访问训练缓存；去掉该参数才走 checkpoint 加载。[聚合脚本](scripts/d13_aggregate.py)只读取 JSONL 并重算，完全不导入 torch。新复现脚本的 `mode` 分支只在 measure 模式启动 benchmark，rebuild 直接调用聚合；运行任一步之前登记 argv，结束后保存退出码。由代码而非猜测得出依赖：

| 路径 | 必需输入 | 不应暗藏的条件 |
|---|---|---|
| CPU 随机模型重新测量 | 仓库源码、已有 torch 环境、完整参数 | 不依赖个人训练权重或外部 tokenizer |
| 只重建统计 | 标准库 Python、原始文件、聚合脚本版本 | 不需要 GPU、torch 或旧终端变量 |
| 可选训练权重测量 | 匹配权重/meta/tokenizer及固定身份 | 资源和设备另列，不代替 CPU 主线 |

如确实选择最后一行，在单独小节补上实际 `NANOCHAT_BASE_DIR`、source/tag/step、配置和文件指纹。没有资产时省略这项，不回头要求补训练；MiniMind 权重不直接兼容。

## 6. 把失败入口也写给读者

- `import torch` 失败：核对解释器和 D02 环境；统计重建不受影响，不尝试训练解决。
- `early_stop`、error 或非零退出：保留 log/raw，查结束规则或异常阶段，换新目录重试，不删失败行。
- 无匹配 raw：核对实际目录、文件名与是否只有矩阵；不要生成空表冒充成功。
- 重复 run_id 或坏 JSON：保留原件，查复制/写入问题；不能修改 ID 凑次数。

最后只看自己写的入口，从新终端核查所选路径，记录实际执行范围。正文要让读者知道哪里需要替换、哪里停止和怎样判断；不依靠聊天临时补变量。无需上传或公开材料。

## 本日脚本与产物清单

路径相对于项目根目录。attempt01/rebuild01 是本次尝试的明确目录；已有目标需更换新名称，只有选择并执行相应路径才生成。

| 脚本完整路径 | 用途与输入 | 条件 |
|---|---|---|
| [.agents/docs/career_roadmap/w3/scripts/d15_prepare_materials.py](scripts/d15_prepare_materials.py) | 创建 entry 骨架、可选 evidence 指纹 | 标准库，新材料目标 |
| [.agents/docs/career_roadmap/w3/scripts/d20_reproduce_cpu.py](scripts/d20_reproduce_cpu.py) | 固定 CPU 测量或已有 raw 重建；完整 argv 与退出码登记 | measure 需 torch，rebuild 只需标准库；新 summary 目标 |
| [.agents/docs/career_roadmap/w2/scripts/d12_benchmark.py](../w2/scripts/d12_benchmark.py) | 路径 A：P8/G4，B1/B4 | CPU、已有 torch 环境，无训练 |
| [.agents/docs/career_roadmap/w3/scripts/d13_aggregate.py](scripts/d13_aggregate.py) | 两条路径的统计重建 | 标准库，实际 raw |

| 产物完整路径 | 生成者 | 条件与含义 |
|---|---|---|
| `experiments/inference_infra/reports/w3_reproduction_v1/d20_reproduction_entry.md` | 材料工具后本人完善 | 必需实践材料；完整命令、输入、判据及边界 |
| `experiments/inference_infra/reports/w3_reproduction_v1/d20_entry_evidence_inventory.json` | 材料工具 | 文件指纹；不是内容验收 |
| `experiments/inference_infra/results/raw/w3_d20_attempt01/d20_p8_g4_b1_random_cpu.jsonl`、`experiments/inference_infra/results/raw/w3_d20_attempt01/d20_p8_g4_b4_random_cpu.jsonl` | 路径 A 的 benchmark | 新测逐次记录；第二组仅前组退出 0 时运行 |
| `experiments/inference_infra/results/summary/w3_d20_attempt01/d20_baseline.csv`、`experiments/inference_infra/results/summary/w3_d20_attempt01/d20_baseline_audit.jsonl` | 路径 A 聚合 | 两个 benchmark 均退出 0 后尝试生成 |
| `experiments/inference_infra/results/summary/w3_d20_rebuild01/d20_rebuilt.csv`、`experiments/inference_infra/results/summary/w3_d20_rebuild01/d20_rebuilt_audit.jsonl` | 路径 B 聚合 | 已有 raw 的独立统计重建 |
| `experiments/inference_infra/results/summary/w3_d20_attempt01/d20_reproduction.json`、`experiments/inference_infra/results/summary/w3_d20_rebuild01/d20_reproduction.json` | 复现脚本 | 对应路径的真实 argv、日志位置、子进程退出码；不是样本验收 |
| `experiments/inference_infra/results/summary/w3_d20_attempt01/d20_b1.log`、`experiments/inference_infra/results/summary/w3_d20_attempt01/d20_b4.log`、`experiments/inference_infra/results/summary/w3_d20_attempt01/d20_aggregate.log` | 复现脚本子进程 | 实际启动对应步骤时生成，后续步骤可能因前步失败而未启动 |
| `experiments/inference_infra/results/summary/w3_d20_rebuild01/d20_aggregate.log` | rebuild 子进程 | 统计重建日志 |
| `experiments/inference_infra/reports/d20_checks.log` | 局部命令组 tee | 实际目录、版本、输出、错误和退出码 |

## 本日自验证与交流

1. 展示已填写入口。新读者能否仅凭正文找到起点、输入、命令、成功判据和限制？

   **核对要点**：完整正文，不只是模板；亲自执行范围写实，没有暗藏旧终端变量。

2. 路径 A 与 B 的“结果相同”分别指什么？session/run_id 和毫秒值应完全相同吗？

   **核对要点**：新测有新身份与时间波动；同 raw 重建应保留原身份、分组、样本选择与统计定义。

3. 如果同学只有标准库 Python 和保留的 raw，能做什么，不能做什么？

   **核对要点**：可重建统计，不可据此声称重新跑过模型；要新测先准备 torch，无需训练。

4. 用一个实际或假设失败场景检验正文：读者能定位到日志/原始行并知道怎样恢复吗？

   **核对要点**：具体阶段与证据路径，保留失败、新尝试，不只写“环境问题请检查”。

## 完成核对

- [ ] 已保留可执行的复现正文，选定路径从新终端核查过。
- [ ] 能从 raw 重建并反查统计，未执行的测量明确说明。
- [ ] 训练权重/GPU只是可选分支，材料没有额外强制报告。
