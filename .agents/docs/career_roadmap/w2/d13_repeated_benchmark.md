---
source_files:
  - nanochat/common.py
  - nanochat/tokenizer.py
  - nanochat/dataset.py
  - nanochat/dataloader.py
  - scripts/tok_train.py
  - scripts/tok_eval.py
  - scripts/base_train.py
  - nanochat/checkpoint_manager.py
  - nanochat/gpt.py
  - nanochat/engine.py
  - scripts/infer_bench.py
  - nanochat/flash_attention.py
doc_type: tutorial
knowledge_scope: W2第6天实操与验收
---

# W2 D6：运行三次预热、五次正式测量

前一天：[返回前一天](d12_benchmark_schema.md)；后一天：[继续下一天](d14_matrix_smoke.md)；[原始总计划](../nanochat_daily_execution_2026-10-01.md)。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[12.2 生成控制流与返回格式](../../llm-training-workflow/12-inference-and-generation.md#reading-12-2)；[10.1 日志口径](../../llm-training-workflow/10-logging-checkpoints-and-resume.md#reading-10-1)。
- **今天重点**：回读实际输出长度、终止条件与日志统计范围，逐项对应今天的预热、正式记录和自动汇总。今天主要做应用回查，不增加新训练概念。
- **暂时跳过**：不用重读已掌握的整章；不把观察单次数字当作重复测量完成。

**读完能解释**：如果一次生成提前结束，直接用要求的生成长度作吞吐分子，会产生什么问题？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 开始前：固定 shell 与结果位置

所有命令从仓库根目录执行。当前终端会自动激活 nanochat 环境，无需重复激活；先运行以下检查。这里使用本项目的 Conda 环境；不要混用系统 Python。CPU实验建议先 `export NANOCHAT_DTYPE=float32`；脚本只在变量未设置时选择FP32，已有变量会遵从并记录，不能假定CPU必定FP32。GPU实验另按协议显式选dtype。

日志只在 `{ ...; }` 命令组执行期间记录；<abbr title="Bash process substitution · 进程替换。这里把命令组的输出交给 tee，同时显示并追加到日志；重定向在命令组结束后恢复。">进程替换</abbr>不会持续接管交互终端，因此提示符和 VS Code 终端集成控制字符不会混入日志。命令组在当前 shell 执行，变量设置会保留；代码块返回最后一条命令的退出码，需核对单条命令时立即记录 `$?`。已有的显式文件重定向仍写入各自文件。复制时包含外层花括号和末尾重定向。若之前执行过旧版全局日志命令，先关闭旧终端并新开终端；仅修改文档不会停止旧终端的记录。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
{
python -c "import sys; print(sys.executable); print(sys.version)"
} > >(tee -a experiments/inference_infra/reports/d13_session.log) 2>&1
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。若环境不存在或第三方依赖缺失，先解决环境，不把 `--help` 成功当作模型实验通过。后面写入的 `experiments/` 文件是你实际执行教程时产生的正式成果；本指导不预先生成成绩。大数据、tokenizer、权重放在 `NANOCHAT_BASE_DIR`，不要加进 Git。表中的 `$NANOCHAT_BASE_DIR` 代表当前 shell 选定的真实绝对数据根；合成分支示例为 `/tmp/nanochat-w2-toy`，未设置时当前用户默认根为 `/home/lizishuo/.cache/nanochat`。交流时给出实际展开后的绝对路径；`*.parquet` 表示该目录内逐一列出的实际数据文件，旧数据回退目录为 `$NANOCHAT_BASE_DIR/base_data/`。

本页各 Bash 代码块内的输出追加保存在 `experiments/inference_infra/reports/d13_session.log`；命令返回后立即执行 `echo "exit=$?"`，避免丢失退出码。帮助、源码检索、解释器与 Git 检查只打印，不另有检查结果文件。日志是追加写入，重跑时交流中注明会话边界或日志中的起止位置。

## 今日目标与时间盒

用 15 分钟审查脚本、20 分钟 CPU 契约试跑、20 分钟真实checkpoint试跑（具备资源才做）、20 分钟自动汇总。前置是 [原始数据协议](d12_benchmark_schema.md)。今天的核心是重复与自动统计，不是比较某次最快耗时。

## 第一步：检查入口与保存路径

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py --help
} > >(tee -a experiments/inference_infra/reports/d13_session.log) 2>&1
```

打开 [完整重复测量实现](scripts/d12_benchmark.py)，依次找 `load()`、`measure()`、`main()`。确认模型加载/hash计算在计时外；每次run创建新Engine/cache；预热与正式均保存；测量每步用 `perf_counter` 并在CUDA同步；CPU不会调用CUDA显存函数。

本脚本不用 `scripts.infer_bench` 的单次JSON反复拼接，因此能保留逐步间隔与有效token。官方脚本仍可用于阅读，但不能直接作为本计划重复基线。当前模式是uncompiled eager Engine；以后启用编译必须新建协议版本。

## 第二步：先做无需训练模型的契约验证

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py --device cpu --random-model --prompt-tokens 8 --decode-tokens 4 --batch-size 1 --warmup 3 --repeats 5 --seed 42 --output experiments/inference_infra/results/raw/d13_p8_g4_b1_random_cpu.jsonl
} > >(tee -a experiments/inference_infra/reports/d13_session.log) 2>&1
```

<abbr title="Random model · 随机初始化且没有经过训练的模型。本文只用于检查张量、生成器、计时与落盘契约；随机模型的工作负载与真实checkpoint不能混入同一性能组。">random-model</abbr> 使用2层、C=64、H=2、V=512模型和合成token映射，没有加载真实tokenizer，因此它不能代替两步训练门槛。输出应8行：3行warmup、5行measure。模型初始化有随机性，输出只用于契约检查，不是生成质量。

打开JSONL逐行看phase/status，确认 `actual_decode_steps=4`、`per_row_generated_tokens=[4]`、`generated_tokens=4`、`len(decode_step_ms)=3`。检查 `elapsed_ms` 近似等于 `ttft_ms + sum(decode_step_ms)`，`tokens_per_s = 4000 / elapsed_ms`。允许微小浮点舍入差，不允许把ms直接当s。

## 第三步：使用已冻结checkpoint重复测量

准备好昨天记录的数据根与模型后运行：

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py --device cpu --source base --model-tag d11_w2_20 --step 20 --prompt-tokens 32 --decode-tokens 8 --batch-size 1 --warmup 3 --repeats 5 --seed 42 --output experiments/inference_infra/results/raw/d13_p32_g8_b1_checkpoint_cpu.jsonl
} > >(tee -a experiments/inference_infra/reports/d13_session.log) 2>&1
```

CPU记录标 `CPU_FUNCTION_ONLY`，有timing不等于可以评价GPU性能。合成数据训练出的checkpoint还要在模型来源中注明synthetic；它只是一套合法的系统实验权重。

若已有GPU、对应PyTorch且完成环境核对，把 `--device cpu` 改成 `--device cuda`，将输出明确改为 `experiments/inference_infra/results/raw/d13_p32_g8_b1_checkpoint_cuda.jsonl`，并在运行前设置明确的 `NANOCHAT_DTYPE`。精度在模块import时确定，不能同一Python进程跑一半再改变量。GPU驱动用 `nvidia-smi` 单独留在会话日志或聊天协议确认中；脚本环境字段包含PyTorch/CUDA、设备名、线程数、源码commit、dirty与后端。

## 第四步：自动汇总，保留失败行

使用下一周共用的 [统计脚本](../w3/scripts/d13_aggregate.py)，先检查实际CLI：

```bash
{
python .agents/docs/career_roadmap/w3/scripts/d13_aggregate.py --help
} > >(tee -a experiments/inference_infra/reports/d13_session.log) 2>&1
```

实际汇总命令如下。它读取共享目录 `experiments/inference_infra/results/raw/` 下全部匹配的原始文件，按 session 和实验身份分别汇总；不会只凭 d13 文件名前缀筛选。本日验收须依据审计记录中的完整输入文件路径和 run_id，选出本次生成的两组，不能把其它日期的组数算作今天完成数：

```bash
{
python .agents/docs/career_roadmap/w3/scripts/d13_aggregate.py --raw-dir experiments/inference_infra/results/raw --output experiments/inference_infra/results/summary/d13_w2_baseline.csv --audit experiments/inference_infra/results/summary/d13_w2_audit.jsonl
} > >(tee -a experiments/inference_infra/reports/d13_session.log) 2>&1
```

然后按 [W3统计复核教程](../w3/d19_statistics_audit.md) 的命令生成CSV。要求把 `model_kind/device/precision/attention_backend/P/G/B/tag/step` 不同的行分开；不同session和checkpoint hash也不能无判断合并。median是正式run的中位数；<abbr title="P95 / 95th percentile · 95%样本不超过的分位值。本文汇总采用nearest-rank：排序后取ceil(0.95N)对应样本，N=5时等于最大值；不能据此断言尾延迟稳定。">P95</abbr> 在仅五个样本时信息很有限，因此交流时同时给出样本数、最小/最大值。

## 常见失败、产物与验收

- `early_stop`：至少一行遇到terminal、forced输出或有效长度不足；保留raw行，并记录重新选择工作负载的理由。不要把实际少于G的值当固定G性能。同一prompt与greedy配置重跑通常仍会早停；可用 `--prompt-fill-id 2` 改变普通ID工作负载（须在词表范围且非特殊ID），换新raw文件并按 `prompt_sha256` 单独分组，不从若干prompt挑出最好数字。
- `P+G exceeds`：缩短P/G或准备更长sequence_len的模型；脚本拒绝静默截断。
- 每次都追加到旧raw：脚本是append-only，新实验使用新文件名；若重复执行，汇总前按session_id区分。
- `status=error`：查看error字段和退出码；GPU OOM行不得删除后只展示剩余成功数据。
- 验收：至少一个真实跑出的8行文件（含预热），全部字段可解释；汇总来自脚本。CUDA不可用写 `PENDING_GPU`，训练模型不可用写 `PENDING_CHECKPOINT`。

## 今日脚本与产物清单

每日编号使用全程 d01–d21；跨日复用保留首次实际执行日。以下项目相对路径均以仓库根目录为基准，脚本清单直接列出本日执行入口。

| 脚本完整项目路径 | 用途 | 执行阶段 | 输入 | 运行条件 |
|---|---|---|---|---|
| `.agents/docs/career_roadmap/w2/scripts/d12_benchmark.py` | 复用重复测量入口 | 第二、三步 | 随机模型无需文件；真实模型读 `$NANOCHAT_BASE_DIR/base_checkpoints/d11_w2_20/model_000020.pt`、`$NANOCHAT_BASE_DIR/base_checkpoints/d11_w2_20/meta_000020.json`、`$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl`；固定 token ID prompt | CPU 契约可先跑；checkpoint/GPU 分支分别满足条件 |
| `.agents/docs/career_roadmap/w3/scripts/d13_aggregate.py` | 按身份分组统计并审计原始记录 | 第四步 | experiments/inference_infra/results/raw 中本次 JSONL | raw 文件实际存在；错误行保留 |

| 产物完整项目相对路径 | 用途 | 生成者 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/results/raw/d13_p8_g4_b1_random_cpu.jsonl` | 随机 CPU 重复契约记录 | d12_benchmark.py | 条件自动；成功 8 行，失败记录实际行数 |
| `experiments/inference_infra/results/raw/d13_p32_g8_b1_checkpoint_cpu.jsonl` | 真实 checkpoint CPU 记录 | d12_benchmark.py | 条件自动；模型分支实际运行才存在 |
| `experiments/inference_infra/results/raw/d13_p32_g8_b1_checkpoint_cuda.jsonl` | GPU checkpoint 重复记录 | d12_benchmark.py | 条件自动；仅 CUDA 分支实际运行后存在 |
| `experiments/inference_infra/results/summary/d13_w2_baseline.csv` | 分组汇总 | d13_aggregate.py | 条件自动；实际执行统计 |
| `experiments/inference_infra/results/summary/d13_w2_audit.jsonl` | 逐行审计 | d13_aggregate.py | 条件自动；实际执行统计 |
| `experiments/inference_infra/reports/d13_session.log` | 当天终端输出与手动记录的退出码 | 各 Bash 代码块的 tee | 自动；仅追加各代码块内的输出 |

原生 `tokenizer.pkl`、`token_bytes.pt`、`model_<step>.pt`、`meta_<step>.json` 与 `optim_<step>_rank0.pt` 是 loader 的格式契约，保留其名称；用带日编号的数据文件或模型标签归属日期。已有历史结果不原位重命名，新的命令使用本页命名。没有执行或条件不满足的必需检查均标 PENDING；可选 Markdown 未整理不影响验收。

## 今日自验证与交流问题

按题号在聊天中回答即可。概念题说明理由或推导；实测题引用当天 JSONL、audit 或 CSV 中能支撑结论的关键片段，同一份证据可复用。开头一次注明本次分支与 session、raw/汇总路径，不必逐题重复命令和环境。未执行标 `PENDING`；历史结果注明日期与版本。建议用约 20–25 分钟讨论，不要求重跑已有有效实验或另写统计报告。

1. 这次文件有 8 行，是否就完成了“三次预热、五次正式成功”？请按本次 session 的 phase/status 判断实际完成情况，并解释预热在解决什么问题、为何仍须保存。

   **核对要点**：用实际记录区分预热、正式、提前结束和错误，排除追加的旧 session；预热不进入正式统计，但参与判断运行条件是否满足。不能把输出列数、文件行数和独立重复次数混为一谈。

2. 选一条本次 `measure+ok` 原始行，不采用记录中现成的 TPOT 和 tokens/s，直接从逐步间隔和有效 token 数复算它们。能否同时核对总时长的一致性？

   **核对要点**：给出 run_id 或文件行号，展示所用原始值、公式和单位，核对 TTFT 加后续间隔与 elapsed 的关系；没有有效行就说明原因，不用教学算例冒充本次实测。

3. 从 CSV 选本次的一组，沿 `run_ids→audit→raw` 追到一条有效记录。为什么共享目录里同样 P/G/B 的其他日期或其他模型记录没有合并进来？

   **核对要点**：用一条实际追溯链说明身份分组，包括 session、模型/输入哈希及环境等直接相关差异；汇总文件名不限制输入日期，audit 不包含预热，预热须回 raw 核对。不要求抄录全部分组字段。

4. 对同一组的五次正式有效 TTFT 排序，核对中位数与 nearest-rank P95。若只有五个样本，P95 的位置意味着你能对尾延迟作出多强的判断？

   **核对要点**：引用同一组的实际样本和 CSV，展示排序与取位方法，同时报告样本数及范围；区分对每 run 指标取分位数与把全部 decode 间隔混合取分位数。样本不足时说明统计边界，不挑最快的一次代替。

5. 若一条正式记录 `early_stop`、另一次命令因加载异常退出，该怎样保留和解释结果？如果改 fill ID 后才完成固定长度，为什么必须作为另一种工作负载？

   **核对要点**：根据实际异常或明确标注的推演区分退出码与行状态，说明错误可能使循环提前结束；保留失败证据，依据 audit/CSV 的排除原因与 `conclusion_status` 判断放行范围。输入哈希变化不能混组或挑最佳成绩，CPU/random 仍只支持功能契约结论。

离开前保留本日会话日志与自动结果；若追加重跑，注明会话边界。交流时可附 `git rev-parse HEAD`、`git status --short` 及实际设备/精度以定位证据，不必另填通用表。下一天可直接复用本聊天中已确认的结论与证据路径。共享前检查数据路径和日志是否包含私人信息；若提交成果，只纳入自己的轻量记录与脚本，不提交权重、数据、环境目录或缓存。
