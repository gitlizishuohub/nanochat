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

前一天：[返回前一天](D5_benchmark_schema.md)；后一天：[继续下一天](D7_matrix_smoke.md)；[原始总计划](../nanochat_daily_execution_2026-10-01.md)。

## 开始前：固定 shell 与结果位置

所有命令从仓库根目录执行。首次打开终端先运行以下准备；`conda activate` 若已可用，可省略 `source` 一行。这里使用本项目的 Conda 环境；不要混用系统 Python。CPU实验建议先 `export NANOCHAT_DTYPE=float32`；脚本只在变量未设置时选择FP32，已有变量会遵从并记录，不能假定CPU必定FP32。GPU实验另按协议显式选dtype。

```bash
source /home/lizishuo/miniconda3/etc/profile.d/conda.sh
conda activate nanochat
cd /home/lizishuo/_program_/Project/nanochat
python -c "import sys; print(sys.executable); print(sys.version)"
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。若环境不存在或第三方依赖缺失，先解决环境，不把 `--help` 成功当作模型实验通过。后面写入的 `experiments/` 文件是你实际执行教程时产生的正式成果；本指导不预先生成成绩。大数据、tokenizer、权重放在 `NANOCHAT_BASE_DIR`，不要加进 Git。

## 今日目标与时间盒

用 15 分钟审查脚本、20 分钟 CPU 契约试跑、20 分钟真实checkpoint试跑（具备资源才做）、20 分钟自动汇总。前置是 [原始数据协议](D5_benchmark_schema.md)。今天的核心是重复与自动统计，不是比较某次最快耗时。

## 第一步：检查入口与保存路径

```bash
python .agents/docs/career_roadmap/w2/scripts/benchmark.py --help
```

打开 [完整重复测量实现](scripts/benchmark.py)，依次找 `load()`、`measure()`、`main()`。确认模型加载/hash计算在计时外；每次run创建新Engine/cache；预热与正式均保存；测量每步用 `perf_counter` 并在CUDA同步；CPU不会调用CUDA显存函数。

本脚本不用 `scripts.infer_bench` 的单次JSON反复拼接，因此能保留逐步间隔与有效token。官方脚本仍可用于阅读，但不能直接作为本计划重复基线。当前模式是uncompiled eager Engine；以后启用编译必须新建协议版本。

## 第二步：先做无需训练模型的契约验证

```bash
python .agents/docs/career_roadmap/w2/scripts/benchmark.py --device cpu --random-model --prompt-tokens 8 --decode-tokens 4 --batch-size 1 --warmup 3 --repeats 5 --seed 42 --output experiments/inference_infra/results/raw/p8_g4_b1_random_cpu.jsonl
```

<abbr title="Random model · 随机初始化且没有经过训练的模型。本文只用于检查张量、生成器、计时与落盘契约；随机模型的工作负载与真实checkpoint不能混入同一性能组。">random-model</abbr> 使用2层、C=64、H=2、V=512模型和合成token映射，没有加载真实tokenizer，因此它不能代替两步训练门槛。输出应8行：3行warmup、5行measure。模型初始化有随机性，输出只用于契约检查，不是生成质量。

打开JSONL逐行看phase/status，确认 `actual_decode_steps=4`、`per_row_generated_tokens=[4]`、`generated_tokens=4`、`len(decode_step_ms)=3`。检查 `elapsed_ms` 近似等于 `ttft_ms + sum(decode_step_ms)`，`tokens_per_s = 4000 / elapsed_ms`。允许微小浮点舍入差，不允许把ms直接当s。

## 第三步：使用已冻结checkpoint重复测量

准备好昨天记录的数据根与模型后运行：

```bash
python .agents/docs/career_roadmap/w2/scripts/benchmark.py --device cpu --source base --model-tag d4-w2-20 --step 20 --prompt-tokens 32 --decode-tokens 8 --batch-size 1 --warmup 3 --repeats 5 --seed 42 --output experiments/inference_infra/results/raw/p32_g8_b1_checkpoint_cpu.jsonl
```

CPU记录标 `CPU_FUNCTION_ONLY`，有timing不等于可以评价GPU性能。合成数据训练出的checkpoint还要在模型来源中注明synthetic；它只是一套合法的系统实验权重。

若已有GPU、对应PyTorch且完成环境核对，把 `--device cpu` 改成 `--device cuda`，另换输出文件名，并在运行前设置明确的 `NANOCHAT_DTYPE`。精度在模块import时确定，不能同一Python进程跑一半再改变量。GPU驱动用 `nvidia-smi` 单独补录到env/protocol；脚本环境字段包含PyTorch/CUDA、设备名、线程数、源码commit、dirty与后端。

## 第四步：自动汇总，保留失败行

使用下一周共用的 [统计脚本](../w3/scripts/aggregate.py)，先检查实际CLI：

```bash
python .agents/docs/career_roadmap/w3/scripts/aggregate.py --help
```

实际汇总命令如下（该目录只放本次测量，避免把不相关试验混入）：

```bash
python .agents/docs/career_roadmap/w3/scripts/aggregate.py --raw-dir experiments/inference_infra/results/raw --output experiments/inference_infra/results/summary/w2_baseline.csv --audit experiments/inference_infra/results/summary/w2_audit.jsonl
```

然后按 [W3统计复核教程](../w3/D5_statistics_audit.md) 的命令生成CSV。要求把 `model_kind/device/precision/attention_backend/P/G/B/tag/step` 不同的行分开；不同session和checkpoint hash也不能无判断合并。median是正式run的中位数；<abbr title="P95 / 95th percentile · 95%样本不超过的分位值。本文汇总采用nearest-rank：排序后取ceil(0.95N)对应样本，N=5时等于最大值；不能据此断言尾延迟稳定。">P95</abbr> 在仅五个样本时信息很有限，因此同时报告样本数、最小/最大值。

## 常见失败、产物与验收

- `early_stop`：至少一行遇到terminal、forced输出或有效长度不足；保留raw行，并记录重新选择工作负载的理由。不要把实际少于G的值当固定G性能。同一prompt与greedy配置重跑通常仍会早停；可用 `--prompt-fill-id 2` 改变普通ID工作负载（须在词表范围且非特殊ID），换新raw文件并按 `prompt_sha256` 单独分组，不从若干prompt挑出最好数字。
- `P+G exceeds`：缩短P/G或准备更长sequence_len的模型；脚本拒绝静默截断。
- 每次都追加到旧raw：脚本是append-only，新实验使用新文件名；若重复执行，汇总前按session_id区分。
- `status=error`：查看error字段和退出码；GPU OOM行不得删除后只展示剩余成功数据。
- 验收：至少一个真实跑出的8行文件（含预热），全部字段可解释；汇总来自脚本。CUDA不可用写 `PENDING_GPU`，训练模型不可用写 `PENDING_CHECKPOINT`。

## 当天记录与离开前核对

把以下模板填写到当天报告末尾；运行命令原样保留，失败也写退出码。`PENDING` 表示没有执行；不要用空值代替失败事实。

```text
日期：
源码 commit / dirty 状态：
解释器 / 设备 / NANOCHAT_DTYPE：
NANOCHAT_BASE_DIR：
实际用时：
运行命令与退出码：
原始结果路径：
已验证的事实：
未验证或阻塞：
下一次恢复的第一条命令：
```

记录 commit 使用 `git rev-parse HEAD`，改动状态使用 `git status --short`。共享给别人前检查数据路径和日志是否包含私人信息。只提交本次自己的轻量报告和脚本，不提交权重、数据、环境目录或机器特有缓存。
