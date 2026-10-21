---
source_files:
  - scripts/infer_bench.py
  - nanochat/engine.py
  - nanochat/checkpoint_manager.py
  - nanochat/gpt.py
source_symbols:
  - bench_generate
  - build_prompt
  - Engine.generate
  - load_model
  - GPT.generate
doc_type: tutorial
knowledge_scope: W1 D7 基准边界审计与周验收
---

# W1 D7：审计现有基准，冻结下一周要修的测量契约

今天不是立刻跑一组数字，而是明确原入口能测什么、还缺什么。预计 75–90 分钟：20 分钟依赖与输入，25 分钟计时边界，20 分钟统计问题，15 分钟核验问题，10 分钟 W1 口述验收。无 GPU/权重时照常完成静态审计，GPU 运行标待完成。

前置：[D4 模型路径](d04_model_shapes.md)、[D5 缓存](d05_cache_lifecycle.md)、[D6 后端](d06_attention_backends.md)。这三份是理解测量的必要背景。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[13.2 能力与系统性能的边界](../../llm-training-workflow/13-evaluation-and-metrics.md#reading-13-2)；[12.2 生成控制流与返回格式](../../llm-training-workflow/12-inference-and-generation.md#reading-12-2)。
- **今天重点**：先分清能力指标与系统性能，再回查终止词元、返回序列和有效生成长度，帮助审计计时边界。
- **暂时跳过**：13.1 的能力指标公式留到 D19；训练命令和 <abbr title="RL / Reinforcement Learning · 强化学习；此处指利用生成回答的数学奖励进一步更新模型。">RL</abbr> 不影响今天的基准审计。

**读完能解释**：为什么生成出来的文本看起来正常，还不足以证明首词元延迟或吞吐测量可信？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 当日脚本与产物清单

所有项目内路径均以 nanochat 项目根目录为基准；外部 MiniMind 只读历史证据不改名。`dNN_` 表示全程首次使用或产生日，跨日复用不改前缀；上游源码和测试保留原生名称。

| 角色与脚本 | 用途与执行阶段 | 输入路径或来源 | 运行条件 |
|---|---|---|---|
| 只读审计对象：[scripts/infer_bench.py](../../../../scripts/infer_bench.py) | 第 1–4 节用 sed/rg 阅读原基准，不执行 GPU 基准 | `scripts/infer_bench.py`、`nanochat/engine.py`、`nanochat/checkpoint_manager.py` | 源码可读即可；原生上游入口保留原名；真正运行还需 CUDA、匹配 tokenizer/checkpoint |
| 可选复用：[`.agents/docs/career_roadmap/w1/scripts/d04_probe_model_and_cache.py`](scripts/d04_probe_model_and_cache.py) | 缺 CPU 模型证据时用 shapes/cache 补测 | 内部随机权重；`nanochat/gpt.py`、`nanochat/engine.py` | 同 D4/D5；已有合格证据时不必重跑 |
| 可选复用：[`.agents/docs/career_roadmap/w1/scripts/d03_run_inference_tests.py`](scripts/d03_run_inference_tests.py) | 缺测试证据时补测 | `tests/test_engine.py`、`tests/test_attention_fallback.py` | 同 D3；已有完整证据时不必重跑 |

| 完整项目相对路径 | 用途 | 生成者 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/reports/d07_benchmark_audit.md` | 计时契约差距和 W1 验收 | 第 5–6 节问答可选整理 | 可选人工整理；静态审计不产生性能 CSV、图片或权重 |
| `experiments/inference_infra/results/w1/d04_shapes_probe.json` | 可选补充 shapes 证据 | 复用 `d04_probe_model_and_cache.py --mode shapes` | 仅执行该模式且断言通过时自动生成/覆盖 |
| `experiments/inference_infra/results/w1/d05_cache_probe.json` | 可选补充 cache 证据 | 复用 `d04_probe_model_and_cache.py --mode cache` | 仅执行该模式且断言通过时自动生成/覆盖 |
| `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_stdout.log` | 可选补测 stdout | 复用 `d03_run_inference_tests.py` | 仅运行补测时自动生成 |
| `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_stderr.log` | 可选补测 stderr | 同上 | 仅运行补测时自动生成，可为空 |
| `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_run.json` | 可选补测元数据 | 同上 | 仅补测子进程正常返回后自动生成 |
| `experiments/inference_infra/reports/d07_checks.log` | 本日终端检查输出与错误 | 首段 Bash 的 exec / tee | 自动捕获启用后的输出；命令及退出码在交流时补充，重复运行追加 |

清单是执行后的预期，不表示文件已经存在。已有未加前缀的历史实验文件保留，新执行按上述路径写入。 `<UTC时间戳>` 由脚本按协调世界时生成，格式为 `YYYYmmddTHHMMSSffffffZ`；以终端 `Logs:` 打印的真实目录为准。

正文的 `sed`、`rg`、`head`、解释器检查等只向终端打印，不会自动生成独立结果文件；输出在执行日志捕获命令后保存到 `experiments/inference_infra/reports/d07_checks.log`；交流时附所用命令、退出码和关键结果，自动结果已有的字段无需手抄。 测试输出由清单中的 stdout/stderr 日志捕获。 盘点或探查的结构化结果保存为 <abbr title="JSON / JavaScript Object Notation · 用键值和数组组织数据的文本格式；这里保存脚本实际采集的事实或断言通过后的结果，可供程序读取。">JSON</abbr>。 未执行或失败时按真实状态记录，不能把旧同名文件当作本次成功。

## 1. 从 main 的前置条件开始

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
exec > >(tee -a experiments/inference_infra/reports/d07_checks.log) 2>&1
source /home/lizishuo/miniconda3/etc/profile.d/conda.sh
conda activate nanochat
rg -n 'def bench_generate|def build_prompt|def main|assert|warmup|decode_steps|step_times|json.dumps' scripts/infer_bench.py
sed -n '120,205p' scripts/infer_bench.py
rg -n '^def load_model|checkpoint|tokenizer' nanochat/checkpoint_manager.py
```

原入口依赖：可用 CUDA、单进程、合法 nanochat checkpoint、匹配 tokenizer，以及对应模型配置。<abbr title="checkpoint · 保存的模型状态与配套元数据。nanochat 加载模型需要匹配架构配置与 tokenizer，不能直接把 MiniMind 权重文件替换进去。">checkpoint</abbr> 不是只有 `.pt` 文件就算具备条件；W2 会逐项核对。

今天不要运行完整 `runs/runcpu.sh` 制造权重，它包含下载与长训练。也不要无卡运行 `scripts.infer_bench` 然后把 CUDA 断言失败当算法错误。查看源码无需依赖导入，`sed` 和 `rg` 足够。

## 2. 手动划出 TTFT 的开始和结束

```bash
sed -n '45,90p' scripts/infer_bench.py
sed -n '205,315p' nanochat/engine.py
```

<abbr title="TTFT / Time To First Token · 从约定开始点到第一生成 token 可用的时间。本入口输入已是 token ID，因此不包含文本分词；但包含生成器首次执行中的缓存分配、prefill、复制与采样。">TTFT</abbr> 起点在 `t_start=time.perf_counter()`，终点在第一次 `next(generator)` 后 CUDA synchronize 完成。按这个边界列清单：

| 操作 | 是否在 TTFT 内 | 原因 |
|---|---|---|
| load_model | 否 | main 中先加载 |
| build_prompt/tokenizer.encode | 否 | 计时前完成 |
| `engine.generate(...)` 返回生成器对象 | 否 | 计时前创建；主要函数体尚未执行 |
| prefill 缓存分配与 forward | 是 | 首次 next 才执行 |
| decode 缓存分配与批量复制 | 是 | 首次 yield 之前 |
| 首 token 采样与状态处理 | 是 | 首次 yield 之前 |
| 首 token 后的下一次 forward | 否 | 下次 next 才恢复 |

<abbr title="CUDA synchronize · 等待当前设备已提交的 GPU 工作完成。GPU 执行通常异步；只用 CPU 时钟包住提交动作，会把尚未完成的工作遗漏。">CUDA 同步</abbr> 使墙钟时间包含实际设备工作；同步也有成本，协议必须保持两版边界一致。

不要把代码里的“prefill_time”理解成纯 attention kernel 时间；这里仍是上述较宽的 TTFT 边界，只是用于 prefill 的近似指标。

## 3. 区分生成 token、decode steps 与吞吐分母

后续 `next(generator)` 每产出一列就记录一个 step time。第一列已计入 TTFT，因此理想情况下 G 个生成 token 对应 `G-1` 个后续 decode steps。

<abbr title="TPOT / Time Per Output Token · 首 token 之后生成后续 token 的时间指标。当前入口取单次运行内 step_times 的排序中间值；它不是五次独立运行的中位数。">TPOT</abbr> 的现有计算是 `sorted(step_times)[num_steps//2]`。偶数步数取较高的中间项，不等于两个中间项平均值；W2 必须明确统计定义。

<abbr title="throughput / tokens per second · 单位时间产出的 token 数。此入口以 B*decode_steps 除以所有后续 step 时间之和，排除 TTFT；是 decode 聚合吞吐，不是端到端文本服务吞吐。">tokens/s</abbr> 是 `batch_size*num_steps/sum(step_times)`，需要特别记录以下限制：

- Engine 遇到 `<|assistant_end|>` 或 `<|bos|>` 会结束行；所有行完成才结束整体生成。
- 部分行先结束时，后续列并不代表每行都有同等有用输出。原入口只计列数，`B*steps` 不一定等于实际有效 token 总数。
- tool forced token 有 mask=0；原入口没有细分采样与强制注入的有效 token。
- 发现 `StopIteration` 的那次 `next()` 可能先执行末尾额外 forward，但没有被 append 到 step_times；不能用 raw step sum 当成整个调用的完整耗时。
- `decode_tokens=1` 会出现零后续步，脚本跳过该组；W2 应拒绝不适合 TPOT 的配置或显式标无该指标。

这些是审计事实与口径问题，不是今天授权直接修改上游源码。后续包装应保存逐列、逐行 mask/完成状态，以及真实有效 token 计数，才能作更严格报告。

## 4. 列出重复测量和输入校验的缺口

```bash
sed -n '215,310p' scripts/infer_bench.py
```

对照下面差异表核验总计划硬门槛，可直接说明缺口：

| 项目 | 原入口行为 | W2 要补的契约 |
|---|---|---|
| 预热 | 每 batch 一次，生成 8 token | 至少 3 次，记录边界/配置 |
| 正式运行 | 每组一次 | 至少 5 次独立重复，保留每次原始值 |
| 统计 | 单次内 TPOT 中间值 | 跨运行 median、P95 或范围；定义分位数 |
| 输入长度 | 自动 clamp prompt | 保存 requested/actual P；防 P<=0 |
| 输出长度 | 检查列数是否 G-1 | 保存有效行 token、提前结束原因 |
| dtype/后端 | 有部分信息 | 环境、seed、真实dtype、选中后端均入 raw |
| 显存 | peak allocated | 说明不是 reserved/整卡显存/纯KV增量 |
| 错误 | 运行可能抛异常 | 每组错误也保留，不只保存成功组 |

<abbr title="P95 / 95th percentile · 95% 样本不超过的分位位置；具体计算可用最近秩或插值，两者在五个样本时可能不同。必须写明方法，不能把 P95 当作误差置信区间。">P95</abbr> 对五个样本非常粗糙，因此还应报告样本数与范围。<abbr title="peak allocated memory · PyTorch 记录的峰值已分配张量内存，包含模型与此次运行中活跃的张量。它不同于缓存池 reserved 内存和驱动看到的整进程占用。">峰值 allocated 显存</abbr> 也不等于 KV 本体字节数。

`P+G<=sequence_len` 是计划使用的安全契约；源码还会 clamp P，负/零长度的输入不能靠 clamp 自动修好。不能把 requested P=512 的截短运行仍记作 512 组。

## 5. 当天自验证与交流问题

完成前面的检查后，按下面当天问题自行验证，再直接在聊天里与我交流。每题可用短句、数值和证据路径回答；未执行、失败或不理解的地方如实说明，无需排版或长文。`experiments/inference_infra/reports/d07_benchmark_audit.md` 仅供需要时可选整理，不是必交文件。

1. 原基准入口要满足哪些 CUDA、进程、checkpoint 与 tokenizer 条件？哪些已由真实证据确认，哪些还阻塞？只静态阅读就明确未运行。
2. 对照源码划出 TTFT 起止：加载模型、分词、创建生成器、缓存分配、批量复制、首 token 采样哪些在计时内？为什么变量名 `prefill_time` 不能当纯 attention 时间？
3. 生成 G 个 token 为什么通常只有 G-1 个后续 decode step？原 TPOT 在偶数步数时怎样取中间项，与五次独立运行统计有什么不同？
4. 吞吐分子/分母是什么？提前结束行、tool mask=0、发现 StopIteration 前的额外 forward 各会造成什么统计边界问题？指出对应源码。
5. 对照第 4 节差异表，W2 在 warmup/repeats、requested/actual P、有效 token、错误记录和峰值显存定义上要补什么？选一组 P/G 说明合法条件与截短风险。
6. 你今天有哪些结论仅来自静态审计，哪些另有 CPU/GPU 运行证据？给出对应路径或函数位置，并明确仍未测量的性能项与 W2 首要目标。

今天审计没有必须新增的 GPU 实验脚本，因为 W2 会提供真正重复测量入口。需补 CPU 证据时直接用[真实小模型探查](scripts/d04_probe_model_and_cache.py)或[测试捕获](scripts/d03_run_inference_tests.py)，不要另写一个冒充 GPU 的计时脚本。

## 6. W1 gate：用五分钟口述检验自己

合上源码，逐条口述或直接在聊天中回答“能解释 / 需补读”，再打开源码核验：

1. 无缓存 GPT.generate 为什么每步处理越来越长的前缀？
2. Engine 第一 token 为什么只需要一次 B=1 prefill，再扩到 B 个样本？
3. K/V shape、位置向量和 prev_embedding 分别存什么？
4. CPU 测试通过后，为什么 FA3 仍未验收？
5. 现有 TTFT 是否包括 tokenizer、KV 分配和批量复制？
6. 为什么一次正式运行以及五个 decode step 都不等于五次独立重复？

W1 最低完成要求是各日特定问题已有回答，以及所执行检查的 JSON、测试日志等原始证据可打开；可直接引用聊天回答和自动结果路径，所有人工 Markdown 报告均可选。没有执行的模型/测试明确标 BLOCKED，不能因文档齐全就把运行门槛判通过。

交流本周四项：已验证内容、可打开的证据路径、最大阻塞、下一周单一目标。附真实日期、commit、实际命令与状态；用时可选。下一周先做 tokenizer/数据检查，再准备合法 nanochat checkpoint，不直接跳到性能加速。

上一篇：[D6 后端路由](d06_attention_backends.md)。下一阶段：[W2 D1 数据前置核查](../w2/d08_data_prerequisites.md)。
