---
source_files:
  - nanochat/engine.py
  - nanochat/gpt.py
  - nanochat/checkpoint_manager.py
  - nanochat/flash_attention.py
  - scripts/infer_bench.py
  - tests/test_engine.py
  - tests/test_attention_fallback.py
doc_type: tutorial
knowledge_role: supporting
knowledge_scope: W3第3日实操与验收
---

# W3 D3 · 单样本基线与无 GPU 分流

[上一日](d16_baseline_matrix.md) · [下一日](d18_batch_runs.md)。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[8.1 显式混合精度与梯度缩放](../../llm-training-workflow/08-precision-and-training-acceleration.md#reading-8-1)；[8.3 编译与动态输入](../../llm-training-workflow/08-precision-and-training-acceleration.md#reading-8-3)。
- **今天重点**：结合今天的真实设备和 dtype，区分精度选择、编译准备与正式测量。仅在需要理解 <abbr title="FP8 · 8 位浮点计算；本项目的可选训练加速方式，需要兼容硬件。">FP8</abbr> 模型来源时选读 8.2 的实现限制。
- **暂时跳过**：没有对应 GPU 时不做 FP8 实测；本日不要求理解动态量化或优化内核。

**读完能解释**：编译或预热耗时与正式生成延迟怎样区分？CPU 实测能支持哪些结论，哪些仍需 GPU？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 开始前：固定终端和记录方式

本日命令均从 nanochat 仓库根执行。当前终端会自动激活 nanochat 环境，无需重复激活；先核对解释器路径，不创建另一个 `.venv`。

日志只在 `{ ...; }` 命令组执行期间记录；<abbr title="Bash process substitution · 进程替换。这里把命令组的输出交给 tee，同时显示并追加到日志；重定向在命令组结束后恢复。">进程替换</abbr>不会持续接管交互终端，因此提示符和 VS Code 终端集成控制字符不会混入日志。命令组在当前 shell 执行，变量设置会保留；代码块返回最后一条命令的退出码，需核对单条命令时立即记录 `$?`。已有的显式文件重定向仍写入各自文件。复制时包含外层花括号和末尾重定向。若之前执行过旧版全局日志命令，先关闭旧终端并新开终端；仅修改文档不会停止旧终端的记录。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
{
python -c "import sys; print(sys.executable)"
git status --short
} > >(tee -a experiments/inference_infra/reports/d17_checks.log) 2>&1
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。依赖是否已装好要另查，激活成功不代表 PyTorch 可用。保留已有改动，遇到未认识的变更不要还原。教程脚本保存在本周 `scripts/`，结果写入 `experiments/inference_infra/`，两者不能混为同一类产物。

本日验证完成后，直接按文末“本日自验证与交流”逐题和我交流，附关键输出或自动日志/原始结果路径即可。无需手写报告，也不要求固定排版或长文。实际日期、设备、命令参数与退出码以日志和聊天核对；标题里的 W3 Dn 是学习顺序，不倒填完成状态。若想自行整理，可选用 `experiments/inference_infra/reports/d15_experiment-log.md`，后续跨日也可引用本次聊天和自动结果。

## 今天的目标（约 2–4 小时，计时任务可缩短）

今天检查 B=1 的三个输入长度与两个生成长度中可运行的子集。每个有效 GPU case 至少 5 次正式运行；无 GPU 时验证 CPU 契约和执行入口，GPU 结果留 `PENDING_GPU`。前置：昨天的固定模型、矩阵和协议；不因为日历推进而跳过 W2 加载检查。

<abbr title="TTFT / Time To First Token · 首token时延；本教程从第一次next开始到首个yield完成并同步，包含Engine缓存分配、预填、复制及采样，不包括tokenizer与模型加载。">TTFT</abbr> 包含首个输出前的 Engine 工作。<abbr title="TPOT / Time Per Output Token · 后续输出步平均耗时；本周每次运行用首步之后的逐步同步时间求算术平均，B大于1时为一列token的步时间。">TPOT</abbr> 衡量首 token 之后每步时间，不能把 TTFT 简单当成纯矩阵乘的 prefill 时间。

## 第一步：核对设备和依赖

```bash
{
python -c "import torch; print(torch.__version__); print('cuda_available=',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py --help
} > >(tee -a experiments/inference_infra/reports/d17_checks.log) 2>&1
```

若 import 失败，记录解释器和错误，先完成环境修复，不把“无 torch”误写成“CPU 测试通过”。有 GPU 但返回 false 时检查所装 PyTorch 版本与驱动；今天不自动租卡或联网安装。

## 第二步 A：有 GPU，显式执行 B=1

在同一终端恢复昨天的真实 META、MODEL_TAG、CHECKPOINT_STEP 和 NANOCHAT_BASE_DIR。以下命令使用[矩阵驱动](scripts/d16_matrix.py)，从项目根运行：

```bash
{
python .agents/docs/career_roadmap/w3/scripts/d16_matrix.py \
  --artifact-day 17 --meta "$META" --source base --model-tag "$MODEL_TAG" --step "$CHECKPOINT_STEP" \
  --batch 1 --device cuda --execute \
  --output-dir experiments/inference_infra/results/raw/w3_b1_attempt01
} > >(tee -a experiments/inference_infra/reports/d17_checks.log) 2>&1
```

`--execute` 是实际运行开关；去掉它仅列计划。程序逐 case 等待并保存完整 log；`PROCESS_OK` 仅说明子进程退出码为 0，不等于全部正式行都有效。其余 B=4 case 仍列 PENDING。顺序固定有利于首次核查，正式分析仍需考虑温度、时钟和背景负载；不要把一次跑出来的最小值当成绩。

在运行中观察机器负载和日志；一组耗时明显超出预算可用 Ctrl+C 停止，保留已落盘行，在聊天注明中断时间，或可选补入日志。中断可能导致矩阵缺未落盘 case，不能标成全矩阵完成；恢复用新的 attempt 目录。

## 第二步 B：无 GPU，执行一个小 CPU 契约

先保留昨天 GPU 计划不动，再用[W2 基准脚本](../w2/scripts/d12_benchmark.py) 的随机小模型做短输入验证。<abbr title="Smoke test · 冒烟验证；用很小工作量检查接口、文件格式与执行路径，不证明模型质量或GPU性能。">smoke</abbr> 只检查程序是否形成闭环。

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py \
  --device cpu --random-model --prompt-tokens 32 --decode-tokens 8 --batch-size 1 \
  --warmup 3 --repeats 5 --seed 42 \
  --output experiments/inference_infra/results/raw/d17_p32_g8_b1_cpu_contract.jsonl
python -m pytest tests/test_engine.py tests/test_attention_fallback.py -q
} > >(tee -a experiments/inference_infra/reports/d17_checks.log) 2>&1
```

随机权重/测试 tokenizer 是接口夹具，不能代替实际 checkpoint 的性能报告。CPU SDPA 测试通过也不证明 FA3 通过，GPU 测试被跳过必须单列 skip 数。CPU 机器资源有限时先只跑 Engine 测试，完整 fallback 单测后续补记。

测试入口为 `python -m pytest`，输入为 `tests/test_engine.py` 与 `tests/test_attention_fallback.py`；需已安装 pytest/PyTorch。两者是上游原生测试文件，保留命名；测试输出保存在 `experiments/inference_infra/reports/d17_checks.log`，未配置报告插件，不会自动生成独立测试报告。

## 第三步：看一行真实原始记录

用下面只读命令，替换为你刚生成的 case 文件；不要拿 matrix.jsonl 当 benchmark raw。

```bash
{
python -c "import json,pathlib; p=pathlib.Path('experiments/inference_infra/results/raw/d17_p32_g8_b1_cpu_contract.jsonl'); r=[json.loads(x) for x in p.read_text().splitlines()]; print(json.dumps(r[-1],indent=2,ensure_ascii=False))"
} > >(tee -a experiments/inference_infra/reports/d17_checks.log) 2>&1
```

检查 `phase` 区分 warmup/measure；正式至少 5 行，`status=ok` 才候选有效。`actual_decode_steps` 是 yield 列数，`generated_tokens` 是排除终止、forced 和完成行 padding 后的有效 token 总数，两者不可互换；B=1 也可能因工具规则而不同。检查 TTFT、decode_step_ms、elapsed_ms 都为正且单位 ms，`env`、model_config、后端、精度都有记录。

## 失败分流、产物与验收

- `early_stop`：保留原始行，检查 assistant_end/bos 或工具强制 token；不能通过修改 EOS 语义把它包装成原协议成功。
- OOM/error：查看同名 log 的异常；模型加载期失败可能没有 raw 行，但进程退出码仍有记录。
- `INVALID_CONTEXT`：今天不运行；应扩大合法 checkpoint 上下文并重建不同实验，而非悄悄截断 P。
- 正式有效不足 5 次：可继续补跑同条件新 attempt，但跨目录合并前审计模型、env 和 run_id；今天可标不足样本。

按文末问题在聊天核对每 case 的计划状态、退出码、measure总数/有效数、raw与log路径；`experiments/inference_infra/reports/d17_baseline_b1.md` 仅为可选整理位置。验收：所有 B1 候选都可追踪；至少一个可用设备的功能验证或明确依赖阻塞；没有将 CPU/random 数字写成 GPU baseline。下一天专看批量行为。

## 本日脚本与产物清单

下列路径以项目根目录为基准；脚本编号保留首次使用日。外部历史归档与上游 checkpoint 原生文件名保留，按正文真实绝对路径或环境变量解析；不冒充项目新产物。<abbr title="条件生成：只有指定分支执行、前置文件满足且实际写入后，文件才算存在；清单中的目标路径本身不证明生成成功。">条件生成</abbr>须在验收时核对。只打印的检查统一进入本日 checks 日志，不另称生成报告。

| 脚本完整项目相对路径 | 用途 | 阶段 | 输入路径或来源 | 运行条件 |
|---|---|---|---|---|
| [.agents/docs/career_roadmap/w3/scripts/d16_matrix.py](scripts/d16_matrix.py) | 固定12个组合，按开关执行子集 | 计划/显式运行 | META 指向真实上游 meta；MODEL_TAG、CHECKPOINT_STEP | 计划需真实 meta；--execute 需权重/tokenizer，CUDA运行需GPU |
| [.agents/docs/career_roadmap/w2/scripts/d12_benchmark.py](../w2/scripts/d12_benchmark.py) | 矩阵内部调用；CPU分支直接调用 | 逐次测量 | checkpoint及合成prompt；CPU示例使用随机夹具 | 依赖 PyTorch；真实基准需匹配 tokenizer |

| 产物完整项目相对路径 | 用途 | 生成者/步骤 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/reports/d17_checks.log` | 环境、只读检查、命令输出及错误 | 各 Bash 代码块的 tee | 自动追加各代码块内的输出；退出码在聊天核对，或可选补记日志 |
| `experiments/inference_infra/reports/d15_experiment-log.md` | 每日事实、退出码和阻塞 | 自愿整理；跨日可引用聊天和自动结果 | 可选人工笔记，不影响验收 |
| `experiments/inference_infra/results/raw/w3_b1_attempt01/d17_matrix.jsonl` | 12个候选状态、命令与原始输出目标 | d16_matrix.py | 自动生成；计划中的 raw_file 仅为目标，不保证文件存在 |
| `experiments/inference_infra/results/raw/w3_b1_attempt01/d17_p32_g32_b1.log` | 对应组合的子进程输出/错误 | d16_matrix.py | 满足上下文并选中该批量、--execute 时生成 |
| `experiments/inference_infra/results/raw/w3_b1_attempt01/d17_p32_g32_b1.jsonl` | 该组合逐次测量行 | d12_benchmark.py（矩阵调用） | 条件生成；模型加载异常写 error 原始行；解释器启动、导入或参数解析失败可能只有 log，无 raw |
| `experiments/inference_infra/results/raw/w3_b1_attempt01/d17_p32_g128_b1.log` | 对应组合的子进程输出/错误 | d16_matrix.py | 满足上下文并选中该批量、--execute 时生成 |
| `experiments/inference_infra/results/raw/w3_b1_attempt01/d17_p32_g128_b1.jsonl` | 该组合逐次测量行 | d12_benchmark.py（矩阵调用） | 条件生成；模型加载异常写 error 原始行；解释器启动、导入或参数解析失败可能只有 log，无 raw |
| `experiments/inference_infra/results/raw/w3_b1_attempt01/d17_p128_g32_b1.log` | 对应组合的子进程输出/错误 | d16_matrix.py | 满足上下文并选中该批量、--execute 时生成 |
| `experiments/inference_infra/results/raw/w3_b1_attempt01/d17_p128_g32_b1.jsonl` | 该组合逐次测量行 | d12_benchmark.py（矩阵调用） | 条件生成；模型加载异常写 error 原始行；解释器启动、导入或参数解析失败可能只有 log，无 raw |
| `experiments/inference_infra/results/raw/w3_b1_attempt01/d17_p128_g128_b1.log` | 对应组合的子进程输出/错误 | d16_matrix.py | 满足上下文并选中该批量、--execute 时生成 |
| `experiments/inference_infra/results/raw/w3_b1_attempt01/d17_p128_g128_b1.jsonl` | 该组合逐次测量行 | d12_benchmark.py（矩阵调用） | 条件生成；模型加载异常写 error 原始行；解释器启动、导入或参数解析失败可能只有 log，无 raw |
| `experiments/inference_infra/results/raw/w3_b1_attempt01/d17_p512_g32_b1.log` | 对应组合的子进程输出/错误 | d16_matrix.py | 满足上下文并选中该批量、--execute 时生成 |
| `experiments/inference_infra/results/raw/w3_b1_attempt01/d17_p512_g32_b1.jsonl` | 该组合逐次测量行 | d12_benchmark.py（矩阵调用） | 条件生成；模型加载异常写 error 原始行；解释器启动、导入或参数解析失败可能只有 log，无 raw |
| `experiments/inference_infra/results/raw/w3_b1_attempt01/d17_p512_g128_b1.log` | 对应组合的子进程输出/错误 | d16_matrix.py | 满足上下文并选中该批量、--execute 时生成 |
| `experiments/inference_infra/results/raw/w3_b1_attempt01/d17_p512_g128_b1.jsonl` | 该组合逐次测量行 | d12_benchmark.py（矩阵调用） | 条件生成；模型加载异常写 error 原始行；解释器启动、导入或参数解析失败可能只有 log，无 raw |
| `experiments/inference_infra/results/raw/d17_p32_g8_b1_cpu_contract.jsonl` | 无 GPU 分支的短输入随机模型记录 | d12_benchmark.py | 仅执行 CPU 分支且成功写入后存在 |
| `experiments/inference_infra/reports/d17_baseline_b1.md` | 候选状态、有效数及失败边界 | 末尾交流后自愿整理 | 可选人工笔记，聊天和原始结果可替代 |

## 本日自验证与交流

按题号在聊天中回答即可，预计交流约 15–20 分钟，不要求另写报告。开头一次说明本次实际走 GPU checkpoint 还是 CPU 随机模型分支，并附日期/版本及一份可定位的日志/raw；之后只引用支撑判断的关键片段，不必重复环境、命令和路径。未执行写 `PENDING`，历史结果注明归属；GPU 待测保留 `PENDING_GPU`，不为答题增加长任务。

1. 根据今天真实检查的结果，你有条件完成哪一级验证？哪一条前置证据决定了不能把它叫作更高一级的基线？

   **核对要点**：结合依赖可用性、设备、模型与 tokenizer 加载情况作判断，区分依赖阻塞、CPU/random 契约和真实 GPU checkpoint 测量。引用实际关键输出；不能因计划存在、CUDA 可用或脚本帮助能打开就声称模型基准通过。

2. 抽一条实际正式运行，判断它是否完成了要求的固定长度工作量。假如 `status=ok`，你仍要核对哪些相互约束的数据？

   **核对要点**：用该行的 G、实际输出列数、后续间隔数、每行有效 token 和总数逐项交叉核查；检查时间为正且有限。能解释终止/forced token 为何可能使列数与有效输出不同；没有真实行时可推演字段关系，实际结果继续待验。

3. 不看进程的成功提示，只依据一个 session 的原始行，今天有多少次预热、多少次正式运行、多少次完整有效？这些证据允许它进入什么类型的后续统计？

   **核对要点**：从 `phase/status` 与第 2 题的完整性条件核数，保留失败和不足样本。说明预热与正式样本的用途，核对至少 3 次成功预热及至少 5 次有效正式运行的门槛；CPU/random 即使满足次数仍属于契约验证。`PROCESS_OK` 不能替代这份判断。

4. 选今天一处真实失败或可疑结果，说明你会如何定位并恢复。若未遇到失败，请推演“只看到非零退出码、没有 raw”的情况。

   **核对要点**：先区分解释器/导入/参数解析、模型加载和生成阶段，再说明查哪段 case log 或原始错误行；证据不足就标未知。恢复保留旧文件并用新 attempt，说明下一次最小检查；不通过删失败行、改 EOS 规则或换模型沿用原名制造成功，也不要求故意注入故障。

## 核对记录与完成标记

本文基于当前仓库相关生成、加载、注意力路由及原基准代码编写；源码阅读不能代替执行。本教程的交付状态不表示读者已经完成当天实验。按当天问题核对真实证据后再勾选；聊天回答即可，无需额外报告：

- [ ] 输入条件与前一天产物已核实。
- [ ] 本日命令在明确设备执行，或具体阻塞已记录。
- [ ] 自动日志和原始输出可定位，实践材料或聊天回答可回溯；可选笔记无需生成。
- [ ] 示例数值与真实结果已区分，失败和待测未删除。
- [ ] 能说明本日证据证明什么、尚不能证明什么。
