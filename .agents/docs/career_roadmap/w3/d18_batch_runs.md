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
knowledge_scope: W3第4日实操与验收
---

# W3 D4 · 批量生成与失败记录

[上一日](d17_single_batch_runs.md) · [下一日](d19_statistics_audit.md)。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[12.2 生成控制流与返回格式](../../llm-training-workflow/12-inference-and-generation.md#reading-12-2)；[5.3 batch、epoch 和每卡分配](../../llm-training-workflow/05-data-encoding-and-batching.md#reading-5-3)。
- **今天重点**：回看一次提示预填充后复制到多样本行的行为，结合有效输出数理解批量对总耗时、总吞吐及缓存容量的影响。
- **暂时跳过**：多卡通信和独立请求调度不属于今天的同提示批量实验；不由批量增大直接推断一定加速。

**读完能解释**：B=4 在这个 Engine 中代表什么？为什么不能直接称为四条独立请求的服务吞吐？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 开始前：固定终端和记录方式

本日命令均从 nanochat 仓库根执行。当前终端会自动激活 nanochat 环境，无需重复激活；先核对解释器路径，不创建另一个 `.venv`。

日志只在 `{ ...; }` 命令组执行期间记录；<abbr title="Bash process substitution · 进程替换。这里把命令组的输出交给 tee，同时显示并追加到日志；重定向在命令组结束后恢复。">进程替换</abbr>不会持续接管交互终端，因此提示符和 VS Code 终端集成控制字符不会混入日志。命令组在当前 shell 执行，变量设置会保留；代码块返回最后一条命令的退出码，需核对单条命令时立即记录 `$?`。已有的显式文件重定向仍写入各自文件。复制时包含外层花括号和末尾重定向。若之前执行过旧版全局日志命令，先关闭旧终端并新开终端；仅修改文档不会停止旧终端的记录。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
{
python -c "import sys; print(sys.executable)"
git status --short
} > >(tee -a experiments/inference_infra/reports/d18_checks.log) 2>&1
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。依赖是否已装好要另查，激活成功不代表 PyTorch 可用。保留已有改动，遇到未认识的变更不要还原。教程脚本保存在本周 `scripts/`，结果写入 `experiments/inference_infra/`，两者不能混为同一类产物。

本日验证完成后，直接按文末“本日自验证与交流”逐题和我交流，附关键输出或自动日志/原始结果路径即可。无需手写报告，也不要求固定排版或长文。实际日期、设备、命令参数与退出码以日志和聊天核对；标题里的 W3 Dn 是学习顺序，不倒填完成状态。若想自行整理，可选用 `experiments/inference_infra/reports/d15_experiment-log.md`，后续跨日也可引用本次聊天和自动结果。

## 今天的目标与前置（约 3 小时）

先确认昨天 B=1 至少有可解释的执行结果，再测 B=4；如果 B1 就 OOM，今天优先记录边界，不通过换成另一模型后仍沿用同名基线。<abbr title="Batch size · 批大小；Engine此处为一次prefill后复制缓存的num_samples，每行共享同一prompt，不能直接等同四个独立请求。">batch</abbr> 增大可能提高总 token 吞吐，也可能增加单步耗时和显存；不能预设“四倍 batch 几乎免费”。

时间分配：30 分钟检查内存预算，60–90 分钟执行，40 分钟检查逐行完成数，30 分钟核对失败证据与交流。

## 第一步：重看昨天配置与缓存公式

从昨天 log 确认模型、dtype、后端、P/G 和 seed。比较 B=1 与 B=4 时保持这些条件相同。Engine 在 batch=1 预填后复制 KV，因此 B4 的 TTFT 多了缓存复制和更大分配；不是四次完整 prefill。

<abbr title="OOM / Out Of Memory · 内存不足；本处应区分模型加载期、缓存分配期和计算期。失败说明当前配置资源不足，不能作为正常延迟样本。">OOM</abbr> 必须保留错误与输入组合。使用第 2 天公式估计 B4 解码缓存约为 B1 的四倍，但总峰值显存通常不会精确四倍，因为权重等共享部分仍存在。GPU reserved 与 allocated 不同：前者包括分配器保留，后者反映当前张量占用。

## 第二步：执行 B4 的可运行子集

使用[矩阵驱动](scripts/d16_matrix.py)，仍把实际 META 等变量从协议恢复。默认至少 5 次正式运行，CPU 分支不以速度为今天的目标。

```bash
{
python .agents/docs/career_roadmap/w3/scripts/d16_matrix.py \
  --artifact-day 18 --meta "$META" --source base --model-tag "$MODEL_TAG" --step "$CHECKPOINT_STEP" \
  --batch 4 --device cuda --execute \
  --output-dir experiments/inference_infra/results/raw/w3_b4_attempt01
} > >(tee -a experiments/inference_infra/reports/d18_checks.log) 2>&1
```

若无 CUDA，先不要执行以上命令，GPU case 保留计划状态；另跑下面短输入的 CPU 合约例子：

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py \
  --device cpu --random-model --prompt-tokens 32 --decode-tokens 8 --batch-size 4 \
  --warmup 3 --repeats 5 --seed 42 \
  --output experiments/inference_infra/results/raw/d18_p32_g8_b4_cpu_contract.jsonl
} > >(tee -a experiments/inference_infra/reports/d18_checks.log) 2>&1
```

没有必要在 CPU 上强跑长 P512×G128 并等待数小时。GPU 候选完整性由 matrix 状态表示，CPU 小样本只用于检查 B4 接口与有效 token 记账。

## 第三步：检查有效 token，而非假定 B×G

打开 raw 的 `per_row_generated_tokens`。满长度正常运行时应有 4 项，每项 G，且和为 `generated_tokens`。某行提前终止后 Engine 仍可能产生列；不能把列数乘 B 当全部有效输出。forced token 是工具返回等引入的 token，其工作量和普通采样不同，本周统计排除并将未达到完整条件的行留作异常。

<abbr title="EOS / End Of Sequence · 序列结束标记；Engine实际上按assistant_end或bos等项目专用标记终止行，不能假定所有生成器都共享同一个EOS规则。">EOS</abbr> 在此是结束行为的统称；GPT.generate 简单循环不会采用同一套 Engine 行状态规则。由此带来的有效 token 差异不应自动判作 cache 数值错误。

对比 `actual_decode_steps`、`decode_step_ms` 长度和有效 token 数。G 步输出通常包含一次 TTFT 和 G−1 个后续步；最后 StopIteration 的额外边界不作为已输出 token 的延迟。基准封装每步同步会引入开销，应保持 B1/B4 方法相同。

## 第四步：核对成功与失败的边界

每个 P/G 下对 B1 与 B4 核查以下内容，直接在聊天交流即可；可选整理为 `experiments/inference_infra/reports/d18_baseline_batch.md`，无需另写报告：

```text
P/G/B：
checkpoint / device / precision / backend：
raw / process log：
measure数 / 完整有效数 / early_stop数 / error数：
实际每行有效token：
OOM发生阶段（从异常位置核查，不能猜）：
峰值allocated/reserved（CPU写N/A）：
是否满足进入正式统计门槛：
```

可以新增 B2 作为资源边界探索，但必须新目录、原始字段 batch_size=2，并在交流中声明它不属于原定 12 组；本周 d16_matrix.py 仅接受原计划 B1/B4。不要覆盖 B4 失败行。

## 验收与下一步

B4 在同条件下成功或失败都可以验收“边界记录完成”；只有 >=5 有效 GPU 正式行才能写该 case 基线。若 B1/B4 工作量不等、模型或后端不同，先标不可直接比较。完成后保留全部 raw、log 和失败原因，明天统一自动汇总，不手工抄最快值。

## 本日脚本与产物清单

下列路径以项目根目录为基准；脚本编号保留首次使用日。外部历史归档与上游 checkpoint 原生文件名保留，按正文真实绝对路径或环境变量解析；不冒充项目新产物。<abbr title="条件生成：只有指定分支执行、前置文件满足且实际写入后，文件才算存在；清单中的目标路径本身不证明生成成功。">条件生成</abbr>须在验收时核对。只打印的检查统一进入本日 checks 日志，不另称生成报告。

| 脚本完整项目相对路径 | 用途 | 阶段 | 输入路径或来源 | 运行条件 |
|---|---|---|---|---|
| [.agents/docs/career_roadmap/w3/scripts/d16_matrix.py](scripts/d16_matrix.py) | 固定12个组合，按开关执行子集 | 计划/显式运行 | META 指向真实上游 meta；MODEL_TAG、CHECKPOINT_STEP | 计划需真实 meta；--execute 需权重/tokenizer，CUDA运行需GPU |
| [.agents/docs/career_roadmap/w2/scripts/d12_benchmark.py](../w2/scripts/d12_benchmark.py) | 矩阵内部调用；CPU分支直接调用 | 逐次测量 | checkpoint及合成prompt；CPU示例使用随机夹具 | 依赖 PyTorch；真实基准需匹配 tokenizer |

| 产物完整项目相对路径 | 用途 | 生成者/步骤 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/reports/d18_checks.log` | 环境、只读检查、命令输出及错误 | 各 Bash 代码块的 tee | 自动追加各代码块内的输出；退出码在聊天核对，或可选补记日志 |
| `experiments/inference_infra/reports/d15_experiment-log.md` | 每日事实、退出码和阻塞 | 自愿整理；跨日可引用聊天和自动结果 | 可选人工笔记，不影响验收 |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_matrix.jsonl` | 12个候选状态、命令与原始输出目标 | d16_matrix.py | 自动生成；计划中的 raw_file 仅为目标，不保证文件存在 |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p32_g32_b4.log` | 对应组合的子进程输出/错误 | d16_matrix.py | 满足上下文并选中该批量、--execute 时生成 |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p32_g32_b4.jsonl` | 该组合逐次测量行 | d12_benchmark.py（矩阵调用） | 条件生成；模型加载异常写 error 原始行；解释器启动、导入或参数解析失败可能只有 log，无 raw |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p32_g128_b4.log` | 对应组合的子进程输出/错误 | d16_matrix.py | 满足上下文并选中该批量、--execute 时生成 |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p32_g128_b4.jsonl` | 该组合逐次测量行 | d12_benchmark.py（矩阵调用） | 条件生成；模型加载异常写 error 原始行；解释器启动、导入或参数解析失败可能只有 log，无 raw |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p128_g32_b4.log` | 对应组合的子进程输出/错误 | d16_matrix.py | 满足上下文并选中该批量、--execute 时生成 |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p128_g32_b4.jsonl` | 该组合逐次测量行 | d12_benchmark.py（矩阵调用） | 条件生成；模型加载异常写 error 原始行；解释器启动、导入或参数解析失败可能只有 log，无 raw |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p128_g128_b4.log` | 对应组合的子进程输出/错误 | d16_matrix.py | 满足上下文并选中该批量、--execute 时生成 |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p128_g128_b4.jsonl` | 该组合逐次测量行 | d12_benchmark.py（矩阵调用） | 条件生成；模型加载异常写 error 原始行；解释器启动、导入或参数解析失败可能只有 log，无 raw |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p512_g32_b4.log` | 对应组合的子进程输出/错误 | d16_matrix.py | 满足上下文并选中该批量、--execute 时生成 |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p512_g32_b4.jsonl` | 该组合逐次测量行 | d12_benchmark.py（矩阵调用） | 条件生成；模型加载异常写 error 原始行；解释器启动、导入或参数解析失败可能只有 log，无 raw |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p512_g128_b4.log` | 对应组合的子进程输出/错误 | d16_matrix.py | 满足上下文并选中该批量、--execute 时生成 |
| `experiments/inference_infra/results/raw/w3_b4_attempt01/d18_p512_g128_b4.jsonl` | 该组合逐次测量行 | d12_benchmark.py（矩阵调用） | 条件生成；模型加载异常写 error 原始行；解释器启动、导入或参数解析失败可能只有 log，无 raw |
| `experiments/inference_infra/results/raw/d18_p32_g8_b4_cpu_contract.jsonl` | 无 GPU 分支的短输入随机模型记录 | d12_benchmark.py | 仅执行 CPU 分支且成功写入后存在 |
| `experiments/inference_infra/reports/d18_baseline_batch.md` | 候选状态、有效数及失败边界 | 末尾交流后自愿整理 | 可选人工笔记，聊天和原始结果可替代 |

## 本日自验证与交流

可直接在聊天中按题号回答，不要求手写报告。每题给出结论和可定位依据（源码位置、命令与退出码、日志片段或自动结果字段）；同一份证据可复用。概念推演须注明假设，实际运行须区分本次与历史结果。未执行写 `PENDING`，失败写真实错误和影响范围，不能据此勾选实验通过。

1. **本日分支与 B1 前置**：D17 的 B1 结果是否足以解释配置和资源边界？本次走 GPU checkpoint B4 还是 CPU random B4，判断依据是什么？

   **通过标准**：引用真实模型加载、设备与B1日志，缺GPU保留 PENDING_GPU；CPU短输入只验批量契约，不把另一随机模型计作同一checkpoint的比较。

2. **比较条件是否一致**：选相同 P/G 的 B1与B4，模型/词表/输入哈希、seed、dtype、后端、环境与计时方法是否一致？不同session如何追溯？

   **通过标准**：列两份raw/log及身份字段，说明改变B以外条件就不能直接归因于batch；session分开统计，不能把同prompt复制说成独立请求服务吞吐。

3. **TTFT 与批量复制**：B4 从 prefill 到首次yield怎样分配和复制缓存，哪些工作在TTFT内？为什么不是四次完整prefill，TTFT也不保证与B1相同？

   **通过标准**：引用 Engine两次分配与measure首next边界，说明分配、复制、首采样和同步成本；这是机制解释，不能未测就写出B4的性能增减。

4. **缓存预算与峰值**：用实际配置手算B1/B4的解码K/V容量及复制时共存量；与实测peak_allocated/reserved怎样比较？

   **通过标准**：列L/Hkv/Dh/P/G/元素字节来源及公式，区分KV四倍与共享权重/临时张量；CPU字段不适用，估算不证明一定装得下，缺实测留待验。

5. **四行有效输出**：抽一条B4正式行，逐项读出per_row_generated_tokens、generated_tokens、actual_decode_steps和间隔数；G=8完整时各应满足什么关系？

   **通过标准**：真实行应与4行各8、总32、8列、7间隔的契约核对；如不完整解释真实status和证据，不用名义B×G遮蔽终止/forced。

6. **终止与工具边界**：Engine某行结束后仍可能产出列，但测量封装怎样处理任一行终止？mask=0的工具输出是否计为普通有效token？

   **通过标准**：沿Engine completed/forced_tokens与measure any(completed)/counts/forced对应；不把EOS语义差异直接判成cache数值错误，没有故障实例可按源码推演。

7. **复算批量指标**：从一个有效B4 run复算TPOT与tokens/s，再与同条件B1组的统计比较；为何TPOT不除4、聚合吞吐也不能重复乘4？

   **通过标准**：附run_id、原始间隔和有效数，统一单位；先说明每run均值再跨run统计，不从单个最快值或不同工作量得出加速结论。

8. **候选状态与执行证据**：六个B4候选哪些成功、失败、非法上下文或未执行？matrix进程结束与PROCESS_OK能否保证每组至少5个有效正式样本？

   **通过标准**：对应d18_matrix.jsonl、case log/raw、returncode及phase/status计数；保留未选中B1候选状态，不把manifest目标路径当成已存在测量。

9. **OOM 与资源探索**：如发生OOM，证据能否定位加载、cache分配或计算阶段？若改B2继续探索，怎样保留原B4失败并区别原矩阵？

   **通过标准**：引用首个异常栈，不能定位就写未知；B2须用benchmark直接CLI与新目录/文件、真实batch_size=2，matrix仅接受B1/B4，不删除或覆盖失败记录。

10. **边界记录与性能放行**：本日哪些case只完成边界记录，哪些具备进入正式GPU统计的条件？缺GPU、有效样本不足或B1/B4条件不同时分别标什么？

    **通过标准**：至少核对正式有效数、同session预热和模型身份；成功/失败可完成事实记录，但GPU基线需真实数据与样本门槛，保留全部raw/log，不为问答补跑长实验。

## 核对记录与完成标记

本文基于当前仓库相关生成、加载、注意力路由及原基准代码编写；源码阅读不能代替执行。本教程的交付状态不表示读者已经完成当天实验。按当天问题核对真实证据后再勾选；聊天回答即可，无需额外报告：

- [ ] 输入条件与前一天产物已核实。
- [ ] 本日命令在明确设备执行，或具体阻塞已记录。
- [ ] 自动日志和原始输出可定位，实践材料或聊天回答可回溯；可选笔记无需生成。
- [ ] 示例数值与真实结果已区分，失败和待测未删除。
- [ ] 能说明本日证据证明什么、尚不能证明什么。
