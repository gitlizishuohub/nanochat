# nanochat AI Infra 逐日执行指导（2026-10-01 起）

> 版本：2026-09-29；时区：Asia/Shanghai。日期是行动日历，不是保证完成的期限。每个阶段以验收门槛放行；没有通过就继续修复并重排后续技术任务。导师联系和实习投递有外部时效，应按日期推进。

## 一、从已有成果出发

这份计划参照 MiniMind 的 [逐周路线](../../../../minimind/.agents/docs/career_roadmap/1-1-weekly-timeline-2026-10-01.md)、[验收标准](../../../../minimind/.agents/docs/career_roadmap/1-2-execution-standards-and-outreach.md)和[资源与延期规则](../../../../minimind/.agents/docs/career_roadmap/1-3-resources-delays-and-extensions.md)，并按本地仓库实况修正。

截至本计划编写时，MiniMind 的 `experiments/minimind_baseline/README.md` 已记录 D1–D14 的 CPU/FP32 训练、SFT、恢复、KV 一致性、基准和干净 shell 复现。其 D12 八步 logits 最大绝对差约 `2.38e-6`；D13 有 3 次预热、5 次重复的 60 行原始数据。该证据属于 **CPU 功能验证**，没有 GPU 性能结论。MiniMind 在计划中视为已收尾，10 月前两周转给 nanochat；如复核发现缺失，只补相应证据，不重做全套。

当前 nanochat 的 `nanochat/engine.py` 已有预分配 `KVCache` 和 `Engine.generate`；`nanochat/gpt.py` 已有无缓存 `GPT.generate`，并以 `F.rms_norm` 实现无参数 norm；`scripts/infer_bench.py` 已能输出 TTFT、TPOT、吞吐、显存及 JSON，但它只测一次正式运行、只做一次预热，且要求 CUDA 和现成 checkpoint。故本项目主线是 **建立可信重复基线 → trace 定位 → Triton RMSNorm 的受控实验 → 现有 KV Cache 的正确性和瓶颈优化**，不是再写一个同名功能。当前环境能找到 `uv`，未发现 `nvidia-smi` 或项目 `.venv`，不能预设本机可跑 GPU benchmark。

本仓库已有未提交的 `.agents/` 改动，执行时保留它们；实验分支只纳入自己产生的文件。不要把 MiniMind 的训练权重直接给 nanochat 加载：模型结构和 tokenizer 不兼容。

## 二、项目边界、工作量和成果结构

- 基准工作量：每周约 12 小时，工作日用 45–90 分钟推进小任务，两个较完整时段各 3–4 小时做实现或实验。若一周仅有 6–8 小时，按顺序推迟技术日历，不压缩测试和复现。
- 主线只做 nanochat 推理性能。训练仅用于制造小型合法 checkpoint；不以 GPT-2 效果或大型训练为目标。SFT、RL、MoE、WebUI 不进入本期项目。
- `experiments/inference_infra/` 是计划中的正式产物目录，随着进度逐日创建，不搭空文件树。建议含 `README.md`、`env.md`、`protocol.md`、`scripts/`、`tests/`、`results/raw/`、`results/summary/`、`results/traces/`、`reports/`。大 checkpoint、原始数据和临时 profiler 缓存放在可配置数据目录，不提交 Git。
- 每天留一条记录：日期、当前 commit、工作时长、命令/配置、结果文件、通过/失败、下一步。没有实际运行的数据标 `PENDING`，不能填预计加速比。
- 每周结束更新四项：完成物、证据路径、最大阻塞、下一周单一目标。若技术门槛失败，优先修门槛；职业节点仍照日历执行。

### 实验的硬门槛

1. **基线**：固定 commit、GPU/驱动、PyTorch/CUDA/Triton、模型 checkpoint、dtype、batch、prompt tokens、生成 tokens、seed、attention 后端；每组至少 3 次预热、5 次正式独立运行。CUDA 计时须同步；保存每次原始值，汇总中位数和 P95 或范围。早停导致有效生成步数不足的样本须记录并重测。TTFT 包含 tokenizer 与否必须写清；本计划默认仅计 token ID 输入后的模型及 Engine 工作。
2. **Profiler**：分别覆盖 prefill 和 decode，保存可打开的 trace。每个瓶颈结论写出 trace 中的事件名、耗时/占比、shape、推断及下一步实验；不把 profiler 的时间当成正常 benchmark 时间。
3. **RMSNorm**：至少覆盖 token 数 `1/16/128/512`、hidden size `256/768/1024/非 2 的幂`、可用 dtype；非理想 shape 不能省。先在 FP32 累加的参考实现上核对输出，记录最大绝对/相对误差及近零值处理，再测 kernel。仅当相关 shape、dtype、后端均通过时才在推理路径启用；保留 PyTorch 回退。若实测没有端到端收益，报告负结果。
4. **KV Cache**：同一随机权重或 checkpoint 下，cache 与无 cache 的逐步最后位置 logits、greedy token、各层 K/V 写入位置和 `prev_embedding` 状态均通过检查。测试 `P=32/128/512`（受模型 `sequence_len` 限制时缩短生成长度），记录 TTFT、TPOT、tokens/s、分配峰值显存。不要声称 KV Cache 降低总显存。
5. **复现**：另开干净 shell 按 README 重跑 smoke、correctness 和代表性 benchmark；记录失败原因和环境要求。模型质量与系统性能分开叙述。

### 资源门槛与退出条件

- 本地无 CUDA 时，先做 CPU 正确性、接口、测试和实验脚本。Triton kernel 和 GPU 性能数值保持 `PENDING_GPU`，绝不把 CPU 结果冒充 GPU 收益。借卡或租卡前确认可用 GPU、驱动、PyTorch/Triton 兼容性；固定一次 2–4 小时会话，预设费用/时间上限，结束即关机。
- 若到 GPU 实验日仍无卡，按日完成脚本、测试矩阵、预期输出和 trace 分析模板，然后把需卡任务移到获得 GPU 后的第一个会话。不得因为等 GPU 停止可离线推进的工作。
- **最低完整版本**：可信基线 + 一份 trace 与有证据的瓶颈分析 + 一项深做的优化实验（RMSNorm 若有 GPU）+ 回归测试 + 可复现 README。KV 改进和连续批处理属于更深版本；如果 GPU 长期不可用，交付 CPU 正确性报告与可直接在 CUDA 上运行的实验脚本，并明确该版本没有性能结论。

## 三、运行起点与命令边界

从本仓库根目录操作。先运行 `uv sync --extra cpu --group dev`；上 GPU 节点时用 `uv sync --extra gpu --group dev`，不要在同一环境同时选 cpu/gpu extra。随后 `uv run --extra cpu --group dev pytest -q tests/test_engine.py tests/test_attention_fallback.py` 作为本地推理相关起点。若依赖下载受网络限制，记录阻塞，不把未运行写成通过。

要运行 `scripts.infer_bench`，必须先准备 **nanochat 自己的** tokenizer、数据和 base checkpoint。`runs/runcpu.sh` 是完整教学流程，包含大数据下载和 5000 次训练，不能直接当小 smoke 用。小配置可从 `scripts/base_train.py` 顶部示例的 `--depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20` 适配，补 `--run=dummy --sample-every=-1`；运行前先检查 tokenizer 与 dataset 是否齐备。没有数据时，可先用随机权重 GPT 做形状与正确性测试，但这样的数字不能称为真实模型推理性能。

正式 GPU 基准的原始入口示例：`uv run --extra gpu --group dev python -m scripts.infer_bench -i base -g d4 --prompt-tokens 128 --decode-tokens 32 --batch-sizes 1,4`。该原入口 **本身不满足重复测量标准**；完成 W2 的重复运行包装/增强后才用于报告。输入 `P+G` 不得超过 checkpoint 的 `sequence_len`；若脚本自动截短 prompt，必须将实际长度写入结果。

## 四、逐日任务

表中“验收/产物”是当天结束条件。代码和脚本名是建议落点；如实际实现需调整，实验契约与可复现证据不变。周次按原路线的 10 月 1 日为起点，每周 7 天。外部邮件和投递都由本人审核并发送，项目文档只准备材料。

### W1 · 10 月 1–7 日：确认 MiniMind 已收尾，摸清 nanochat 推理路径

| 日期 | 当天任务 | 验收/产物 |
|---|---|---|
| 10-01 | 对照 MiniMind D14 报告、README、KV 测试和 benchmark 原始 CSV；列出已证明与未证明的事。 | `reports/starting_point.md` 写清 CPU 证据与 GPU 空白；不重跑全量训练。 |
| 10-02 | 记录本仓库 commit、dirty 状态、Python/uv/torch、GPU 能力和可用数据目录；规划不覆盖用户改动的实验目录。 | `env.md` 首版；每项缺失标 `N/A`。 |
| 10-03 | 安装 CPU 开发依赖并跑推理相关 pytest；如失败，逐个定位是依赖、环境还是代码。 | 原始命令、退出码和测试摘要保存。 |
| 10-04 | 阅读 `gpt.py` 的 `norm`、Q/K norm、`GPT.forward`、无缓存 `generate`；画张量 shape 图。 | `reports/model_path.md` 能从 `[B,T]` 写到 logits。 |
| 10-05 | 阅读 `engine.py` 的 `KVCache`、prefill、批量复制和 decode；列出 cache 生命周期。 | `reports/cache_path.md` 含 `[L,B,T,Hkv,Dh]`、`cache_seqlens`、`prev_embedding`。 |
| 10-06 | 阅读 `flash_attention.py` 的 FA3/SDPA 路由，并检查 CPU fallback 测试覆盖。 | 一张后端条件表；已知不能在 CPU 验证的路径明确标出。 |
| 10-07 | 复盘现有 `infer_bench.py`：列出 checkpoint 依赖、单次测量、结束 token、TTFT 边界等问题。 | `reports/benchmark_audit.md`；W1 gate：能口述两条生成路径。 |

### W2 · 10 月 8–14 日：做出合法的小 checkpoint 与可信基准协议

| 日期 | 当天任务 | 验收/产物 |
|---|---|---|
| 10-08 | 检查 nanochat tokenizer 和训练数据是否存在；阅读 `dataset.py`、`tok_train.py` 的路径规则。 | `reports/data_prerequisites.md` 逐项说明来源和缺口。 |
| 10-09 | 用最小数据/合法 tokenizer 验证 token round-trip；没有外部数据时先做无需下载的模型形状实验。 | round-trip 断言或带错误原因的阻塞记录。 |
| 10-10 | 用小配置生成 nanochat base checkpoint；先做 2 step smoke，确认元数据与模型配置能加载。 | checkpoint 路径、配置、退出码；不能混用 MiniMind 权重。 |
| 10-11 | 扩到足够跑推理的最小 checkpoint，运行 `Engine.generate` 和无缓存 `GPT.generate`。 | 同一 prompt 有确定输出与两条路径日志；效果差属预期。 |
| 10-12 | 为基准定义 CSV/JSON schema：环境、输入、每次 TTFT、逐步 TPOT、有效 token、显存、错误。 | `protocol.md` 和一个 schema 样例，不填虚构数字。 |
| 10-13 | 编写或增强重复运行包装：预热 3 次、正式至少 5 次、同步 CUDA、记录完整 raw row。 | 自动汇总 median、P95/范围，并能发现提前结束。 |
| 10-14 | 在可用设备上做小矩阵试跑，核对输出文件和统计方法；本机无 CUDA 则仅报告 CPU 功能验证。 | `results/raw/` 首批数据；W2 gate：报告代码不依赖手工抄数。 |

### W3 · 10 月 15–21 日：基线矩阵与求职材料初稿

| 日期 | 当天任务 | 验收/产物 |
|---|---|---|
| 10-15 | 写一页简历初稿：教育/课程、MiniMind 已验证成果、nanochat 正在做的事。 | 一页 PDF/文档初稿；未测量的加速比不写。 |
| 10-16 | 固定基线 checkpoint、seed、dtype、后端，定义 `P=32/128/512, G=32/128, B=1/4` 的可运行子集。 | `protocol.md` 矩阵与内存预估；每组满足 `P+G<=sequence_len`。 |
| 10-17 | 有 GPU 则集中测 B=1 的三种 prompt；无 GPU 则跑 CPU 契约测试并封装 GPU 执行脚本。 | 原始值不少于每组 5 次或 `PENDING_GPU`。 |
| 10-18 | 测 B=4 或硬件能承受的 batch，记录 OOM/截断/提前终止情况。 | 批量对比表；失败行保留而非隐藏。 |
| 10-19 | 复算 TTFT、TPOT、tokens/s，核查单位、样本数和分位数。 | `results/summary/baseline.csv` 与原始行一一对应。 |
| 10-20 | 给 `README` 写项目目的、模型来源、最短运行命令及当前结果状态。 | 同学按文档知道需要何种硬件和 checkpoint。 |
| 10-21 | 为导师准备三分钟介绍和一张 MiniMind→nanochat 的证据图。 | W3 gate：至少能展示 CPU 闭环及 nanochat 可信的运行入口。 |

### W4 · 10 月 22–28 日：导师首次联系，建立 trace 证据

| 日期 | 当天任务 | 验收/产物 |
|---|---|---|
| 10-22 | 核查 GitHub 可见内容、README、简历与成绩单；选择研究方向匹配的目标导师。 | 每位导师一条具体匹配理由；材料不存在则如实说明进度。 |
| 10-23 | 在 prefill 和 decode 的关键边界加 profiler 标记；先用短输入验证 trace 能打开。 | trace 文件与事件命名说明。 |
| 10-24 | GPU 可用时对 prefill 做代表性 trace；CPU 环境只验证标记和 trace 格式。 | `reports/profile_prefill.md` 引用具体算子，不用主观“利用率低”。 |
| 10-25 | 对 decode 做短 trace；撰写个性化首封邮件和附件清单。 | `reports/profile_decode.md` 与邮件草稿；由本人核对敏感信息。 |
| 10-26 | **最迟今天本人发送**第一封导师邮件；登记时间与预定工作日跟进日期。 | 联系记录；若未发送，记录现实原因并在下一工作日补发。 |
| 10-27 | 汇总 prefill/decode 至少两个候选瓶颈，用 trace 事件、占比和输入形状支持。 | `reports/bottlenecks.md`；不能把 CPU 推断写成 GPU 结论。 |
| 10-28 | 复盘基线和 trace，选 RMSNorm 为第一个受控优化点；明确可证伪假设。 | W4 gate：一个可执行实验假设和冻结的对照基线。 |

### W5 · 10 月 29 日–11 月 4 日：RMSNorm 正确性优先

| 日期 | 当天任务 | 验收/产物 |
|---|---|---|
| 10-29 | 列举 `norm` 的调用点：embedding、Q/K、Block、末层；统计 trace 中所占比例。 | `reports/norm_scope.md`；说明哪些输入形状适合单 kernel。 |
| 10-30 | 写独立 PyTorch 参考函数与误差测量，明确 accumulation dtype 和 epsilon 契约。 | FP32/bf16 输入的参考测试；epsilon 与仓库 `F.rms_norm` 对齐。 |
| 10-31 | 学 Triton program/block/mask/reduction，写单行或多行 RMSNorm 第一版；没 GPU 则只完成源码与 launch 参数检查。 | 独立算子入口，未运行 GPU 时标 `UNVERIFIED`。 |
| 11-01 | 增加 token 数与 hidden size 参数化 correctness；覆盖非 2 的幂和接近 0 的值。 | 测试保存 max_abs/max_rel，遇到零分母按协议处理。 |
| 11-02 | 有 GPU 时跑 correctness，先修错误；无 GPU 时准备一键运行脚本与输出目录。 | 所有启用 shape 通过才进入性能测试。 |
| 11-03 | 写单算子 benchmark：warm-up、同步、重复、计时事件、带宽估算；若 10-26 的导师邮件未获回复，按 5–7 个工作日规则礼貌跟进一次。 | 原始逐次数据与版本信息自动保存；联系记录只记一次跟进。 |
| 11-04 | review kernel 边界、strides、mask 和 dtype；冻结 correctness 用例。 | W5 gate：正确性通过或清晰标明 GPU 阻塞及剩余工作。 |

### W6 · 11 月 5–11 日：Triton shape 矩阵和调参

| 日期 | 当天任务 | 验收/产物 |
|---|---|---|
| 11-05 | 跑 `M=1/16/128/512` 与 `D=256/768/1024/非 2 的幂` 的基线，排除 OOM/不支持组合。 | 每组含 dtype、shape、误差、耗时，失败组合也记录。 |
| 11-06 | 比较不同 `BLOCK_SIZE`，不改算法，只改一个变量。 | raw CSV 标明配置和统计值。 |
| 11-07 | 比较 `num_warps`，同时观察小 M 与大 M；防止以大形状掩盖小 decode 退化。 | 每 shape 相对 PyTorch 的比值与波动。 |
| 11-08 | 对性能最好和最差形状各采短 trace，核查 launch、访存及额外拷贝。 | 两条有事件名的解释。 |
| 11-09 | 重跑候选配置并随机化比较顺序，检查时钟/热机噪声。 | 结果稳定性表；无稳定收益就接受负结果。 |
| 11-10 | 依据结果制定形状 dispatch 条件，默认 PyTorch 回退。 | 条件列出每个通过的 dtype 与 shape。 |
| 11-11 | 汇总单算子报告和风险。 | W6 gate：正确性完整、原始数据可追溯、没有只展示最快一组。 |

### W7 · 11 月 12–18 日：接入推理并测端到端

| 日期 | 当天任务 | 验收/产物 |
|---|---|---|
| 11-12 | 在 `gpt.py` 中通过受控开关接入 RMSNorm，默认保持原路径。 | 关闭开关时结果与基线一致。 |
| 11-13 | 测所有实际调用位置的形状、dtype、非连续布局；任何不支持输入走回退。 | 推理路径的形状清单与断言/测试。 |
| 11-14 | 同一模型、输入和 dtype 下逐层比 logits，最后比 greedy 输出。 | 输出一致性及最大误差表。 |
| 11-15 | 运行旧推理测试和新增 correctness，排查 FA3/SDPA 分支。 | 测试摘要；不能通过则禁用集成。 |
| 11-16 | 跑端到端 `P=32/128/512`、`B=1/4` 可用子集，原版与 Triton 版交替测。 | 两版各至少 5 次正式运行、含 TTFT/TPOT/显存。 |
| 11-17 | 计算收益与不确定性，并用 norm 在总时间中的占比解释。 | 单算子与端到端并列图，列出退化形状。 |
| 11-18 | 整理可审阅 diff、命令和实验结论。 | W7 gate：即使收益为零，也有可信负结果和回退。 |

### W8 · 11 月 19–25 日：扩展证据并开始投递

| 日期 | 当天任务 | 验收/产物 |
|---|---|---|
| 11-19 | 检查 kernel 的不同 dtype/硬件边界；只在可用设备上声明性能。 | 支持矩阵与限制说明。 |
| 11-20 | 复跑最能代表项目场景的基线和优化版，检查与 W7 是否一致。 | 原始重复数据、当前 commit、环境。 |
| 11-21 | 完成 RMSNorm 技术报告：假设、设计、正确性、单算子、端到端、负例。 | `reports/rmsnorm.md` 可独立阅读。 |
| 11-22 | 更新一页简历：仅把已实测数字写入；列出 AI Infra/推理引擎岗位关键词。 | 简历第二版与岗位清单。 |
| 11-23 | 开始投递寒假/日常/远程相关岗位，记录投递链接、日期、状态。 | 至少第一批真实投递；机会较少也保留检索记录。 |
| 11-24 | 给愿意沟通的导师准备阶段进展摘要：一图、一数字、一限制、一下一步。 | 邮件草稿经本人核实；无可信 GPU 数字时只汇报方法和进度。 |
| 11-25 | 发进展更新给适合的已联系导师；明确下一阶段是现有 KV Cache 优化。 | W8 gate：成果可展示，外部沟通记录真实。 |

### W9 · 11 月 26 日–12 月 2 日：拆解现有 KV Cache 的正确性

| 日期 | 当天任务 | 验收/产物 |
|---|---|---|
| 11-26 | 对比 `GPT.generate` 与 `Engine.generate`，标出终止 token、采样及模型调用差异。 | `reports/kv_design.md` 的状态图。 |
| 11-27 | 记录每层 K/V 的 `[L,B,T,Hkv,Dh]`、dtype、字节数及理论公式。 | 估算表与实测分配区分开。 |
| 11-28 | 为随机权重小模型构造 cache/no-cache 逐步最后 logits 对照；先固定 greedy。 | 长度 1、短 prompt、多个 decode step 的自动化测试。 |
| 11-29 | 覆盖 `prev_embedding` 和 RoPE 位置推进，比较 cache_seqlens 每步变化。 | 每层/每步状态 CSV；预填后位置正确。 |
| 11-30 | 处理 Engine 与裸 GPT 的输出规则差异，避免把工具调用或 EOS 处理误判为数值错误。 | 对照测试只比较同一 token 前缀和最后位置 logits。 |
| 12-01 | 覆盖 batch=1 与复制到 batch=4 的 KV 内容一致性；查预分配越界边界。 | 测试有清晰失败提示。 |
| 12-02 | 冻结正确性回归集。 | W9 gate：greedy token 一致，logits 在 dtype 容差内，K/V 写入位置准确。 |

### W10 · 12 月 3–9 日：只改一处可量化的 KV 成本

| 日期 | 当天任务 | 验收/产物 |
|---|---|---|
| 12-03 | 从 trace 选最有证据的 KV 成本：预分配、prefill 复制、位置读取或 decode 访存。 | `reports/kv_hypothesis.md` 写预计影响哪个指标。 |
| 12-04 | 为该成本写隔离 microbenchmark，固定 shape 和 batch。 | 原始延迟/显存数据，不靠单次测量。 |
| 12-05 | 实现一项最小改动；保留可切换回退。 | 小 diff，代码路径清楚。 |
| 12-06 | 跑 W9 正确性集，特别检查 RoPE、smear 和短窗口注意力。 | 全部通过才继续，否则回退并分析。 |
| 12-07 | 加入极端输入：短 prompt、接近最大上下文、不同 batch；做边界测试。 | 无越界/OOM 误判，真实 OOM 有记录。 |
| 12-08 | 在同一硬件/配置下重新跑 microbenchmark，分析效果是否与假设相符。 | 原始数据和 median/P95。 |
| 12-09 | 决定保留或撤销改动；写原因。 | W10 gate：正确性通过且收益/负结果可追溯。 |

### W11 · 12 月 10–16 日：KV 的端到端与显存报告

| 日期 | 当天任务 | 验收/产物 |
|---|---|---|
| 12-10 | 固定 `P=32/128/512`、生成长度和 batch；保证 checkpoint 上下文允许。 | 实验矩阵注明实际 prompt 长度。 |
| 12-11 | 运行 cache on/off 对照：逐步 logits 使用 GPT 路径，时间使用同等工作量的封装。 | 同条件原始结果；Engine 不可直接切 cache 时写单独对照 harness。 |
| 12-12 | 在三种长度下测 TTFT、TPOT、tokens/s、峰值显存。 | 3+ 组完整原始记录，或 GPU 阻塞表。 |
| 12-13 | 用理论 KV bytes 解释实测峰值差异，指出权重/激活/allocator 的影响。 | 显存表与计算过程。 |
| 12-14 | 测原版与 KV 改动版的端到端表现；如只有正确性改动，不虚构速度收益。 | 两版配置和顺序一致。 |
| 12-15 | 整理失败案例、长序列限制和 batch 行为。 | `reports/kv_cache.md`。 |
| 12-16 | 建立项目中期版本记录。 | W11 gate：三长度、正确性、性能/阻塞状态、局限齐备。 |

### W12 · 12 月 17–23 日：选择系统进阶路径

| 日期 | 当天任务 | 验收/产物 |
|---|---|---|
| 12-17 | 根据资源选 **稳定性压测** 或 **连续批处理最小版**，只选一个。默认稳定性压测；连续批处理需要先证明 batch 变化可正确处理。 | `reports/advanced_choice.md` 写选择理由和停止条件。 |
| 12-18 | 设计输入矩阵：空输入、1 token、长输入、极限 batch、重复请求、提前 EOS。 | 矩阵列出预期结果/允许失败类型。 |
| 12-19 | 写自动化驱动与日志收集，不让错误被吞掉。 | 每次运行有退出码、种子、输入摘要。 |
| 12-20 | 跑 1 小时以内的首轮压测或调度正确性测试。 | 原始日志、失败分类。 |
| 12-21 | 修复最重要的一类真实问题，加入对应回归测试。 | 修改前可复现、修改后测试通过。 |
| 12-22 | 重测同一矩阵，标出改善与新限制。 | 前后对照表。 |
| 12-23 | W12 复盘，确定 W13 的单一收尾目标。 | 已知问题按严重程度排序。 |

### W13 · 12 月 24–30 日：完成进阶验证并收束范围

| 日期 | 当天任务 | 验收/产物 |
|---|---|---|
| 12-24 | 稳定性路线：重复长时间运行；批处理路线：加入新请求。 | 日志体现请求/时间线。 |
| 12-25 | 稳定性路线：逼近 OOM 边界；批处理路线：完成请求退出并回收槽位。 | 成功/失败条件明确。 |
| 12-26 | 测吞吐和延迟分布，记录 batch 或负载形状。 | 原始数据与有效样本数。 |
| 12-27 | 分析一次故障或性能异常的具体调用位置。 | 一条可重放的案例与原因。 |
| 12-28 | 修复或写清无法修复的限制；补回归测试。 | 测试与 issue 清单。 |
| 12-29 | 对当前阶段所有实验生成统一结果索引。 | `results/README.md` 指向 raw、summary、trace。 |
| 12-30 | 冻结新功能，只保留复现和文档修复。 | W13 gate：一项系统进阶证据或如实放弃说明。 |

### W14 · 12 月 31 日–1 月 6 日：从干净环境复现

| 日期 | 当天任务 | 验收/产物 |
|---|---|---|
| 12-31 | 清点依赖、缓存、下载数据、checkpoint、硬件要求，写环境清单。 | `env.md` 可由他人逐项核查。 |
| 01-01 | 清理 README 路径和命令，明确预期耗时/显存与失败恢复办法。 | Quickstart 不依赖个人 shell profile。 |
| 01-02 | 在干净 shell 运行 CPU smoke 和核心单测。 | 完整 stdout/stderr 和退出码。 |
| 01-03 | 有 GPU 则重跑代表性 3 组基准；无 GPU 则检查一键脚本与 `PENDING_GPU`。 | 原始结果与报告一致，未测不填。 |
| 01-04 | 重新生成所有图表，检查图轴、单位、样本数。 | 图可从 raw CSV 自动重建。 |
| 01-05 | 请另一人按 README 复现，或自己用新目录/新 shell 模拟。 | 记录困惑点并修文档。 |
| 01-06 | 完成一份从问题到证据到结论的技术报告。 | W14 gate：不靠口头补充可复现最小成果。 |

### W15 · 1 月 7–13 日：成果叙述和集中投递

| 日期 | 当天任务 | 验收/产物 |
|---|---|---|
| 01-07 | 用已验证数字更新一页简历，保留硬件与条件。 | 每条数字能定位 raw CSV。 |
| 01-08 | 写 3 分钟项目讲解：问题、基线、优化、验证、限制。 | 不看稿可讲完且不夸大。 |
| 01-09 | 写 10 分钟深讲：KV shape、RMSNorm 数值、Profiler、实验设计。 | 有代码位置和图表支持。 |
| 01-10 | 列 15 个面试追问并口述回答，重点解释负结果和测量误差。 | Q&A 文档。 |
| 01-11 | 给曾回复的导师发送完整阶段更新；无回复者按新匹配名单联系。 | 本人核对后发送，联系记录更新。 |
| 01-12 | 集中投递 AI Infra、推理引擎、CUDA/HPC、训练系统岗位。 | 岗位、版本、日期和状态表。 |
| 01-13 | 检查 GitHub 可公开内容，脱敏日志与路径，准备演示。 | W15 gate：仓库、简历、讲解一致。 |

### 收尾 · 1 月 14–17 日：第一版验收

| 日期 | 当天任务 | 验收/产物 |
|---|---|---|
| 01-14 | 修复影响复现的最高优先级问题，停止无证据的调参。 | 关键命令均有明确通过/失败记录。 |
| 01-15 | 完成最终结果表：基线、RMSNorm、KV、进阶、限制；逐项审查数据出处。 | 所有声称可追溯；无 GPU 结果明确空缺。 |
| 01-16 | 用全新终端重跑最小复现；核对仓库状态和 README。 | 最终复现日志与版本号。 |
| 01-17 | 冻结 v1.0 成果清单，列下一阶段问题；标签只在代码和文档达到门槛时创建。 | `reports/final_review.md`；未过门槛则转入延长日历。 |

## 五、门槛失败时的日历延长规则

“完成时间不限”意味着不拿日期伪造完成。发生阻塞时，在 `reports/experiment-log.md` 记录原因、下一次检查时间、可并行的离线任务；把尚未完成的技术日顺延到后续自然日，保留同样的任务顺序和验收。每完成一个 7 天块再复盘一次，不无限重复调参。

优先级依次为：基线和正确性 → trace → 一项深入优化 → KV 证据 → 稳定性/批处理。若时间不足，先取消连续批处理，再缩减 RMSNorm 调参组合，但保留非理想 shape、正确性和端到端对照；KV 可先限单 batch，但保留无缓存对照。导师联系、简历和投递不等待 GPU。

2027 年 1 月之后若仍有精力，先补完本项目未过的门槛，再根据实际岗位或组内任务选择单卡算子深化，或从数据并行/集合通信进入分布式训练。不要同时开两个主项目。最终要能准确回答：瓶颈证据是什么、改了什么、怎样证明正确、在什么条件下带来多少收益或退化、还缺什么证据。

## 六、来源与代码锚点

- 源路线：`/home/lizishuo/_program_/Project/minimind/.agents/docs/career_roadmap/1-1-weekly-timeline-2026-10-01.md`、`1-2-execution-standards-and-outreach.md`、`1-3-resources-delays-and-extensions.md`。
- 已完成 MiniMind 证据：`/home/lizishuo/_program_/Project/minimind/experiments/minimind_baseline/README.md` 与 `reports/reproduction.md`。
- nanochat 代码：`nanochat/gpt.py`、`nanochat/engine.py`、`nanochat/flash_attention.py`、`scripts/infer_bench.py`、`scripts/base_train.py`、`runs/runcpu.sh`、`tests/test_engine.py`。
- Triton 官方 [LayerNorm 教程](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)可学习 reduction、mask 和 benchmark 写法；LayerNorm 会减均值并含可学习仿射参数，不能直接当作本仓库无参数 RMSNorm 的正确性参考。
- PyTorch 官方 [`torch.profiler` 文档](https://docs.pytorch.org/docs/stable/profiler.html)用于核对 trace 导出与 profiler 使用方式。
