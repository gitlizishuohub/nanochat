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
knowledge_scope: W3第2日实操与验收
---

# W3 D2 · 冻结 checkpoint 与基线矩阵

[上一日](d15_resume_draft.md) · [下一日](d17_single_batch_runs.md)。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[4.1 配置不是可学习权重](../../llm-training-workflow/04-model-construction-and-weight-loading.md#reading-4-1)；[4.3 从配置重建模型并加载参数](../../llm-training-workflow/04-model-construction-and-weight-loading.md#reading-4-3)；[12.2 生成控制流与返回格式](../../llm-training-workflow/12-inference-and-generation.md#reading-12-2)。
- **今天重点**：把模型配置、所选权重、配套词表和缓存容量连起来，核对今天固定的输入长度、生成长度与样本数。
- **暂时跳过**：不更换模型来填补同一个基线；初始化数值表和 <abbr title="SFT / Supervised Fine-Tuning · 监督微调；用示范对话中的助手目标更新基础模型。">SFT</abbr> 参数不影响今天冻结身份。

**读完能解释**：为什么只记一个 d4 标签不够？你的基线还需要固定哪些身份信息和输入条件？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 开始前：固定终端和记录方式

本日命令均从 nanochat 仓库根执行。先打开终端，激活已创建的 Conda 环境；若新终端提示找不到 `conda`，先运行 `source /home/lizishuo/miniconda3/etc/profile.d/conda.sh` 再激活，不创建另一个 `.venv`。

```bash
conda activate nanochat
cd /home/lizishuo/_program_/Project/nanochat
python -c "import sys; print(sys.executable)"
git status --short
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
exec > >(tee -a experiments/inference_infra/reports/d16_checks.log) 2>&1
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。依赖是否已装好要另查，激活成功不代表 PyTorch 可用。保留已有改动，遇到未认识的变更不要还原。教程脚本保存在本周 `scripts/`，结果写入 `experiments/inference_infra/`，两者不能混为同一类产物。

本日验证完成后，直接按文末“本日自验证与交流”逐题和我交流，附关键输出或自动日志/原始结果路径即可。无需手写报告，也不要求固定排版或长文。实际日期、设备、命令参数与退出码以日志和聊天核对；标题里的 W3 Dn 是学习顺序，不倒填完成状态。若想自行整理，可选用 `experiments/inference_infra/reports/d15_experiment-log.md`，后续跨日也可引用本次聊天和自动结果。

## 今天的目标和前置（约 3 小时）

前置：W2 有可加载的 nanochat checkpoint、配套 tokenizer 和 `meta_NNNNNN.json`。今天把“测什么”冻结成 12 个候选组合，先检查再运行。P 是输入 token 数，G 是要求生成的步数，B 是同一提示词的生成样本数；这里不是 B 条独立提示词的服务端并发吞吐。

分配：40 分钟核对模型，40 分钟生成矩阵，40 分钟估算内存，40 分钟核对协议与交流。<abbr title="Checkpoint · 检查点；本项目由模型权重与 JSON 元数据等配套文件构成。必须与 nanochat 架构和 tokenizer 匹配，不能直接加载 MiniMind 权重。">checkpoint</abbr> 必须有固定 source、model_tag 和 step；省略后自动选择最大模型或最新 step 会导致重跑时换模型。

## 第一步：找出你实际使用的文件

从 W2 训练日志抄录 `NANOCHAT_BASE_DIR`、model_tag 和 step。下面的 `d11_w2_20`、`20` 仅演示如何填变量，不表示这台机器已有对应 checkpoint；先把它们替换成 W2 验收过的值。

```bash
export NANOCHAT_BASE_DIR=/实际/nanochat缓存根目录
MODEL_TAG=d11_w2_20
CHECKPOINT_STEP=20
META="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG/meta_$(printf '%06d' "$CHECKPOINT_STEP").json"
python -m json.tool "$META"
sha256sum "$META"
```

读取 `model_config.sequence_len`，再核对 `vocab_size/n_layer/n_head/n_kv_head/n_embd/window_pattern`。`P+G<=sequence_len` 是本周人为固定的可比较范围；不能用模型 RoPE 预计算更长作为超出训练上下文仍有效的证明。Meta 不能证明权重存在，还需核对同 step 的 `model_NNNNNN.pt` 与 W2 加载成功日志。

## 第二步：仅生成计划，不自动执行

使用[矩阵驱动脚本](scripts/d16_matrix.py)。脚本复用[W2 基准程序](../w2/scripts/d12_benchmark.py)，默认只生成清单，既不租 GPU 也不启动训练。

```bash
python .agents/docs/career_roadmap/w3/scripts/d16_matrix.py \
  --artifact-day 16 --meta "$META" --source base --model-tag "$MODEL_TAG" --step "$CHECKPOINT_STEP" \
  --output-dir experiments/inference_infra/results/raw/w3_plan
```

`--artifact-day 16` 标记本次新计划的首次生成日；后续复用此脚本新建执行目录时使用当天编号，已有文件不改号。默认计划不生成 raw 或子进程 log。

预期打印 12 个 case：`P=32/128/512` × `G=32/128` × `B=1/4`。可用上下文的行标 `PENDING_GPU`，超出上下文标 `INVALID_CONTEXT`。例如 sequence_len=128 时只有 P32+G32 可行，其他不能偷偷截短成同名 P128/P512。矩阵中的 command 是参数数组，便于复核，没有 shell 字符串拼接执行。

W2基准默认使用 `[BOS]+[1]×(P−1)` 的固定合成token输入，主要控制计算工作量，不是自然语言质量测试；原始行含prompt_sha256和prompt_fill_id。正式对比必须固定这一构造，若为排查提前终止更换fill ID则作为新输入实验，不能混入旧组。

打开 `experiments/inference_infra/results/raw/w3_plan/d16_matrix.jsonl` 检查每行 `sequence_len` 和 `raw_file`。`--output-dir` 每次使用新目录，已有矩阵时拒绝覆盖，避免失败证据丢失。本周 W2 包装器的 source 仅允许 `base`；生产加载器另支持 sft/rl，但当前教程不扩展该分支。上游 infer_bench 帮助里出现的 `mid` 不能直接沿用。

## 第三步：手算缓存，预估而非承诺显存

<abbr title="GQA / Grouped-Query Attention · 分组查询注意力；多组 Q 共享较少的 K/V 头。缓存按 n_kv_head 存储，不能按 n_head 直接估算。">GQA</abbr> 下缓存主项的字节数为：`2 × L × B × T × Hkv × Dh × s`。2 表示 K 和 V，L 是层数，Dh=`n_embd/n_head`，s 是计算 dtype 每元素字节数。<abbr title="dtype / Data type · 数据类型；float32 每元素4字节，bfloat16每元素2字节。参数dtype、计算dtype和缓存dtype可能不同，应分别记录。">dtype</abbr> 必须读取实际环境与运行记录，不能因为文件是 bf16 就认定运行所有部分都 bf16。

Engine 先建 batch=1、T=P 的预填缓存，再建 batch=B、T=P+G 的解码缓存并复制，因此复制阶段同时存在两个缓存：缓存峰值主项约 `2×L×Hkv×Dh×s×(P+B×(P+G))`。窗口注意力减少读取范围，但此 Engine 的分配仍按完整 T；不要把滑窗长度直接代入分配容量。

这还不包括权重、logits、激活、RoPE、prev_embedding、allocator 保留块和 CUDA context。在聊天给出公式、配置、s 的来源和粗估值，实测峰值另列；可选整理到 `experiments/inference_infra/d12_protocol.md`。模型很小时 `vocab_size` 大仍可能使 logits/权重占显著内存。

## 第四步：冻结运行协议

先在聊天或既有自动记录中明确以下条件，后续据此恢复变量与比较实验；无需额外报告。若希望留文件，可选在 W2 `d12_protocol.md` 追加 W3 小节。以下是核查项目，不要求逐项另抄一份：

```text
模型：source/model_tag/step；meta和权重SHA-256；tokenizer来源与哈希
输入：P集合、G集合、B集合；prompt构造方法与实际token长度
采样：temperature=0、seed=42；Engine工具/结束规则
环境：解释器、torch版本、CPU/GPU型号、驱动、compute dtype、实际注意力后端
运行：每case预热3次、正式至少5次；同一软件版本；每次next后同步
边界：超上下文拒绝；不悄悄截断；提前结束/OOM保留
输出：matrix.jsonl、每case raw JSONL和log；summary来自自动聚合
```

后端自动路由在 CPU 用 SDPA，CUDA 的 FA3 还受硬件、库及计算 dtype 影响；“设备有 CUDA”不等于“运行 FA3”。本驱动不强制后端，冻结实际解析出来的 `attention_backend` 与 precision，变化时另建实验目录。

## 验收和阻塞处理

验收不是 12 组都必须跑通，而是 12 组都有明确状态、可行组满足长度条件、模型来源与测量协议固定。缺 checkpoint 时交流缺失条件并返回 W2；不要用虚构 meta 冒充实际模型。没有 GPU 保留 `PENDING_GPU`。下一天先测 B=1，并将 CPU 功能验证与 GPU 基线分开。

## 本日脚本与产物清单

下列路径以项目根目录为基准；脚本编号保留首次使用日。外部历史归档与上游 checkpoint 原生文件名保留，按正文真实绝对路径或环境变量解析；不冒充项目新产物。<abbr title="条件生成：只有指定分支执行、前置文件满足且实际写入后，文件才算存在；清单中的目标路径本身不证明生成成功。">条件生成</abbr>须在验收时核对。只打印的检查统一进入本日 checks 日志，不另称生成报告。

| 脚本完整项目相对路径 | 用途 | 阶段 | 输入路径或来源 | 运行条件 |
|---|---|---|---|---|
| [.agents/docs/career_roadmap/w3/scripts/d16_matrix.py](scripts/d16_matrix.py) | 固定12个组合，按开关执行子集 | 计划/显式运行 | META 指向真实上游 meta；MODEL_TAG、CHECKPOINT_STEP | 计划需真实 meta；--execute 需权重/tokenizer，CUDA运行需GPU |
| [.agents/docs/career_roadmap/w2/scripts/d12_benchmark.py](../w2/scripts/d12_benchmark.py) | 矩阵内部调用；CPU分支直接调用 | 逐次测量 | checkpoint及合成prompt；CPU示例使用随机夹具 | 依赖 PyTorch；真实基准需匹配 tokenizer |

| 产物完整项目相对路径 | 用途 | 生成者/步骤 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/reports/d16_checks.log` | 环境、只读检查、命令输出及错误 | 开始前 Bash 的 exec / tee | 自动捕获后续终端输出；退出码在聊天核对，或可选补记日志 |
| `experiments/inference_infra/reports/d15_experiment-log.md` | 每日事实、退出码和阻塞 | 自愿整理；跨日可引用聊天和自动结果 | 可选人工笔记，不影响验收 |
| `experiments/inference_infra/results/raw/w3_plan/d16_matrix.jsonl` | 12个候选状态、命令与原始输出目标 | d16_matrix.py | 自动生成；计划中的 raw_file 仅为目标，不保证文件存在 |
| `experiments/inference_infra/d12_protocol.md` | 追加W3固定模型、缓存估算与测量协议 | 第3–4步 | 可选人工整理；协议也可由聊天和自动记录恢复 |

## 本日自验证与交流

按以下问题自行核查后直接发我；每项给关键输出、文件/行号或原始字段即可。没有执行的部分说明原因和未验证范围，不要求写成报告。

1. 实际 source、model_tag、step、tokenizer 来自哪里？给出 meta、权重或加载日志路径与已核对的哈希；哪些文件仍缺失？
2. 你的 sequence_len 是多少？12 个 P/G/B 组合中哪些合法、哪些 INVALID_CONTEXT？指向 d16_matrix.jsonl 的实际行，解释为什么不能偷偷截短。
3. 选一个合法 B4 组合代入缓存公式：L、Hkv、Dh、s 分别来自哪里，两个缓存共存时粗估多少字节？说明它未包含哪些内存，实测峰值是否仍待验证。
4. 你冻结了哪些输入、采样、预热、正式次数、精度和后端条件？用矩阵/自动记录或聊天中的完整参数恢复一次计划；为什么有 CUDA 也不保证 FA3？

## 核对记录与完成标记

本文基于当前仓库相关生成、加载、注意力路由及原基准代码编写；源码阅读不能代替执行。本教程的交付状态不表示读者已经完成当天实验。按当天问题核对真实证据后再勾选；聊天回答即可，无需额外报告：

- [ ] 输入条件与前一天产物已核实。
- [ ] 本日命令在明确设备执行，或具体阻塞已记录。
- [ ] 自动日志和原始输出可定位，实践材料或聊天回答可回溯；可选笔记无需生成。
- [ ] 示例数值与真实结果已区分，失败和待测未删除。
- [ ] 能说明本日证据证明什么、尚不能证明什么。
