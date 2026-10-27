---
source_files:
  - nanochat/engine.py
  - .agents/docs/career_roadmap/w3/scripts/d16_matrix.py
  - .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py
doc_type: tutorial
knowledge_role: supporting
knowledge_scope: W3第2日学习与实践
---

# W3 D2 · 设计可比较的实验与基线矩阵

[上一日](d15_resume_draft.md) · [下一日](d17_single_batch_runs.md)。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[4.1 配置不是可学习权重](../../llm-training-workflow/04-model-construction-and-weight-loading.md#reading-4-1)；[4.3 从配置重建模型并加载参数](../../llm-training-workflow/04-model-construction-and-weight-loading.md#reading-4-3)；[12.2 生成控制流与返回格式](../../llm-training-workflow/12-inference-and-generation.md#reading-12-2)。
- **今天重点**：把模型配置、所选权重、配套词表和缓存容量连起来，核对今天固定的输入长度、生成长度与样本数。
- **暂时跳过**：不更换模型来填补同一个基线；初始化数值表和 <abbr title="SFT / Supervised Fine-Tuning · 监督微调；用示范对话中的助手目标更新基础模型。">SFT</abbr> 参数不影响今天冻结身份。

**读完能解释**：为什么只记一个 d4 标签不够？你的基线还需要固定哪些身份信息和输入条件？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 今天学什么，做到哪里算完成

今天学习设计可比较的实验，约 60–90 分钟。**主线是纸面矩阵与容量推导，不操作模型，不要求训练、GPU 或 checkpoint**。明天的 CPU 小实验也不依赖真实权重。拥有并选择使用训练权重时，才进入文末计划工具分支。

<abbr title="Baseline · 基线；固定对象、输入与方法后得到的参照结果。计划仅规定未来如何测量，还没有测量结果。">基线</abbr>不是某个文件名；它需要说明测的是谁、什么输入、怎样计时。今天完成比较设计即达标，无需把全部候选跑一遍。

## 1. 先判断两个数字是否回答同一个问题

假设甲用随机小模型 CPU、P=8/G=4/B=1；乙用训练模型 GPU、P=512/G=128/B=4。乙吞吐更高，也不能归因于“乙实现了更好的缓存”，因为模型、设备和工作量同时改变。

明天主线固定随机模型配置、seed=42、CPU float32、线程数、普通 fill ID=1、P=8/G=4；后天只改变 B。仍需从 raw 核对实际配置、后端、环境是否一致。即使随机 seed 相同，也不是跨版本逐位相同参数的保证；版本与初始化实现要一致。

真实 <abbr title="Checkpoint · 模型检查点；需匹配权重、模型配置和分词器。仅有配置 JSON 不能证明权重存在或可加载。">checkpoint</abbr> 分支另固定 source、model_tag、step、配置、权重及 tokenizer 指纹。标签叫 d4 或词表大小一样都不足以识别同一模型。MiniMind 权重不能直接作为 nanochat 权重使用。

## 2. 纸面推导 12 个候选格

P 是提示 token 数，G 是要求输出的最大列数，B 是同一提示的样本行数。先假设固定 `sequence_len=128`，不运行命令：

| P | G | P+G | 上下文 128 | 上下文 256 |
|---|---|---|---|---|
| 32 | 32 | 64 | 合法 | 合法 |
| 32 | 128 | 160 | 拒绝 | 合法 |
| 128 | 32 | 160 | 拒绝 | 合法 |
| 128 | 128 | 256 | 拒绝 | 合法 |
| 512 | 32 | 544 | 拒绝 | 拒绝 |
| 512 | 128 | 640 | 拒绝 | 拒绝 |

每行各取 B=1/4，合计 12 格。128 上下文有 2 格合法、10 格拒绝；256 上下文有 8 格合法、4 格拒绝。B 不改变每行长度约束，但会改变缓存量。

这里遵守基准约定 `P+G<=sequence_len`；不能悄悄截短后还保留 P512 的名字。长度合法只是必要条件，不保证内存足够或生成不提前结束。RoPE 预计算更长也不能替代既定上下文约束。

## 3. 从 Engine 的两次分配推导容量

[Engine.generate](../../../../nanochat/engine.py)先以 B=1、T=P 建立 prefill 缓存，再以 B 行、T=P+G 建解码缓存并复制。<abbr title="KV Cache · 键值缓存；每层保存历史位置的 K/V。GQA 使用较少 K/V 头，缓存头数应取 n_kv_head。">K/V</abbr> 主项是 `2×L×B×T×Hkv×Dh×s` 字节，其中 Dh=`n_embd/n_head`，s 是每元素字节数。

做一个假设算例：L=2，Hkv=2，Dh=32，float32 的 s=4，P=8，G=4，B=4。解码缓存为 `2×2×4×12×2×32×4=49152` 字节，即 48 KiB；prefill 为 `2×2×1×8×2×32×4=8192` 字节，即 8 KiB。复制阶段两份共存，主项 56 KiB。这只是玩具配置容量推演，不是实测峰值。

对照 `kv_cache_decode.prefill(kv_cache_prefill)` 后的 `del kv_cache_prefill`：复制前后生命周期解释了为何要暂时相加。还未计入权重、激活、logits、RoPE、`prev_embedding` 和分配器保留块，所以容量公式不能承诺“不 OOM”。滑窗缩小读取范围，也不自动缩小这里按完整 T 分配的缓存。

## 4. 带读计划工具的三个判断

打开[矩阵驱动](scripts/d16_matrix.py)，重点顺序是：

```python
supported = prompt + generation <= limit
# 之后，只有同时满足三个条件才调用 subprocess.run：
# supported and args.execute
# and (args.batch is None or b == args.batch)
```

`--execute` 缺省为 false，所以默认只写 12 行矩阵，不启动模型。`--batch 1` 只是筛选执行子集，B4 仍保留计划行。上下文非法先记 `INVALID_CONTEXT`；支持的未执行行按目标设备记 `PENDING_CPU` 或 `PENDING_GPU`。这些是**所选计划的脚本状态**，不表示你每天必须补做对应设备。

子进程退出 0 时记 `PROCESS_OK`；它不直接检查 raw 是否有至少五次有效正式测量，后续还要 D19 审计。`raw_file` 字段只是未来目标路径，不能据此声称文件已存在。

## 5. 主线收束与可选真实权重计划

主线在聊天给出：两个上下文的合法格数量、上面的缓存推导、明后天只改变 B 的协议。无需创建假 meta，也无需写报告；没有 checkpoint 不阻塞本日。

只有手头已有可加载的真实 nanochat 权重、同 step meta 和匹配 tokenizer，且想练习计划工具时执行以下分支。先将本机值改成真实已核查的身份；示例 tag/step 仅适用于确实做过 W2 可选两步 smoke 的情况，不要求去补训练。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
export NANOCHAT_BASE_DIR=/替换为已核查的缓存根
MODEL_TAG=d10_w2_smoke
CHECKPOINT_STEP=2
META="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG/meta_$(printf '%06d' "$CHECKPOINT_STEP").json"
python .agents/docs/career_roadmap/w3/scripts/d16_matrix.py \
  --artifact-day 16 --meta "$META" --source base \
  --model-tag "$MODEL_TAG" --step "$CHECKPOINT_STEP" --device cpu \
  --output-dir experiments/inference_infra/results/raw/w3_plan
D16_RC=$?
printf 'plan_exit=%s\n' "$D16_RC"
} > >(tee -a experiments/inference_infra/reports/d16_checks.log) 2>&1
```

这是 CPU 目标的**计划命令，没有 `--execute`**；只读 meta、写矩阵，不加载权重或训练。即使计划成功，也需另查真实权重和 tokenizer，不能把 meta 当加载验收。已有矩阵目录拒绝覆盖，应新建尝试目录。GPU 与真实执行是另外自选扩展，本日无须运行。

## 本日脚本与产物清单

路径相对于项目根目录。纸面主线不生成文件；只有选择实际分支才产生下表目标。

| 脚本完整路径 | 用途与输入 | 条件 |
|---|---|---|
| [.agents/docs/career_roadmap/w3/scripts/d16_matrix.py](scripts/d16_matrix.py) | 从真实 meta 生成 12 格计划 | 标准库；目标身份由本人核实；缺省不执行 |
| [.agents/docs/career_roadmap/w2/scripts/d12_benchmark.py](../w2/scripts/d12_benchmark.py) | 驱动在显式执行时调用 | 执行才需 PyTorch、匹配资产及所选设备 |

| 产物完整路径 | 生成者 | 条件与含义 |
|---|---|---|
| `experiments/inference_infra/reports/d16_checks.log` | 可选命令组 tee | 局部输出、退出码；命令组后恢复终端 |
| `experiments/inference_infra/results/raw/w3_plan/d16_matrix.jsonl` | d16_matrix.py | 选择真实 meta 计划后写入；只有计划，未生成 case raw/log |
| `experiments/inference_infra/d12_protocol.md` | 本人自愿整理 | 可选保存固定条件和容量推导，聊天可替代 |

## 本日自验证与交流

1. 对照表格，若 sequence_len 变为 160，有几格合法？B 为什么不参与这一步判断？

   **核对要点**：前三种 P/G 组合合法，各取两个 B 共 6 格；B 改变并行行数和内存，不改变每行 P+G。

2. 把容量算例的 B 从 4 改成 1，复制阶段主项是多少？它为什么不一定等于工具记录的峰值？

   **核对要点**：解码 12 KiB，prefill 8 KiB，主项 20 KiB；列出其他占用，解释两份共存时机。

3. 同学拿着 `PROCESS_OK` 说“性能基线已完成”，你还要看什么？

   **核对要点**：真实 raw、有效正式次数、预热、实际配置和计数/时间；计划与子进程退出码不是统计验收。

4. 明天若更换 fill ID、精度或后端，能否把变化都归为 B 的影响？

   **核对要点**：输入或执行条件也变了，需要新实验身份；B 表示同一提示的样本数，并非独立用户请求。

## 完成核对

- [ ] 能推导合法格与缓存主项，而非只背公式。
- [ ] 明确 CPU 主线的比较条件，分清计划和测量。
- [ ] 未选择权重/GPU扩展时不留每日欠账；未把假设当资产事实。
