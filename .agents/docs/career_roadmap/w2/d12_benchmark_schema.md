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
knowledge_scope: W2第5天实操与验收
---

# W2 D5：冻结原始数据协议和统计边界

前一天：[返回前一天](d11_checkpoint_generation.md)；后一天：[继续下一天](d13_repeated_benchmark.md)；[原始总计划](../nanochat_daily_execution_2026-10-01.md)。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[10.1 日志口径](../../llm-training-workflow/10-logging-checkpoints-and-resume.md#reading-10-1)；[13.2 能力与系统性能的边界](../../llm-training-workflow/13-evaluation-and-metrics.md#reading-13-2)。
- **今天重点**：了解训练日志的耗时与吞吐口径，再对照今天的推理协议，明确计时起止、分母、预热和失败行。
- **暂时跳过**：本栏目的训练吞吐不是推理指标公式；具体测量定义仍以当天教程和脚本为准。

**读完能解释**：为什么训练日志中的 tokens/s 不能直接与 Engine 的生成吞吐放在一列比较？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 开始前：固定 shell 与结果位置

所有命令从仓库根目录执行。当前终端会自动激活 nanochat 环境，无需重复激活；先运行以下检查。这里使用本项目的 Conda 环境；不要混用系统 Python。CPU实验建议先 `export NANOCHAT_DTYPE=float32`；脚本只在变量未设置时选择FP32，已有变量会遵从并记录，不能假定CPU必定FP32。GPU实验另按协议显式选dtype。

日志只在 `{ ...; }` 命令组执行期间记录；<abbr title="Bash process substitution · 进程替换。这里把命令组的输出交给 tee，同时显示并追加到日志；重定向在命令组结束后恢复。">进程替换</abbr>不会持续接管交互终端，因此提示符和 VS Code 终端集成控制字符不会混入日志。命令组在当前 shell 执行，变量设置会保留；代码块返回最后一条命令的退出码，需核对单条命令时立即记录 `$?`。已有的显式文件重定向仍写入各自文件。复制时包含外层花括号和末尾重定向。若之前执行过旧版全局日志命令，先关闭旧终端并新开终端；仅修改文档不会停止旧终端的记录。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports experiments/inference_infra/results/raw experiments/inference_infra/results/summary
{
python -c "import sys; print(sys.executable); print(sys.version)"
} > >(tee -a experiments/inference_infra/reports/d12_session.log) 2>&1
```

解释器应指向 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。若环境不存在或第三方依赖缺失，先解决环境，不把 `--help` 成功当作模型实验通过。后面写入的 `experiments/` 文件是你实际执行教程时产生的正式成果；本指导不预先生成成绩。大数据、tokenizer、权重放在 `NANOCHAT_BASE_DIR`，不要加进 Git。表中的 `$NANOCHAT_BASE_DIR` 代表当前 shell 选定的真实绝对数据根；合成分支示例为 `/tmp/nanochat-w2-toy`，未设置时当前用户默认根为 `/home/lizishuo/.cache/nanochat`。交流时给出实际展开后的绝对路径；`*.parquet` 表示该目录内逐一列出的实际数据文件，旧数据回退目录为 `$NANOCHAT_BASE_DIR/base_data/`。

本页各 Bash 代码块内的输出追加保存在 `experiments/inference_infra/reports/d12_session.log`；命令返回后立即执行 `echo "exit=$?"`，避免丢失退出码。帮助、源码检索、解释器与 Git 检查只打印，不另有检查结果文件。日志是追加写入，重跑时交流中注明会话边界或日志中的起止位置。

## 今日目标与时间盒

今天用 20 分钟读测量边界、25 分钟确认协议并交流、20 分钟核对 schema、15 分钟用错误分支验证落盘。前置是 [模型身份记录](d11_checkpoint_generation.md)。即便没有 checkpoint，协议设计和错误行检查仍能完成。

## 第一步：先定义被测服务与计时范围

阅读 [重复基准脚本](scripts/d12_benchmark.py) 的 `measure()`。本周固定 `mode=engine`，输入为长度恰好 P 的 token ID 列表，首位 BOS，剩余均为普通 ID1（可显式调整 `--prompt-fill-id`，记录独立prompt hash）；不在测量中做文本 tokenization。这是“固定形状 ID 工作负载”，不是用户真实问答延迟。

<abbr title="TTFT / Time to First Token · 从开始处理已编码的 token ID 输入，到第一个生成 token 列准备好的时间。本文包括 Engine generator 首次执行、cache分配、prefill、cache复制、首token采样及GPU同步，不包括模型加载和tokenizer。">TTFT</abbr> 的起点是首次 `next(stream)` 前；终点是第一次 yield 后完成设备同步。Engine generator 在创建对象时不执行函数体，因此计时不能从创建前后随意截。

<abbr title="TPOT / Time Per Output Token · 首token之后相邻输出步骤之间的耗时。本文每个值是一个包含B行的同步decode列间隔，单位ms，不除以B。每次run的TPOT是G−1个间隔的均值。">TPOT</abbr> 记录为 `decode_step_ms` 数组；P95可以对“每次run的平均TPOT”计算，也可对“所有步延迟”计算，但二者不是同一个统计量，本项目汇总采用前者并说明样本数。

`elapsed_ms` 等于 TTFT 加所有 decode 间隔，终点为最后一列 yield。脚本收满 G 列立即关闭 generator，避免 Engine 在最后 yield 之后再计算一个无用 forward。CUDA 每个输出边界同步，测得的是带同步开销的流式延迟；以后用事件测纯 kernel 必须分开命名。

## 第二步：确认当天协议并在聊天中交流

下面列出本次固定设置与要核对的字段，作为阅读清单；结合实际参数、自动环境字段与文末问题，在聊天中确认协议。缺少的身份或环境信息据实标记未验证，后续可引用本聊天、原始记录与会话日志，不依赖手写文件。`experiments/inference_infra/d12_protocol.md` 仅为可选整理位置：

```text
协议版本：1
源码commit / dirty摘要：
工作负载：固定token-ID prompt；同一prompt复制B份输出
模型来源 / tag / step / 权重hash / tokenizer hash：
设备 / GPU驱动 / PyTorch / CUDA / dtype / attention backend：
seed：42
P / G / B：先32 / 8 / 1
边界：P+G不超过sequence_len；不自动裁短
预热：3次；正式：至少5次；每次新建Engine cache
TTFT范围：token ID输入到首列yield，含分配/prefill/复制/采样，不含加载和tokenizer
TPOT：G-1个同步输出列间隔；每次run先取均值
有效token：每行尚未结束且mask=1且不是terminal的token
吞吐：有效token总数 / elapsed秒；B乘法不再重复计算
汇总：仅measure+ok；median与nearest-rank P95；失败/提前结束保留
CPU或random：仅功能/契约，不用于GPU加速结论
GPU驱动补录命令：nvidia-smi（不可用则记录缺失）
```

注意 B=4 的 prefill 仍只有一个 prompt，而不是四个不同请求的吞吐；不能把这样的 TTFT 与普通 batch-prefill 工作负载混为一谈。每次调用新建 cache 并预分配零张量；这部分开销明确计入 TTFT。

## 第三步：逐字段认识 JSONL

<abbr title="JSONL / JSON Lines · 每行一个独立JSON对象。适合追加保存每次实验原始记录；读取时每行分别解析，不能把整个文件当成一个JSON对象。">JSONL</abbr> 文件每行一个 run，预热也保留，错误不会抹除。schema由脚本输出定义：

| 类别 | 实际字段与约束 |
|---|---|
| 标识 | schema_version=1、session_id、run_id；每行run_id唯一 |
| 运行角色 | phase=warmup/measure；repeat从0开始；setup_or_run是异常行 |
| 状态 | ok/early_stop/error；只有固定G步、所有B行都有G个有效token才ok |
| 形状 | prompt_tokens、decode_tokens、batch_size、actual_decode_steps、prompt_fill_id、prompt_sha256 |
| 计数 | generated_tokens、per_row_generated_tokens；前者为后者之和 |
| 延迟 | ttft_ms、decode_step_ms、tpot_ms、elapsed_ms；毫秒单位 |
| 吞吐 | tokens_per_s；实际有效token数除以含TTFT总秒数 |
| 内存 | peak_allocated_bytes、peak_reserved_bytes；CPU为null，不能写0假装测得 |
| 归因 | model_kind、source、model_tag、checkpoint_step、model_config、checkpoint_sha256 |
| 可复现 | seed、env、precision、attention_backend、timing_scope、batch_semantics、evidence_scope |
| 异常 | error；setup失败行允许缺少延迟和环境字段，汇总必须容忍 |

GPU峰值在模型已加载之后重置，allocated仍包含活跃模型内存；reserved是分配器保留值，不是“缓存比无缓存节省显存”。每次重复不清空CUDA allocator，所以reserved可能保留前次状态，交流时说明此边界。

## 第四步：制造明确的缺失模型错误行

不要手填一组漂亮数字作为 schema 样例。用一个不存在的标签生成真实错误样例：

```bash
{
python .agents/docs/career_roadmap/w2/scripts/d12_benchmark.py --device cpu --model-tag absent-w2-schema-check --step 123456 --output experiments/inference_infra/results/raw/d12_schema_error.jsonl
} > >(tee -a experiments/inference_infra/reports/d12_session.log) 2>&1
```

预期非零退出，JSONL有 `status=error` 和真实异常原因。这只验证错误落盘；模型缺失或依赖缺失都可能触发，不是一次成功测量。参数非法（例如 `--warmup 0`）由 argparse 退出2，尚未进入实验，因此不会写raw行。

## 失败分流与验收

- JSONL不能解析：检查是否把普通终端日志追加到了同一个raw文件；stdout日志与JSONL分开保存。
- 字段缺失：首先检查status/phase，setup错误缺timing是合法的；ok行缺字段才需修复。
- G=1：只有TTFT没有后续间隔，TPOT应null；不要除以G或强填0。
- 验收：聊天中确认的协议边界完整且能由实际参数/自动记录追溯；能用字段说明吞吐与B的关系；错误样例来自真实命令；未运行成功的数据不填入基线表。

## 今日脚本与产物清单

每日编号使用全程 d01–d21；跨日复用保留首次实际执行日。以下项目相对路径均以仓库根目录为基准，脚本清单直接列出本日执行入口。

| 脚本完整项目路径 | 用途 | 执行阶段 | 输入 | 运行条件 |
|---|---|---|---|---|
| `.agents/docs/career_roadmap/w2/scripts/d12_benchmark.py` | 检验错误行落盘与测量字段 | 第四步 | `$NANOCHAT_BASE_DIR/base_checkpoints/absent-w2-schema-check/`（故意不存在） | 依赖可加载时尝试缺模型；非零退出是预期 |

| 产物完整项目相对路径 | 用途 | 生成者 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/d12_protocol.md` | 固定工作负载、计时与统计边界 | 第二步 | 可选人工整理；可复用本聊天确认的协议与自动记录，文件若生成仍保留 d12 |
| `experiments/inference_infra/results/raw/d12_schema_error.jsonl` | 真实错误样例 | d12_benchmark.py | 条件自动；通过参数校验进入实验后写错误行；argparse 错误不写 |
| `experiments/inference_infra/reports/d12_session.log` | 当天终端输出与手动记录的退出码 | 各 Bash 代码块的 tee | 自动；仅追加各代码块内的输出 |

原生 `tokenizer.pkl`、`token_bytes.pt`、`model_<step>.pt`、`meta_<step>.json` 与 `optim_<step>_rank0.pt` 是 loader 的格式契约，保留其名称；用带日编号的数据文件或模型标签归属日期。已有历史结果不原位重命名，新的命令使用本页命名。没有执行或条件不满足的必需检查均标 PENDING；可选 Markdown 未整理不影响验收。

## 今日自验证与交流问题

自行验证后，直接在聊天中逐题简答即可，不要求正式报告、排版或长文。每个结论引用相应日志、自动结果中的字段或源码位置；没有执行的检查明确标为 `PENDING`，遇到失败给出真实退出码与错误。需要本地留存时再选用上表的可选 Markdown。

1. 请按实际脚本说明 TTFT 的起止点及包含/排除的工作：tokenizer、模型加载、cache分配、prefill、复制、采样与同步分别在哪一侧？引用 `measure()` 的相关位置。

2. 给定 P=32/G=8/B=1 与预热3次/正式5次，哪些是本次固定协议？B=4时 prefill 的请求语义是什么，为什么 TPOT 不除以B、吞吐不能重复乘B？

3. 若 G=1，TPOT 应是什么？若 CPU 没测显存，相关字段应是什么？结合字段表解释有效 token、elapsed 与吞吐的单位关系。

4. 缺失模型命令实际退出码和错误行是什么？引用 `d12_schema_error.jsonl` 的 status/phase/error；它验证了什么，为什么不能算成功基线，argparse失败又有什么不同？

5. 后续如何按身份分组、排除预热与失败行并计算 P95？说明 reserved/allocated 的边界和未测条件；协议可在本聊天确认，哪些字段仍缺证据就明确标记。

离开前保留本日会话日志与自动结果；若追加重跑，注明会话边界。交流时可附 `git rev-parse HEAD`、`git status --short` 及实际设备/精度以定位证据，不必另填通用表。下一天可直接复用本聊天中已确认的结论与证据路径。共享前检查数据路径和日志是否包含私人信息；若提交成果，只纳入自己的轻量记录与脚本，不提交权重、数据、环境目录或缓存。
