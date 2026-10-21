---
doc_role: training-workflow
source_revision: fbc049f379926ecdb49071a129586278245d71c2
review_date: 2026-10-12
source_files:
  - nanochat/gpt.py
  - nanochat/tokenizer.py
  - nanochat/dataset.py
  - nanochat/dataloader.py
  - nanochat/common.py
  - nanochat/optim.py
  - nanochat/fp8.py
  - nanochat/checkpoint_manager.py
  - nanochat/loss_eval.py
  - nanochat/core_eval.py
  - nanochat/engine.py
  - scripts/tok_train.py
  - scripts/tok_eval.py
  - scripts/base_train.py
  - scripts/chat_sft.py
  - scripts/chat_rl.py
  - scripts/base_eval.py
  - scripts/chat_eval.py
  - scripts/chat_cli.py
  - tasks/common.py
  - tasks/smoltalk.py
  - tasks/mmlu.py
  - tasks/gsm8k.py
  - tasks/arc.py
  - tasks/humaneval.py
source_symbols:
  - GPT.forward
  - GPT.setup_optimizer
  - RustBPETokenizer.render_conversation
  - tokenizing_distributed_data_loader_with_state_bos_bestfit
  - MuonAdamW.step
  - Engine.generate_batch
---

[栏目目录](README.md) · [上一篇](14-gsm8k-reinforcement-learning.md) · [术语查阅](glossary/README.md)

> 原技术正文沿用 2026-10-12 的源码核对记录；2026-10-21 调整结构。新增 SFT 串联说明做了局部静态核对，未执行训练。

# 15. 运行示例与模型交付

<a id="sec-running"></a>
本章在理解训练、SFT、推理和评价后使用。下面命令保留原文的静态参考性质，并非已运行的实验结果。

<a id="reading-15-1"></a>
## 15.1 运行前的条件

本地学习沿用 `nanochat` <abbr title="Conda · 管理独立 Python 解释器与依赖的工具；此处使用名为 nanochat 的既有环境。">Conda</abbr> 环境，激活和复查见[环境记录](../project-maintenance/nanochat-conda-environment-2026-10-15.md)。先确认数据、原分词器、模型来源和输出目录；逐日小实验优先按[每日指导](../career_roadmap/w1/d01_starting_point.md)执行。

## 15.2 训练、微调与评价参考命令

以下CPU极小模型示例需先准备第2章数据与词表。参数名、batch整除及源码路径已静态核对，未运行、不承诺训练质量或当前依赖可用。保留2048长度以减少SFT默认2048上限与短行冲突。

~~~bash
NANOCHAT_DTYPE=float32 python -m scripts.base_train \
  --device-type=cpu --depth=4 --head-dim=64 --window-pattern=L \
  --max-seq-len=2048 --device-batch-size=1 --total-batch-size=2048 \
  --num-iterations=20 --eval-tokens=2048 --eval-every=10 \
  --core-metric-every=-1 --sample-every=-1 --save-every=10 \
  --model-tag=teach-d4 --run=dummy

NANOCHAT_DTYPE=float32 python -m scripts.base_eval \
  --device-type=cpu --model-tag=teach-d4 --eval=bpb \
  --device-batch-size=1 --split-tokens=2048

# SFT默认消费混合数据一轮，CPU可能很久；
# --num-iterations不能据帮助文字承诺精确优化更新步数，此处不使用。
NANOCHAT_DTYPE=float32 python -m scripts.chat_sft \
  --device-type=cpu --model-tag=teach-d4 --load-optimizer=0 \
  --eval-tokens=2048 --eval-every=200 --chatcore-every=-1 --run=dummy

NANOCHAT_DTYPE=float32 python -m scripts.chat_eval \
  --device-type=cpu -i sft -g teach-d4 -a ARC-Easy -x 16 -b 1
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli \
  --device-type=cpu -i sft -g teach-d4 -p "Why is the sky blue?"
~~~

20步例大半位于warmup，不是充分收敛。SFT仍有截断、全mask和CPU编译等边界，不能因为命令通过静态检查就说运行通过。GPU参考来自当前runs/speedrun.sh主体：

~~~bash
# 已准备依赖、170训练分片、原词表及兼容多GPU；本次未执行。
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=24 --target-param-data-ratio=8 --device-batch-size=16 \
  --fp8 --fp8-recipe=tensorwise --run=dummy
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
  --device-batch-size=16
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=dummy
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
python -m scripts.chat_cli -i sft -g d24
~~~

脚本当前d24/ratio8/batch16/FP8，与入口d20/ratio12/batch32及旧d20注释不同。整个bash脚本会安装、下载、训练，本次未执行。显存不足可减device-batch-size，同时保持总量整除，不能宣称当前显卡可承载。

<a id="reading-15-3"></a>
## 15.3 交付资产与部署边界

交付选定model.pt、同一步meta.json、原tokenizer.pkl/token_bytes.pt、仓库版本、启动配置与实际评估范围/结果；恢复base另需匹配rank优化器分片，RL没有该产物。能力与性能结果必须真实执行后填写。

现有形态是PyTorch state_dict+JSON+CLI/Engine；没有ONNX/GGUF/Safetensors导出、量化部署、Web服务或发布管线。[scripts/infer_bench.py](/home/lizishuo/_program_/Project/nanochat/scripts/infer_bench.py)是单CUDA卡性能入口，不是训练，打印依赖meta['step']；RL meta没有step，直接测RL可能失败。本次没有运行，不编造缺失功能命令。
