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

[栏目目录](README.md) · [上一篇](01-training-pipeline-and-optional-branches.md) · [下一篇](03-tokenizer-and-conversation-protocol.md) · [术语查阅](glossary/README.md)

> 原技术正文沿用 2026-10-12 的源码核对记录；2026-10-21 调整结构。新增 SFT 串联说明做了局部静态核对，未执行训练。

# 2. 环境与数据准备

<a id="sec-3"></a>

<a id="reading-2-1"></a>
## 2.1 本地环境入口

本项目每日学习采用已有的 `nanochat` <abbr title="Conda · 管理独立 Python 解释器与依赖的工具；此处使用名为 nanochat 的既有环境。">Conda</abbr> 环境，先按[环境验证记录](../project-maintenance/nanochat-conda-environment-2026-10-15.md)激活并重新盘点。历史版本和测试记录不代替当前复查。以下 uv 命令是上游独立环境的复建参考，不是要求在现有 Conda 环境外再创建一套环境。

### 2.1.1 上游依赖与缓存参考

[pyproject.toml](/home/lizishuo/_program_/Project/nanochat/pyproject.toml) 要求 Python≥3.10，[.python-version](/home/lizishuo/_program_/Project/nanochat/.python-version) 写 3.10，PyTorch 固定 2.9.1。核心依赖包括 rustbpe、tiktoken、pyarrow、kernels、filelock、numpy、wandb。uv 的 GPU extra 使用 CUDA 12.8 PyTorch 源，CPU extra 使用 CPU 源，两者互斥。以下命令仅记录，不在本任务执行：

~~~bash
uv sync --extra gpu
# CPU/MPS 环境改选：uv sync --extra cpu
source .venv/bin/activate
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export OMP_NUM_THREADS=1
~~~

compute_init 设 torch 种子 42，设备优先 CUDA→MPS→CPU。<a id="ref-distributed-001"></a><abbr title="Rank / World Size / torchrun / NCCL｜分布式进程与启动&#10;定义：rank为全局进程编号，world size为总进程数，local rank为节点内设备编号，torchrun设置这些环境变量。&#10;作用：分配样本与梯度计算到多卡，同时通过通信保持各卡模型数值一致。&#10;本项目：compute_init只在CUDA建NCCL组；预训练按row-group stride，SFT/RL按样本索引stride分配。&#10;注意：变量ddp表示分布式请求，不代表模型包了DistributedDataParallel；优化器同步梯度。CPU/MPS不可照搬多rank训练命令。&#10;位置：nanochat/common.py:151 get_dist_info；nanochat/common.py:174 compute_init；nanochat/optim.py:429 step">torchrun</abbr>（[释义](glossary/03-batching-and-distributed-training.md#term-distributed)） 多进程仅 CUDA 初始化 NCCL。<a id="ref-precision-001"></a><abbr title="Mixed Precision / bf16 / fp16 / fp32｜混合精度与数值类型&#10;定义：对不同张量/运算选不同位宽，在显存和吞吐与数值精度之间取平衡。&#10;作用：降低激活/矩阵乘法开销，同时保留重要统计和损失计算精度。&#10;本项目：COMPUTE_DTYPE默认SM80+ CUDA bf16、其它fp32；Linear临时转换fp32权重，嵌入通常直接低精度，logits/loss fp32。&#10;注意：不用autocast，不是全权重fp32；fp16时嵌入保留fp32方便unscale，dtype导入时选择，device-type覆盖不重选。&#10;位置：nanochat/common.py:17 dtype选择；nanochat/gpt.py:45 Linear；nanochat/gpt.py:265 嵌入类型">COMPUTE_DTYPE</abbr>（[释义](glossary/05-loss-and-parameter-updates.md#term-precision)） 在模块导入时选择：SM80+ CUDA 默认 bf16，其它默认 fp32，可用 NANOCHAT_DTYPE 覆盖。--device-type=cpu 不会重新选择 dtype，若同机仍可发现 CUDA，CPU 示例应显式 NANOCHAT_DTYPE=float32。

实现边界：dataset.py import requests，base_eval/core_eval 使用 yaml/jinja2，但核心依赖列表未逐项显式声明它们，可能经传递依赖获得；本次未验证安装或导入，不能保证所有入口现已可运行。kernels 首次加载可能联网，本文未触发。

<a id="reading-2-2"></a>
## 2.2 语料下载、划分与适配

<a id="ref-parquet-001"></a><abbr title="Parquet / Shard / Row Group｜列式文件、分片与行组&#10;定义：Parquet是列式文件，shard为数据文件分片，row group是一个文件内部按块可读取的行集合。&#10;作用：让大规模语料按文件下载、按行组读列和分配给rank，避免整库载入内存。&#10;本项目：ClimbMix文件读取text列，默认base_data_climbmix；-n8取8训练文件并追加06542验证，装载器末文件作val。&#10;注意：row group不同于模型行或batch；自备文件需保证最后文件的验证安排，MAX_SHARD6542是末索引并非总文件数。&#10;位置：nanochat/dataset.py:23 BASE_URL；nanochat/dataset.py:67 parquets_iter_batched；nanochat/dataloader.py:25 _document_batches">Parquet 分片</abbr>（[释义](glossary/02-data-and-protocol.md#term-parquet)）当前使用代码配置的 karpathy/climbmix-400b-shuffle，默认 base_data_climbmix。新目录不存在时可回退旧 base_data。装载器按文件排序后以最后一个文件为 val、其它为 train，不是随机 9:1。官方下载器总会额外下载固定 shard_06542.parquet 作验证。

~~~bash
# 8 个训练分片 + 额外固定验证分片
python -m nanochat.dataset -n 8
python -m scripts.tok_train
python -m scripts.tok_eval
# 较大训练增加语料；170 是 speedrun 示例，不是通用固定要求
python -m nanochat.dataset -n 170
~~~

MAX_SHARD=6542 是最后索引，总共可列 6543 个索引；训练为 0…6541、验证为 6542。-n=-1 实际下载全部训练分片，不是帮助文字所说的 disable。默认 4 个下载进程；先写 .tmp、成功后改名，已有文件跳过。装载器没有内容校验、清洗去重或污染检查。[dev/repackage_data_reference.py](/home/lizishuo/_program_/Project/nanochat/dev/repackage_data_reference.py) 是离线制备参考，本次未运行，也未核验远端最新状态。

<a id="ref-task-001"></a><abbr title="Task / TaskMixture / HubDataset｜任务适配、混合与数据包装&#10;定义：Task把数据变为对话并提供评分，TaskMixture组合任务实例，HubDataset包装Arrow表与索引置换。&#10;作用：使SFT示范、RL奖励和任务评估共用协议，同时隔离不同源数据格式。&#10;本项目：load_hub_dataset缓存Hub Parquet；TaskMixture列所有索引并Random(42)洗牌，SFT以重复MMLU/GSM8K实例提高权重。&#10;注意：1/3/4是重复实例次数，不是精确抽样概率；无需datasets库，实际条数以len而非近似注释为准。&#10;位置：tasks/common.py:21 HubDataset；tasks/common.py:45 load_hub_dataset；tasks/common.py:129 TaskMixture">任务适配器</abbr>（[释义](glossary/02-data-and-protocol.md#term-task)）用 Hub Parquet API、FileLock 缓存到 task_data，不依赖 datasets.load_dataset。SmolTalk、MMLU、GSM8K 按种子 42 洗牌；实际行数以下载后 len 为准，注释中的 460K/100K/8K 是近似，本次未下载实物。
