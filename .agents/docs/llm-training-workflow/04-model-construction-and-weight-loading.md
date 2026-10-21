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

[栏目目录](README.md) · [上一篇](03-tokenizer-and-conversation-protocol.md) · [下一篇](05-data-encoding-and-batching.md) · [术语查阅](glossary/README.md)

> 原技术正文沿用 2026-10-12 的源码核对记录；2026-10-21 调整结构。新增 SFT 串联说明做了局部静态核对，未执行训练。

# 4. 模型构建、初始化与参数加载

<a id="sec-5"></a>

<a id="reading-4-1"></a>
## 4.1 配置不是可学习权重

<a id="ref-config-001"></a><abbr title="Configuration / Hyperparameter｜配置与超参数&#10;定义：由命令行或代码设定、控制结构与训练行为的数值，如层数、长度、学习率和评估周期。&#10;作用：定义实验，并在结构meta或用户配置中记录以便重建和理解训练。&#10;本项目：base入口默认depth20/aspect64/head_dim128，覆盖GPTConfig自身12层/768宽/6头；SFT可继承base配置。&#10;注意：CLI的“参数”不是可学习权重，matrix-lr不会被反向学习；SFT user_config在继承前复制可能仍None，结构看model_config。&#10;位置：scripts/base_train.py:41 argparse；nanochat/gpt.py:29 GPTConfig；scripts/chat_sft.py:68 user_config">配置与超参数</abbr>（[释义](glossary/04-model-and-forward-pass.md#term-config)）决定结构/训练行为；<a id="ref-weights-001"></a><abbr title="Parameters / Weights / state_dict｜可学习参数、权重与状态字典&#10;定义：可学习参数是由优化器按梯度更新的张量，state_dict按名字收集参数和持久缓冲。&#10;作用：保存模型学到的数值，与仅说明层数/维度的配置文件分工。&#10;本项目：包含wte、独立lm_head、投影、值表与层标量；cos/sin persistent=False，加载时重建，不在权重文件中。&#10;注意：state_dict不含完整训练游标、随机状态和数据缓冲；严格加载要求名字/形状匹配，不能与CLI超参数混为同义。&#10;位置：nanochat/gpt.py:156 GPT；nanochat/gpt.py:200 register_buffer；nanochat/checkpoint_manager.py:76 build_model">可学习参数</abbr>（[释义](glossary/04-model-and-forward-pass.md#term-weights)）是反向后更新的张量。base_train 覆盖 GPTConfig 自身默认：

| 项 | GPTConfig 自身默认 | base_train 默认实际值 |
| --- | --- | --- |
| 层数 | 12 | depth=20 |
| 宽度 C | 768 | ceil(depth×64/128)×128=1280 |
| query 头 / KV 头 | 6 / 6 | 10 / 10 |
| 每头 D | 128 | head_dim=128 |
| 长度 T | 2048 | 2048 |
| 词表 | 32768 | 从实际 tokenizer 读取，默认 32768 |
| 窗口模式 | SSSL | SSSL，最后一层强制 L |

模型支持<a id="ref-gqa-001"></a><abbr title="GQA / Grouped-Query Attention；MHA｜分组查询与多头注意力&#10;定义：GQA让多个查询头共享较少K/V头，MHA则查询头与K/V头数量相等。&#10;作用：共享K/V可降低缓存与投影开销，但收益只有配置真正使用共享头时才存在。&#10;本项目：GPT支持Hkv≤Hq且整除，SDPA根据头数不同启用enable_gqa；base入口设置Hkv=Hq，默认10/10。&#10;注意：支持不等于默认启用，没有n-kv-head预训练CLI；不能声称默认缓存因GQA缩小或给不存在参数命令。&#10;位置：nanochat/gpt.py:76 头数检查；scripts/base_train.py:138 GPTConfig；nanochat/flash_attention.py:134 enable_gqa">GQA</abbr>（[释义](glossary/04-model-and-forward-pass.md#term-gqa)），但入口默认 Hq=Hkv，实际 MHA；没有 --n-kv-head CLI。宽度向 head_dim 倍数对齐，任意自选 head_dim 不保证满足 RoPE/内核隐含约束，默认 128 是已知配置。

<a id="reading-4-2"></a>
## 4.2 从结构到权重

先区分三个动作：配置确定张量形状；初始化为新模型填入起始数值；加载用已有参数替换这些数值。第一次阅读掌握这个顺序即可；下面的逐模块分布用于源码对照。

### 4.2.1 深入：初始化顺序与数值

<a id="ref-init-001"></a><abbr title="Meta Device / Initialization｜元设备与权重初始化&#10;定义：meta只保留形状/类型，不分配实际数据，随机初始化需在真实设备分配存储后执行。&#10;作用：减少构建大型模型时的临时内存峰值，并统一明确训练起点。&#10;本项目：build_model_meta→to_empty→init_weights；wte正态std0.8，lm_head0.001，Q/K/V均匀，块输出投影零初始化。&#10;注意：to_empty不是合法权重起点；加载也先重建RoPE再覆盖参数。docstring std1.0旧值和构造meta假初始化不代表真实训练值。&#10;位置：scripts/base_train.py:129 build_model_meta；nanochat/gpt.py:204 init_weights；nanochat/checkpoint_manager.py:103 初始化加载">meta 初始化</abbr>（[释义](glossary/04-model-and-forward-pass.md#term-init)）调用顺序是 build_model_meta→to_empty(device)→init_weights。meta 只有形状，to_empty 分配未初始化存储，最后一步才填入数值。

| 模块 | init_weights 实际值 |
| --- | --- |
| 输入 wte | Normal std=0.8；docstring 的 1.0 旧值不作依据。 |
| 独立 lm_head | Normal std=0.001。 |
| Q/K/V | Uniform(−s,s)，s=√3/√C。 |
| MLP c_fc | Uniform(−0.4s,0.4s)。 |
| 注意力/MLP 输出 c_proj | 全 0，仍可获得梯度训练。 |
| resid_lambdas / x0_lambdas | 前者从 1.15 到 1.05，后者从 0.20 到 0.05。 |
| smear_lambda / backout_lambda | 0 / 0.2；门控 Uniform(0,0.02)。 |
| value_embeds | Uniform(−s,s)，值门控 Uniform(0,0.02)。 |
| cos/sin | 按实际设备重建；无可学习参数，不随 state_dict 保存。 |

固定种子不是完整确定性保证；未启用 torch.use_deterministic_algorithms。输入与输出词表权重独立，不绑定。

<a id="reading-4-3"></a>
## 4.3 从配置重建模型并加载参数

<a id="ref-warmstart-001"></a><abbr title="Warm Start / Weight Loading｜热启动与权重加载&#10;定义：用已有权重作为新阶段起点，可导入优化器统计，但数据与循环通常重新开始。&#10;作用：让SFT继承基础能力、RL继承对话/工具能力，减少从零学习成本。&#10;本项目：load_model映射base/sft/rl，默认最深tag/最新step；SFT默认读base完整optimizer后只重置lr，RL只读sft模型。&#10;注意：不是断点恢复；state_dict还覆盖beta/decay，日志“仅momentum”不准确。最好显式tag/step避免自动挑错权重。&#10;位置：nanochat/checkpoint_manager.py:148 load_model_from_dir；scripts/chat_sft.py:134 热启动；scripts/chat_rl.py:74 加载">加载已有权重</abbr>（[释义](glossary/06-checkpoints-and-resume.md#term-warmstart)）为 load_model→load_model_from_dir→build_model→load_checkpoint。以 meta.model_config 重建 GPT，先初始化重建 RoPE，再 strict=True、assign=True 加载权重。CPU/MPS 把保存的 bf16 张量转 float32；旧模型兼容补丁仅明确覆盖缺失 window_pattern、resid_lambdas、x0_lambdas，不能承诺所有历史版本兼容。

默认选最深 d<number> tag；无匹配则按目录修改时间，step 取最大 model 文件编号。交付应指定来源/tag/step与配套词表。各阶段的起点和热启动选项见[SFT 全流程](11-supervised-fine-tuning.md)与[可选 RL](14-gsm8k-reinforcement-learning.md)；恢复训练所需的状态统一见[检查点](10-logging-checkpoints-and-resume.md)。
