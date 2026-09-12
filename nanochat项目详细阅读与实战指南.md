# nanochat 项目详细阅读与实战指南

> 配套课件：`LLM_PyTorch_Transformer_Foundations_Detailed.pptx`  
> 对照项目：[karpathy/nanochat](https://github.com/karpathy/nanochat)  
> 推荐前置：配套 PPT → `nanoGPT项目详细阅读与实战指南.md` → MiniMind Dense/SFT 主线  
> 核对版本：2026-09-08，GitHub `master` 分支。nanochat 仍在快速演进，命令、默认值和实验结构可能变化，应以当前源码和 `runs/` 脚本为准。

---

## 1. nanochat 是什么

nanochat 不是简单的“新版 nanoGPT 模型文件”，而是一套端到端的小型聊天模型实验系统，覆盖：

```text
预训练语料下载
  → BPE Tokenizer 训练与评测
  → Base Model 预训练
  → Base Model 能力评测
  → SFT 对话微调
  → Chat Model 评测
  → RL 数学能力强化
  → KV Cache 推理与 CLI 对话
```

项目强调三个目标：

1. 单节点运行，而不是大型生产集群框架。
2. 代码相对集中、可读、可修改。
3. 端到端训练出可以聊天的模型，而不只训练 Base Model。

它比 nanoGPT 更现代、更完整，也明显更难。当前源码中既有通用现代组件，也有为了训练速度或小模型能力而加入的研究性技巧。阅读时必须把两者分开。

---

## 2. 完成本指南后应具备的能力

你应该能够：

1. 解释 Parquet 文档如何经过 BPE 变成训练 token。
2. 解释 BOS 对齐和 best-fit packing 的数据组织方式。
3. 从 `input_ids [B,T]` 追踪到 token loss。
4. 解释 RoPE、RMSNorm、QK Norm、GQA、滑动窗口注意力和 KV Cache。
5. 区分常规 Transformer 结构和 nanochat 的实验性增强。
6. 解释 Meta Device 初始化和显式 `COMPUTE_DTYPE`。
7. 解释 Muon 与 AdamW 为什么服务不同参数组。
8. 解释 `--depth` 如何影响宽度、头数和训练预算。
9. 读懂预训练循环中的梯度累积、调度、评测和 checkpoint。
10. 解释 SFT 的对话渲染和 assistant-only loss mask。
11. 解释 nanochat RL 为什么更接近 on-policy REINFORCE，而不是完整 PPO/标准 GRPO。
12. 使用 CPU 或单 GPU 完成小规模冒烟测试。

---

## 3. 前置知识自测

### 3.1 必须掌握

- Python 模块、dataclass、生成器、上下文管理器。
- `nn.Module`、Parameter、buffer、`state_dict`。
- `[B,T,C]`、`[B,H,T,Dh]`、`[B,T,V]`。
- Embedding、Linear、Cross Entropy。
- Q/K/V、causal mask、多头注意力。
- Pre-Norm、残差、MLP。
- 训练/验证模式、反向传播、AdamW。
- 梯度累积、梯度裁剪、checkpoint。
- tokenizer、BOS、特殊 token、SFT 基本概念。

### 3.2 建议提前掌握

- nanoGPT 的数据—模型—训练—生成闭环。
- RoPE、RMSNorm、GQA、KV Cache。
- DDP 中 rank、world size、all-reduce。
- bf16、fp16、fp32 的差异。
- SFT 中 prompt token 与 response token 的 loss mask。

如果第二组大部分不会，不要直接读 `base_train.py` 和 `chat_rl.py`，先完成 nanoGPT 和 MiniMind 指南对应章节。

---

## 4. 三层阅读法

### 4.1 第一层：基础主线

这是必须读懂的部分：

- tokenizer 编码/解码；
- 数据错位 inputs/targets；
- GPTConfig；
- RoPE、RMSNorm、Attention、MLP、Block；
- 预训练 loss；
- SFT loss mask；
- KV Cache 推理；
- checkpoint。

### 4.2 第二层：现代工程

主线跑通后再读：

- DDP；
- Flash Attention 3 与 SDPA fallback；
- 显式混合精度；
- Meta Device；
- MuonAdamW；
- 训练吞吐、MFU、显存指标；
- BPB、CORE、ChatCORE、pass@k。

### 4.3 第三层：研究性增强

不要把下面内容误认为所有现代 LLM 的标准配置：

- ReLU² MLP；
- Value Embedding / Value Residual；
- `resid_lambdas`；
- `x0_lambdas`；
- Smear；
- Backout；
- 特殊初始化方案；
- 自动训练预算和超参数缩放；
- FP8 speedrun 配置。

这些适合在标准 Transformer 主线理解后研究其动机和消融结果。

---

## 5. 仓库地图

```text
nanochat/
├─ nanochat/
│  ├─ gpt.py                 # 模型主干
│  ├─ tokenizer.py           # BPE、特殊 token、对话渲染
│  ├─ dataset.py             # 预训练数据下载和 Parquet 读取
│  ├─ dataloader.py          # 文档分片、在线分词、packing
│  ├─ engine.py              # KV Cache 推理和批量生成
│  ├─ optim.py               # Muon + AdamW
│  ├─ checkpoint_manager.py  # 模型与优化器保存/加载
│  ├─ loss_eval.py           # bits per byte
│  ├─ core_eval.py           # Base Model CORE 评测
│  ├─ execution.py           # Python 工具执行
│  ├─ flash_attention.py     # FA3 / SDPA 适配
│  └─ common.py              # 设备、分布式、dtype、公共工具
├─ scripts/
│  ├─ tok_train.py           # 训练 tokenizer
│  ├─ tok_eval.py            # tokenizer 压缩率评测
│  ├─ base_train.py          # Base Model 预训练
│  ├─ base_eval.py           # Base Model 评测
│  ├─ chat_sft.py            # SFT
│  ├─ chat_rl.py             # GSM8K on-policy RL
│  ├─ chat_eval.py           # Chat Model 评测
│  ├─ chat_cli.py            # 命令行对话
│  └─ infer_bench.py         # 推理延迟、吞吐和显存测试
├─ tasks/
│  ├─ common.py              # Task、TaskMixture 等抽象
│  ├─ smoltalk.py            # 通用对话数据
│  ├─ mmlu.py                # 多选知识任务
│  ├─ gsm8k.py               # 小学数学任务与 reward
│  └─ humaneval.py           # 简化 Python 代码任务
├─ runs/
│  ├─ runcpu.sh              # CPU/MPS 教学流程
│  ├─ speedrun.sh            # 多 GPU GPT-2 能力级流程
│  ├─ miniseries.sh          # 不同深度模型系列
│  └─ scaling_laws.sh        # Scaling Laws 实验
└─ tests/                    # tokenizer、attention、engine、optim 等测试
```

---

## 6. 推荐阅读顺序

不要按目录字母顺序阅读。推荐分八轮：

| 轮次 | 文件 | 目标 |
|---|---|---|
| 1 | `runs/runcpu.sh`、README | 建立端到端地图 |
| 2 | `dataset.py`、`tok_train.py`、`tokenizer.py` | 理解文本如何变 token |
| 3 | `gpt.py` 基础主干 | 理解模型结构和 shape |
| 4 | `dataloader.py`、`base_train.py` | 理解预训练 |
| 5 | `loss_eval.py`、`base_eval.py` | 理解 Base Model 评测 |
| 6 | `chat_sft.py`、`tasks/` | 理解对话微调 |
| 7 | `engine.py`、`chat_cli.py` | 理解 KV Cache 推理 |
| 8 | `chat_rl.py`、`optim.py`、Scaling Laws | 研究进阶 |

第一遍明确跳过：FP8、Flash Attention 内核细节、分布式 optimizer 内部通信、完整 RL 推导。

---

## 7. 环境与运行边界

### 7.1 安装

nanochat 使用 `uv` 管理依赖：

```powershell
git clone https://github.com/karpathy/nanochat.git
cd nanochat

uv sync --extra cpu
.venv\Scripts\Activate.ps1
```

CUDA 环境：

```powershell
uv sync --extra gpu
.venv\Scripts\Activate.ps1
```

开发和测试依赖：

```powershell
uv sync --extra gpu --group dev
```

### 7.2 Windows 注意事项

- `runs/*.sh` 是 Bash 脚本，PowerShell 中应逐条执行其中的 Python 命令。
- Bash 的 `export VAR=value` 在 PowerShell 中应写为 `$env:VAR = "value"`。
- 默认缓存位置由 `NANOCHAT_BASE_DIR` 控制，建议明确设到空间足够的磁盘。
- FA3、FP8 和部分高性能路径通常依赖特定 CUDA/Linux 环境；Windows/CPU 应接受 SDPA 或普通路径。

示例：

```powershell
$env:NANOCHAT_BASE_DIR = "D:\nanochat_cache"
```

### 7.3 算力预期

`runs/speedrun.sh` 面向 8×H100 节点，不是小白第一次运行的命令。CPU/MPS 脚本是教育演示，即使完整跑完也不会得到强模型。

第一次目标应是：

- tokenizer 能训练和 round-trip；
- Base Model 能执行 10～20 个优化步骤；
- loss 有限；
- checkpoint 能保存；
- SFT 数据 mask 正确；
- Engine 能加载已有 checkpoint 时完成生成。

---

## 8. 阶段 1：数据与 Tokenizer

### 8.1 `dataset.py`

预训练语料以多个 Parquet shard 保存。当前数据模块负责：

1. 按编号下载 shard；
2. 下载失败时重试；
3. 使用临时文件避免把半下载文件当成完整数据；
4. 最后一个 shard 作为验证集；
5. 逐 row group 读取 `text` 列。

数据流：

```text
远程 Parquet shards
  → 本地 NANOCHAT_BASE_DIR/base_data_*
  → PyArrow 读取 row group
  → list[str]
```

小规模下载：

```powershell
python -m nanochat.dataset -n 1
```

注意：即便 `-n 1`，脚本还会下载固定的验证 shard。先确认磁盘和网络条件。

### 8.2 BPE 训练

目标文件：[scripts/tok_train.py](https://github.com/karpathy/nanochat/blob/master/scripts/tok_train.py)

```text
Parquet 文档
  → 每文档最多 doc_cap 字符
  → 总计最多 max_chars
  → rustbpe 训练 merge rules
  → 构造 tiktoken.Encoding
  → tokenizer.pkl
  → token_bytes.pt
```

默认词表大小是 32768。特殊 token 不参与普通 merge 学习，而是在训练后加入词表。

极小教学配置：

```powershell
python -m scripts.tok_train --max-chars=1000000 --doc-cap=2000 --vocab-size=4096
python -m scripts.tok_eval
```

小数据可能导致质量较差，但足以验证流程。

### 8.3 `tokenizer.py`

重点类：`RustBPETokenizer`。

它使用：

- `rustbpe`：训练 BPE；
- `tiktoken`：快速编码和解码；
- 固定 split pattern：先把文本预切分，再学习 byte merge。

必须理解三类 token：

1. 普通 BPE token；
2. `<|bos|>` 文档边界 token；
3. SFT/Tool Use 特殊 token。

当前对话特殊 token 包括 user、assistant、Python 调用和 Python 输出的开始/结束标记。

### 8.4 Round-trip 验证

```python
text = "你好，nanochat! 123"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)
assert decoded == text
```

不要只测试英文。至少测试：中文、emoji、换行、数字、标点和混合语言。

### 8.5 为什么保存 `token_bytes.pt`

不同 tokenizer 的词表大小和压缩率不同，直接比较“每 token 的平均 loss”可能不公平。`token_bytes.pt` 记录每个 token 对应多少原始字节，用于计算 bits per byte（BPB）。特殊 token 的字节数记为 0。

---

## 9. 阶段 2：对话渲染与 Loss Mask

`render_conversation` 返回：

```text
ids：完整对话 token id
mask：同长度的 0/1 序列
```

典型格式：

```text
<|bos|>
<|user_start|>用户问题<|user_end|>
<|assistant_start|>助手回答<|assistant_end|>
```

mask 原则：

| 内容 | 是否参与 SFT loss |
|---|---:|
| BOS | 0 |
| user 标记和正文 | 0 |
| assistant_start | 0 |
| assistant 正文 | 1 |
| assistant_end | 1 |
| Python 调用 | 1 |
| Python 执行结果 | 0 |

Python 输出不参与监督，因为推理时它来自外部工具，而不是模型应凭空生成的内容。

### 9.1 System Message

当前实现把开头 system message 与随后 user message 合并。它不是所有聊天模板的通用做法，而是本项目当前的数据约定。

### 9.2 Shift 后的 mask 对齐

SFT 构造：

```python
inputs  = ids[:, :-1]
targets = ids[:, 1:]
mask_targets = mask[:, 1:]
targets[mask_targets == 0] = -1
```

关键点：mask 必须和 targets 一样向左错位。忘记 `mask[:,1:]` 会让监督位置整体错一位。

### 9.3 必做调试

使用 `visualize_tokenization`，逐 token 检查：

- user token 是否为 0；
- assistant 正文是否为 1；
- assistant_end 是否被监督；
- tool output 是否为 0；
- 截断后 ids/mask 长度是否一致。

---

## 10. 阶段 3：模型主干 `gpt.py`

目标文件：[nanochat/gpt.py](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py)

推荐顺序：

```text
GPTConfig
→ norm
→ Linear
→ apply_rotary_emb
→ CausalSelfAttention
→ MLP
→ Block
→ GPT.__init__
→ init_weights
→ GPT.forward
→ setup_optimizer
→ FLOPs/KV 估算函数
```

### 10.1 `GPTConfig`

| 参数 | 含义 |
|---|---|
| `sequence_len` | 最大训练上下文 |
| `vocab_size` | tokenizer 词表大小 |
| `n_layer` | Block 数 |
| `n_head` | Query 头数 |
| `n_kv_head` | Key/Value 头数 |
| `n_embd` | 残差流维度 |
| `window_pattern` | 各层长/短窗口模式 |

约束：

```text
n_embd % n_head == 0
n_kv_head <= n_head
n_head % n_kv_head == 0
head_dim = n_embd / n_head
```

### 10.2 参数为空的 RMSNorm

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

它不创建可训练 scale 参数。输入输出均保持 `[B,T,C]`。这与许多 LLM 中带可训练 weight 的 RMSNorm 不同。

### 10.3 自定义 `Linear`

模型主权重保持 fp32 以服务优化器精度，forward 时把 weight 转成输入 dtype 做矩阵乘法：

```text
fp32 master weight
  → forward 临时转换到 COMPUTE_DTYPE
  → bf16/fp16/fp32 matmul
```

这是显式精度控制，不依赖 `torch.amp.autocast`。

### 10.4 RoPE

RoPE 只作用于 Q/K：

```text
q, k [B,T,H,Dh]
  → 最后一维拆成两半
  → 与 cos/sin 组合旋转
  → shape 不变
```

当前实现的旋转符号与部分教材约定相反，但 Q/K 使用一致约定，关键的相对旋转关系保持等价。

### 10.5 GQA 与 QK Norm

```text
q [B,T,Hq,Dh]
k [B,T,Hkv,Dh]
v [B,T,Hkv,Dh]
```

当 `Hkv < Hq` 时是 GQA；多组 Query 共享较少的 K/V 头，从而减少 KV Cache。Q/K 在 RoPE 后进行 RMSNorm，然后进入注意力。

### 10.6 Flash Attention 与窗口

训练时：

- 优先使用 FA3；
- 不可用时回退到 PyTorch SDPA；
- `causal=True`；
- 每层窗口由 `window_pattern` 决定。

`L` 表示长窗口，`S` 表示较短窗口；pattern 循环分配到各层，最后一层强制使用长窗口。

第一遍阅读可设置：

```text
window_pattern = "L"
```

这样先理解普通全上下文 causal attention，再研究滑动窗口。

### 10.7 Attention Shape

| 步骤 | Shape |
|---|---|
| 输入 x | `[B,T,C]` |
| Q 投影 | `[B,T,Hq,Dh]` |
| K/V 投影 | `[B,T,Hkv,Dh]` |
| RoPE / QK Norm | shape 不变 |
| Attention 输出 | `[B,T,Hq,Dh]` |
| 合头 | `[B,T,C]` |
| 输出投影 | `[B,T,C]` |

当前 layout 是 `[B,T,H,D]`，与 nanoGPT 手工注意力常用的 `[B,H,T,D]` 不同。不要机械照搬 transpose。

### 10.8 ReLU² MLP

```text
x [B,T,C]
→ Linear(C,4C)
→ ReLU
→ square
→ Linear(4C,C)
→ [B,T,C]
```

ReLU² 是当前实验选择，不是理解 Transformer 的必要前提。先把它当作激活函数 `f(x)=max(0,x)^2`。

### 10.9 Block

```python
x = x + attention(norm(x))
x = x + mlp(norm(x))
```

基础结构仍是 Pre-Norm Transformer。研究性残差混合发生在更高层 GPT forward 中，不要和 Block 主线混为一谈。

---

## 11. 模型中的研究性结构

### 11.1 Value Embedding / Value Residual

部分层为 token 创建独立 value embedding，并通过输入相关 gate 加到 V 上。直觉是把早期 token 表示直接注入后续注意力 Value 路径。

阅读问题：

- 为什么只影响 V，不影响 Q/K？
- gate 的 shape 如何广播到 head_dim？
- 哪些层启用 value embedding？
- 它增加多少 embedding 参数？

### 11.2 `resid_lambdas`

每层有可学习标量控制残差流尺度。它属于显式残差缩放，不是标准 Transformer 的必需项。

### 11.3 `x0_lambdas`

把初始 embedding `x0` 按层混回残差流，为深层网络保留低层 token 信息。

### 11.4 Smear

把前一个 token 的归一化 embedding 以门控方式混入当前位置，提供廉价的局部 bigram 信息。推理时还要在 KV Cache 中保留前一个 embedding 状态。

### 11.5 Backout

在最终归一化前减去缓存的中间层残差，试图移除部分低层特征。这是项目当前研究技巧，应结合 commit、实验记录或消融结果理解，不要当作架构定理。

### 11.6 正确学习方式

对每个增强做四步记录：

1. 标准基线是什么？
2. 增强修改哪条张量路径？
3. 参数量、FLOPs、KV 内存增加多少？
4. 仓库有什么实验指标支持它？

---

## 12. Meta Device 与权重初始化

`base_train.py` 的模型创建分三步：

```text
with torch.device("meta") 创建模型
  → 只有 shape/dtype，没有真实存储
model.to_empty(device)
  → 在目标设备分配未初始化存储
model.init_weights()
  → 显式初始化全部参数和 buffer
```

好处：

- 避免先在 CPU 创建完整参数再复制到 GPU；
- 对大模型更节省峰值内存；
- 可以低成本构建参考深度模型，只用于参数量和 scaling 计算。

风险：任何遗漏初始化的参数都可能保留垃圾值。因此 `__init__` 与 `init_weights` 的职责必须严格区分。

必做检查：遍历参数，确认没有 NaN/Inf，并打印均值和标准差。

---

## 13. 阶段 4：预训练数据加载

目标文件：[nanochat/dataloader.py](https://github.com/karpathy/nanochat/blob/master/nanochat/dataloader.py)

### 13.1 文档级 DDP 分片

每个 rank 从不同 row group 开始，步长为 world size：

```text
rank 0：0, world_size, 2×world_size, ...
rank 1：1, 1+world_size, ...
```

这样不同 GPU 大体读取不同数据。

### 13.2 BOS-aligned best-fit packing

每个训练行容量是 `T+1`，因为之后要构造：

```text
inputs  = row[:-1]  # T 个
targets = row[1:]   # T 个
```

packing 算法：

1. 缓冲一批已分词文档；
2. 选择能完整放入剩余空间的最长文档；
3. 重复填充；
4. 没有文档能完整放入时，裁剪一个文档填满；
5. 每个文档以 BOS 开始；
6. 达到无 padding、100% token 槽位利用。

权衡：它可能裁掉部分文档尾部，但确保每行开头和文档边界更清晰。

### 13.3 预分配 buffer

代码预分配：

- `row_buffer`：构造 `[B,T+1]`；
- pinned CPU buffer：连续存放 inputs 和 targets；
- GPU buffer：一次 HtoD copy；
- `inputs/targets` 是这些大 buffer 的 view。

这是吞吐优化。第一遍只需保证数据语义正确，再分析为什么减少分配和传输次数。

### 13.4 恢复状态

dataloader state 记录 Parquet 文件、row group 和 epoch，用于近似恢复数据位置。它不一定提供严格逐 token 可复现，但能避免从头重读。

---

## 14. 阶段 5：`base_train.py`

目标文件：[scripts/base_train.py](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py)

### 14.1 阅读分块

```text
CLI 参数
→ 设备/DDP/dtype 初始化
→ tokenizer
→ Meta Device 创建模型
→ FP8 可选转换
→ torch.compile
→ 自动计算训练预算
→ MuonAdamW
→ dataloader
→ scheduler
→ train/eval/sample/checkpoint 循环
```

### 14.2 `--depth` 不是唯一实际参数

项目把 depth 作为主要复杂度旋钮：

```text
base_dim = depth × aspect_ratio
model_dim = 向上取整到 head_dim 的倍数
num_heads = model_dim / head_dim
```

随后参数量、目标 token 数、batch size、学习率和 weight decay 还会参与自动缩放。这是项目约定，不是普适的模型规模定律。

### 14.3 三种训练时长来源

按优先级：

1. `--num-iterations`：直接指定步数；
2. `--target-flops`：按计算预算反推步数；
3. `--target-param-data-ratio`：按参数—数据比决定 token 数。

小白冒烟测试应显式设置 `--num-iterations`，避免意外运行很久。

### 14.4 有效 batch

```text
单 rank 单次 token = device_batch_size × max_seq_len
全 rank 单次 token = 上式 × world_size
grad_accum_steps = total_batch_size / 全 rank 单次 token
```

必须满足整除，否则脚本会 assert。

### 14.5 学习率、动量与 Weight Decay 调度

当前 Base Train 不只是一个 scheduler：

- LR：warmup → constant → linear warmdown；
- Muon momentum：前期升高，warmdown 阶段降低；
- weight decay：随训练余弦衰减到 0。

阅读时分别画三条曲线，不要把它们混成一个“学习率调度”。

### 14.6 训练循环主线

```text
周期性 BPB / CORE / sample
  → 取预取好的 x,y
  → grad accumulation micro-steps
  → model(x,y)
  → loss 按累计步数缩放
  → backward
  → optimizer.step
  → 更新调度参数
  → 日志：loss、tok/s、MFU、VRAM
  → checkpoint
```

第一遍关闭或降低昂贵评测：

```text
--core-metric-every=-1
--sample-every=-1
--save-every=-1
```

### 14.7 CPU 冒烟命令

在已经准备好 tokenizer 和至少少量数据后：

```powershell
python -m scripts.base_train --depth=4 --head-dim=64 --window-pattern=L --max-seq-len=128 --device-batch-size=1 --total-batch-size=128 --eval-tokens=256 --eval-every=10 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --num-iterations=20 --run=dummy --device-type=cpu
```

参数受仓库更新影响；若当前版本 assert，应优先检查 `total_batch_size` 是否是 `device_batch_size × max_seq_len × world_size` 的整数倍。

---

## 15. Muon + AdamW

目标文件：[nanochat/optim.py](https://github.com/karpathy/nanochat/blob/master/nanochat/optim.py)

先理解参数分组，再读公式：

| 参数类型 | 示例 | 优化方式 |
|---|---|---|
| Transformer 二维矩阵 | Q/K/V、MLP 权重 | Muon |
| Token Embedding | `wte` | AdamW 类路径 |
| LM Head | unembedding | AdamW 类路径 |
| 标量参数 | residual lambdas 等 | AdamW 类路径 |

Muon 的核心直觉：先用动量累积矩阵更新，再对更新矩阵做近似正交化/谱方向处理，使不同方向的更新尺度更均衡。

学习顺序：

1. 参数如何分组；
2. 单卡 update；
3. Newton-Schulz 类迭代的输入输出；
4. 分布式参数切分/通信；
5. momentum 与 weight decay 调度。

不要只记住“Muon 更快”。它的适用参数、学习率尺度和分布式实现均不同于 AdamW。

---

## 16. 精度体系

`COMPUTE_DTYPE` 根据硬件自动选择，也可由环境变量覆盖：

```powershell
$env:NANOCHAT_DTYPE = "float32"
python -m scripts.base_train ...
```

理解三层 dtype：

1. optimizer master weight：通常 fp32；
2. embedding 存储：可能直接使用 compute dtype；
3. forward matmul：自定义 Linear 转换到 compute dtype。

fp16 训练会启用 GradScaler；bf16 与 fp32 通常不使用。FP8 是额外实验路径，首次阅读应关闭。

---

## 17. 评测体系

### 17.1 Training Loss

用于优化，但受 tokenizer 粒度影响，不适合直接比较不同 tokenizer。

### 17.2 BPB

Bits per byte 使用 token 对应的原始字节数归一化，更适合比较不同词表和 tokenizer。

### 17.3 CORE

Base Model 的综合能力指标，聚合多个任务。阅读时关注：

- prompt 如何构造；
- token probability 如何转为选项分数；
- 是否做长度归一化；
- 多任务分数如何聚合。

### 17.4 ChatCORE

面向聊天模型的综合评测，通常在 SFT 阶段周期性执行。

### 17.5 pass@k

同一问题生成 k 个答案，只要至少一个正确便记为成功。它衡量“多次采样时找到正确答案”的概率，不等同于单次回答准确率。

---

## 18. 阶段 6：SFT

目标文件：[scripts/chat_sft.py](https://github.com/karpathy/nanochat/blob/master/scripts/chat_sft.py)

### 18.1 Base Model 与 Chat Model

```text
Base Model：学习通用 next-token distribution
SFT Model：学习用户/助手格式、回答方式、工具调用格式
```

SFT 从 Base checkpoint 加载模型，并可继承预训练的上下文长度、batch、学习率和 optimizer momentum。

### 18.2 数据混合

当前训练混合包含：

- SmolTalk：通用对话；
- MMLU：知识与多选；
- GSM8K：数学推理。

`TaskMixture` 把不同任务统一成 conversation 接口。阅读每个 Task 时只追踪：

```text
原始样本
→ conversation messages
→ render_conversation
→ ids + mask
```

### 18.3 SFT Packing 与预训练 Packing 的差异

预训练无法完整放入时允许裁剪文档；SFT 为避免破坏完整对话，无法再放入时使用 BOS token 补齐剩余空间，并把 padding mask 设为 0。

### 18.4 SFT Loss

```text
完整对话作为输入上下文
  → 只有 assistant answer / tool call token 保留 targets
  → user、BOS、padding、tool output targets = -1
  → Cross Entropy ignore_index=-1
```

### 18.5 SFT 验收

- 一条 conversation 能正确渲染；
- ids 与 mask 等长；
- assistant 文本被监督；
- user 和 tool output 不被监督；
- padding 不被监督；
- targets 至少存在一个非 `-1` token；
- Base checkpoint 与 SFT 配置匹配。

---

## 19. 阶段 7：推理引擎与 KV Cache

目标文件：[nanochat/engine.py](https://github.com/karpathy/nanochat/blob/master/nanochat/engine.py)

### 19.1 分层职责

- Tokenizer：字符串 ↔ token ids；
- Engine：token ids → next token；
- GPT：给定 token ids 与 cache 计算 logits；
- Chat CLI：维护对话、格式化输入、展示输出。

Engine 不应了解自然语言字符串，这种分层有利于测试和复用。

### 19.2 KV Cache Shape

当前缓存：

```text
k_cache / v_cache
[L,B,Tmax,Hkv,Dh]
```

与 FA3 对接时每层 view 是 `[B,Tmax,Hkv,Dh]`，并用 `cache_seqlens [B]` 记录各 batch 当前长度。

### 19.3 Prefill 与 Decode

```text
Prefill：一次处理完整 prompt，写入各层 K/V
Decode：每次只送入新 token，读取历史 cache
```

nanoGPT 没有 cache，会重复计算历史；nanochat 通过预分配 cache 换取更快生成，但消耗额外显存。

### 19.4 批量采样

Engine 可以先用 batch=1 prefill prompt，再把 cache 扩展/复制给多个样本并行 decode，避免为相同 prompt 重复 prefill。

### 19.5 采样参数

- temperature=0：通常走确定性选择；
- temperature>0：按概率采样；
- top-k：限制候选集；
- seed：控制可复现性；
- max tokens：限制生成长度。

### 19.6 工具执行安全

仓库含 Python/计算器工具执行路径。阅读源码时不要默认任何“过滤 + eval/exec”方案可安全处理不可信代码。只在隔离环境和可信实验输入中运行；不要把它直接暴露为公网执行服务。

---

## 20. 阶段 8：RL

目标文件：[scripts/chat_rl.py](https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py)

### 20.1 当前算法的准确定位

源码虽然使用“GRPO”称呼，但主动说明它更接近简化的 on-policy REINFORCE：

- 没有 reference model KL trust region；
- 没有 PPO ratio 和 clipping；
- rollout 来自当前 policy；
- 使用 token-level 归一化；
- advantage 使用 `reward - group_mean`，不除标准差。

不要把这份实现当作标准 GRPO 或 PPO 模板。

### 20.2 一次 RL Batch

```text
取 GSM8K 问题
→ 删除真实 assistant answer
→ 保留 assistant_start 作为生成起点
→ 对同一问题采样 N 个回答
→ 解码并计算答案 reward
→ padding 到同长度
→ prompt/tool token mask 掉
→ advantage = reward - mean(rewards)
→ 计算生成 token log probability
→ policy gradient objective
```

### 20.3 核心公式

对同一问题的样本 i：

```text
A_i = r_i - mean(r)
objective ∝ Σ_t log π(token_t | prefix) × A_i
loss = -objective
```

若某答案高于组内平均 reward，梯度提高其生成 token 的概率；低于平均则降低。

### 20.4 Token Mask

只训练模型实际采样的回答 token：

- prompt token：mask 0；
- padding：mask 0；
- 外部工具强制插入内容：mask 0；
- policy 自己生成的有效回答：mask 1。

### 20.5 RL 前置门槛

在进入代码前应能解释：

- policy、trajectory、reward；
- log probability；
- REINFORCE；
- baseline 降方差；
- on-policy；
- PPO ratio/clip；
- KL regularization；
- sequence-level 与 token-level normalization；
- pass@k。

---

## 21. 从零实践路线

### 路线 A：只读源码，不训练大模型

1. 跑 tokenizer 小数据 round-trip。
2. 手工创建极小 GPTConfig。
3. 随机 token 做 forward/backward。
4. 打印 Attention shape。
5. 人工构造 conversation，检查 ids/mask。
6. 单 batch 过拟合。
7. 阅读 Engine cache shape。
8. 用伪 reward 手算一个 RL objective。

### 路线 B：CPU/MPS 教学流程

参考 `runs/runcpu.sh`，但逐条运行并缩短：

1. 下载少量 shard；
2. 小字符数训练 tokenizer；
3. depth 4～6 小模型预训练；
4. 只跑少量 SFT step；
5. CLI 验证链路。

目标是流程正确，不是模型能力。

### 路线 C：单 GPU

- 从较小 depth 开始；
- `device_batch_size` 从 1/2/4 向上测试；
- 先用 `window_pattern=L`；
- 禁用 FP8；
- CORE/sample 低频或关闭；
- 先确认 20 step 稳定，再扩大数据和步数。

### 路线 D：官方 speedrun

只适合明确拥有相应多 GPU 预算并理解成本、数据量、FP8、FA3、DDP 和 checkpoint 的情况。运行前固定仓库 commit，因为 master 的结果和参数会变化。

---

## 22. 必做实验

### 实验 A：Tokenizer Round-trip

对中英文、emoji、数字和换行执行 encode/decode，确保完全还原。

### 实验 B：Token 压缩率

比较字符 tokenizer、nanoGPT GPT-2 BPE、nanochat BPE 对同一文本的 token 数和 bytes/token。

### 实验 C：对话 Loss Mask

逐 token 输出：token id、字符串、mask、shift 后 target。确认监督位置没有偏移。

### 实验 D：基础模型 Shape

使用 B=2、T=8，记录 embedding、Q/K/V、attention output、Block output 和 logits。

### 实验 E：MHA 与 GQA Cache

固定 L、C、Dh、T，分别令 `Hkv=Hq` 和 `Hkv=Hq/4`，计算：

```text
KV bytes = L × 2 × T × Hkv × Dh × dtype_bytes
```

### 实验 F：因果性

改变未来 token，确认过去位置 logits 不变。关闭 dropout，固定随机种子。

### 实验 G：滑动窗口

比较 `L` 与 `SSSL`：每层可见范围、估算 FLOPs、长距离依赖和运行速度。

### 实验 H：单 Batch 过拟合

固定一个预训练 batch，训练到 loss 明显下降。失败时先排查实现和 dtype。

### 实验 I：Meta 初始化

比较普通初始化与 meta → to_empty → init_weights 的峰值内存；确认所有参数已正确初始化。

### 实验 J：KV Cache

记录 prefill 与每个 decode step 的输入长度、cache position、耗时和显存。

### 实验 K：BPB

构造两个不同 tokenization 的同一文本，解释为什么 raw token loss 不宜直接比较，而 BPB 更稳定。

### 实验 L：RL Advantage

假设同一问题四个 reward 为 `[0,0,1,1]`，手算 group mean、advantage 和每条序列的梯度方向。

---

## 23. 常见错误排查

### 23.1 找不到 tokenizer

- 是否运行 `scripts.tok_train`；
- `NANOCHAT_BASE_DIR` 是否与训练时一致；
- `tokenizer.pkl` 和 `token_bytes.pt` 是否同时存在。

### 23.2 找不到数据

- 是否运行 `python -m nanochat.dataset -n N`；
- 数据缓存目录是否正确；
- 是否至少有训练 shard 和固定验证 shard；
- Parquet 文件是否完整。

### 23.3 `total_batch_size` assert

检查：

```text
total_batch_size % (device_batch_size × max_seq_len × world_size) == 0
```

### 23.4 FA3 不可用

- 首次实验接受 SDPA fallback；
- 使用 `--window-pattern=L` 避免 fallback 下低效滑动窗口；
- 不要为了启用 FA3 随意更换 CUDA/PyTorch 破坏环境。

### 23.5 OOM

依次降低：

1. `device-batch-size`；
2. `max-seq-len`；
3. `depth`；
4. head/width 相关参数；
5. 评测 batch 和生成样本数。

### 23.6 NaN/Inf

- 强制 `NANOCHAT_DTYPE=float32`；
- 关闭 FP8；
- 检查所有参数初始化；
- 检查 targets 范围和有效 token 数；
- 检查 LR、Muon momentum、grad norm；
- 使用单 batch 和较小模型复现。

### 23.7 SFT Loss 异常

- 检查 mask 是否全 0；
- 检查 mask 是否正确 shift；
- 检查完整 conversation 是否被截断到没有 assistant 回答；
- 检查 padding 是否设为 `-1`；
- 检查 Base checkpoint 和 tokenizer 是否配套。

### 23.8 恢复训练不连续

- 模型 checkpoint、optimizer checkpoint、meta 是否同一步；
- dataloader state 是否加载；
- dtype、depth、sequence length 是否一致；
- world size 改变是否会影响精确复现；
- 随机数状态是否按当前实现保存。

### 23.9 CLI 无法生成

- 是否已有正确阶段的 checkpoint；
- `model-tag`、step 和 phase 是否正确；
- tokenizer 特殊 token 是否与 checkpoint 一致；
- cache 最大长度是否够 prompt + output；
- 当前设备是否支持 checkpoint dtype。

---

## 24. nanoGPT、MiniMind、nanochat 对照

| 维度 | nanoGPT | MiniMind | nanochat |
|---|---|---|---|
| 最适合阶段 | 第一套 GPT 源码 | 现代 LLM 常规组件 | 端到端现代训练工程 |
| Tokenizer | 字符/GPT-2 BPE | 项目 tokenizer | 自训练 GPT-4 风格 BPE |
| 数据 | 连续 bin token 流 | JSONL Dataset | Parquet + 在线分词 + packing |
| 位置 | 绝对 Embedding | RoPE | RoPE |
| Norm | LayerNorm | RMSNorm | 无参数 RMSNorm + QK Norm |
| Attention | MHA | GQA | GQA + 滑动窗口 + FA3 |
| FFN | GELU | 门控 FFN | ReLU² |
| KV Cache | 无 | 有 | 完整推理 Engine |
| 优化器 | AdamW | AdamW | Muon + AdamW |
| SFT | 简单文本微调 | 完整 SFT | 任务混合、packing、工具格式 |
| RL | 无 | 多种算法 | GSM8K 简化 on-policy PG |
| 研究特性 | 少 | MoE 等 | Scaling、FP8、Value Residual 等 |

最稳妥路线：

```text
PPT
→ nanoGPT：掌握不变的基本骨架
→ MiniMind Dense + SFT：掌握常见现代 LLM
→ nanochat：掌握完整流水线和研究工程
```

---

## 25. 源码标注模板

```text
名称：CausalSelfAttention
类别：基础主线 / 现代工程 / 研究增强
职责：执行因果自注意力
输入：x [B,T,C]
Q：[B,T,Hq,Dh]
K/V：[B,T,Hkv,Dh]
位置编码：RoPE
归一化：QK RMSNorm
窗口：本层 window_size
训练路径：FA3/SDPA causal attention
推理路径：读写 KV Cache
输出：[B,T,C]
参数：c_q/c_k/c_v/c_proj、可选 ve_gate
与 nanoGPT 差异：layout、RoPE、GQA、QK Norm、cache、window
待验证：mask、cache position、dtype、shape
```

对 tokenizer、dataloader、optimizer、task 和 engine 也使用同样模板：职责、输入、输出、状态、失败边界、与前一项目差异。

---

## 26. 完成标准

### Tokenizer 与数据

- [ ] 能解释 byte-level BPE 和 split pattern。
- [ ] 能解释普通 token 与特殊 token。
- [ ] 能完成多语言 round-trip。
- [ ] 能解释 `token_bytes.pt` 与 BPB。
- [ ] 能解释 Parquet shard 和验证集划分。
- [ ] 能解释 BOS-aligned best-fit packing。
- [ ] 能正确推导 inputs/targets shift。

### 模型

- [ ] 能推导 Hq、Hkv、Dh 和所有 Attention shape。
- [ ] 能解释 RoPE、RMSNorm、QK Norm 和 GQA。
- [ ] 能解释长/短滑动窗口。
- [ ] 能解释 ReLU² MLP。
- [ ] 能区分基础结构与 Value Embedding、Smear、Backout。
- [ ] 能解释显式 dtype 和 Meta Device 初始化。

### 训练

- [ ] 能解释 depth 如何派生 width/head。
- [ ] 能计算有效 token batch 和累积步数。
- [ ] 能解释训练时长的三种来源。
- [ ] 能画出 LR、Muon momentum、weight decay 调度。
- [ ] 能解释 Muon 与 AdamW 参数分组。
- [ ] 能完成小模型 forward/backward/checkpoint。

### SFT、推理与 RL

- [ ] 能解释 conversation 特殊 token。
- [ ] 能逐 token验证 assistant-only loss mask。
- [ ] 能解释 prefill、decode 和 KV Cache shape。
- [ ] 能解释批量生成如何复用 prompt cache。
- [ ] 能解释 CORE、ChatCORE、BPB 和 pass@k。
- [ ] 能准确说明当前 RL 与标准 PPO/GRPO 的差异。
- [ ] 能手算 group mean advantage 和 policy-gradient 方向。

完成上述清单，才算真正具备修改 nanochat 的基础，而不只是运行了官方脚本。

---

## 27. 关键链接

- [nanochat README](https://github.com/karpathy/nanochat)
- [CPU/MPS 教学流程](https://github.com/karpathy/nanochat/blob/master/runs/runcpu.sh)
- [多 GPU speedrun](https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh)
- [模型 gpt.py](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py)
- [Tokenizer](https://github.com/karpathy/nanochat/blob/master/nanochat/tokenizer.py)
- [预训练数据](https://github.com/karpathy/nanochat/blob/master/nanochat/dataset.py)
- [DataLoader](https://github.com/karpathy/nanochat/blob/master/nanochat/dataloader.py)
- [预训练脚本](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py)
- [SFT 脚本](https://github.com/karpathy/nanochat/blob/master/scripts/chat_sft.py)
- [RL 脚本](https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py)
- [推理 Engine](https://github.com/karpathy/nanochat/blob/master/nanochat/engine.py)
- [优化器](https://github.com/karpathy/nanochat/blob/master/nanochat/optim.py)
- [Chat CLI](https://github.com/karpathy/nanochat/blob/master/scripts/chat_cli.py)

## 28. 最后建议

阅读 nanochat 时同时追踪五条线：

1. **数据线**：文档、conversation、token、mask 如何变化？
2. **形状线**：当前 tensor 的 B、T、Hq、Hkv、Dh、V 是什么？
3. **状态线**：checkpoint、dataloader、KV Cache、optimizer 保存了什么状态？
4. **性能线**：该优化减少计算、显存、通信还是 Python 开销？
5. **证据线**：这个结构是通用原理，还是当前 speedrun 的实验选择？

只要始终区分“Transformer 不变主干”和“nanochat 当前实验设计”，即使仓库继续更新，你也能根据数据流、shape 和训练目标重新定位代码，而不依赖死记文件行号。
