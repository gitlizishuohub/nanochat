---
source_files:
  - nanochat/gpt.py
  - nanochat/common.py
  - nanochat/optim.py
source_symbols:
  - GPTConfig
  - norm
  - CausalSelfAttention.forward
  - GPT.forward
  - GPT.generate
doc_type: tutorial
knowledge_scope: W1 D4 无缓存模型路径和张量形状
---

# W1 D4：从 token ID 走到 logits

今天完成一张能自己讲出来的形状图，并用随机小模型核查。预计 90 分钟：15 分钟配置，25 分钟注意力，20 分钟完整 forward，15 分钟脚本，15 分钟报告。进阶阅读可另加 30 分钟，不挤掉验收。

前置：[D3 测试](D3_inference_smoke.md)。只使用随机权重，不要等待真实 checkpoint。随机权重能验证接口和形状，不能衡量语言质量或实际模型性能。

## 1. 建立符号与尺寸的对应

```bash
cd /home/lizishuo/_program_/Project/nanochat
source /home/lizishuo/miniconda3/etc/profile.d/conda.sh
conda activate nanochat
rg -n '^class GPTConfig|^def norm|^class CausalSelfAttention|^class Block|def forward|def generate' nanochat/gpt.py
sed -n '25,160p' nanochat/gpt.py
```

在笔记页首写下 `B=批量大小，T=本次输入长度，C=n_embd，Hq=n_head，Hkv=n_kv_head，Dh=C/Hq，V=词表大小`。<abbr title="tensor shape · 张量每个轴的长度。相同数字在不同轴代表不同含义；例如 T 是当前输入长度，Tmax 是缓存容量，两者不能混用。">shape</abbr> 既要记数字，也要记轴含义。

脚本固定 `B=1,T=8,C=64,Hq=4,Hkv=2,Dh=16,V=128,L=2`，足够小且满足整除要求。不能任意改成 `C=16`，当前 smear gate 要读取前 24 个通道，极小配置可能不合法。head dim 同时要兼容当前旋转位置编码的成对拆分。

## 2. 读 norm 与注意力，不跳过布局

`norm(x)` 调用 `F.rms_norm(x, (x.size(-1),))`。<abbr title="RMSNorm / Root Mean Square Normalization · 按最后一维的均方根缩放输入。此项目没有可学习的归一化权重，epsilon 未在调用处显式指定，不能擅自声称与另一个实现的常数完全相同。">RMSNorm</abbr> 保持 shape；这里没有可学习 norm 参数，但不能据此说整个模型无参数。

按 `CausalSelfAttention.forward` 的顺序画表：

| 节点 | 形状 | 你要注意的事 |
|---|---|---|
| 残差输入 | `[1,8,64]` | B、T、C |
| `c_q` 后 | `[1,8,64]` | 下一步 view 为 `[1,8,4,16]` |
| `c_k/c_v` 后 | `[1,8,32]` | view 为 `[1,8,2,16]` |
| RoPE 与 Q/K norm 后 | Q `[1,8,4,16]`；K `[1,8,2,16]` | 轴不变；Q/K 各乘 1.2 |
| attention 输出 | `[1,8,4,16]` | 回退内部会 transpose |
| 合并 heads 与投影 | `[1,8,64]` | 返回残差通道 |

<abbr title="GQA / Grouped-Query Attention · 分组查询注意力。多组 Query heads 共享较少的 Key/Value heads，本文 Hq=4、Hkv=2；缓存按 Hkv 存储，不按 Hq 存储。">GQA</abbr> 解释了 K/V 比 Q 窄。<abbr title="RoPE / Rotary Position Embedding · 旋转位置编码。对 Q 和 K 的通道对做随位置变化的旋转，使注意力利用相对位置信息；这里不改变张量尺寸。">RoPE</abbr> 不是独立加在 token embedding 上的位置向量。

Value embedding 会在特定层通过 gate 混入 V；不要把 `c_v(x)` 当作 V 的全部来源。第一次阅读只标出混入位置，不需要今天推导训练效果。

## 3. 沿 GPT.forward 走一遍完整链路

```bash
sed -n '460,565p' nanochat/gpt.py
```

逐项写入报告草稿：

1. 整数 ID `[B,T]` 查 embedding 得 `[B,T,C]`，转 compute dtype，再 norm。
2. smear 混合相邻 token embedding。无 cache 时通过切片读取前一个位置，因此当前代码断言 `T>1`。本教程用 P=8；不能用单 token 无缓存输入作为本版本的通过条件。
3. 每层先结合 `resid_lambdas`、`x0_lambdas`，再 attention 和 MLP；形状仍 `[B,T,C]`。
4. 中间层残差参与 backout；末层 norm 后，`lm_head` 输出 padded vocab，再裁回 `V`。
5. logits 转 FP32、经过 `15*tanh(logits/15)` softcap。<abbr title="softcap · 用平滑函数限制分数幅度。此处最终分数被压到接近正负 15 的范围，数值不再是裸线性投影输出。">softcap</abbr> 不改变 `[B,T,V]`。
6. 不传 targets 返回 logits；传 targets 返回交叉熵损失，不能仍按三维 logits 解读返回值。

`GPT.generate` 每次把完整已生成前缀重新传入 forward，只取最后一个位置分数选下一 token，再拼回前缀；它是无缓存、自回归、batch=1 路径。

## 4. 用脚本验证形状

打开[随机模型与缓存探查脚本](scripts/probe_model_and_cache.py)，重点看 `GPTConfig`、`init_weights()` 和 `register_forward_hook()`：

```bash
python .agents/docs/career_roadmap/w1/scripts/probe_model_and_cache.py --help
python .agents/docs/career_roadmap/w1/scripts/probe_model_and_cache.py --mode shapes
```

脚本在导入前隐藏 CUDA、固定 FP32、seed=42 和单 CPU 线程，避免 GPU 内核发现；不下载 tokenizer、数据或权重。默认输出 `experiments/inference_infra/results/w1/shapes_probe.json`。

预期可检查的结构是 `logits_shape=[1,8,128]`、`logits_dtype=torch.float32`、`USE_FA3=false`。这是契约示例，不是预先填好的执行结果。`shapes` 的 c_q/c_k/c_v hook 截获的是线性层输出，尚未经过 view，所以 K/V 是 `[1,8,32]`；你需要根据源码补上 heads 轴。

脚本在初始化后给残差输出投影填入非零随机值，并启用非零 smear 系数。原因是 `init_weights()` 默认把部分输出投影置零，直接使用默认值可能让后续缓存错误被“零输出”掩盖。它依旧是随机功能实验，不是训练好的模型。

编写时验证记录（2026-10-15）：使用 Conda `nanochat` 的 Python 3.10.20 / torch 2.9.1+cpu 运行同一脚本，shape 与有限值断言通过，实际 logits 为 `[1,8,128]`、FP32、SDPA。该记录是交付脚本烟测，不替代你当天生成的实验报告。

## 5. 记录图、处理失败并验收

新建 `experiments/inference_infra/reports/model_path.md`，至少包含：配置表、上面的形状链、smear/QK norm/backout 的位置、GPT.generate 伪流程、实际 JSON 链接。下图采用纵向布局，节点内标注输出 shape，表示不传 targets、无缓存且 `T>1` 的 forward 路径：

```mermaid
%%{init: {"theme":"base","themeVariables":{"lineColor":"#3b82f6","defaultLinkColor":"#3b82f6","primaryColor":"#123d57","primaryTextColor":"#ffffff","primaryBorderColor":"#7dd3fc","fontSize":"16px"},"themeCSS":".marker { fill: #3b82f6 !important; stroke: #3b82f6 !important; }","flowchart":{"curve":"linear","nodeSpacing":32,"rankSpacing":48,"wrappingWidth":240}}}%%
flowchart TD
  ids["Token IDs<br/>[B, T]"]
  embed["Embedding → compute dtype<br/>[B, T, C]"]
  first_norm["RMSNorm<br/>[B, T, C]"]
  smear["Smear：混入前一位置的 embedding<br/>[B, T, C]"]
  blocks["逐层残差混合 → Attention → MLP<br/>Q/K norm 位于 Attention 内<br/>[B, T, C]"]
  backout["Backout：减去缩放的中间层残差<br/>[B, T, C]"]
  final_norm["末层 RMSNorm<br/>[B, T, C]"]
  head["lm_head：投影到补齐后的词表<br/>[B, T, Vpad]"]
  logits["裁回 V → FP32 → softcap<br/>Logits [B, T, V]"]

  ids --> embed --> first_norm --> smear --> blocks
  blocks --> backout --> final_norm --> head --> logits
    classDef default fill:#123d57,stroke:#7dd3fc,color:#ffffff,stroke-width:2px;
    linkStyle default stroke:#3b82f6,stroke-width:3px;
```

`Vpad` 表示模型为对齐而补齐后的词表大小，`V` 是实际词表大小。将上图中的符号替换为本次实验的具体尺寸；Attention 内部的 Q/K/V 布局另用第 2 节的表补充。不要将 hook 看到的线性层输出错标成多头布局。

| 失败 | 检查 |
|---|---|
| DEPENDENCY_MISSING | 返回 D2 确认依赖与解释器，保存错误 |
| T>1 断言 | 确认 `--prompt-tokens` 至少 2，不绕开断言 |
| dtype 断言 | 环境变量必须在导入前设置；新开进程重跑 |
| shape 不一致 | 先核对配置和线性层输出/reshape 的区别 |
| 有 NaN/Inf | 保留输出与 seed，不能用异常数值继续对比 |

验收：能从 `[B,T]` 推到 `[B,T,V]`；能说明 Hq/Hkv；真实运行成功或明确依赖阻塞；报告不含随机模型的质量结论。当天记录填写真实日期、commit、用时、命令、结果、状态和明日待办。

上一篇：[D3 推理测试](D3_inference_smoke.md)。下一篇：[D5 缓存生命周期](D5_cache_lifecycle.md)。
