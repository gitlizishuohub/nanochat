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

[栏目目录](README.md) · [上一篇](06-forward-pass-and-prediction-loss.md) · [下一篇](08-precision-and-training-acceleration.md) · [术语查阅](glossary/README.md)

> 原技术正文沿用 2026-10-12 的源码核对记录；2026-10-21 调整结构。新增 SFT 串联说明做了局部静态核对，未执行训练。

# 7. 反向传播、优化器与训练预算

<a id="sec-8"></a>
## 7.1 一次更新的真实次序

预训练/SFT用<a id="ref-compile-001"></a><abbr title="torch.compile / Static Shape｜图编译与固定形状&#10;定义：把PyTorch运算路径捕获为可优化计算图，通过融合等方式降低重复调度成本。&#10;作用：加速固定B/T的重复前向/反向和优化器步骤，不改变训练目标定义。&#10;本项目：base/SFT模型dynamic=False，优化器fused函数也编译；保留orig_model供保存/变长生成，FP8替换在编译前。&#10;注意：编译可能有初期开销，不是算法或质量验证；没有实际跑过就不能承诺加速，RL没有torch.compile(model)调用。&#10;位置：scripts/base_train.py:246 orig_model/compile；scripts/chat_sft.py:117 compile；nanochat/optim.py:23 编译装饰器">torch.compile(dynamic=False)</abbr>（[释义](glossary/05-loss-and-parameter-updates.md#term-compile)）优化固定形状路径，保留orig_model保存原始权重键名与变长生成。每微批forward求loss，除累积次数A，<a id="ref-backprop-001"></a><abbr title="Autograd / Backward｜自动微分与反向传播&#10;定义：沿前向计算图使用链式法则计算损失对可学习张量的导数，并写入参数梯度。&#10;作用：告诉优化器沿哪些方向调整数值可改善当前目标，连接预测误差与参数更新。&#10;本项目：训练调用loss.backward或scaler.scale(loss).backward，微批累加，更新后zero_grad(set_to_none=True)。&#10;注意：backward只求梯度，optimizer.step才改权重；model.eval不关闭求导，验证no_grad、Engine inference_mode才限制梯度记录。&#10;位置：scripts/base_train.py:511 单步；scripts/chat_sft.py:430 backward；nanochat/engine.py:176 generate">backward</abbr>（[释义](glossary/05-loss-and-parameter-updates.md#term-backprop)）累加梯度；完成A批后应用调度、优化器同步并更新，最后zero_grad(set_to_none=True)。日志取最后微批loss再平滑，不是精确整步平均。

<a id="ref-accum-001"></a><abbr title="Gradient Accumulation｜梯度累积&#10;定义：多个微批分别反向、梯度相加，全部完成后再进行一次优化更新。&#10;作用：在每卡显存限制下扩大统计批量，降低device batch时可维持全局token容量。&#10;本项目：A=total_batch_size/(B×T×W)须整除，每微批mean loss除A，最后optimizer.step后清梯度。&#10;注意：SFT有效位置数不同，等权微批均值不等于全局有效token均值；rank平均也不重加权。日志仅末微批值不能当精确整步loss。&#10;位置：scripts/base_train.py:410 grad_accum_steps；scripts/base_train.py:513 loss缩放；scripts/chat_sft.py:425 累积">梯度累积</abbr>（[释义](glossary/05-loss-and-parameter-updates.md#term-accum)）对等长全监督batch得到等权均值；SFT每微批有效token数不同，mean(loss)/A是等权微批均值，不严格等于所有有效token统一均值，rank间AVG也不重新按有效token加权。

混合精度和编译选项见[训练加速](08-precision-and-training-acceleration.md)；多卡通信细节见[优化器通信](09-distributed-optimizer-and-communication.md)。

## 7.2 优化器与参数分组

<a id="ref-optim-001"></a><abbr title="Optimizer / MuonAdamW｜优化器与混合更新器&#10;定义：根据梯度和历史动量/统计调整模型数值，并保存跨步更新状态的对象。&#10;作用：将损失导数转为实际参数变化，支持不同类别权重使用不同更新算法。&#10;本项目：GPT.setup_optimizer给lm_head/wte/值表/标量建AdamW组，块内矩阵建Muon组；同一step还同步分布式梯度与参数。&#10;注意：不负责计算loss或shift标签；并非所有二维权重Muon。恢复原更新轨迹需每rank状态，只有model文件不足。&#10;位置：nanochat/gpt.py:419 setup_optimizer；nanochat/optim.py:184 MuonAdamW；nanochat/optim.py:429 step">MuonAdamW</abbr>（[释义](glossary/05-loss-and-parameter-updates.md#term-optim)）组合<a id="ref-adamw-001"></a><abbr title="AdamW / Momentum / Bias Correction｜AdamW、动量与偏差修正&#10;定义：维护梯度一阶/二阶指数平均并修正初期偏差，再按自适应幅度更新，权重衰减与梯度分离。&#10;作用：平滑噪声与调节不同权重的更新尺度，用于嵌入、输出层和标量。&#10;本项目：fused步骤转fp32计算回写原dtype；lm_head betas(0.8,0.96)，wte/value(0.8,0.995)，eps1e−10。&#10;注意：zeros_like状态可能低精度存储，计算fp32不等于始终fp32保存；各组beta/decay不同，不能套0.9/0.999常见默认。&#10;位置：nanochat/optim.py:24 adamw_step_fused；nanochat/optim.py:320 _compute_adamw；nanochat/gpt.py:439 分组">AdamW</abbr>（[释义](glossary/05-loss-and-parameter-updates.md#term-adamw)）与<a id="ref-muon-001"></a><abbr title="Muon / Polar Express / NorMuon｜矩阵动量正交化更新&#10;定义：对矩阵动量方向做近似正交化及尺度调整，减少单一方向主导整个矩阵更新的情况。&#10;作用：为Transformer内部矩阵提供适合其结构的更新，同时对同形矩阵批量计算。&#10;本项目：包括Nesterov、行均衡、五次Polar Express、Frobenius再归一化、分解二阶矩、cautious decay。&#10;注意：不是精确SVD最近正交矩阵或梯度裁剪；ns_steps5，实际lr乘sqrt(max(1,行/列))，不能仅看组内显示lr理解幅度。&#10;位置：nanochat/optim.py:112 muon_step_fused；nanochat/optim.py:362 _compute_muon；nanochat/gpt.py:449 分组">Muon</abbr>（[释义](glossary/05-loss-and-parameter-updates.md#term-muon)）。wte、value_embeds、lm_head、外层标量/Smear参数用AdamW；transformer.h内部矩阵（含值门控）按同形组用Muon，不只看张量是否二维。

### 7.2.1 深入：参数组数值与 Muon 更新细节

| AdamW组 | 组构造学习率（再经入口倍率） | betas / decay |
| --- | --- | --- |
| lm_head | unembedding_lr×(C/768)^−1/2 | (0.8,0.96) / 0.01 |
| wte | embedding_lr×(C/768)^−1/2 | (0.8,0.995) / 0.001 |
| value_embeds | 上项×0.5 | (0.8,0.995) / 0.01 |
| resid_lambdas | scalar_lr×0.01 | (0.8,0.95) / 0.05 |
| x0_lambdas | scalar_lr | (0.96,0.95) / 0 |
| smear_gate、smear_lambda、backout_lambda | 固定0.2 | (0.8,0.95) / 0 |

AdamW eps=1e-10，计算中转fp32再回写原类型，状态zeros_like可能低精度保存。Muon初组momentum=0.95、ns_steps=5、beta2=0.9，含Nesterov、行均衡、五次Polar Express、范数再归一化、分解二阶矩、<a id="ref-wd-001"></a><abbr title="Weight Decay / Cautious Decay｜权重衰减与谨慎衰减&#10;定义：优化更新时抑制权重增长，cautious变体只对指定更新与参数方向关系施加衰减。&#10;作用：控制参数尺度，需区别算法与组/阶段实际设置，不是loss里任意统一L2项。&#10;本项目：AdamW头0.01、输入表0.001、值表0.01、resid0.05；Muon基准0.28经预算缩放再余弦降，按更新×参数非负mask。&#10;注意：SFT新建只令Muon decay0，热启动加载元数据可覆盖它，AdamW仍非全0；RL帮助中的作用组描述与实参也不同。&#10;位置：nanochat/gpt.py:439 分组；nanochat/optim.py:178 cautious更新；scripts/base_train.py:385 调度；scripts/chat_sft.py:143 加载">cautious decay</abbr>（[释义](glossary/05-loss-and-parameter-updates.md#term-wd)）；实际lr再乘sqrt(max(1,行数/列数))。这些不是梯度范数裁剪。

## 7.3 预训练默认预算与调度

| CLI项 | 默认 | 注意 |
| --- | --- | --- |
| device-batch-size / total-batch-size | 32 / −1 | 总量自动估计、按2的幂取整，单位token。 |
| num-iterations / target-flops / target-param-data-ratio | −1 / −1 / 12 | 正明确步数→正FLOPs→正比例，按优先级。 |
| embedding-lr / unembedding-lr | 0.3 / 0.008 | 与函数自身0.2/0.004不同。 |
| matrix-lr / scalar-lr / weight-decay | 0.02 / 0.5 / 0.28 | decay是Muon基准，不是全组。 |
| warmup-steps / warmdown-ratio / final-lr-frac | 40 / 0.65 / 0.05 | 线性升温、平台、线性降温。 |
| eval-every / eval-tokens | 250 / 80×524288 | 正周期启用，预算取整成batch次数。 |
| core-metric-every / max-per-task | 2000 / 500 | 启用时末尾也评。 |
| sample-every / save-every | 2000 / −1 | 保存默认仅末尾。 |

N_scale=transformer_matrices+lm_head，排除wte、值表与部分外层参数；D_target=12×N_scale。批量估计为：

$$
B_{\rm pred}=2^{19}(D_{\rm target}/D_{\rm ref,d12})^{0.383},
\quad B=2^{\operatorname{round}(\log_2B_{\rm pred})}.
$$

默认20层/32768词表的静态形状计算得到自动总batch=1048576；Bdevice32、T2048时，8卡A=2、1卡A=16。换词表/预算必须重算。数据比停止时Nstep=floor(D_target/B)，比例12不是12个epoch。

入口学习率乘sqrt(B/524288)，再由组构造处理宽度等倍率，Smear组固定率不经过传入lr的缩放；每step再乘<a id="ref-lr-001"></a><abbr title="Learning Rate / Warmup / Warmdown｜学习率与升降温&#10;定义：学习率控制更新幅度，warmup初期增大，warmdown后期减小，不是模型前向可学习权重。&#10;作用：兼顾训练初期稳定、主体推进和结束收敛，不同参数组可有不同基准。&#10;本项目：base40步升温、后65%降到倍率0.05；SFT初始0.8、后半降0；RL初始0.05、按step线性降。&#10;注意：入口lr还受batch/宽度与组倍率影响；短跑warmup条件优先，RL最后更新不精确0，SFT progress未clamp可越界。&#10;位置：scripts/base_train.py:360 get_lr_multiplier；scripts/chat_sft.py:307 调度；scripts/chat_rl.py:210 调度">学习率调度</abbr>（[释义](glossary/05-loss-and-parameter-updates.md#term-lr)）倍率。Muon decay先按sqrt(B/Bref)×Dref/Dtarget缩放，再余弦下降，AdamW组固定decay。Muon momentum前400步优先从0.85向0.97变，后续降温段向0.90变；短跑不会完全经历理想调度曲线。

## 7.4 一次更新的流程图

从上到下；实线为常规更新，虚线为fp16选择路径，循环箭头表示继续微批。直接路线和缩放路线择一；fp16必须先unscale/溢出判断。没有把不存在的裁剪插成必需步骤。

~~~mermaid
%%{init: {"theme":"base","themeVariables":{"lineColor":"#3b82f6","defaultLinkColor":"#3b82f6","primaryColor":"#123d57","primaryTextColor":"#ffffff","primaryBorderColor":"#7dd3fc","fontSize":"16px"},"themeCSS":".marker { fill: #3b82f6 !important; stroke: #3b82f6 !important; }","flowchart":{"curve":"linear","nodeSpacing":32,"rankSpacing":48,"wrappingWidth":240}}}%%
flowchart TD
    A["起点：各rank编码 + 装箱 + shift + mask"] --> B["当前权重前向 → logits → 有效目标损失"]
    B --> C["loss / 累积次数A"]
    C --> D["反向：累加参数梯度"]
    C -.-> S["fp16可选：scale后反向"]
    S -.-> D
    D --> E{"A个微批已完成？"}
    E --> N["尚未完成：继续微批"]
    N --> A
    E --> Y["已完成：准备参数更新"]
    Y --> F["lr / momentum / decay调度"]
    F -.-> U["fp16：unscale + 各卡统一溢出决定"]
    F --> G["优化器：归约 → 分片更新 → 收集权重"]
    U -.-> G
    G --> H["清梯度 + 日志 + step前进"]
    H --> I["周期/末尾：验证出口 + checkpoint"]
    classDef data fill:#123d57,stroke:#7dd3fc,color:#ffffff,stroke-width:2px;
    classDef compute fill:#49306b,stroke:#d8b4fe,color:#ffffff,stroke-width:2px;
    classDef eval fill:#674518,stroke:#fcd34d,color:#ffffff,stroke-width:2px;
    class A,H data;
    class B,C,D,E,N,Y,F,G,S,U compute;
    class I eval;
    linkStyle default stroke:#3b82f6,stroke-width:3px;
~~~

原循环把评估/保存放在下一次更新之前，图末概括两次更新之间的周期动作；如果fp16溢出，该步可跳过权重更新但循环计数仍推进。
