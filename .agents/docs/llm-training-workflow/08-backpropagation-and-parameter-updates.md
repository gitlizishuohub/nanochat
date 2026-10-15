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

[栏目目录](README.md) · [上一篇](07-forward-pass-and-prediction-loss.md) · [下一篇](09-logging-checkpoints-and-resume.md)

> 拆分自原文；源码核对日期为 2026-10-12。本次仅整理文档结构，未重新核验源码或运行训练。

<a id="sec-8"></a>
# 8. 反向、混合精度、累积、优化器和学习率

## 8.1 一次更新的真实次序

预训练/SFT用<a id="ref-compile-001"></a><abbr title="torch.compile / Static Shape｜图编译与固定形状&#10;定义：把PyTorch运算路径捕获为可优化计算图，通过融合等方式降低重复调度成本。&#10;作用：加速固定B/T的重复前向/反向和优化器步骤，不改变训练目标定义。&#10;本项目：base/SFT模型dynamic=False，优化器fused函数也编译；保留orig_model供保存/变长生成，FP8替换在编译前。&#10;注意：编译可能有初期开销，不是算法或质量验证；没有实际跑过就不能承诺加速，RL没有torch.compile(model)调用。&#10;位置：scripts/base_train.py:246 orig_model/compile；scripts/chat_sft.py:117 compile；nanochat/optim.py:23 编译装饰器">torch.compile(dynamic=False)</abbr>（[释义](12-5-loss-and-parameter-updates.md#term-compile)）优化固定形状路径，保留orig_model保存原始权重键名与变长生成。每微批forward求loss，除累积次数A，<a id="ref-backprop-001"></a><abbr title="Autograd / Backward｜自动微分与反向传播&#10;定义：沿前向计算图使用链式法则计算损失对可学习张量的导数，并写入参数梯度。&#10;作用：告诉优化器沿哪些方向调整数值可改善当前目标，连接预测误差与参数更新。&#10;本项目：训练调用loss.backward或scaler.scale(loss).backward，微批累加，更新后zero_grad(set_to_none=True)。&#10;注意：backward只求梯度，optimizer.step才改权重；model.eval不关闭求导，验证no_grad、Engine inference_mode才限制梯度记录。&#10;位置：scripts/base_train.py:511 单步；scripts/chat_sft.py:430 backward；nanochat/engine.py:176 generate">backward</abbr>（[释义](12-5-loss-and-parameter-updates.md#term-backprop)）累加梯度；完成A批后应用调度、优化器同步并更新，最后zero_grad(set_to_none=True)。日志取最后微批loss再平滑，不是精确整步平均。

<a id="ref-accum-001"></a><abbr title="Gradient Accumulation｜梯度累积&#10;定义：多个微批分别反向、梯度相加，全部完成后再进行一次优化更新。&#10;作用：在每卡显存限制下扩大统计批量，降低device batch时可维持全局token容量。&#10;本项目：A=total_batch_size/(B×T×W)须整除，每微批mean loss除A，最后optimizer.step后清梯度。&#10;注意：SFT有效位置数不同，等权微批均值不等于全局有效token均值；rank平均也不重加权。日志仅末微批值不能当精确整步loss。&#10;位置：scripts/base_train.py:410 grad_accum_steps；scripts/base_train.py:513 loss缩放；scripts/chat_sft.py:425 累积">梯度累积</abbr>（[释义](12-5-loss-and-parameter-updates.md#term-accum)）对等长全监督batch得到等权均值；SFT每微批有效token数不同，mean(loss)/A是等权微批均值，不严格等于所有有效token统一均值，rank间AVG也不重新按有效token加权。

<a id="ref-precision-002"></a><abbr title="Mixed Precision / bf16 / fp16 / fp32｜混合精度与数值类型&#10;定义：对不同张量/运算选不同位宽，在显存和吞吐与数值精度之间取平衡。&#10;作用：降低激活/矩阵乘法开销，同时保留重要统计和损失计算精度。&#10;本项目：COMPUTE_DTYPE默认SM80+ CUDA bf16、其它fp32；Linear临时转换fp32权重，嵌入通常直接低精度，logits/loss fp32。&#10;注意：不用autocast，不是全权重fp32；fp16时嵌入保留fp32方便unscale，dtype导入时选择，device-type覆盖不重选。&#10;位置：nanochat/common.py:17 dtype选择；nanochat/gpt.py:45 Linear；nanochat/gpt.py:265 嵌入类型">混合精度</abbr>（[释义](12-5-loss-and-parameter-updates.md#term-precision)）通过Linear把权重临时转为激活dtype实现，不用autocast；嵌入通常直接低精度存储，logits/loss为fp32。<a id="ref-scaler-001"></a><abbr title="GradScaler / Loss Scaling｜梯度缩放器与损失缩放&#10;定义：先放大低精度loss再反向，更新前还原梯度，检查非有限数值并动态改变缩放倍数。&#10;作用：缓解fp16小梯度下溢，避免溢出结果被写进权重。&#10;本项目：base/SFT仅float16启用GradScaler，unscale后对found_inf做rank MAX，统一决定跳步，随后step/update。&#10;注意：bf16/fp32不启用，RL未支持；checkpoint未存scaler状态，fp16断点恢复不保证原缩放轨迹精确接续。&#10;位置：scripts/base_train.py:323 GradScaler；scripts/base_train.py:529 scaler步骤；scripts/chat_sft.py:151 GradScaler">fp16 GradScaler</abbr>（[释义](12-5-loss-and-parameter-updates.md#term-scaler)）路径为scale→backward→unscale→多rank MAX归约found_inf→step/update；bf16/fp32不用，RL没有这一路径。没有<a id="ref-clip-001"></a><abbr title="Gradient Clipping｜梯度裁剪&#10;定义：更新前按范数或数值范围缩小极端梯度，范数裁剪通常用统一比例限制总范数。&#10;作用：在实现时缓解梯度爆炸，但应确认阈值、作用范围与混合精度先unscale的顺序。&#10;本项目：当前三个训练入口没有clip_grad_norm_/clip_grad_value_调用或裁剪阈值，不可写入通用默认1.0。&#10;注意：Muon矩阵正交化、RMSNorm、logit softcap、PPO ratio clip都不是梯度范数裁剪；本项目此项是缺失边界。&#10;位置：scripts/base_train.py:511 单步；scripts/chat_sft.py:425 单步；scripts/chat_rl.py:300 更新">梯度裁剪</abbr>（[释义](12-5-loss-and-parameter-updates.md#term-clip)）调用/阈值，不填写通用clip=1.0。

<a id="ref-fp8-001"></a><abbr title="FP8 / Tensorwise Scaling｜八位浮点矩阵计算与整张量缩放&#10;定义：将矩阵操作数按最大幅值动态缩放量化成八位浮点，乘法结合逆缩放恢复数值尺度。&#10;作用：在兼容硬件加速主要Linear前向和反向计算，是预训练执行选项。&#10;本项目：Float8Linear用e4m3输入/权重、e5m2梯度，torch._scaled_mm执行三个乘法；只转维度整除16且最小维≥128层。&#10;注意：权重保持原存储精度，只有tensorwise；CLI rowwise会被本地配置拒绝，CUDA存在不保证硬件支持，评估临时换回普通Linear。FP8替换类不被num_matmul_params的自定义Linear判别计入，FLOPs/MFU估计可能偏差。&#10;位置：nanochat/fp8.py:82 _to_fp8；nanochat/fp8.py:195 Float8Linear；nanochat/fp8.py:230 配置；scripts/base_train.py:178 filter">FP8</abbr>（[释义](12-5-loss-and-parameter-updates.md#term-fp8)）仅base_train可选，替换维度能被16整除且最小维度≥128的Linear；共享原权重，对三次矩阵乘法动态量化，输入/权重e4m3、梯度e5m2，调用torch._scaled_mm。仅检测CUDA不完整验证硬件兼容；预期H100+。本地仅支持tensorwise，CLI接受rowwise但配置会拒绝。验证临时换回普通Linear，训练FP8不等于交付权重转FP8。另有当前实现边界：GPT.num_matmul_params只统计自定义Linear，FP8替换类继承的是nn.Linear；替换后部分矩阵不再被该函数计数。因此FP8路径的estimate_flops、target-flops步数和MFU估计可能偏差，不能把这些日志当作精确算力/预算证据；参数比例预算使用的结构计数是另一条路径。对应源码为[GPT.num_matmul_params](/home/lizishuo/_program_/Project/nanochat/nanochat/gpt.py:341)和[Float8Linear](/home/lizishuo/_program_/Project/nanochat/nanochat/fp8.py:195)。

## 8.2 优化器分组与通信

<a id="ref-optim-001"></a><abbr title="Optimizer / MuonAdamW｜优化器与混合更新器&#10;定义：根据梯度和历史动量/统计调整模型数值，并保存跨步更新状态的对象。&#10;作用：将损失导数转为实际参数变化，支持不同类别权重使用不同更新算法。&#10;本项目：GPT.setup_optimizer给lm_head/wte/值表/标量建AdamW组，块内矩阵建Muon组；同一step还同步分布式梯度与参数。&#10;注意：不负责计算loss或shift标签；并非所有二维权重Muon。恢复原更新轨迹需每rank状态，只有model文件不足。&#10;位置：nanochat/gpt.py:419 setup_optimizer；nanochat/optim.py:184 MuonAdamW；nanochat/optim.py:429 step">MuonAdamW</abbr>（[释义](12-5-loss-and-parameter-updates.md#term-optim)）组合<a id="ref-adamw-001"></a><abbr title="AdamW / Momentum / Bias Correction｜AdamW、动量与偏差修正&#10;定义：维护梯度一阶/二阶指数平均并修正初期偏差，再按自适应幅度更新，权重衰减与梯度分离。&#10;作用：平滑噪声与调节不同权重的更新尺度，用于嵌入、输出层和标量。&#10;本项目：fused步骤转fp32计算回写原dtype；lm_head betas(0.8,0.96)，wte/value(0.8,0.995)，eps1e−10。&#10;注意：zeros_like状态可能低精度存储，计算fp32不等于始终fp32保存；各组beta/decay不同，不能套0.9/0.999常见默认。&#10;位置：nanochat/optim.py:24 adamw_step_fused；nanochat/optim.py:320 _compute_adamw；nanochat/gpt.py:439 分组">AdamW</abbr>（[释义](12-5-loss-and-parameter-updates.md#term-adamw)）与<a id="ref-muon-001"></a><abbr title="Muon / Polar Express / NorMuon｜矩阵动量正交化更新&#10;定义：对矩阵动量方向做近似正交化及尺度调整，减少单一方向主导整个矩阵更新的情况。&#10;作用：为Transformer内部矩阵提供适合其结构的更新，同时对同形矩阵批量计算。&#10;本项目：包括Nesterov、行均衡、五次Polar Express、Frobenius再归一化、分解二阶矩、cautious decay。&#10;注意：不是精确SVD最近正交矩阵或梯度裁剪；ns_steps5，实际lr乘sqrt(max(1,行/列))，不能仅看组内显示lr理解幅度。&#10;位置：nanochat/optim.py:112 muon_step_fused；nanochat/optim.py:362 _compute_muon；nanochat/gpt.py:449 分组">Muon</abbr>（[释义](12-5-loss-and-parameter-updates.md#term-muon)）。wte、value_embeds、lm_head、外层标量/Smear参数用AdamW；transformer.h内部矩阵（含值门控）按同形组用Muon，不只看张量是否二维。

| AdamW组 | 组构造学习率（再经入口倍率） | betas / decay |
| --- | --- | --- |
| lm_head | unembedding_lr×(C/768)^−1/2 | (0.8,0.96) / 0.01 |
| wte | embedding_lr×(C/768)^−1/2 | (0.8,0.995) / 0.001 |
| value_embeds | 上项×0.5 | (0.8,0.995) / 0.01 |
| resid_lambdas | scalar_lr×0.01 | (0.8,0.95) / 0.05 |
| x0_lambdas | scalar_lr | (0.96,0.95) / 0 |
| smear_gate、smear_lambda、backout_lambda | 固定0.2 | (0.8,0.95) / 0 |

AdamW eps=1e-10，计算中转fp32再回写原类型，状态zeros_like可能低精度保存。Muon初组momentum=0.95、ns_steps=5、beta2=0.9，含Nesterov、行均衡、五次Polar Express、范数再归一化、分解二阶矩、<a id="ref-wd-001"></a><abbr title="Weight Decay / Cautious Decay｜权重衰减与谨慎衰减&#10;定义：优化更新时抑制权重增长，cautious变体只对指定更新与参数方向关系施加衰减。&#10;作用：控制参数尺度，需区别算法与组/阶段实际设置，不是loss里任意统一L2项。&#10;本项目：AdamW头0.01、输入表0.001、值表0.01、resid0.05；Muon基准0.28经预算缩放再余弦降，按更新×参数非负mask。&#10;注意：SFT新建只令Muon decay0，热启动加载元数据可覆盖它，AdamW仍非全0；RL帮助中的作用组描述与实参也不同。&#10;位置：nanochat/gpt.py:439 分组；nanochat/optim.py:178 cautious更新；scripts/base_train.py:385 调度；scripts/chat_sft.py:143 加载">cautious decay</abbr>（[释义](12-5-loss-and-parameter-updates.md#term-wd)）；实际lr再乘sqrt(max(1,行数/列数))。这些不是梯度范数裁剪。

模型不包DDP，<a id="ref-zero-001"></a><abbr title="ZeRO-2 Style / Reduce-scatter / All-gather｜优化器状态分片与归约收集&#10;定义：归约梯度并分配给负责进程，各rank更新其参数片段，再收集更新参数以恢复一致副本。&#10;作用：减少每卡优化器状态与重复更新计算，用异步操作重叠通信和计算。&#10;本项目：大AdamW沿首维分片，小参数all_reduce；Muon同形矩阵堆叠分块，reduce_scatter平均后更新、all_gather。&#10;注意：仓库自定义ZeRO-2风格，无DeepSpeed；模型权重/反向梯度仍有副本，不是完整模型分片或FSDP。&#10;位置：nanochat/optim.py:274 _reduce_adamw；nanochat/optim.py:295 _reduce_muon；nanochat/optim.py:419 _finish_gathers">ZeRO-2风格通信</abbr>（[释义](12-3-batching-and-distributed-training.md#term-zero)）在optimizer.step：异步归约→负责rank更新→all_gather。大AdamW参数沿首维切片且须整除world_size；小于1024元素用all_reduce；Muon同形矩阵堆叠按矩阵块分配并补通信空位。每卡保存其状态，模型权重仍复制；不是DeepSpeed/FSDP。

## 8.3 预训练默认预算与调度

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

入口学习率乘sqrt(B/524288)，再由组构造处理宽度等倍率，Smear组固定率不经过传入lr的缩放；每step再乘<a id="ref-lr-001"></a><abbr title="Learning Rate / Warmup / Warmdown｜学习率与升降温&#10;定义：学习率控制更新幅度，warmup初期增大，warmdown后期减小，不是模型前向可学习权重。&#10;作用：兼顾训练初期稳定、主体推进和结束收敛，不同参数组可有不同基准。&#10;本项目：base40步升温、后65%降到倍率0.05；SFT初始0.8、后半降0；RL初始0.05、按step线性降。&#10;注意：入口lr还受batch/宽度与组倍率影响；短跑warmup条件优先，RL最后更新不精确0，SFT progress未clamp可越界。&#10;位置：scripts/base_train.py:360 get_lr_multiplier；scripts/chat_sft.py:307 调度；scripts/chat_rl.py:210 调度">学习率调度</abbr>（[释义](12-5-loss-and-parameter-updates.md#term-lr)）倍率。Muon decay先按sqrt(B/Bref)×Dref/Dtarget缩放，再余弦下降，AdamW组固定decay。Muon momentum前400步优先从0.85向0.97变，后续降温段向0.90变；短跑不会完全经历理想调度曲线。

## 8.4 SFT继承与热启动边界

SFT默认继承base meta长度、每卡行数、总batch，以及base user_config中三个lr；缺失回退2048、32、524288、0.3、0.004、0.02。load_optimizer=1加载完整state_dict后**只显式重置lr**，再乘init_lr_frac=0.8；betas/decay等元数据也可能继承，日志“momentum buffers only”不符合完整行为。

新建Muon传decay=0，但默认加载状态可能恢复base最后实际更新的微小非零decay；AdamW本来还有固定decay，不能说SFT全部无衰减。load_optimizer=0才保留新建组元数据。SFT没有应用base_train全部预算/batch倍率重算。

默认warmup_ratio=0，后半消费进度降lr到0；Muon前300更新从0.85到0.95。progress来自预取/consumed，可能超1，调度不clamp，边界设置下不保证lr非负。SFT的user_config在继承值前复制，meta内可能保留None；末尾BPB即使eval_every≤0仍执行。

## 8.5 一个batch更新图

从上到下；实线为常规更新，虚线为fp16选择路径，循环箭头表示继续微批。直接路线和缩放路线择一；fp16必须先unscale/溢出判断。没有把不存在的裁剪插成必需步骤。

~~~mermaid
%%{init: {'theme':'base','themeVariables':{'lineColor':'#f8fafc','fontSize':'15px'},'flowchart':{'curve':'linear'}}}%%
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
    linkStyle default stroke:#f8fafc,stroke-width:3px;
~~~

原循环把评估/保存放在下一次更新之前，图末概括两次更新之间的周期动作；如果fp16溢出，该步可跳过权重更新但循环计数仍推进。
