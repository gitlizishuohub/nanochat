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

[栏目目录](README.md) · [上一篇](07-backpropagation-and-parameter-updates.md) · [下一篇](09-distributed-optimizer-and-communication.md) · [术语查阅](glossary/README.md)

> 原技术正文沿用 2026-10-12 的源码核对记录；2026-10-21 调整结构。新增 SFT 串联说明做了局部静态核对，未执行训练。

# 8. 混合精度与训练加速

先理解第 7 章的普通更新循环，再查阅本章的精度与加速选项。它们改变计算方式，不构成新的训练阶段。

<a id="reading-8-1"></a>
## 8.1 显式混合精度与梯度缩放

<a id="ref-precision-002"></a><abbr title="Mixed Precision / bf16 / fp16 / fp32｜混合精度与数值类型&#10;定义：对不同张量/运算选不同位宽，在显存和吞吐与数值精度之间取平衡。&#10;作用：降低激活/矩阵乘法开销，同时保留重要统计和损失计算精度。&#10;本项目：COMPUTE_DTYPE默认SM80+ CUDA bf16、其它fp32；Linear临时转换fp32权重，嵌入通常直接低精度，logits/loss fp32。&#10;注意：不用autocast，不是全权重fp32；fp16时嵌入保留fp32方便unscale，dtype导入时选择，device-type覆盖不重选。&#10;位置：nanochat/common.py:17 dtype选择；nanochat/gpt.py:45 Linear；nanochat/gpt.py:265 嵌入类型">混合精度</abbr>（[释义](glossary/05-loss-and-parameter-updates.md#term-precision)）通过Linear把权重临时转为激活dtype实现，不用autocast；嵌入通常直接低精度存储，logits/loss为fp32。<a id="ref-scaler-001"></a><abbr title="GradScaler / Loss Scaling｜梯度缩放器与损失缩放&#10;定义：先放大低精度loss再反向，更新前还原梯度，检查非有限数值并动态改变缩放倍数。&#10;作用：缓解fp16小梯度下溢，避免溢出结果被写进权重。&#10;本项目：base/SFT仅float16启用GradScaler，unscale后对found_inf做rank MAX，统一决定跳步，随后step/update。&#10;注意：bf16/fp32不启用，RL未支持；checkpoint未存scaler状态，fp16断点恢复不保证原缩放轨迹精确接续。&#10;位置：scripts/base_train.py:323 GradScaler；scripts/base_train.py:529 scaler步骤；scripts/chat_sft.py:151 GradScaler">fp16 GradScaler</abbr>（[释义](glossary/05-loss-and-parameter-updates.md#term-scaler)）路径为scale→backward→unscale→多rank MAX归约found_inf→step/update；bf16/fp32不用，RL没有这一路径。没有<a id="ref-clip-001"></a><abbr title="Gradient Clipping｜梯度裁剪&#10;定义：更新前按范数或数值范围缩小极端梯度，范数裁剪通常用统一比例限制总范数。&#10;作用：在实现时缓解梯度爆炸，但应确认阈值、作用范围与混合精度先unscale的顺序。&#10;本项目：当前三个训练入口没有clip_grad_norm_/clip_grad_value_调用或裁剪阈值，不可写入通用默认1.0。&#10;注意：Muon矩阵正交化、RMSNorm、logit softcap、PPO ratio clip都不是梯度范数裁剪；本项目此项是缺失边界。&#10;位置：scripts/base_train.py:511 单步；scripts/chat_sft.py:425 单步；scripts/chat_rl.py:300 更新">梯度裁剪</abbr>（[释义](glossary/05-loss-and-parameter-updates.md#term-clip)）调用/阈值，不填写通用clip=1.0。

## 8.2 深入：FP8 实现与计数限制

<a id="ref-fp8-001"></a><abbr title="FP8 / Tensorwise Scaling｜八位浮点矩阵计算与整张量缩放&#10;定义：将矩阵操作数按最大幅值动态缩放量化成八位浮点，乘法结合逆缩放恢复数值尺度。&#10;作用：在兼容硬件加速主要Linear前向和反向计算，是预训练执行选项。&#10;本项目：Float8Linear用e4m3输入/权重、e5m2梯度，torch._scaled_mm执行三个乘法；只转维度整除16且最小维≥128层。&#10;注意：权重保持原存储精度，只有tensorwise；CLI rowwise会被本地配置拒绝，CUDA存在不保证硬件支持，评估临时换回普通Linear。FP8替换类不被num_matmul_params的自定义Linear判别计入，FLOPs/MFU估计可能偏差。&#10;位置：nanochat/fp8.py:82 _to_fp8；nanochat/fp8.py:195 Float8Linear；nanochat/fp8.py:230 配置；scripts/base_train.py:178 filter">FP8</abbr>（[释义](glossary/05-loss-and-parameter-updates.md#term-fp8)）仅base_train可选，替换维度能被16整除且最小维度≥128的Linear；共享原权重，对三次矩阵乘法动态量化，输入/权重e4m3、梯度e5m2，调用torch._scaled_mm。仅检测CUDA不完整验证硬件兼容；预期H100+。本地仅支持tensorwise，CLI接受rowwise但配置会拒绝。验证临时换回普通Linear，训练FP8不等于交付权重转FP8。另有当前实现边界：GPT.num_matmul_params只统计自定义Linear，FP8替换类继承的是nn.Linear；替换后部分矩阵不再被该函数计数。因此FP8路径的estimate_flops、target-flops步数和MFU估计可能偏差，不能把这些日志当作精确算力/预算证据；参数比例预算使用的结构计数是另一条路径。对应源码为[GPT.num_matmul_params](/home/lizishuo/_program_/Project/nanochat/nanochat/gpt.py:341)和[Float8Linear](/home/lizishuo/_program_/Project/nanochat/nanochat/fp8.py:195)。

<a id="reading-8-3"></a>
## 8.3 编译与动态输入

预训练和 SFT 用 `torch.compile(dynamic=False)` 处理固定形状训练路径，另保留未编译的 `orig_model` 用于权重保存和变长生成。编译是运行方式，不改变监督目标；首次编译开销和正常训练时间应分别理解。
