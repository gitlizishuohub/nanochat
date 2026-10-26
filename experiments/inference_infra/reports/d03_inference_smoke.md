# W1 D03：推理测试交流验收记录

- 交流日期：2026-10-23（Asia/Shanghai）。
- 教程：[D03 推理测试](../../../.agents/docs/career_roadmap/w1/d03_inference_smoke.md)。
- 范围：第 1、2 题由助手补充回答，第 3 题记录用户回答与核对结果；本次仅核对已有日志和源码，未重新运行测试。

## 1. 完整测试命令与运行结果

用户回答“略”，本节由助手依据已有日志补充。

已有完整运行的实际子进程命令如下，工作目录为项目根目录：

```bash
/home/lizishuo/miniconda3/envs/nanochat/bin/python -m pytest -q tests/test_engine.py tests/test_attention_fallback.py -p no:cacheprovider
```

环境限制为 `NANOCHAT_DTYPE=float32`、`CUDA_VISIBLE_DEVICES=''`，即 CPU、FP32（32 位浮点精度）。退出码为 **0**；结果为 **13 passed、10 skipped、0 failed、0 errors**。测试输出用时为 5.74 秒，脚本记录的子进程用时约为 7.78 秒，两者计时范围不同。

原始日志目录为 `experiments/inference_infra/results/w1/d03_pytest/20261022T112433868825Z/`。目录时间对应 2026-10-22 11:24:33 UTC，即北京时间 2026-10-22 19:24:33；不将其记为 2026-10-23 新运行的结果。

- [d03_run.json](../results/w1/d03_pytest/20261022T112433868825Z/d03_run.json)：精确参数、工作目录、退出码、用时及环境限制。
- [d03_stdout.log](../results/w1/d03_pytest/20261022T112433868825Z/d03_stdout.log)：原始摘要 `13 passed, 10 skipped in 5.74s`，其余输出仅为测试进度。
- [d03_stderr.log](../results/w1/d03_pytest/20261022T112433868825Z/d03_stderr.log)：本次读取为空。

结论：该次所选两文件运行成功；10 项跳过不计为通过，不能据此声称 GPU 分支或真实模型权重已验证。日志未记录当时的 commit，因此无法从这三份文件确认运行时的源码版本。

## 2. 缓存容器测试与真实模型的区别

用户回答“略”，本节由助手依据当前源码补充。

以 `test_kv_cache_basic` 为例，它创建 CPU 上的 <abbr title="KV Cache / Key-Value Cache · 键值缓存。保存注意力计算中的键和值，生成后续 token 时可复用历史计算。这里测试的是保存这些张量的容器。">KV Cache</abbr>，检查以下断言：

- 初始位置为 0。
- K、V 张量形状均为 `(6, 2, 64, 3, 5)`，依次对应层数、批大小、容量、键值头数、每头维度。
- 调用 `advance(10)` 后位置为 10，再调用 `advance(5)` 后为 15。
- 调用 `reset()` 后位置回到 0；此断言不证明缓存张量内容被清零。
- `get_layer_cache(0)` 返回的 K、V 形状均为 `(2, 64, 3, 5)`；此处只检查形状，未通过修改返回值验证其与原缓存共享存储。

该测试验证容器的形状、位置推进、重置与分层访问接口，并未运行真实 GPT 前向计算。

`MockModel` 是 <abbr title="mock · 测试替身。通过可控的简化行为隔离待测组件，适合检查接口与控制流，但不能替代真实模型的数值验证。">测试替身</abbr>：其 `forward` 返回全零分数，使各 token 在采样前具有相同分数；传入缓存时只推进位置，没有真实计算和写入各层 K、V。`ByteTokenizer` 用 UTF-8 字节与少量特殊 token 模拟分词，也不代表真实模型使用的分词器。因此这些替身参与的 Engine 测试通过，只能支持相应控制流或接口断言，不能证明真实 GPT 的缓存数值等价。

要验证真实 GPT 的缓存数值等价，还需在相同模型权重和输入下，对比完整前向与使用缓存逐步解码得到的分数，并按设备、精度设定合理误差容限；本次未执行该验证。源码依据：[tests/test_engine.py](../../../tests/test_engine.py) 中的 `test_kv_cache_basic`、`MockModel.forward` 和 `ByteTokenizer`。

## 3. 多样本首 token 多样性、固定 seed 复现与温度为零

**验收结论**：基本通过；用户已抓住三项断言。固定 seed 一项需明确比较范围，核对后的准确表述如下。

1. **多样本首 token 多样性**：在 `test_multi_sample_first_token_diversity` 中，固定输入提示（prompt），以 `temperature=1.0`、`seed=42` 生成 16 个样本，取每个样本的首个 <abbr title="token · 模型处理和生成的离散单位。此处比较生成的 token ID，而不是解码后字符串的首个字符。">token</abbr>，断言 `len(set(first_tokens)) > 1`。它用于防止一次采样得到的 token 被广播给所有样本，不要求 16 个 token 两两不同；该断言也不能单独证明采样具有完整的统计独立性。
2. **固定 seed 复现**：在 `test_seed_reproducibility` 中，对 <abbr title="seed · 随机种子。用于控制伪随机采样序列；复现比较须保持输入、模型、生成参数及运行条件一致。">seed</abbr> 为 1、42、123、999 的情况分别重复生成三次，每组断言 `r1 == r2 == r3`。验证的是同一个 seed 下重复调用的输出一致，不要求不同 seed 的输出互相相等。用户原表述“各个种子值断言均相等”应按此理解。
3. **温度为零**：在 `test_temperature_zero_determinism` 中，固定 prompt 和其他条件，设置 <abbr title="temperature · 温度，控制采样分布的参数。这里温度为零时选择得分最高的 token，属于贪心生成，不使用随机采样来选择 token。">temperature</abbr> 为 0，分别使用 seed 1、42、123，断言 `r1 == r2 == r3`，验证该测试条件下输出不受采样种子影响。

源码依据：[tests/test_engine.py](../../../tests/test_engine.py) 中上述三个测试函数。以上记录是对回答和当前源码断言的核对，不代表本次执行过这些测试，也不代表真实模型或其他设备上的行为已验收。

## 4. 跳过条件与未覆盖范围

未回答、未验收。

## 5. 失败分类与重跑证据

未回答、未验收。本次未执行定向重跑或完整两文件测试。

下一步：继续回答第 4、5 题；完整实验验收仍需结合实际命令、退出码和原始日志证据。
