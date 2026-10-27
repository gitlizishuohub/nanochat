---
source_files:
  - tests/test_engine.py
  - tests/test_attention_fallback.py
  - nanochat/engine.py
  - nanochat/flash_attention.py
  - pyproject.toml
source_symbols:
  - MockModel.forward
  - test_multi_sample_first_token_diversity
  - test_seed_reproducibility
  - TestSDPAOnly
  - TestOverrideMechanism
doc_type: tutorial
knowledge_scope: W1 D3 CPU推理测试及失败定位
---

# W1 D3：用测试判断推理代码究竟验证了什么

D2 确认“这个解释器里有什么”，今天进一步判断：“哪些推理行为真的执行并符合预期？”你将读懂测试为什么这样设计，运行 CPU 测试，再从日志提出有边界的结论。预计 60–90 分钟。

前置：[D2 环境盘点](d02_environment_inventory.md)。**今天不训练、不下载数据或权重，不需要 GPU 或 checkpoint。** 测试主要使用受控替身和随机注意力张量；其中一个测试计算梯度，但没有优化器更新参数，不能称为训练模型。

## 当天的流程阅读

用约 10–20 分钟读下面的小节；只需理解并口述，不新增报告或实验任务。

- **阅读入口**：[4.1 配置不是可学习权重](../../llm-training-workflow/04-model-construction-and-weight-loading.md#reading-4-1)；[4.2 从结构到权重](../../llm-training-workflow/04-model-construction-and-weight-loading.md#reading-4-2)。
- **今天重点**：区分配置、随机初始化、已有权重；理解没有训练好的模型也能检查接口或计算行为。
- **暂时跳过**：4.2.1 初始化分布表只看标题；权重加载细节在 D11 学习。

**读完能解释**：一个测试输出符合预期，与模型已经学会语言，分别需要什么证据？

## 1. 先把“正确”改写成一个可能失败的断言（概念推演，无需运行）

假设 Engine 要从同一个提示生成 16 个样本。一个很具体的错误是：代码只采样一次，把同一个首 token 复制给全部 16 行。只检查“返回了 16 行”发现不了它，因为行数依旧正确。

怎样使错误暴露？先让模型给全部词表项相同分数，让采样结果不被模型偏好支配；再检查 16 个首 token 中是否至少有两种值。例如 `[7,7,7,7]` 满足“有 4 个结果”，却不满足“有多个不同值”；`[7,7,12,7]` 满足后一个条件，即使并非每行互异。

这就是<abbr title="assertion · 断言。把预期行为写成可检查的条件；执行时条件不成立便报告失败。它只能检查写出来的条件，不能证明全部程序正确。">断言</abbr>的价值：**规定什么观察结果会使我们收回结论。** 对本例，断言针对“首 token 被广播”的缺陷，并不检验回答内容是否合理。

### 1.1 从真实测试看输入、执行、判断

下面摘自 [test_multi_sample_first_token_diversity](../../../../tests/test_engine.py)，省略了提示和循环收集部分；这是源码带读，不需单独执行此片段：

```python
model = MockModel(vocab_size=262)
engine = Engine(model, ByteTokenizer())
# generate(..., num_samples=16, max_tokens=1, temperature=1.0, seed=42)
unique_tokens = set(first_tokens)
assert len(unique_tokens) > 1
```

把它拆成三件事：`MockModel` 控制输入分数；`generate` 执行真正的 Engine 生成控制流；最后的断言判断首 token 是否只有一种。这种安排使失败尽量指向采样流程，而非语言模型本身。

替身 [MockModel.forward](../../../../tests/test_engine.py) 的核心是：

```python
B, T = ids.shape
if kv_cache is not None:
    kv_cache.advance(T)
logits = torch.zeros(B, T, self.vocab_size)
return logits
```

<abbr title="mock · 测试替身。用简单、可控行为代替真实组件，让一个测试聚焦另一个组件。本文替身返回全零词表分数，不执行真实 GPT 的投影、注意力或缓存写入。">Mock</abbr> 返回的 logits 全为零；采样中的 softmax 因此给 262 个词表项相同概率。它还推进缓存位置，以满足 Engine 对状态的要求，但没有真正计算 K/V。`ByteTokenizer` 用字节及少量特殊 ID 支持 Engine 的接口，也不加载训练好的分词器。

如果 16 行独立均匀采样，全部恰巧一样的概率为 `(1/262)^15`，极低但不是数学上的零。测试固定 seed，使当前执行可复查；`len(set(...)) > 1` 也不等于严格证明每一行都独立。它是针对广播缺陷的强检测条件。

再回看 [Engine.generate](../../../../nanochat/engine.py) 的两个相邻动作：

```python
logits = logits[:, -1, :].expand(num_samples, -1)
# 后面在循环里：
next_ids = sample_next_token(logits, rng, temperature, top_k)
```

复制的是**首 token 的分数**，随后按行采样。16 行可以拥有相同概率分布，同时抽出不同 token；如果复制的是已抽出的 token，就会产生开头的缺陷。该区别是本测试要保护的行为。

### 1.2 三个“随机性测试”为什么都要有

| 真实用例 | 控制什么、比较什么 | 可以发现的错误 |
|---|---|---|
| `test_multi_sample_first_token_diversity` | 同次生成多行，比较首 token 的种类 | 把一个采样结果广播给多行 |
| `test_seed_reproducibility` | 同一提示、同一 seed，重复生成并比较序列 | 随机数状态没有按 seed 初始化或使用 |
| `test_temperature_zero_determinism` | temperature=0，换 seed 仍比较相同输出 | 贪心分支错误地继续随机采样 |

同次运行的行间变化，与重复运行的可复现并不矛盾：一个确定的随机数序列内可以有不同数。temperature=0 时 [sample_next_token](../../../../nanochat/engine.py) 直接 `argmax`，应与 seed 无关。变式推演：如果代码永远返回同一 token，后两项可能通过，第一项却会失败。因此“一个确定性测试通过”并不足以证明采样正确。

## 2. 换一种真实计算：注意力测试覆盖了什么

[tests/test_attention_fallback.py](../../../../tests/test_attention_fallback.py) 中的 `TestSDPAOnly.test_basic_forward` 使用真正的随机 Q/K/V 张量，并执行注意力计算，关键判断为：

```python
y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
assert y.shape == (B, T, H, D)
assert not torch.isnan(y).any()
```

<abbr title="SDPA / Scaled Dot Product Attention · 缩放点积注意力。这里指 PyTorch 提供的计算接口；nanochat 在无法使用 FA3 时通过它执行注意力回退路径。">SDPA</abbr> 不是上面的模型替身：这里确实进行了张量计算。但断言只检查形状和没有 NaN，没有与独立参考值比较，也没有检查所有 Inf，所以不能把它描述成“数值结果已经全面证明正确”。

`test_backward` 调用 `loss.backward()`，检查 Q/K/V 梯度存在及 Q 梯度无 NaN；它不调用 `optimizer.step()`，不训练 GPT。`test_kvcache` 执行随机注意力 prefill 和单 token 续算，检查输出形状和推进后的位置。`test_engine.py` 中的两个 `kv_cache` 测试则直接检查缓存容器的分配、推进、reset 和已写入区域复制。**容器能复制，与真实 GPT 缓存和全前缀计算数值相符，是两种不同性质**；后一项在 D5 用随机 GPT 比较。

另一组 `TestFA3VsSDPA` 要让 FA3 与 SDPA 在相同输入上比较数值和梯度，需要适用硬件/内核。今天脚本隐藏 CUDA，相关测试跳过是正确的条件分支；GPU 对照是可选扩展，未做不算每日欠账。

## 3. 现在执行 CPU 测试并读结果（实际操作）

下面会导入依赖并运行两份现有测试，写测试日志和运行元数据；不下载、不启动训练。使用当前终端的 `nanochat` 环境，先确认解释器：

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
python -c 'import sys,torch,pytest; print(sys.executable); print(torch.__version__); print(pytest.__version__)'
d03_status=$?
printf 'dependency_check_exit=%s\n' "$d03_status"
} > >(tee -a experiments/inference_infra/reports/d03_checks.log) 2>&1
```

出现导入错误先返回 D2 处理相应依赖，不将其归因于模型算法，也不假定后台正在安装依赖。检查成功后运行[测试捕获脚本](scripts/d03_run_inference_tests.py)：

```bash
{
python .agents/docs/career_roadmap/w1/scripts/d03_run_inference_tests.py
d03_status=$?
printf 'inference_tests_exit=%s\n' "$d03_status"
} > >(tee -a experiments/inference_infra/reports/d03_checks.log) 2>&1
```

脚本用当前 `sys.executable -m pytest`，只选 `tests/test_engine.py` 和 `tests/test_attention_fallback.py`，设置 `CUDA_VISIBLE_DEVICES=""` 与 `NANOCHAT_DTYPE=float32`。它关闭 pytest 本地缓存，把字节码缓存放 `/tmp/nanochat_w1_pycache`。每次创建新时间戳目录，终端的 `Logs:` 给出本次实际位置。

### 3.1 先读摘要，再读三份证据

| 观察 | 含义 | 应怎样表述 |
|---|---|---|
| `passed` / `.` | 该用例执行后断言满足 | 指明测试及输入范围，不扩大成全模型正确 |
| `skipped` / `s` | 条件不满足，测试主体没执行 | 记录跳过条件；不算通过或失败 |
| `failed` / `F` | 测试执行后断言不满足 | 找到用例名与首个失败断言，保留输入及错误 |
| `ERROR collecting ...` | 收集/导入阶段出错 | 不能报告相应测试已执行 |

这是状态解释，不是本次预填成绩。退出码为 0 表示所选测试成功，允许其中有 skip；脚本 JSON 的 `status=PASS` 同样只表示 pytest 返回 0。没有运行时不能套用历史通过数。

`experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/` 内：

- `d03_stdout.log`：读最终摘要、失败用例名，再向上读相应栈和断言值。
- `d03_stderr.log`：保留额外诊断；为空也正常，导入错误也可能出现在 stdout 的 pytest 收集报告中，因此两份都要查看。
- `d03_run.json`：核对 `argv` 中确实选了两文件，`cwd`、真实 `exit_code` 和环境限制是否符合这次运行；外部中断时可能来不及生成。

由这些证据，可以形成这样的结论：“本次 CPU/FP32 下，日志列出的 Engine 替身测试及 SDPA 基础计算测试通过；FA3 条件用例没有执行；未测试训练好 GPT 的语言质量。”具体状态必须替换为自己的日志结果。

### 3.2 失败栈怎样把问题缩小

以下是**假设失败**，无需制造故障：若日志定位到 `test_multi_sample_first_token_diversity` 的 `len(unique_tokens) > 1`，且显示所有行同一首 token，先问“分数是否先扩到 B 行再采样？”而不是先换 tokenizer 或重新训练。测试用的是均匀分数替身，训练权重不参与这条路径。

若失败在 `test_kv_cache_basic` 的 `get_pos() == 15`，就沿 `advance(10)`、`advance(5)` 检查状态累加；如果失败在 `test_basic_forward` 的输出形状，则从 Q/K/V 输入布局和注意力返回布局定位。测试函数名只提供入口，**真正缩小问题的是失败断言、实际值和调用栈**。

| 实际问题 | 下一步 |
|---|---|
| 缺 torch/pytest/tiktoken 等模块 | 核对 `sys.executable` 与依赖；Engine 的传递导入也需要项目依赖 |
| 动态库加载失败 | 保存完整诊断，回查 D2 的解释器与安装版本 |
| 行为断言失败 | 定位用例、输入、断言值和选中后端；只重跑相关项辅助诊断 |
| FA3 条件跳过 | 记录条件未执行；CPU 学习主线可完成 |
| 进程被终止、缺完整记录 | 保存中断事实，不能据残留旧文件报告本次完成 |

例如只重跑缓存容器与 SDPA（有相关问题时再执行）：

```bash
{
python .agents/docs/career_roadmap/w1/scripts/d03_run_inference_tests.py --select 'kv_cache or SDPAOnly'
d03_status=$?
printf 'targeted_tests_exit=%s\n' "$d03_status"
} > >(tee -a experiments/inference_infra/reports/d03_checks.log) 2>&1
```

`--select` 传给 pytest 的 `-k` 名称筛选表达式。定向通过只覆盖这部分；若修改环境或代码解决了问题，再补完整两文件检查，以免别的测试退化。现有 `test_override_auto` 把 `USE_FA3 == HAS_FA3` 作为断言，但生产路由还检查 compute dtype；以后换到有 FA3 且强制 FP32 的环境时应重新核实该测试假设。今天 CPU 隔离下两者均应为 false。

## 4. 用四个问题检验理解

直接在聊天按题号回答即可；实测信息只需附一次命令、退出码和本次目录。没有执行的 CPU 操作说明原因；引用历史证据注明日期与版本，不为答题额外训练或做 GPU 实验。

1. **广播缺陷**：假设 16 行都返回首 token 7，哪些已讲过的断言能发现问题，哪些仍可能通过？为什么测试要用均匀分数替身？

   **核对要点**：能区分样本数、行间变化与重复运行确定性，并将替身和待测缺陷联系起来。

2. **证据边界**：从本次摘要中分别选一个执行过的测试与一个跳过条件，说明它们支持或不支持什么结论。若本次没跑，改用源码推演并明确无运行证据。

   **核对要点**：不把 JSON 的 PASS 等同所有分支通过，不把形状/无 NaN 等同完整数值正确。

3. **从失败走向代码**：假设缓存位置断言预期 15、实际 5，你会先查看哪两个动作和哪一个方法？与缺少 torch 的错误为何应走不同排查路径？

   **核对要点**：用具体状态和调用定位，区分收集/导入错误与执行后的行为失败。

4. **下一层验证**：为什么这些测试通过后，D4 仍要运行随机 GPT、D5 仍要比较缓存与全前缀 logits？

   **核对要点**：指出替身没执行的真实模型计算，以及基础注意力测试未证明的端到端等价；无需讨论语言质量评估方案。

## 5. 脚本、产物与日志回查

所有项目内路径相对项目根目录。清单表示执行时预期的产物，不代表文件已经生成；不覆盖或改名外部历史证据。

| 脚本完整路径 | 阶段与用途 | 输入及运行条件 |
|---|---|---|
| [`.agents/docs/career_roadmap/w1/scripts/d03_run_inference_tests.py`](scripts/d03_run_inference_tests.py) | 第 3 节完整测试及按需定向重跑 | `tests/test_engine.py`、`tests/test_attention_fallback.py`；当前 nanochat 解释器可导入项目和 pytest，CPU/FP32，无 checkpoint |

| 完整项目相对路径 | 用途与生成条件 |
|---|---|
| `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_stdout.log` | 捕获脚本自动打开并写入 pytest stdout，包括摘要与失败栈 |
| `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_stderr.log` | 同脚本自动打开并捕获 stderr，可为空 |
| `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_run.json` | 子进程正常返回后由同脚本写入命令、退出码、用时、限制；外部中断可能缺失 |
| `experiments/inference_infra/reports/d03_checks.log` | 各命令组通过 tee 自动追加的输出与即时退出码 |
| `experiments/inference_infra/reports/d03_inference_smoke.md` | 可选人工整理聊天答案和证据，不是完成本日的前置 |

`<UTC时间戳>` 是脚本按协调世界时生成的 `YYYYmmddTHHMMSSffffffZ`，以 `Logs:` 实际输出为准。<abbr title="JSON / JavaScript Object Notation · 用键值和数组保存结构化数据的文本格式；此处记录本次真实命令与退出状态，便于复查。">JSON</abbr> 中自动已有的信息不必手抄。

日志命令采用局部 <abbr title="Bash process substitution · 进程替换。命令组输出同时交给 tee 显示并追加；命令组结束后恢复终端输出，不持续记录提示符。">进程替换</abbr>，复制时包含外层花括号及末尾重定向。`$?` 必须紧跟被检查的命令保存；组内最后 `printf` 的状态不代表测试状态，以显式打印值和 JSON 为准。若旧终端仍有全局重定向，关闭后新开终端；修改教程不会终止旧重定向。

本日完成标准是能解释断言针对的缺陷、实际运行结论及未覆盖范围。CPU 依赖阻塞不妨碍概念阅读，但需明确尚无本次运行证据；不将其写成性能验收。

上一篇：[D2 环境盘点](d02_environment_inventory.md)。下一篇：[D4 模型与形状](d04_model_shapes.md)。
