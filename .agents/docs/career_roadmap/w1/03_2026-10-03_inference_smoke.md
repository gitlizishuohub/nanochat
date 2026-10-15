---
source_files:
  - tests/test_engine.py
  - tests/test_attention_fallback.py
  - pyproject.toml
source_symbols:
  - test_multi_sample_first_token_diversity
  - TestSDPAOnly
  - TestOverrideMechanism
doc_type: tutorial
knowledge_scope: W1 D3 CPU推理测试及失败定位
---

# W1 D3 · 2026-10-03：运行推理测试，保留失败与跳过证据

今天让“环境看起来装好了”变成“具体测试在这个环境执行过”。预计 60–90 分钟：10 分钟检查依赖，15 分钟首次运行，20 分钟读测试，20 分钟针对失败定位，15 分钟记录。若首次导入/测试较慢，以真实用时为准；不要为满足日历删掉失败用例。

前置：[D2 环境卡](02_2026-10-02_environment_inventory.md)。本轮 Conda 环境已创建；依赖安装完成情况必须由下面命令确认。没有 GPU 走 CPU，不跳过 CPU 验证。

## 1. 检查同一解释器内的依赖

```bash
cd /home/lizishuo/_program_/Project/nanochat
source /home/lizishuo/miniconda3/etc/profile.d/conda.sh
conda activate nanochat
python -c 'import sys,torch,pytest; print(sys.executable); print(torch.__version__); print(pytest.__version__)'
```

看到 `ModuleNotFoundError` 就先处理环境，不能把它解释成模型算法错误。本次任务由主流程安装 CPU 和 dev 依赖；安装未结束时不要同时启动第二次安装。准备好后用[测试捕获脚本](scripts/run_inference_tests.py)：

```bash
python .agents/docs/career_roadmap/w1/scripts/run_inference_tests.py --help
python .agents/docs/career_roadmap/w1/scripts/run_inference_tests.py
```

脚本使用当前 `sys.executable -m pytest`，显式限制 CPU 和 FP32；每次建立独立时间戳目录，不覆盖上一轮日志。它关闭 pytest 本地缓存，把 Python 字节码缓存放 `/tmp/nanochat_w1_pycache`，避免无意义的仓库文件变动。

## 2. 找到三份原始证据

终端打印 `Logs:` 后的目录位于 `experiments/inference_infra/results/w1/pytest/`：

- `stdout.log`：测试收集、通过数、跳过数与失败栈。
- `stderr.log`：导入失败等诊断信息；为空也正常。
- `run.json`：精确参数数组、工作目录、真实退出码、用时和环境限制。

<abbr title="pytest · Python 的测试运行器。它收集以 test 命名的函数和类，执行断言，并区分通过、失败、跳过及收集错误。测试数量可能随代码变化，应记录当次实际结果。">pytest</abbr> 的 `exit_code=0` 表示所选测试运行成功；不等于所有 GPU 分支通过。`skipped` 是未执行条件分支，不能计入已验证 GPU 正确性。记录实际摘要，不在本教程预填固定通过数。

## 3. 逐个读懂现有用例证明什么

```bash
rg -n '^def test_|^class Test|def test_' tests/test_engine.py tests/test_attention_fallback.py
sed -n '15,130p' tests/test_engine.py
rg -n 'skipif|HAS_FA3|DEVICE|DTYPE|test_override_auto' tests/test_attention_fallback.py
```

按这条阅读顺序做笔记：

1. `MockModel` 返回均匀分数，`ByteTokenizer` 用字节和少量特殊 token 模拟分词。<abbr title="mock · 测试替身。用受控的简单对象隔离一个组件，例如检验生成控制流而无需真实权重。通过 mock 测试不能说明真实 GPT 数值计算已正确。">mock 测试</abbr> 用于隔离 Engine 控制流。
2. `test_kv_cache_basic` 检查形状、推进和 reset；`test_kv_cache_prefill` 检查复制。写清“验证缓存容器”，不要写“真实模型 KV 等价已证明”。
3. 多样本首 token 测试防止一个采样结果被广播到所有行；同 seed 可复现与温度为零的确定性检验的是不同性质。
4. `TestSDPAOnly` 在可用设备执行基础前向、反向和缓存操作。当前脚本隐藏 CUDA，因此它选 CPU。
5. FA3 对照测试有硬件/内核条件，CPU 上跳过符合预期。<abbr title="SDPA / Scaled Dot Product Attention · 缩放点积注意力接口。这里由 PyTorch 执行回退计算，CPU 路线验证的是该分支，不是 FA3 内核。">SDPA</abbr> 成功不能替代 FA3 对照。

留一个三列表：`测试组 / 这组证明什么 / 仍未证明什么`。真实 GPT 的缓存数值与 smear 状态在 D5 的小模型探查中补充。

## 4. 失败分类后只重跑相关用例

| 错误 | 类型 | 下一步 |
|---|---|---|
| `No module named torch/pytest` | 依赖/解释器 | 重看 `sys.executable`；等待安装完成后重试 |
| `No module named tiktoken` 等 | 传递导入依赖 | Engine 导入 checkpoint_manager 会拉入别的依赖；仅有 torch 不一定够 |
| 动态库加载失败 | 环境 | 保存完整 stderr，核对 CPU wheel 与解释器 |
| assertion failure | 行为或环境条件 | 看第一个失败栈、输入、dtype、选中后端 |
| FA3 required 跳过 | 预期硬件限制 | 记 `PENDING_GPU`，不安装随机内核碰运气 |
| 进程被终止 | 资源/外部中断 | 保存中断事实；无完整退出码则未验收 |

例如只重跑缓存容器与 SDPA：

```bash
python .agents/docs/career_roadmap/w1/scripts/run_inference_tests.py --select 'kv_cache or SDPAOnly'
```

`--select` 是 pytest 的名字表达式。定向通过只说明筛选后的用例通过；最终仍需完整两文件运行。现有 `test_override_auto` 用 `USE_FA3 == HAS_FA3` 作为断言，而生产路由还看 compute dtype；未来在有 FA3 且强制 FP32 的机器运行时，应按源码核实这一测试假设，不能盲目认定是内核坏了。当前 CPU 隔离下这两个值都应为 false。

## 5. 写测试小结并验收

新建 `experiments/inference_infra/reports/inference_smoke.md`：

```markdown
# CPU 推理测试
- 实际日期 / commit / Python：
- 原始日志目录：
- 原始 argv / exit_code：
- 实际摘要：passed / skipped / failed / errors
- 第一个失败（若有）：环境 / 依赖 / 行为 / 未分类
- 解决动作与定向重跑证据：
- 完整重跑证据：
- 已覆盖：
- 未覆盖：FA3、真实 checkpoint 等
- 结论：PASS_CPU / BLOCKED_ENV / FAIL_BEHAVIOR
```

验收要求：有精确命令与退出码；理解 mock 与真实模型区别；明确 skip；没有把未运行当 PASS。失败未解决也要保留可复查日志，然后按依赖阻塞继续读源码，不能进入“性能已通过”状态。

追加当天记录：`实际日期 / commit / 用时 / 命令 / 日志目录 / 状态 / 下一步`。

上一篇：[D2 环境盘点](02_2026-10-02_environment_inventory.md)。下一篇：[D4 模型与形状](04_2026-10-04_model_shapes.md)。
