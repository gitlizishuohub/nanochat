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

# W1 D3：运行推理测试，保留失败与跳过证据

今天让“环境看起来装好了”变成“具体测试在这个环境执行过”。预计 60–90 分钟：10 分钟检查依赖，15 分钟首次运行，20 分钟读测试，20 分钟针对失败定位，15 分钟记录。若首次导入/测试较慢，以真实用时为准；不要为满足日历删掉失败用例。

前置：[D2 环境盘点](d02_environment_inventory.md)。本轮 Conda 环境已创建；依赖安装完成情况必须由下面命令确认。没有 GPU 走 CPU，不跳过 CPU 验证。

## 当天的流程阅读

先用约 10–20 分钟读下面指定的小节，再回到当天操作；已读内容按问题回查即可。阅读只需口述理解，不新增手写报告或实验产物。

- **阅读入口**：[4.1 配置不是可学习权重](../../llm-training-workflow/04-model-construction-and-weight-loading.md#reading-4-1)；[4.2 从结构到权重](../../llm-training-workflow/04-model-construction-and-weight-loading.md#reading-4-2)。
- **今天重点**：掌握配置、随机初始化、已有权重三个概念，理解随机小模型为什么也能做接口测试。结合今天的测试区分容器、替身模型和真实模型。
- **暂时跳过**：4.2.1 的初始化分布表只看标题；不要求背数值，加载细节到 D11。

**读完能解释**：随机模型的测试通过，能说明形状、生成控制流和语言质量中的哪几项？可直接在聊天中回答，再用本日原有自验证问题核对实际实验。

## 当日脚本与产物清单

所有项目内路径均以 nanochat 项目根目录为基准；外部 MiniMind 只读历史证据不改名。`dNN_` 表示全程首次使用或产生日，跨日复用不改前缀；上游源码和测试保留原生名称。

| 角色与脚本 | 用途与执行阶段 | 输入路径或来源 | 运行条件 |
|---|---|---|---|
| 入口：[`.agents/docs/career_roadmap/w1/scripts/d03_run_inference_tests.py`](scripts/d03_run_inference_tests.py) | 首次完整测试；第 4 节按需定向重跑 | `tests/test_engine.py`、`tests/test_attention_fallback.py` | 激活 nanochat，已安装项目与 dev 依赖；CPU/FP32；不需 checkpoint |

| 完整项目相对路径 | 用途 | 生成者 | 生成方式与条件 |
|---|---|---|---|
| `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_stdout.log` | 完整测试结果与失败栈 | `d03_run_inference_tests.py` | 自动：启动 pytest 前打开，运行期间写入 |
| `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_stderr.log` | 诊断输出，可为空 | 同上 | 自动：启动 pytest 前打开 |
| `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_run.json` | 命令、退出码、用时和 CPU 限制 | 同上 | 条件生成：子进程正常返回后写入；外部中断可能缺失 |
| `experiments/inference_infra/reports/d03_inference_smoke.md` | 测试解释、skip 和未覆盖范围 | 第 5 节问答可选整理 | 可选人工整理；直接聊天回答即可，不影响验收 |
| `experiments/inference_infra/reports/d03_checks.log` | 本日终端检查输出与错误 | 各 Bash 代码块的 tee | 自动追加各代码块内的输出；命令及退出码在交流时补充 |

清单是执行后的预期，不表示文件已经存在。已有未加前缀的历史实验文件保留，新执行按上述路径写入。 `<UTC时间戳>` 由脚本按协调世界时生成，格式为 `YYYYmmddTHHMMSSffffffZ`；以终端 `Logs:` 打印的真实目录为准。

正文的 `sed`、`rg`、`head`、解释器检查等只向终端打印，不会自动生成独立结果文件；各 Bash 代码块内的输出追加保存到 `experiments/inference_infra/reports/d03_checks.log`；交流时附所用命令、退出码和关键结果，自动结果已有的字段无需手抄。 测试输出由清单中的 stdout/stderr 日志捕获。 未执行或失败时按真实状态记录，不能把旧同名文件当作本次成功。

## 1. 检查同一解释器内的依赖

日志只在 `{ ...; }` 命令组执行期间记录；<abbr title="Bash process substitution · 进程替换。这里把命令组的输出交给 tee，同时显示并追加到日志；重定向在命令组结束后恢复。">进程替换</abbr>不会持续接管交互终端，因此提示符和 VS Code 终端集成控制字符不会混入日志。命令组在当前 shell 执行，变量设置会保留；代码块返回最后一条命令的退出码，需核对单条命令时立即记录 `$?`。已有的显式文件重定向仍写入各自文件。复制时包含外层花括号和末尾重定向。若之前执行过旧版全局日志命令，先关闭旧终端并新开终端；仅修改文档不会停止旧终端的记录。

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
python -c 'import sys,torch,pytest; print(sys.executable); print(torch.__version__); print(pytest.__version__)'
} > >(tee -a experiments/inference_infra/reports/d03_checks.log) 2>&1
```

看到 `ModuleNotFoundError` 就先处理环境，不能把它解释成模型算法错误。本次任务由主流程安装 CPU 和 dev 依赖；安装未结束时不要同时启动第二次安装。准备好后用[测试捕获脚本](scripts/d03_run_inference_tests.py)：

```bash
{
python .agents/docs/career_roadmap/w1/scripts/d03_run_inference_tests.py --help
python .agents/docs/career_roadmap/w1/scripts/d03_run_inference_tests.py
} > >(tee -a experiments/inference_infra/reports/d03_checks.log) 2>&1
```

脚本使用当前 `sys.executable -m pytest`，显式限制 CPU 和 FP32；每次建立独立时间戳目录，不覆盖上一轮日志。它关闭 pytest 本地缓存，把 Python 字节码缓存放 `/tmp/nanochat_w1_pycache`，避免无意义的仓库文件变动。

## 2. 找到三份原始证据

终端打印 `Logs:` 后的目录位于 `experiments/inference_infra/results/w1/d03_pytest/`：

- `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_stdout.log`：测试收集、通过数、跳过数与失败栈。
- `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_stderr.log`：导入失败等诊断信息；为空也正常。
- `experiments/inference_infra/results/w1/d03_pytest/<UTC时间戳>/d03_run.json`：精确参数数组、工作目录、真实退出码、用时和环境限制。

<abbr title="pytest · Python 的测试运行器。它收集以 test 命名的函数和类，执行断言，并区分通过、失败、跳过及收集错误。测试数量可能随代码变化，应记录当次实际结果。">pytest</abbr> 的 `exit_code=0` 表示所选测试运行成功；不等于所有 GPU 分支通过。`skipped` 是未执行条件分支，不能计入已验证 GPU 正确性。记录实际摘要，不在本教程预填固定通过数。

## 3. 逐个读懂现有用例证明什么

```bash
{
rg -n '^def test_|^class Test|def test_' tests/test_engine.py tests/test_attention_fallback.py
sed -n '15,130p' tests/test_engine.py
rg -n 'skipif|HAS_FA3|DEVICE|DTYPE|test_override_auto' tests/test_attention_fallback.py
} > >(tee -a experiments/inference_infra/reports/d03_checks.log) 2>&1
```

按这条阅读顺序理解，可直接在聊天中说明：

1. `MockModel` 返回均匀分数，`ByteTokenizer` 用字节和少量特殊 token 模拟分词。<abbr title="mock · 测试替身。用受控的简单对象隔离一个组件，例如检验生成控制流而无需真实权重。通过 mock 测试不能说明真实 GPT 数值计算已正确。">mock 测试</abbr> 用于隔离 Engine 控制流。
2. `test_kv_cache_basic` 检查形状、推进和 reset；`test_kv_cache_prefill` 检查复制。写清“验证缓存容器”，不要写“真实模型 KV 等价已证明”。
3. 多样本首 token 测试防止一个采样结果被广播到所有行；同 seed 可复现与温度为零的确定性检验的是不同性质。
4. `TestSDPAOnly` 在可用设备执行基础前向、反向和缓存操作。当前脚本隐藏 CUDA，因此它选 CPU。
5. FA3 对照测试有硬件/内核条件，CPU 上跳过符合预期。<abbr title="SDPA / Scaled Dot Product Attention · 缩放点积注意力接口。这里由 PyTorch 执行回退计算，CPU 路线验证的是该分支，不是 FA3 内核。">SDPA</abbr> 成功不能替代 FA3 对照。

对每组测试确认“证明什么、仍未证明什么”，可直接口述，不要求另做表格。真实 GPT 的缓存数值与 smear 状态在 D5 的小模型探查中补充。

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
{
python .agents/docs/career_roadmap/w1/scripts/d03_run_inference_tests.py --select 'kv_cache or SDPAOnly'
} > >(tee -a experiments/inference_infra/reports/d03_checks.log) 2>&1
```

`--select` 是 pytest 的名字表达式。定向通过只说明筛选后的用例通过；最终仍需完整两文件运行。现有 `test_override_auto` 用 `USE_FA3 == HAS_FA3` 作为断言，而生产路由还看 compute dtype；未来在有 FA3 且强制 FP32 的机器运行时，应按源码核实这一测试假设，不能盲目认定是内核坏了。当前 CPU 隔离下这两个值都应为 false。

## 5. 当天自验证与交流问题并验收

可直接在聊天中按题号回答，不要求手写报告。每题给出结论和可定位依据（源码位置、命令与退出码、日志片段或自动结果字段）；同一份证据可复用。概念推演须注明假设，实际运行须区分本次与历史结果。未执行写 `PENDING`，失败写真实错误和影响范围，不能据此勾选实验通过。

1. **精确测试入口**：本次是否完整执行两个测试文件？给出实际解释器、pytest argv、cwd、退出码及对应时间戳目录，是否使用了 `--select`？

   **通过标准**：从 `d03_run.json` 的 `argv/cwd/exit_code` 核对；完整与筛选运行明确区分，不能把 `--help`、日志目录存在或一次定向通过作为完整验收。

2. **CPU 隔离如何生效**：捕获脚本在何时设置 `CUDA_VISIBLE_DEVICES`、`NANOCHAT_DTYPE`？为什么在有 GPU 的主机上，本次仍验证 CPU/FP32 路线？

   **通过标准**：引用脚本创建子进程环境的代码及元数据；说明变量在测试进程导入模块前生效，FA3 发现与计算精度受其影响，不能由主机有卡推断本次用卡。

3. **通过、跳过与错误**：本次 passed、skipped、failed、errors 实际各多少？跳过的是哪些类或函数、条件是什么；若尚未收集成功能否填通过数？

   **通过标准**：引用 stdout 摘要、skip 条件和首个失败栈；未执行或收集错误保持真实状态，不套用教程历史数量，退出码 0 也不表示被跳过分支通过。

4. **容器用例的断言范围**：`test_kv_cache_basic` 与 `test_kv_cache_prefill` 分别检查了哪些形状、位置和复制数据？哪些真实模型行为没有被它们检查？

   **通过标准**：至少各举一条真实断言；说明容器推进/复制不等于注意力数值等价，原复制用例没有覆盖真实 smear 状态和所有层非零数据。

5. **测试替身与真实 GPT**：`MockModel.forward` 产生什么 logits、如何处理 cache；`ByteTokenizer` 怎样编码普通文本和特殊 token？这些替身隔离了什么？

   **通过标准**：指向对应实现，说明它们便于检验生成控制流，但不执行真实 GPT 注意力/训练权重，也不证明自然语言质量；真实随机模型数值对照留到 D5。

6. **采样的三种性质**：首 token 多样性测试、同 seed 复现测试和温度为零测试分别断言什么？为什么“不同样本必须全部不同”不是首 token 测试的要求？

   **通过标准**：给出三个函数名和比较对象；区分行间随机性、相同输入/seed 下复现及 greedy 不依赖 seed，多样性实际只要求不全相同。

7. **长度与批量契约**：`test_max_tokens_respected` 为什么用“小于等于”上限，而 `test_num_samples_count` 检查返回序列条数？是否保证每行都产生完整 G 个有效 token？

   **通过标准**：引用两个断言与 Engine 的结束 token 行为；区分 prompt 长度、生成上限和实际输出，不把返回 B 行解释成每行固定 G 个 token。

8. **SDPA 与 FA3 的覆盖**：`TestSDPAOnly` 的前向、反向和缓存用例分别有哪些断言？它们是否比较了 FA3 输出、检查了每个梯度的全部有限性、覆盖了真实 checkpoint？

   **通过标准**：按实际代码区分 shape/NaN 检查、梯度存在及手动 cache 推进，不将检查 NaN 扩大为检查全部 Inf；指出 FA3 对照与真实权重仍缺哪些条件。

9. **证据留存与失败恢复**：本次 stdout、stderr、run JSON 是否完整；若失败或中断，第一处错误属于依赖、环境还是断言？已有定向重跑是否随后补了完整运行？

   **通过标准**：给出三份原始路径与真实状态，stderr 为空不单独判失败；中断可能没有 run JSON，定向结果不覆盖旧记录，未补全的范围标 PENDING。

验收要求：有精确命令与退出码；理解 mock 与真实模型区别；明确 skip；没有把未运行当 PASS。失败未解决也要保留可复查日志，然后按依赖阻塞继续读源码，不能进入“性能已通过”状态。

交流时附实际日期、commit、命令、原始日志目录、状态与下一步；用时可选，不需再手抄自动记录。

上一篇：[D2 环境盘点](d02_environment_inventory.md)。下一篇：[D4 模型与形状](d04_model_shapes.md)。
