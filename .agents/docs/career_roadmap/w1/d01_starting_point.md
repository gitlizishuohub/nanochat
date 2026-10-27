---
source_files:
  - README.md
  - nanochat/gpt.py
  - scripts/base_train.py
  - nanochat/checkpoint_manager.py
  - .agents/docs/career_roadmap/w1/scripts/d01_inventory_starting_point.py
source_symbols:
  - GPT.forward
  - load_checkpoint
doc_type: tutorial
knowledge_scope: W1 D1 项目全景与证据边界
---

# W1 D1：先理解 nanochat 在做什么，再确定自己的起点

今天要建立一张能解释代码职责的项目地图：文本怎样变成模型输入，训练怎样改变参数，生成又怎样使用参数。随后学习区分“程序能运行”“回答有质量”“运行得快”三种结论。预计 45–70 分钟；正文推演无需执行，末尾操作只读文件，不训练、不下载、不需要 GPU 或模型权重。

有其他语言模型项目经验，可以迁移学习方法；没有 MiniMind 文件或训练经历，也能完整学习本日主线。已有历史证据只在第 4 节作为可选练习使用。

## 当天的流程阅读

用约 10–20 分钟阅读 [1.1 整体训练主线图](../../llm-training-workflow/01-training-pipeline-and-optional-branches.md#reading-1-1)，再结合下文理解。图描述项目支持的流程，不表示你今天需要执行这些阶段。

- **今天重点**：区分分词器、随机初始化、基础训练、对话微调、评估和生成各自的输入与输出。
- **暂时跳过**：1.2 的可选分支和算法对照，以及图中脚本的实际执行。
- **口述自检**：拿到一个模型文件，为什么还不能断言它能在当前项目中正确回答问题？

## 1. 用一个例子建立项目地图

### 1.1 同一段文字，在训练与生成中扮演不同角色

以下是纸面推演。假设一个玩具词表把“猫”“在”“睡觉”分别映射为整数 `7、12、9`，省略真实文档中的边界标记；这不是 nanochat 的真实词表。

<abbr title="Tokenizer · 分词器。把文本编码为整数 token ID，并按同一映射解码。token 不一定是一个字或一个完整单词；这里的映射仅用于教学。">分词器</abbr>解决“文字怎样交给模型”的问题。模型实际接收整数序列，而不是直接接收中文字形。这个例子的两个用途是：

| 场景 | 给模型什么 | 希望拿到什么 | 是否更新参数 |
|---|---|---|---|
| 训练中的下一词元预测 | 输入 `[7,12]`，目标 `[12,9]` | 第一个位置预测“在”，第二个位置预测“睡觉”的损失 | 计算损失后，还需反向传播与优化器更新 |
| 生成中的下一词元预测 | 提示 `[7,12]` | 最后位置对所有候选词元的分数，从中选出下一个 ID | 生成过程不更新参数 |

目标为什么错开一位？模型在“猫”后学习预测“在”，在“猫 在”后学习预测“睡觉”；目标文本已经给出。生成时后面的正确答案没有给出，只能用自己的预测接着往后写。

**随机初始化只是给参数赋初值。** 即使随机模型能输出 ID，也不能说明它学会了语言。训练把预测误差转成参数更新；推理使用当前参数进行预测。训练后的参数可以保存为 <abbr title="Checkpoint · 检查点。保存的模型参数及相关配置；恢复训练通常还需要优化器与循环状态。文件可读取不等于模型质量已通过评估。">checkpoint</abbr>，以后加载使用，但文件存在本身没有证明质量。

### 1.2 文件职责与刚才的例子怎样对应

这张表用于解释职责，今天不要求逐个读完。后续各日会选取其中一段深入。

| 项目位置 | 对应的工作 | 应当追问什么 |
|---|---|---|
| [分词器](../../../../nanochat/tokenizer.py) | 文字与 ID 互相转换 | 同一个 ID 在当前词表里代表什么？ |
| [模型](../../../../nanochat/gpt.py) | 输入 ID，算出预测分数或损失 | 输入/输出形状与目标怎样对应？ |
| [基础训练脚本](../../../../scripts/base_train.py) | 准备数据，计算损失，更新参数 | 哪一步真正改变参数？ |
| [checkpoint 管理](../../../../nanochat/checkpoint_manager.py) | 保存/读取参数、配置与训练状态 | 权重与配置、词表是否配套？ |
| [推理引擎](../../../../nanochat/engine.py) | 用模型逐步生成并管理缓存 | 每一步输入什么，保留什么状态？ |
| [评估](../../../../nanochat/core_eval.py)与[推理基准](../../../../scripts/infer_bench.py) | 分别观察任务表现与运行成本 | 数字回答的是质量还是速度问题？ |

预训练从普通文档学习续写；<abbr title="SFT / Supervised Fine-Tuning · 监督微调。使用示范回答继续更新已有模型参数，学习对话等目标行为。它也是训练，不是普通生成。">对话监督微调</abbr>则在已有模型上使用示范继续训练。W1 先理解推理执行与验证，不要求你先完成这两种训练。

## 2. 从真实代码辨认“预测”与“训练”

只读 [GPT.forward](../../../../nanochat/gpt.py) 末尾的返回分支。下面省略了交叉熵参数，保留真实控制结构：

```python
if targets is not None:
    loss = F.cross_entropy(...)
    return loss
else:
    return logits
```

`targets` 是已知目标 ID。传入目标时，模型把预测与目标比较，返回一个损失；不传目标时，返回 <abbr title="Logits · 模型对词表内每个候选 token 的未归一化分数。最高分对应的 ID 只是一个离散选择，完整分数还携带候选之间的差距。">logits</abbr>，由调用者继续选择 token。**仅计算并返回损失，也还没有更新参数**；验证集评估同样可以计算损失。

在 [base_train.py](../../../../scripts/base_train.py) 的训练步里，实际还有后续操作。下面是省略累积与精度分支后的执行关系，不是独立运行代码：

```python
loss = model(x, y)   # 对照目标计算误差
loss.backward()     # 计算参数梯度
optimizer.step()    # 按梯度更新参数
```

因此判断是否训练，要追踪是否执行了反向传播和参数更新。读到 `init_weights()`、加载 `.pt` 或一次 `forward()`，都不能独自得出“正在训练”的结论。今天只阅读这些语句，不执行训练脚本。

再看 [load_checkpoint](../../../../nanochat/checkpoint_manager.py)：它按步号读取 `model_000002.pt` 这样的权重与 `meta_000002.json` 元数据；需要恢复优化器时还读对应分片。这里的 `2` 只是命名示例，不表示你已有这些文件。加载成功回答“文件和模型能否兼容”，质量仍要由适合任务的评估回答。

## 3. 什么证据能支持什么结论

假设两个实现对同一前缀输出两类分数：A 为 `[2.0,1.0]`，B 为 `[2.0,1.9]`。两者都选第一个 ID，但分数最大差是 `0.9`。这说明“选出的 token 相同”不能代替“数值结果接近”。D3、D5 会把这种比较用于缓存实现；今天只需理解它们是不同强度的检查。

| 你观察到的事情 | 可以支持的结论 | 还缺什么才能扩大结论 |
|---|---|---|
| CPU 上小型随机模型跑通，形状符合断言 | 这组条件下控制流/接口可运行 | 不能据此判断语言质量；需要训练过的兼容权重和任务评估 |
| 同一参数下两种路径的 logits 误差在选定容差内 | 这组输入下，两种实现数值接近 | 其他长度、批量和边界条件仍需覆盖 |
| 某次运行测到一个吞吐值 | 该设备与配置下的一次耗时观察 | 需要明确计时边界、预热与重复；不能直接外推其他设备 |

CPU 上也能测耗时，但 CPU 的数字属于 CPU。GPU 性能是可选的另一个研究范围，不是完成 W1 所欠的实验。类似地，随机模型很适合做接口与实现对照；其输出文本不应当拿来评价训练后的模型能力。

可迁移的经验是“固定条件、比较实现、追溯证据、解释边界”。另一个项目的词表 ID、权重布局、误差容差和速度数值则需要在 nanochat 中重新核对。即使两个词表大小相同，ID 映射不同也会使同一输入整数代表不同内容。

## 4. 实际操作：定位项目；历史证据作为可选应用

本节操作不运行模型。终端沿用已自动激活的 `nanochat` Conda 环境，先确认解释器与仓库：

```bash
cd /home/lizishuo/_program_/Project/nanochat
mkdir -p experiments/inference_infra/reports
{
python -c 'import sys; print(sys.executable)'
d01_python_rc=$?; echo "python_exit_code=$d01_python_rc"
git rev-parse HEAD
d01_git_rc=$?; echo "git_exit_code=$d01_git_rc"
git status --short
} > >(tee -a experiments/inference_infra/reports/d01_checks.log) 2>&1
```

预期解释器为 `/home/lizishuo/miniconda3/envs/nanochat/bin/python`。路径不同先确认当前环境；已有 Git 改动需要保留。日志只在花括号内记录，结束后恢复终端输出；变量仍留在当前 shell。`$?` 必须紧跟被检查命令保存，否则会变成后续命令的退出码。若旧终端仍有全局日志重定向，先新开终端。

**可选：你有 MiniMind 历史材料时**，用[证据盘点脚本](scripts/d01_inventory_starting_point.py)练习追溯。没有材料可以跳过整段，继续 D2。

先读脚本的三个关键动作：`path.is_file()` 区分文件是否存在；`sha256(path.read_bytes())` 标识此次读到的文件内容；`csv.DictReader` 汇总原表。哈希相同说明内容相同，不证明内容结论正确。`repetitions_per_group` 按提示长度、生成长度、batch、cache、device、dtype 分组收集编号；它没有自动剔除预热，也没有替你判定编号是否重复，应回原表核查 `warmup` 和 `repetition`。

```bash
{
python .agents/docs/career_roadmap/w1/scripts/d01_inventory_starting_point.py
d01_inventory_rc=$?; echo "inventory_exit_code=$d01_inventory_rc"
} > >(tee -a experiments/inference_infra/reports/d01_checks.log) 2>&1
```

默认扫描 `../minimind/experiments/minimind_baseline/` 和 `../minimind/experiments/archive/minimind_baseline*/`；非相邻位置传 `--minimind-root /实际路径/minimind`。JSON 中任意目标文件存在就返回 0，所以 **0 不表示整套历史实验完整**；没有目标文件返回 2，仍会写出缺失清单。输入位置缺失只影响这个可选练习，不阻塞 nanochat 学习。

如果找到材料，从报告选一条结论，追到它引用的 CSV 行或日志，再说出项目、设备、配置和日期。旧归档候选包括 `../minimind/experiments/archive/minimind_baseline-before-embedding-20261014/`，实际以本次盘点为准。历史最大误差即使低于当时容差，也只能称为历史记录，不能说今天已复测 nanochat。

## 5. 当天自验证与交流

直接在聊天按题号回答；不要求手写报告。实际执行过的操作附一次命令、退出码与日志路径；概念推演不需要运行证据。可选历史材料未查阅无需列为欠账。

1. **把项目串起来**：用第 1 节的玩具 ID 解释训练输入/目标为什么错开一位；生成第二个新 token 时，输入又从哪里来？

   **核对要点**：能区分已知目标与模型自产预测，不把 token ID 当中文字本身。

2. **辨认训练动作**：一次 `forward` 返回了损失，是否足以说明参数已经变化？从第 2 节的调用关系说明还缺哪些动作。

   **核对要点**：区分初始化、计算损失、反向传播与优化器更新，不凭输出文件判断训练。

3. **控制结论范围**：CPU 随机模型的两条路径输出相同 token，能否宣布模型回答正确、两条实现数值等价，或 GPU 上也一样快？分别需要什么证据？

   **核对要点**：分开功能、数值、质量与设备性能，使用本节算例解释相同 token 的局限。

4. **给 D2 一个具体问题**：选择你现在最不确定的运行条件，说明它可能影响哪个模块、准备如何确认。若做了历史盘点，可用一条实际记录解释哪些经验可迁移。

   **核对要点**：问题具体到解释器、依赖、设备或数据路径之一；不把外部历史分数当当前结果。

## 6. 脚本、产物与结束核对

项目内路径均相对 nanochat 根目录；表中是执行时的生成条件，不表示已经生成。

| 脚本与输入 | 用途与条件 |
|---|---|
| [`.agents/docs/career_roadmap/w1/scripts/d01_inventory_starting_point.py`](scripts/d01_inventory_starting_point.py)；输入为上节列出的外部历史目录 | 可选历史盘点，Python 标准库即可；只读外部文件，不导入模型。重跑覆盖同名 JSON；可用 `--output` 另存快照 |

| 完整项目相对路径 | 用途与生成者 | 生成条件 |
|---|---|---|
| `experiments/inference_infra/reports/d01_checks.log` | 本节 Bash 局部 `tee` 保存输出与即时退出码 | 执行对应命令组后自动追加 |
| `experiments/inference_infra/results/w1/d01_starting_point_inventory.json` | 盘点脚本保存存在性、哈希与 CSV 摘要 | 仅选择历史盘点并完成脚本时自动写入；退出 2 也会生成 |
| `experiments/inference_infra/reports/d01_starting_point.md` | 自己整理起点与证据 | 可选人工编写，聊天回答可替代 |

完成标准是能解释项目职责、训练与生成的区别，以及证据可支持的结论；实际环境采集在 D2，模型测试在 D3。外部文件不存在、没有权重或没有 GPU 都不影响本日概念学习。

下一篇：[D2 环境盘点](d02_environment_inventory.md)。原路线：[逐日总计划](../nanochat_daily_execution_2026-10-01.md)。
