# D01 起点复核交流记录

记录日期：2026-10-21。

本报告整理聊天中的回答与概念核对；不代表今天已完成实验复测。问题见 [D01 教程](../../../.agents/docs/career_roadmap/w1/d01_starting_point.md#4-当天自验证与交流问题)。

## 第 1、2 题

用户略过，本报告不补写答案或判定通过。

## 第 3 题：缓存与无缓存输出的一致性

用户原答：`cache_token` 是有 KV Cache 的逐个输出，`no_cache_token` 是没有 KV Cache 的逐个输出；`max_abs_logit_diff` 是对数几率的差异绝对值。

核对后的表述：

- `cache_token`：使用 <abbr title="KV Cache / Key-Value Cache · 缓存历史 token 的注意力 Key 和 Value，减少后续生成步骤对历史内容的重复计算。">KV Cache</abbr> 时，当前生成步输出的 <abbr title="token ID · 分词器词表中一个 token 对应的整数编号；token 可以是字、子词或其他文本片段，不一定是完整的词。">token ID</abbr>。
- `no_cache_token`：不使用 KV Cache 时，同一步输出的 token ID。两列逐行相等，说明两种方式的生成选择逐步一致。
- `max_abs_logit_diff`：同一步、同一前缀最后位置，两种方式对整个词表输出的 <abbr title="logits · 模型在 softmax 归一化之前，为词表中每个候选 token 输出的分数。这里比较两种计算方式的分数向量，不是比较概率。">logits</abbr> 的最大绝对差，即 `max_i |logits_cache[i] - logits_no_cache[i]|`，其中 `i` 遍历词表。

术语修正：此处将 logits 表述为“未归一化分数”更准确，“对数几率”容易产生歧义。原回答对输出方式的理解基本正确，但需要补充“整个词表”和“最大绝对差”。

token 相同不代表 logits 完全相同：分数有小幅差异时，最高分的 token 仍可能相同。因此，需要同时检查逐步 token 是否一致，以及 logits 误差是否小于实验选定的容差。

### 待补充的实测证据

本次交流尚未核对原始历史表，也未运行新的缓存一致性实验，以下项目仍待确认：

- 历史表中 `cache_token` 与 `no_cache_token` 是否每行相等。
- `max_abs_logit_diff` 全表最大值、实验选定的容差及是否满足该容差。
- 结果来源、证据路径，以及属于历史记录还是今天复测。

当前状态：已完成概念核对，第 3 题的实验事实部分待补充。
