# RLVR 数据集调研与选型

> 基于 2025-2026 年主流 RLVR 论文调研，面向 NeurIPS 2026 投稿，基座模型为 Qwen3-4B。

## 1. 背景

参考论文：*"Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"*
([arXiv:2504.13837](https://arxiv.org/abs/2504.13837), NeurIPS 2025 Oral)

核心发现：RLVR 本质上是推理时优化（inference-time optimization baked into weights），不会突破基座模型的推理能力上限。
评估方法：pass@k（k 从 1 到很大值），k 小时 RLVR 赢，k 大时 base model 赢。

## 2. 最终数据集选型

### 2.1 训练集

| 数据集 | 数量 | 难度 | 来源 |
|--------|------|------|------|
| **MATH (math12k)** | 12k | 高中级 | `hiyouga/math12k` |
| **NuminaMath-1.5-RL-Verifiable** (竞赛子集筛选) | 3-5k | 竞赛级 | `nlile/NuminaMath-1.5-RL-Verifiable` |
| **总计** | **15-17k** | | |

### 2.2 测试集

| 数据集 | 数量 | 难度 | 作用 |
|--------|------|------|------|
| **MATH500** | 500 | 高中 | 基础线 |
| **Minerva Math** | 272 | 大学 | 泛化验证（训练集外分布） |
| **AMC23** | 40 | 竞赛（中） | 中间难度梯度 |
| **AIME25** | 30 | 竞赛（难） | 上限探测 |

## 3. 关键决策及理由

### 3.1 去掉 GSM8K

**原因：**
- Qwen3-4B 在 GSM8K 上准确率 >90%，区分度低
- 作为训练集几步就饱和，无有效 reward signal
- 2025-2026 年主流论文已不再使用（详见下方调研）

**调研结果：** 在 12 篇主流 RLVR 论文中：
- 仅 3 篇仍在训练中使用 GSM8K（SimpleRL-Zoo, Tulu 3, AReaL）
- 完全不使用的有：DeepSeek-R1, DAPO, Open-Reasoner-Zero, STILL-2/3, Sky-T1, PRIME, DeepScaleR, Logic-RL

### 3.2 不使用 Omni-MATH 作为训练集

**原因：**
- 全部为 Olympiad 级别，对 Qwen3-4B 太难，正确率接近 0，无训练信号
- 是知名 benchmark，用作训练会被质疑 data contamination

### 3.3 使用 NuminaMath-1.5-RL-Verifiable 的竞赛子集

**原因：**
- 已预过滤：去除 MCQ、证明题、合成数据、cn_k12、orca_math
- 全部有数值答案，适合 RLVR 的 rule-based reward
- 竞赛级难题来源：olympiads (92k) + cn_contest (16k) + aops_forum (15k) + amc_aime (5k)

**筛选策略（启发式规则）：**
1. 只保留 answer 为纯数字的（正则 `^-?\d+(\.\d+)?$`）
2. 去掉超长 problem（>2000 字符）
3. 去掉超长 solution（>5000 字符，通常对应过难的题）
4. 按 source 分层采样：amc_aime 全留，olympiads 采样 1-2k，aops_forum 采样 500-1k

## 4. NuminaMath-CoT 子集分析

| 子集 | 数量 | 难度 | 是否推荐 | 原因 |
|------|------|------|---------|------|
| olympiads | ~152k | 高 | 推荐 | 国际竞赛真题，核心难题来源 |
| amc_aime | ~5k | 中-高 | 推荐 | 答案为整数，reward 验证零歧义 |
| aops_forum | ~30k | 中-高 | 推荐 | 社区精选，多样性好 |
| cn_contest | ~16k | 中-高 | 可选 | 中国竞赛题 |
| cn_k12 | ~275k | 低 | 不推荐 | 太简单 |
| orca_math | ~150k | 低 | 不推荐 | 合成小学题 |
| synthetic_math | 大量 | 中 | 不推荐 | 合成数据，质量不稳定 |
| synthetic_amc | 大量 | 中 | 不推荐 | 消融实验证明有害，v1.5 已移除 |

## 5. 主流论文训练/测试数据对比

| 论文 | 训练数据 | 测试数据 | GSM8K |
|------|---------|---------|-------|
| DeepSeek-R1 | 纯 RL / cold-start CoT | AIME24, MATH500, GSM8K, LiveCodeBench | 仅测试 |
| DAPO | DAPO-Math-17k | AIME24 | 仅消融 |
| Open-Reasoner-Zero | AIME+MATH+NuminaMath (129k) | AIME24/25, MATH500, GPQA | 不使用 |
| STILL-2/3 | MATH+NuminaMath+AIME历年 (30k) | AIME24, OlymMATH | 不使用 |
| Sky-T1 | AIME+MATH+NuminaMath+APPS (17k) | AIME, MATH | 不使用 |
| PRIME | NuminaMath-CoT (457k) | AMC, AIME | 不使用 |
| DeepScaleR | AIME+AMC+Omni-MATH (40k) | AIME24, MATH500, AMC23 | 不使用 |
| SimpleRL-Zoo | GSM8K+MATH (8k) | GSM8K, MATH500, AIME24 | 训练+测试 |

## 6. 参考链接

- [Does RL Really Incentivize Reasoning (arXiv)](https://arxiv.org/abs/2504.13837)
- [NuminaMath-CoT (HuggingFace)](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
- [NuminaMath-1.5 (HuggingFace)](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5)
- [NuminaMath-1.5-RL-Verifiable (HuggingFace)](https://huggingface.co/datasets/nlile/NuminaMath-1.5-RL-Verifiable)
- [Omni-MATH (HuggingFace)](https://huggingface.co/datasets/KbsdJames/Omni-MATH)
- [awesome-RLVR-boundary (GitHub)](https://github.com/rdi-berkeley/awesome-RLVR-boundary)
