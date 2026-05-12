## 2 相关工作

### 2.1 Search and Policy Projection

MCTS/UCT 提供了这一路线的基本搜索机制 (Kocsis and Szepesvári, 2006; Browne et al., 2012)，AlphaGo Zero、AlphaZero、Expert Iteration 与 MuZero 将其发展为“搜索改进策略、网络拟合搜索”的自博弈学习范式 (Silver et al., 2017, 2018; Anthony et al., 2017; Schrittwieser et al., 2020)，后续 Gumbel AlphaZero / MuZero 与 Sampled MuZero 进一步通过采样动作集合缓解大动作空间中的规划成本 (Danihelka et al., 2022; Hubert et al., 2021)。

### 2.2 Rollout and Policy Gradient

REINFORCE 和 policy gradient theorem 给出了策略梯度估计的基本形式 (Williams, 1992; Sutton et al., 1999)，TRPO、GAE 与 PPO 分别从信赖域约束、优势估计和 clipped surrogate objective 的角度提高更新稳定性 (Schulman et al., 2015, 2016, 2017)。在 LLM 训练中，InstructGPT、DeepSeekMath、DeepSeek-R1、DAPO、REINFORCE++、GPG 以及 RLOO/ReMax 等工作将 PPO/GRPO 或 critic-free policy gradient 用于人类偏好奖励与可验证奖励训练，说明链式 rollout 上的策略梯度可以显著提升模型对齐与推理能力 (Ouyang et al., 2022; Shao et al., 2024; DeepSeek-AI et al., 2025; Yu et al., 2025; Hu et al., 2025; Chu et al., 2025; Ahmadian et al., 2024; Li et al., 2023)。

### 2.3 Search and Policy Gradient

Search and Policy Gradient 在 LLM 中已有若干相邻方向：self-consistency、Tree of Thoughts、RAP、AlphaZero-like tree-search、rStar-Math 以及 verifier/process-supervision 主要用于测试时解码、重排序、轨迹合成或过程奖励建模 (Wang et al., 2022; Yao et al., 2023; Hao et al., 2023; Feng et al., 2023; Guan et al., 2025; Cobbe et al., 2021; Lightman et al., 2024; Wang et al., 2024)；MCTS-DPO 与 SVPO 使用 MCTS 构造 step-level preference/value 信号，而 TreePO、TreeRL、TreeRPO、Tree-GRPO、SEEA-R1、ARPO、VinePPO 与 SPO 则更直接地把树采样、segment/step-level 重采样或中间状态 credit assignment 纳入策略优化过程 (Xie et al., 2024; Chen et al., 2024; Li et al., 2025; Hou et al., 2025; Yang et al., 2025; Ji et al., 2025; Tian et al., 2025; Dong et al., 2025; Kazemnejad et al., 2025; Guo et al., 2025)。这些工作与本文最接近，但多数侧重搜索样本生成、偏好/过程信号构造、segment-level 优势估计或启发式分支；本文进一步刻画同策略树轨迹相对链式 rollout 的梯度测度变化，并给出 branch-corrected policy gradient、TreeGAE 与 OTRC。

---

**参考文献完整题名（写英文 LaTeX 引用用）**

1. Kocsis and Szepesvári (2006): Bandit Based Monte-Carlo Planning.
2. Browne et al. (2012): A Survey of Monte Carlo Tree Search Methods.
3. Silver et al. (2017): Mastering the Game of Go without Human Knowledge.
4. Silver et al. (2018): A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go through Self-Play.
5. Anthony et al. (2017): Thinking Fast and Slow with Deep Learning and Tree Search.
6. Schrittwieser et al. (2020): Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.
7. Danihelka et al. (2022): Policy Improvement by Planning with Gumbel.
8. Hubert et al. (2021): Learning and Planning in Complex Action Spaces.
9. Williams (1992): Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.
10. Sutton et al. (1999): Policy Gradient Methods for Reinforcement Learning with Function Approximation.
11. Schulman et al. (2015): Trust Region Policy Optimization.
12. Schulman et al. (2016): High-Dimensional Continuous Control Using Generalized Advantage Estimation.
13. Schulman et al. (2017): Proximal Policy Optimization Algorithms.
14. Ouyang et al. (2022): Training Language Models to Follow Instructions with Human Feedback.
15. Shao et al. (2024): DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.
16. DeepSeek-AI et al. (2025): DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning.
17. Yu et al. (2025): DAPO: An Open-Source LLM Reinforcement Learning System at Scale.
18. Hu et al. (2025): REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization.
19. Chu et al. (2025): GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning.
20. Ahmadian et al. (2024): Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs.
21. Li et al. (2023): ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models.
22. Wang et al. (2022): Self-Consistency Improves Chain of Thought Reasoning in Language Models.
23. Yao et al. (2023): Tree of Thoughts: Deliberate Problem Solving with Large Language Models.
24. Hao et al. (2023): Reasoning with Language Model is Planning with World Model.
25. Cobbe et al. (2021): Training Verifiers to Solve Math Word Problems.
26. Lightman et al. (2024): Let's Verify Step by Step.
27. Feng et al. (2023): AlphaZero-like Tree-Search can Guide Large Language Model Decoding and Training.
28. Guan et al. (2025): rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking.
29. Wang et al. (2024): Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations.
30. Xie et al. (2024): Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning.
31. Chen et al. (2024): Step-level Value Preference Optimization for Mathematical Reasoning.
32. Li et al. (2025): TreePO: Bridging the Gap of Policy Optimization and Efficacy and Inference Efficiency with Heuristic Tree-Based Modeling.
33. Hou et al. (2025): TreeRL: LLM Reinforcement Learning with On-Policy Tree Search.
34. Yang et al. (2025): TreeRPO: Tree Relative Policy Optimization.
35. Ji et al. (2025): Tree Search for LLM Agent Reinforcement Learning.
36. Tian et al. (2025): SEEA-R1: Tree-Structured Reinforcement Fine-Tuning for Self-Evolving Embodied Agents.
37. Dong et al. (2025): Agentic Reinforced Policy Optimization.
38. Kazemnejad et al. (2025): VinePPO: Refining Credit Assignment in RL Training of LLMs.
39. Guo et al. (2025): Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models.
