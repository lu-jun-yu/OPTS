## 2 相关工作

### 2.1 Search + Policy Projection

Search + Policy Projection 先通过搜索得到改进后的局部决策分布，再将该分布投影回策略网络。MCTS/UCT 提供了这一路线的基本搜索机制 (Kocsis and Szepesvári, 2006; Browne et al., 2012)，AlphaGo Zero、AlphaZero、Expert Iteration 与 MuZero 将其发展为“搜索改进策略、网络拟合搜索”的自博弈学习范式 (Silver et al., 2017, 2018; Anthony et al., 2017; Schrittwieser et al., 2020)，后续 Gumbel AlphaZero / MuZero 与 Sampled MuZero 进一步通过采样动作集合缓解大动作空间中的规划成本 (Danihelka et al., 2022; Hubert et al., 2021)。这些方法证明了搜索作为 policy improvement operator 的强大能力，但其学习目标通常依赖 MCTS 访问分布或搜索策略作为投影目标，更适合中小规模离散动作空间；相比之下，本文不拟合搜索分布，而是在同策略树轨迹上直接构造策略梯度估计。

### 2.2 Rollout + Policy Gradient

Rollout + Policy Gradient 直接从采样轨迹估计策略梯度，不需要枚举动作空间。REINFORCE 和 policy gradient theorem 给出了基本估计形式 (Williams, 1992; Sutton et al., 2000)，TRPO、GAE 与 PPO 分别从信赖域约束、优势估计和 clipped surrogate objective 的角度提高更新稳定性 (Schulman et al., 2015, 2016, 2017)。在 LLM 训练中，InstructGPT、DeepSeekMath、DeepSeek-R1、DAPO、REINFORCE++、GPG 以及 RLOO/ReMax 等工作将 PPO/GRPO 或 critic-free policy gradient 用于人类偏好奖励与可验证奖励训练，说明链式 rollout 上的策略梯度可以显著提升模型对齐与推理能力 (Ouyang et al., 2022; Shao et al., 2024; DeepSeek-AI et al., 2025; Yu et al., 2025; Hu et al., 2025; Chu et al., 2025; Ahmadian et al., 2024; Li et al., 2023)。然而，这一路线通常从初始 prompt 或初始状态生成独立完整轨迹，不能回到中间状态补充 continuation；本文继承 PPO/RLVR 的更新接口，但将链式 rollout 扩展为同策略树轨迹，并用 branch weight、TreeGAE 与加权优势白化修正树采样带来的样本测度变化。

### 2.3 Search + Policy Gradient

Search + Policy Gradient 试图将搜索的局部探索能力与策略梯度的大动作空间适配性结合起来。LLM 推理中的 self-consistency、Tree of Thoughts、RAP 与 verifier/process-supervision 表明，多路径采样、树式推理和过程级反馈可以提高推理质量 (Wang et al., 2022; Yao et al., 2023; Hao et al., 2023; Cobbe et al., 2021; Lightman et al., 2023)，但这些方法多用于测试时解码、重排序或过程奖励建模。更接近本文的是近期将树搜索或中间状态重采样纳入 RL/RLVR 的工作，例如 rStar-Math、TreePO、TreeRL、TreeRPO、SEEA-R1、ARPO、Search-R1 与 R1-Searcher (Guan et al., 2025; Li et al., 2025; Hou et al., 2025; Yang et al., 2025; Tian et al., 2025; Dong et al., 2025; Jin et al., 2025; Song et al., 2025)。这些方法突破了链式 rollout 的探索限制，但通常更关注如何产生树样本或过程奖励；本文则关注树样本如何进入 on-policy 策略梯度目标：TTPO 用 $1/W(x)$ 修正分支区域的重复计数，TreeGAE 将 GAE 推广到多 continuation 的条件平均，OTRC 从 TD 残差分解出发选择重分支位置，从而形成与 PPO/RLVR 工程体系兼容的树轨迹优化框架。

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
10. Sutton et al. (2000): Policy Gradient Methods for Reinforcement Learning with Function Approximation.
11. Schulman et al. (2015): Trust Region Policy Optimization.
12. Schulman et al. (2016): High-Dimensional Continuous Control Using Generalized Advantage Estimation.
13. Schulman et al. (2017): Proximal Policy Optimization Algorithms.
14. Ouyang et al. (2022): Training Language Models to Follow Instructions with Human Feedback.
15. Shao et al. (2024): DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.
16. DeepSeek-AI et al. (2025): DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.
17. Yu et al. (2025): DAPO: An Open-Source LLM Reinforcement Learning System at Scale.
18. Hu et al. (2025): REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization.
19. Chu et al. (2025): GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning.
20. Ahmadian et al. (2024): Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs.
21. Li et al. (2023): ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models.
22. Wang et al. (2022): Self-Consistency Improves Chain of Thought Reasoning in Language Models.
23. Yao et al. (2023): Tree of Thoughts: Deliberate Problem Solving with Large Language Models.
24. Hao et al. (2023): Reasoning with Language Model is Planning with World Model.
25. Cobbe et al. (2021): Training Verifiers to Solve Math Word Problems.
26. Lightman et al. (2023): Let's Verify Step by Step.
27. Guan et al. (2025): rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking.
28. Li et al. (2025): TreePO: Bridging the Gap of Policy Optimization and Efficacy and Inference Efficiency with Heuristic Tree-Based Modeling.
29. Hou et al. (2025): TreeRL: LLM Reinforcement Learning with On-Policy Tree Search.
30. Yang et al. (2025): TreeRPO: Tree Relative Policy Optimization.
31. Tian et al. (2025): SEEA-R1: Tree-Structured Reinforcement Fine-Tuning for Self-Evolving Embodied Agents.
32. Dong et al. (2025): Agentic Reinforced Policy Optimization.
33. Jin et al. (2025): Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning.
34. Song et al. (2025): R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning.
