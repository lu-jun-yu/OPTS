策略优化方法的发展有两个历史性时刻：一是2017年前后 DeepMind 所完成的 Zero 系列的工作，在各种棋类、游戏领域将AI的能力推向超人水平；二是2025年前后DeepSeek-R1所引出的RLVR训练范式，通过纯粹的强化学习激发LLM的推理能力到人类顶尖水平。其中，涉及两类经典的策略优化方法：

- Search & Policy Projection Methods：以 AlphaZero、MuZero 算法为代表，Agent在每次做出动作前进行海量搜索：PUCT Selection $$\rightarrow$$ Expansion $$\rightarrow$$ Evaluation $$\rightarrow$$ Backup。通过利用后验信息调整先验决策，Zero 类算法实现了比纯策略模型更好的表现。在完成搜索后，Zero类算法将得到的更强的搜索概率分布通过 KL 散度投影到神经网络中，实现策略的持续改进。Search and policy projection methods 适用于中小规模离散动作空间，可以显式建模下一个动作完整概率分布的场景，但因为需要建模全概率分布而难以应对动作空间规模进一步扩大的情况。

- Rollout & Policy Gradient Methods：以 PPO、GRPO 算法为代表，策略模型进行大量的 rollout，探索更多的环境奖励信息，随后通过策略梯度方法提升其产生高奖励路径的概率，实现策略期望意义上的持续改进。在这之中，价值模型的作用主要是通过预测未来的期望回报，降低策略梯度的方差，从而稳定RL训练。Rollout and policy gradient methods 在面对大规模、超大规模离散动作空间与连续动作空间的场景时更有优势，但在训练时只进行探索式的 rollout。

As Rich Sutton emphasized in The Bitter Lesson, search and learn are the two methods that scale arbitrarily with computation. AlphaGo Zero 通过 MCTS 将策略性能从 Elo 3000 提升至 5000+，证明了搜索的强大力量。这就引出了一个关键的问题：既然搜索如此强大，为什么当我们使用 policy gradient methods 学习策略时，却放弃了搜索？

核心阻碍有三点：

1. 搜索会造成严重的样本分布偏移，无法简单通过重要性采样和 Clip 修正。由于搜索是后验调整先验的过程，随着搜索次数的增加，样本分布逐渐偏离策略分布，数据将无法避免地走向 off-policy。如果强行使用 on-policy 算法学习 off-policy 的数据，将引入偏差；如果使用重要性采样和 Clip 修正，则会面临无法获得准确的搜索概率以及方差爆炸的问题。

2. 经典搜索算法是 GPU 并行不友好的。经典的类 MCTS 算法通常是 CPU 密集且串行的，这与现代策略梯度方法所依赖的大规模 GPU 并行 Rollout 架构格格不入。

3. 最重要的一点是，搜索与学习的目标的错位。Policy projection methods 要求拟合完整的下一个动作的概率分布，因此需要配合一种广度上扩展的搜索算法，如 MCTS；而 Policy gradient methods 不同，其不需要拟合下一个动作的概率分布，但更关注正回报轨迹样本的挖掘，因此 policy gradient methods 应该配合深度上扩展的搜索算法，而非广度上扩展的。

如果存在第三类高效的策略优化方法：Search & Policy Gradient Methods，那么其一定满足：On-Policy, Parallel-Friendly, Depth-wise Expansion。

为了满足上述要求，本文提出了一种方法，分为学习算法和搜索算法两部分。学习算法被称为：Tree Trajectory Policy Optimization (TTPO) 算法，通过将策略梯度定理与优势估计扩展至同策略树轨迹中，从而保证了——若 value 是公正的，搜索所产生的样本的梯度是统计上无偏的。搜索算法被称为：On-policy Parallel Tree Search (OPTS) 算法，通过构造 Trajectory Optimal Branch Point (TOBP) 公式，在理论层面保证了分支位置在期望意义上最大化地改进轨迹回报。

总而言之，本文的贡献如下：

1. TTPO: 据我们所知，这是首个在同策略树轨迹下严格证明了梯度无偏性的策略梯度方法。基本组件包括：1) Policy Gradient Theorem on Tree Trajectory：树轨迹的每个状态-动作对的策略梯度，需要在原始策略梯度定理的基础上，除以一个权重因子——祖先轨迹所有分支点的分支数的累乘结果；2) Tree-based Generalized Advantage Estimation (TreeGAE)：树轨迹的每个状态-动作对的广义优势估计为当前状态-动作对的分支的广义优势估计的平均。联合二者，再引入归一化、重要性采样和裁剪，最终得到 TTPO's loss function。

2. OPTS: 这是首个专为策略梯度方法设计的、在深度上扩展的并行树搜索算法。核心组件是 TUCT 状态-动作对选择公式，包括开发项与探索项两部分；算法流程为：TUCT Selection $$\rightarrow$$ Expansion (Rollout) $$\rightarrow$$ Evaluation $$\rightarrow$$ Backup (Compute Advantage)。

3. 在严格控制推理预算的前提下，联合 TTPO 和 OPTS 的方法，分别在小规模离散动作空间（Atari）、大规模离散动作空间（LLM）与连续动作空间（MuJoCo）的场景下超越了 PPO。无奖励形式的 OPTS，可以帮助LLM完成基于搜索的在线决策，在 avg@k、pass@k 和 cons@k 三个指标上均实现了 SOTA 的 test-time scaling。