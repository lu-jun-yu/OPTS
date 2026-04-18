## 3.1 问题设定

我们考虑由当前策略 `\pi_\theta` 产生的同策略树轨迹。与标准 rollout 只生成一条链不同，树轨迹允许我们在某个中间状态重新分支，并从该状态继续采样新的后续轨迹。对每个分支样本，我们维护如下元信息：

- `pid`：父轨迹 id。
- `branch_pos`：该轨迹从父轨迹的哪个位置开始分支。
- `cid`：当前轨迹各个位置的子轨迹集合。
- `state_branches`：某个状态处实际产生了多少个后续分支。

在实现上，Atari/MuJoCo 直接对环境状态进行 snapshot 与 restore；LLM 则把“父轨迹前缀 + 新生成后缀”重新拼接成新输入。尽管具体载体不同，这两类实例化都共享同一个抽象：`一个状态之后可以有多个同策略 continuation`。

本文还区分两种预算定义。

- `action-level budget`：Atari 与 MuJoCo 属于此类。一次更深的搜索会多消耗若干环境 action，因此越深的分支点越贵。
- `episode-level budget`：LLM 属于此类。我们按整条回答或整次采样计预算，因此在同一 episode 内更深的位置并不会天然更贵。

预算定义不同，将直接改变 3.4 节中的 TUCT 开发项。

## 3.2 Tree Trajectory Policy Gradient

在链式轨迹中，策略梯度写作

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
\sum_t
\hat A_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)
\right].
$$

树轨迹的关键变化在于：同一个祖先状态之后可能出现多个 continuation。如果我们简单把树上的所有状态-动作对等权平均，那么位于高分支区域的样本会被重复计数，从而偏离原始策略分布。为此，我们给每个状态-动作对定义 branch weight

$$
W(s_t,a_t)
=
M_{\text{root}}
\prod_{u \in \text{Anc}(s_t,a_t)} m(u),
$$

其中 `m(u)` 表示祖先分支点 `u` 处的分支数，`M_{\text{root}}` 表示同一棵树在根部被采样了多少次。于是，树轨迹上的策略梯度可以写成

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{(s_t,a_t)\sim \mathcal{T}_{\pi_\theta}}
\left[
\frac{1}{W(s_t,a_t)}
\hat A(s_t,a_t)
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
\right].
$$

该式给出了 TTPO 的核心思想：树上出现次数越多的状态-动作对，其单次样本权重就应越小。代码中 `compute_branch_weight(...)` 正是在所有域上实现了这一修正。

### 3.2.1 Branch-Weighted PPO Loss

令

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}.
$$

我们将 PPO surrogate 写为

$$
\mathcal{L}_{\pi}^{\text{TTPO}}
=
\frac{
\sum_t
\frac{1}{W_t}
\min
\left(
r_t(\theta)\hat A_t,\;
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
}{
\sum_t \frac{1}{W_t}
}.
$$

价值函数损失同样使用 `1 / W_t` 加权：

$$
\mathcal{L}_{V}^{\text{TTPO}}
=
\frac{
\sum_t \frac{1}{W_t}
\max\left[
(V_\phi(s_t)-\hat R_t)^2,\;
(V_\phi^{\text{clip}}(s_t)-\hat R_t)^2
\right]
}{
\sum_t \frac{1}{W_t}
}.
$$

这与代码中的 `agg_loss(..., branch_weight=...)` 和 `compute_value_loss(..., branch_weight=...)` 完全一致。

### 3.2.2 Branch-Weighted Advantage Whitening

由于树轨迹中不同 token 或 time step 的有效样本数并不相同，我们对优势也采用加权归一化：

$$
\mu_A
=
\frac{\sum_t \hat A_t / W_t}{\sum_t 1/W_t},
\qquad
\sigma_A^2
=
\frac{\sum_t (\hat A_t-\mu_A)^2 / W_t}{\sum_t 1/W_t},
$$

$$
\tilde A_t
=
\frac{\hat A_t-\mu_A}{\sqrt{\sigma_A^2+\varepsilon}}.
$$

LLM 实现中的 `weighted_masked_whiten(...)` 采用的正是这一定义。

## 3.3 TreeGAE

令标准 TD 误差为

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
$$

在链式轨迹中，GAE 的递推是

$$
\hat A_t^{\text{GAE}}
=
\delta_t + \gamma \lambda \hat A_{t+1}^{\text{GAE}}.
$$

树轨迹的核心区别在于，`t` 之后不再只有一个 continuation，而是一个 continuation 集合 `\mathcal{C}(t)`。因此，我们把 GAE 推广为

$$
\hat A_t^{\text{TreeGAE}}
=
\delta_t
\;+\;
\gamma \lambda
\cdot
\frac{1}{|\mathcal{C}(t)|}
\sum_{c \in \mathcal{C}(t)}
\hat A_c^{\text{TreeGAE}}.
$$

这里的 `\mathcal{C}(t)` 包括“沿原轨迹继续前进”的后继，也包括从该位置重新分支后产生的后继。如果某个状态没有分支，那么 TreeGAE 就退化为标准 GAE。

这一定义在 Atari、MuJoCo 与 LLM 三个场景中是完全一致的：branch node 都要把 continuation 与所有子分支对应的后续优势做平均。差别只在实现方式。Atari/MuJoCo 直接沿父指针从 terminal leaf 向上回传；LLM 为了适配 round-based 全局 batch 合并与增量更新，采用了对受新样本影响子图的反向扫描，并用类似 `child_first_adv_sum` 的缓存来维护分支均值所需的子分支首 token 优势之和。换言之，LLM 版 TreeGAE 不是另一套定义，而是同一递推在大批量树轨迹上的工程化实现。

## 3.4 On-Policy Tree Search

TTPO 解决的是“如何学”；OPTS 解决的是“树应当怎样长”。我们的搜索不是传统 MCTS，而是一个面向策略梯度的深度式选点过程。

### 3.4.1 Greedy Optimal Path

在每棵树里，我们先抽取一条当前最优路径：

1. 从根节点开始；
2. 若存在多个子 continuation，则选择优势最大的那个；
3. 重复上述过程直到叶节点。

这样做的目的是把搜索集中在“当前看来最有希望的一条路径”上，而不是在整棵树上做广度展开。它与代码中的 `select_next_states(...)` 完全一致。

### 3.4.2 TUCT 开发项与探索项

设当前最优路径上的优势序列为 `A_k, A_{k+1}, \dots, A_{n-1}`。我们定义从位置 `k` 开始重新分支所对应的开发项

$$
E_k
=
\frac{
-\sum_{t=k}^{n-1}\gamma^{t-k} A_t
}{
(n-k)^\tau
}.
$$

其中分母只在 `action-level budget` 下启用：

- 对 Atari/MuJoCo，`\tau > 0`，对应代码中的 `discounted_sum / ((n-k) ** tau)`。这就是本文所说的长度惩罚。
- 对 LLM，`\tau = 0`，即不做长度惩罚，开发项退化为原始折扣累计优势。这与代码中的 `no length normalization for LLM` 一致。

直观地看，`-A_t` 越大，说明从当前位置往后的已有决策越差，重新分支的潜在改进越大；长度惩罚则用来抵消更深位置在 action budget 下的额外代价。

探索项定义为

$$
U_k = (B_k - 1)\max_j |E_j|,
$$

其中 `B_k` 表示该位置父节点处已经拥有的兄弟分支数。最终的 TUCT 分数为

$$
\text{TUCT}_k = E_k - c U_k.
$$

我们在路径上选择 TUCT 最大的位置，然后从其`父节点`重新分支，而不是直接从该节点分支。代码中的 `selected_to_branch_points(...)` 明确实现了这一点。

### 3.4.3 预算约束与实现启发式

除基本公式外，代码里还实现了若干对训练稳定性很重要的约束：

- 每棵树最多搜索 `max_search_per_tree` 次。
- 只保留那些“最大开发值高于当前均值阈值”的候选树。
- LLM 中还要求分支点满足 prompt 长度约束，并位于 `</think>` 之前。
- 如果没有合格候选，则直接开启新树，而不是强制继续搜索旧树。

这些启发式并不改变 TTPO 的加权目标，但能显著减少无效分支。

## 3.5 跨域实例化

### 3.5.1 Atari 与 MuJoCo

Atari 与 MuJoCo 共享相同的宏观流程：

1. 按 PPO 方式采样一段 rollout。
2. 一旦某个环境终止，就对该环境对应的树执行 TreeGAE 回传。
3. 用 TUCT 选出新的分支点。
4. 通过 snapshot/restore 回到被选中的父状态，继续向前采样。
5. 迭代结束后，按 branch weight 修正策略损失和价值损失。

两者的差别主要在环境封装层：Atari 需要保存 ALE 状态、FrameStack、EpisodicLife 等 wrapper 内部状态；MuJoCo 需要保存物理状态、派生量以及归一化器统计量。

### 3.5.2 LLM 训练

LLM 训练使用 VeRL/Ray 的 round-based 流程。每一轮生成后，我们为样本维护 `rid/pid/branch_pos/cid` 等树结构信息，并把不同轮次的数据合并到 `global_batch` 中。训练更新前，再统一计算 `branch_weight`、加权优势白化以及 branch-weight 修正的 actor/critic 损失。

需要强调的是：LLM 部分并不存在一套不同于 Atari/MuJoCo 的 TreeGAE 定义；三者共享同一个树优势递推。LLM 的特殊之处只在于其实现必须兼容多轮生成、全局 batch 合并、变长前缀对齐以及增量式子图更新。因此，本文在方法上把它视为“同一 TreeGAE 的另一种实现”，而不是单独的算法变体。

### 3.5.3 LLM 测试时搜索

在 `main_opts_generation.py` 中，我们把总推理预算设为

$$
\text{total steps}
=
\left\lceil
\frac{\text{dataset size} \times n_{\text{samples}}}{\text{batch size}}
\right\rceil,
$$

从而与 `pass@k` 的采样成本对齐。算法输出每个响应的 `sample_index`，因此可以离线绘制 `avg@k`、`pass@k` 与 `cons@k` 曲线。这一设计使得 OPTS 能作为一个 test-time scaling 框架，而不只是一个训练算法。

## 3.6 方法小结

`OPTS-TTPO` 的核心并不是“把 MCTS 搬到 PPO 上”，而是重新回答两个更基本的问题：

1. 如果数据来自同策略树轨迹，策略梯度该如何无偏加权？
2. 如果搜索是为策略梯度服务的，分支点又该如何在固定预算下被选择？

TTPO 解决第一个问题，OPTS 解决第二个问题，而 `action-level` 与 `episode-level` 的预算区分则把二者真正统一到了同一个框架之中。
