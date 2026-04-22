## 3 方法

本节给出 OPTS-TTPO 的完整方法推导。第 3.1 节介绍树轨迹策略优化 TTPO，包括同策略树轨迹的形式化定义、树轨迹策略梯度修正及其无偏性、TreeGAE 的定义与无偏性，以及由此导出的 branch-weighted PPO 目标。第 3.2 节介绍同策略并行树搜索 OPTS，说明如何从回报缺口的分解出发构造重分支准则 OTRC，并给出与训练预算对齐的搜索流程。

### 3.1 TTPO：Tree Trajectory Policy Optimization

#### 3.1.1 同策略树轨迹

设当前策略为 $\pi_\theta$。标准 on-policy rollout 从初始状态分布出发仅生成一条链式轨迹
$$
\tau=(s_0,a_0,s_1,a_1,\ldots),\qquad a_t\sim \pi_\theta(\cdot\mid s_t).
$$
本文考虑更一般的**同策略树轨迹** $\mathcal{T}_{\pi_\theta}$：在一条已采样轨迹的某个中间状态处，算法可以回到该状态并再次从同一策略 $\pi_\theta$ 采样新的 continuation。因而，同一状态之后可能对应多个同策略后缀，整体样本结构不再是单链，而是一棵由根轨迹及若干重分支 continuation 组成的树。

形式上，我们把树中的每一个节点记为一个**节点出现** $x=(s,a)$，表示某条根到叶路径上的一个具体状态-动作对；之所以强调“节点出现”，是因为相同的状态值可能在不同深度或不同分支上重复出现。记 $\operatorname{Anc}(x)$ 为从根到 $x$ 的路径上所有祖先分支点的集合，$m(u)$ 为分支点 $u$ 的 continuation 数。若一棵树的根 rollout 被重复采样 $M_{\mathrm{root}}$ 次，则该树对节点 $x$ 的采样放大量定义为
$$
W(x)\;=\;M_{\mathrm{root}}\prod_{u\in\operatorname{Anc}(x)} m(u).
$$
该量刻画了树采样机制相对于标准链式采样对节点 $x$ 的期望重复计数倍数。直观地，$x$ 之前经历的分支点越多、这些分支点的 continuation 数越大，则 $x$ 及其后续片段在训练样本中被过采样的程度越高。

分支权重可以沿树自顶向下递推计算。设根轨迹上节点的初始权重为 $M_{\mathrm{root}}$。若节点 $y$ 的父节点不是分支点，则 $W(y)=W(\operatorname{parent}(y))$；若其父节点是分支点 $u$，且 $u$ 处实际生成了 $m(u)$ 个 continuation，则
$$
W(y)=W(u)\,m(u).
$$
换言之，每穿过一个分支点，就把其所有后代节点的权重再乘以该分支点的 continuation 数。举例而言，若根轨迹仅采样一次，某节点 $x$ 之前依次经过两个分支点，分别产生 $3$ 个和 $2$ 个 continuation，则该节点的分支权重为 $W(x)=1\times 3\times 2=6$；这意味着在树采样下，$x$ 所在后缀的经验出现次数相对于自然 rollout 被放大了 $6$ 倍，因此在梯度估计中必须以 $1/6$ 进行校正。

在实现上，离散/连续控制通过环境状态的 snapshot / restore 实现回溯，LLM 推理通过共享前缀并重新生成后缀实现回溯；两种场景都对应于同一个抽象，即“从同一策略条件状态出发并行采样多个 continuation”。

#### 3.1.2 树轨迹策略梯度修正及其无偏性

链式 on-policy 轨迹上的策略梯度写作
$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim \pi_\theta}\!\left[\sum_t \hat A_t \nabla_\theta\log \pi_\theta(a_t\mid s_t)\right].
$$
如果直接把树上的所有节点当作等权样本，则位于高分支区域的后缀会被重复计数，从而改变经验样本分布并引入偏差。TTPO 的核心思想是：不修改策略梯度的基本形式，而是对树节点按其采样放大量的倒数进行校正。

**定理 1（树轨迹加权恒等式）。** 设 $f$ 为任意可积节点函数，$\tau\sim \pi_\theta$ 为链式 on-policy 轨迹，$\mathcal{T}_{\pi_\theta}$ 为由同一策略生成的树轨迹，且每个分支点的 continuation 在条件于该分支状态时独立采样。则有

$$
\mathbb{E}_{\tau\sim \pi_\theta}\!\left[\sum_t f(s_t,a_t)\right]
=\mathbb{E}_{\mathcal{T}_{\pi_\theta}}\!\left[\sum_{x\in\mathcal{T}_{\pi_\theta}} \frac{f(x)}{W(x)}\right].
$$

**证明。** 对于链式轨迹中的任意位置 $(\tau,t)$，记 $x_t(\tau)$ 为该位置对应的节点出现，$N\!\left(x_t(\tau);\mathcal{T}_{\pi_\theta}\right)$ 为该节点出现在树中被复制的总次数。条件于链式轨迹 $\tau$，每个祖先分支点 $u\in\operatorname{Anc}(x_t(\tau))$ 都会把其后缀独立复制 $m(u)$ 次，而根 rollout 被复制 $M_{\mathrm{root}}$ 次，因此
$$
\mathbb{E}\!\left[N\!\left(x_t(\tau);\mathcal{T}_{\pi_\theta}\right)\mid \tau\right] = M_{\mathrm{root}}\prod_{u\in\operatorname{Anc}(x_t(\tau))} m(u) = W\!\left(x_t(\tau)\right).
$$
利用全期望公式与求和交换，可得
$$
\begin{aligned} \mathbb{E}_{\mathcal{T}_{\pi_\theta}}\!\left[\sum_{x\in\mathcal{T}_{\pi_\theta}}\frac{f(x)}{W(x)}\right] &= \mathbb{E}_{\tau\sim \pi_\theta}\!\left[\mathbb{E}\!\left[\sum_t \frac{N(x_t(\tau);\mathcal{T}_{\pi_\theta})}{W(x_t(\tau))}f(x_t(\tau)) \;\middle|\; \tau\right]\right] \\ &= \mathbb{E}_{\tau\sim \pi_\theta}\!\left[\sum_t \frac{\mathbb{E}[N(x_t(\tau);\mathcal{T}_{\pi_\theta})\mid\tau]}{W(x_t(\tau))} f(x_t(\tau))\right] \\ &= \mathbb{E}_{\tau\sim \pi_\theta}\!\left[\sum_t f(x_t(\tau))\right]. \end{aligned}
$$
证毕。 $\square$

将定理 1 应用于
$$
f(s_t,a_t)=\hat A_t \nabla_\theta \log \pi_\theta(a_t\mid s_t),
$$
即可得到树轨迹上的无偏策略梯度校正。

**推论 1（TTPO 的无偏策略梯度）。**
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\mathcal{T}_{\pi_\theta}}\!\left[
\sum_{x\in\mathcal{T}_{\pi_\theta}}
\frac{1}{W(x)}
\hat A(x)\nabla_\theta\log \pi_\theta(a\mid s)
\right].
$$
因此，$1/W(x)$ 不是启发式重加权，而是由树采样分布与链式 on-policy 分布之间的期望计数关系直接推出的校正因子。

#### 3.1.3 TreeGAE 及其无偏性

标准 GAE 递推
$$
\hat A_t^{\mathrm{GAE}} = \delta_t+\gamma\lambda \hat A_{t+1}^{\mathrm{GAE}}
$$
默认每个位置只有唯一 continuation。树轨迹中这一假设不再成立：从某个节点 $x$ 出发，下一步可能对应多个同策略 continuation。为此，我们把链式 GAE 推广到树结构。

记 $\operatorname{Ch}(x)$ 为节点 $x$ 之后所有 continuation 的首节点集合；若 $x$ 为叶节点，则 $\operatorname{Ch}(x)=\varnothing$。给定 critic $V_\phi$，定义节点 $x$ 的一步 TD 残差为
$$
\delta^\phi(x)=r(x)+\gamma V_\phi(s'(x)) - V_\phi(s(x)),
$$
其中 $s'(x)$ 为执行节点 $x$ 对应动作后的下一状态。进一步定义由 $V_\phi$ 诱导的**树 $\lambda$-优势目标**
$$
A^{\lambda,\phi}(x)=\delta^\phi(x)+\gamma\lambda\,\mathbb{E}_{C\sim p(\cdot\mid x)}\!\left[A^{\lambda,\phi}(C)\right],
$$
其中 $p(\cdot\mid x)$ 表示从节点 $x$ 继续按当前策略采样 continuation 所得到的条件分布；若 $x$ 为叶节点，则 $A^{\lambda,\phi}(x)=\delta^\phi(x)$。这个定义表明，TreeGAE 要估计的并不是“真实优势本身”，而是由当前 critic 基线诱导的 tree-$\lambda$ target；只有当 $V_\phi=V^\pi$ 时，它才退化为对真实 tree-$\lambda$ advantage 的估计。

据此，我们定义 TreeGAE 估计量：
$$
\hat A^{\mathrm{TreeGAE}}(x)=\delta^\phi(x)+\gamma\lambda\,\frac{1}{|\operatorname{Ch}(x)|}\sum_{c\in \operatorname{Ch}(x)}\hat A^{\mathrm{TreeGAE}}(c),
$$
并约定叶节点处
$$
\hat A^{\mathrm{TreeGAE}}(x)=\delta^\phi(x).
$$
当 $|\operatorname{Ch}(x)|=1$ 时，上式恰好退化为标准 GAE；当 $|\operatorname{Ch}(x)|>1$ 时，则对所有子 continuation 的首节点优势取算术平均后再向上回传。

**定理 2（TreeGAE 的无偏性）。** 假设对任意节点 $x$，其子 continuation $\operatorname{Ch}(x)$ 条件于 $x$ 独立同分布地来自同一 on-policy continuation 分布 $p(\cdot\mid x)$。则对任意节点 $x$，
$$
\mathbb{E}\!\left[\hat A^{\mathrm{TreeGAE}}(x)\mid x\right] = A^{\lambda,\phi}(x).
$$

**证明。** 对树深度作反向归纳。若 $x$ 为叶节点，则
$$
\hat A^{\mathrm{TreeGAE}}(x)=\delta^\phi(x)=A^{\lambda,\phi}(x),
$$
结论显然成立。假设对所有比 $x$ 更深的节点 $c\in \operatorname{Ch}(x)$，已有
$$
\mathbb{E}\!\left[\hat A^{\mathrm{TreeGAE}}(c)\mid c\right]=A^{\lambda,\phi}(c).
$$
则由 TreeGAE 定义与条件独立同分布假设，
$$
\begin{aligned} \mathbb{E}\!\left[\hat A^{\mathrm{TreeGAE}}(x)\mid x\right] &= \delta^\phi(x)+\gamma\lambda\,\mathbb{E}\!\left[\frac{1}{|\operatorname{Ch}(x)|}\sum_{c\in \operatorname{Ch}(x)} \hat A^{\mathrm{TreeGAE}}(c)\;\middle|\; x\right] \\ &= \delta^\phi(x)+\gamma\lambda\,\frac{1}{|\operatorname{Ch}(x)|}\sum_{c\in \operatorname{Ch}(x)}\mathbb{E}\!\left[\mathbb{E}\!\left[\hat A^{\mathrm{TreeGAE}}(c)\mid c\right]\middle|\; x\right] \\ &= \delta^\phi(x)+\gamma\lambda\,\frac{1}{|\operatorname{Ch}(x)|}\sum_{c\in \operatorname{Ch}(x)}\mathbb{E}\!\left[A^{\lambda,\phi}(c)\mid x\right] \\ &= \delta^\phi(x)+\gamma\lambda\,\mathbb{E}_{C\sim p(\cdot\mid x)}\!\left[A^{\lambda,\phi}(C)\right] \\ &= A^{\lambda,\phi}(x). \end{aligned}
$$
归纳完成。 $\square$

定理 2 说明，TreeGAE 的算术平均并非启发式“平摊”处理，而是对子 continuation 条件期望的无偏 Monte Carlo 估计。因此，只要所有 continuation 都来自相同的 on-policy 条件分布，TreeGAE 就是标准 GAE 在树结构上的自然推广。

#### 3.1.4 Branch-weighted PPO 目标

在树轨迹上得到无偏的优势估计之后，PPO 风格的 actor 与 critic 更新只需在样本测度上与第 3.1.2 节的校正保持一致即可。记
$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}, \qquad W_t=W(s_t,a_t),
$$
则 TTPO 的策略损失为
$$
\mathcal{L}^{\mathrm{TTPO}}_{\pi}=\frac{
\sum_t \frac{1}{W_t}
\min\!\bigl(
r_t(\theta)\hat A_t,\;
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\bigr)
}{
\sum_t \frac{1}{W_t}
}.
$$
相应地，价值函数损失写为
$$
\mathcal{L}^{\mathrm{TTPO}}_{V}=\frac{
\sum_t \frac{1}{W_t}
\max\!\bigl[
(V_\phi(s_t)-\hat R_t)^2,\;
(V_\phi^{\mathrm{clip}}(s_t)-\hat R_t)^2
\bigr]
}{
\sum_t \frac{1}{W_t}
}.
$$
同样的权重也用于优势归一化。设
$$
\mu_A=\frac{\sum_t \hat A_t/W_t}{\sum_t 1/W_t}, \qquad \sigma_A^2=\frac{\sum_t (\hat A_t-\mu_A)^2/W_t}{\sum_t 1/W_t},
$$
则标准化优势定义为
$$
\tilde A_t=\frac{\hat A_t-\mu_A}{\sqrt{\sigma_A^2+\varepsilon}}.
$$
这一归一化与策略损失使用同一加权测度，从而避免高分支区域在 whiten 过程中再次被隐式放大。

### 3.2 OPTS：On-policy Parallel Tree Search

TTPO 回答“如何从树样本更新策略”，OPTS 回答“树应当在哪里继续生长”。与面向策略投影的宽度式 MCTS 不同，OPTS 是一个面向策略梯度的并行重分支机制：每一轮搜索都在每棵尚未搜满的树上选择一个最值得继续扩展的位置，再从该位置的父节点重新采样新的 continuation。

#### 3.2.1 贪心参考路径

给定当前树轨迹，我们首先在每棵树中定义一条**贪心参考路径**。从根节点出发，若当前节点存在多个 continuation，则选择首节点 TreeGAE 优势最大的那个 continuation 继续前进，直到叶节点为止。该路径并不试图恢复全局最优解，而是把搜索集中在当前样本证据下最有希望的后缀上，从而把“在哪里分支”转化为“在这条参考路径的哪个位置重分支”。

#### 3.2.2 OTRC 的开发项

设当前贪心参考路径为
$$
(s_0,a_0),(s_1,a_1),\ldots,(s_{n-1},a_{n-1}).
$$
若在位置 $k$ 之后继续搜索，一个自然的问题是：原路径从 $k$ 开始的后缀相对于一个“更好的 continuation”究竟还存在多大改进空间。为刻画这一量，我们考虑后缀回报缺口
$$
\Delta_k(V):=V(s_k)-G_k,
\qquad
G_k=\sum_{t=k}^{n-1}\gamma^{t-k}r_t,
$$
其中 $V$ 是任意基线函数。

**引理 1（回报缺口的残差分解）。** 对任意基线函数 $V$，令
$$
\delta_t^{V}=r_t+\gamma V(s_{t+1})-V(s_t),
$$
则有
$$
\Delta_k(V) = -\sum_{t=k}^{n-1}\gamma^{t-k}\delta_t^{V}
+\gamma^{n-k}V(s_n).
$$

**证明。** 直接展开残差和：
$$
\begin{aligned} -\sum_{t=k}^{n-1}\gamma^{t-k}\delta_t^V &= -\sum_{t=k}^{n-1}\gamma^{t-k}r_t-\sum_{t=k}^{n-1}\gamma^{t-k+1}V(s_{t+1})+\sum_{t=k}^{n-1}\gamma^{t-k}V(s_t) \\ &= -G_k-\sum_{t=k+1}^{n}\gamma^{t-k}V(s_t)+\sum_{t=k}^{n-1}\gamma^{t-k}V(s_t) \\ &= -G_k + V(s_k)-\gamma^{n-k}V(s_n). \end{aligned}
$$
移项即得结论。 $\square$

引理 1 表明，决定某一后缀是否值得重分支的关键量不是整条后缀的绝对回报，而是其局部残差沿路径的累计。关键之处在于，**OTRC 并不是把 critic 直接当作真实值函数来使用**；相反，它在残差层面工作，把优势估计作为局部残差的低方差代理。

具体地，令 $\hat A_t$ 为第 3.1.3 节定义的 TreeGAE 估计量，并定义开发项
$$
E_k:=-\sum_{t=k}^{n-1}\gamma^{t-k}\hat A_t.
$$
这里 $\hat A_t$ 不是通过“假设 $V_\phi=V^\pi$”得到的，而是直接作为残差序列的估计量使用：当 $\lambda=0$ 时，TreeGAE 退化为一步 TD 残差 $\delta_t^\phi$；当 $\lambda>0$ 时，它对同一残差序列进行指数平滑，从而以较低方差保留后缀质量信息。

**命题 1（开发项的期望形式）。** 设 $\hat A_t$ 由 TreeGAE 给出，则
$$
\mathbb{E}[E_k] = -\sum_{t=k}^{n-1}\gamma^{t-k}A^{\lambda,\phi}_t,
$$
其中 $A^{\lambda,\phi}_t$ 表示当前参考路径第 $t$ 个节点对应的 tree-$\lambda$ target。特别地，当 $\lambda=0$ 时，
$$
\mathbb{E}[E_k] = -\sum_{t=k}^{n-1}\gamma^{t-k}\delta_t^\phi,
$$
与引理 1 中的 TD 残差累计项一致。

**证明。** 由定理 2 与期望的线性性，
$$
\mathbb{E}[E_k] = -\sum_{t=k}^{n-1}\gamma^{t-k}\mathbb{E}[\hat A_t] = -\sum_{t=k}^{n-1}\gamma^{t-k}A_t^{\lambda,\phi}.
$$
当 $\lambda=0$ 时，$A_t^{0,\phi}=\delta_t^\phi$，因此退化为一步 TD 残差的折扣累加。 $\square$

命题 1 解释了 OTRC 中“用优势替换 $\delta_t$”的含义：替换发生在**局部残差估计层面**，而不是通过把 $V_\phi$ 视作真实值函数 $V^\pi$ 来完成。这样做的好处是，OTRC 可以直接复用训练时已经计算好的优势估计，并获得比原始一步 TD 残差更稳定的分支信号。

#### 3.2.3 OTRC 的探索项与重分支分数

开发项 $E_k$ 只衡量“从位置 $k$ 继续搜索可能带来多大改善”，尚未惩罚那些已经被反复扩展的位置。为此，我们引入 sibling-count 驱动的探索项
$$
U_k=(B_k-1)\max_j |E_j|,
$$
其中 $B_k$ 表示参考路径上位置 $k$ 的父节点已经拥有的 continuation 数。$\max_j |E_j|$ 仅用于把探索项缩放到与开发项相近的量级，从而减少超参数在不同任务上的敏感性。

于是，同策略轨迹重分支准则定义为
$$
\mathrm{OTRC}_k = E_k - c\,U_k,
$$
其中 $c>0$ 为探索系数。对每棵树，我们在当前贪心参考路径上选择
$$
k^\star=\arg\max_k \mathrm{OTRC}_k
$$
作为最优重分支位置；随后从 $\operatorname{parent}(k^\star)$ 重新采样动作并生成新的 continuation。之所以从父节点而非 $k^\star$ 本身分支，是因为“重分支”的语义是：在到达 $k^\star$ 之前的最后一个决策点重新采样不同动作。

#### 3.2.4 预算对齐与长度惩罚

对于 LLM 推理，预算按完整 episode 或完整回答计量，因此 OTRC 直接使用 $E_k$ 即可。对于 Atari 和 MuJoCo 这类 action-level budget 任务，更深的重分支意味着真实消耗更多环境交互步数；因而我们在开发项上加入长度惩罚，得到
$$
E_k^{(\tau)} = \frac{-\sum_{t=k}^{n-1}\gamma^{t-k}\hat A_t}{(n-k)^\tau},
\qquad \tau\ge 0.
$$
其中 $\tau=0$ 对应 episode-level budget，$\tau>0$ 则鼓励在相同预算下优先选择更具单位成本收益的重分支位置。需要强调的是，这一长度惩罚是对开发项的预算化延拓，而不是一个与 OTRC 无关的额外启发项。

#### 3.2.5 训练流程与跨域实例化

综合 TTPO 与 OPTS，单次训练 step 可概括为如下流程。

1. 进行 $R$ 轮搜索 rollout。第一轮从初始状态分布或 prompt 集合采样新的根轨迹；后续轮次对每棵仍继续搜索的树，根据 OTRC 选出的 $\operatorname{parent}(k^\star)$ 并行生成新 continuation。
2. 将本轮新采样得到的节点并入已有树结构，并更新相应的 parent-child 关系、分支位置和 continuation 数。
3. 在合并后的树上自底向上计算 TreeGAE；非分支节点执行标准 GAE 递推，分支节点对所有子 continuation 首节点的优势取平均后再回传。
4. 根据第 3.1.2 节定义的 $W(x)$ 计算 branch weight，并按第 3.1.4 节构造 branch-weighted actor/critic 损失。
5. 使用 $\mathcal{L}_V^{\mathrm{TTPO}}$ 更新 critic，再使用 $\mathcal{L}_\pi^{\mathrm{TTPO}}$ 更新 actor。

三类任务共享上述抽象，但实例化方式不同。LLM 训练中，树通过“共享前缀、扩展后缀”实现；测试时搜索则在固定推理预算下重复执行 OTRC 选点与 continuation 扩展，并输出 `avg@k / pass@k / cons@k` 所需的样本索引。Atari 与 MuJoCo 中，树通过环境状态 snapshot / restore 实现；两者与 LLM 的差别主要体现在预算计量方式和长度惩罚系数 $\tau$，而非方法本身。
