## 3 方法

本节给出 OPTS-TTPO。第 3.1 节介绍树轨迹策略优化 (TTPO)：分支校正策略梯度、TreeGAE 与 TTPO 目标。第 3.2 节推导同策略并行树搜索 (OPTS) 的重分支准则 OTRC。第 3.3 节整理完整训练流程。

### 3.1 Tree Trajectory Policy Optimization

#### 3.1.1 Branch-corrected Policy Gradient for Tree Trajectories

设当前策略为 $\pi_\theta$。下文把采样结构中的每个状态-动作位置称为节点 $x$，并记其状态、动作与真实优势为 $s_x,a_x,A_x$；在树轨迹中，即便两个节点对应相同的状态-动作对，只要它们位于不同分支中，也视为不同节点。标准 on-policy rollout 生成链式轨迹
$$
\tau=(s_0,a_0,s_1,a_1,\ldots),\qquad a_t\sim \pi_\theta(\cdot\mid s_t),
$$
标准策略梯度定理给出
$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}\!\left[
\sum_{t}A_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].
$$

同策略树轨迹 $\mathcal{T}_{\pi_\theta}$ 允许算法回到已采样节点，并再次从 $\pi_\theta$ 采样新的 continuation，即从该节点继续采样得到的后缀轨迹。令 $\operatorname{Anc}(x)$ 表示从根到 $x$ 之前的祖先节点集合；$m(u)$ 表示节点 $u$ 的 continuation 数，非分支节点 $m(u)=1$，叶子节点 $m(u)=0$。

令 $\tau_u$ 表示 $u$ 之后的严格后缀轨迹，并定义其条件期望梯度贡献为
$$
\mu(u)
:=
\mathbb{E}_{\tau_u\sim \pi_\theta(\cdot\mid u)}
\!\left[
\sum_{x\in\tau_u}A_x\nabla_\theta\log\pi_\theta(a_x\mid s_x)
\middle|u
\right],
$$
其中严格后缀不含 $u$ 本身。若从 $u$ 出发有多条 continuation，记 $G_j(u)=\sum_{x\in\tau_{u,j}}A_x\nabla_\theta\log\pi_\theta(a_x\mid s_x)$，则 $\mathbb{E}[G_j(u)\mid u]=\mu(u)$。

等权求和会放大分支点后的节点的梯度贡献：若 $u$ 后有 $b$ 条同分布 continuation，则 $\mathbb{E}[\sum_{j=1}^{b}G_j(u)\mid u]=b\,\mu(u)$，而在标准链式采样中原本只应贡献 $\mu(u)$。对于每个节点，该放大会随着分支增多，沿祖先分支数累乘。下文用 $W(x)$ 表示节点 $x$ 的分支放大因子，用 $1/W(x)$ 作为分支校正权重。

**基础情形 1：从单链到二叉树**。设链式轨迹在节点 $u$ 后被重新采样一次，得到两条 continuation $\tau_{u,1}$ 与 $\tau_{u,2}$。为保持后缀贡献的条件期望仍为 $\mu(u)$，应对二者取平均，因此二叉树上的梯度估计为
$$
\hat g(\mathcal{T})
=
\sum_{x\in\tau_{\preceq u}}A_x\nabla_\theta\log\pi_\theta(a_x\mid s_x)
+
\frac{1}{2}\sum_{i=1}^{2}G_i(u).
$$
由于两条 continuation 条件于 $u$ 独立同分布（i.i.d.），$\frac{1}{2}(G_1(u)+G_2(u))$ 与分支前的后缀贡献同期望；前缀 $\tau_{\preceq u}$ 不变。因此 $u$ 严格后缀的分支校正权重为 $1/2$，分支放大因子为 $W(x)=2$；前缀节点仍有 $W(x)=1$。

**基础情形 2：已有二叉树再分支**。设当前树已有二叉分支点 $v$，现在在 $u$ 处新增一条 continuation。若 $u$ 位于 $v$ 之前，则旧后缀虽含 $v$ 处二叉子树，但已按基础情形 1 聚合，故旧后缀 $G_1(u)$ 与新后缀 $G_2(u)$ 同期望，在 $u$ 处平均即可。若 $u=v$，则把 $\frac{1}{2}\sum_{j=1}^2G_j(v)$ 更新为 $\frac{1}{3}\sum_{j=1}^3G_j(v)$，条件期望仍为 $\mu(v)$。若 $u$ 位于 $v$ 的某条后缀中，来自 $v$ 的校正权重 $1/2$ 是公共因子，且条件于 $u$，
$$
\mathbb{E}\!\left[
\frac{1}{2}\cdot\frac{1}{2}\left(G_1(u)+G_2(u)\right)
\middle|u\right]=\mathbb{E}\!\left[\frac{1}{2}G_1(u)\middle|u\right].
$$
两个基础情形说明，新增 continuation 时只需在 $u$ 严格后缀上把平均从 $b$ 条更新为 $b+1$ 条，并保留既有上游平均因子，则可以保证策略梯度聚合后无偏。

**归纳假设**。假设当前树 $\mathcal{T}$ 按照上述方式得到的加权策略梯度无偏，且任意节点 $x$ 的分支放大因子为 $W(x)=\prod_{z\in\operatorname{Anc}(x)}m(z)$，校正权重为 $1/W(x)$。

**归纳步骤**。选择任意节点 $u$，设其已有 $b=m(u)\ge 1$ 条 continuation；从 $u$ 再采样一条 on-policy continuation 得 $\mathcal{T}^{+}$。令 $\hat g_{\setminus u}$ 表示不属于 $u$ 严格后缀的贡献；$G_j(u)$ 表示去掉上游权重 $1/W(u)$ 与本层平均 $1/b$ 后的第 $j$ 条后缀贡献，内部更深分支已按归纳假设聚合。于是
$$
\begin{aligned}
\hat g(\mathcal{T})
&=\hat g_{\setminus u}
+\frac{1}{W(u)}\frac{1}{b}\sum_{j=1}^{b}G_j(u),\\
\hat g(\mathcal{T}^{+})
&=\hat g_{\setminus u}
+\frac{1}{W(u)}\frac{1}{b+1}\sum_{j=1}^{b+1}G_j(u).
\end{aligned}
$$
由归纳假设与新 continuation 的 on-policy 采样，$\mathbb{E}[G_j(u)\mid u]=\mu(u),\,j=1,\ldots,b+1$。从而
$$
\mathbb{E}\!\left[
\frac{1}{b+1}\sum_{j=1}^{b+1}G_j(u)
\middle|u\right]
=
\mu(u)
=
\mathbb{E}\!\left[
\frac{1}{b}\sum_{j=1}^{b}G_j(u)
\middle|u\right],
$$
故 $\mathbb{E}[\hat g(\mathcal{T}^{+})\mid u]=\mathbb{E}[\hat g(\mathcal{T})\mid u]$，新树仍满足 $W(x)=\prod_{z\in\operatorname{Anc}(x)}m(z)$。

**定理 1（树轨迹策略梯度）** 设 $\mathcal{T}_{\pi_\theta}$ 由同一策略生成，且每个节点的 continuation 条件于该节点以 i.i.d. 方式按 $\pi_\theta$ 采样。对任意节点定义分支放大因子
$$
W(x)=\prod_{u\in\operatorname{Anc}(x)}m(u),
$$
其中空乘积为 $1$。等价地，$W$ 可自顶向下递推：根节点处的放大因子为 $1$；若 $x$ 的父节点为 $p$，则 $W(x)=W(p)m(p)$。由上述归纳可得同策略树轨迹的无偏梯度估计：
$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\mathcal{T}_{\pi_\theta}}\!\left[
\sum_{x\in\mathcal{T}_{\pi_\theta}}
\frac{1}{W(x)}A_x\nabla_\theta\log\pi_\theta(a_x\mid s_x)
\right].
$$

#### 3.1.2 Tree-based Generalized Advantage Estimation

本节先在理想 value $V^\pi$ 下推导 TreeGAE，再替换为实际 critic $\hat V$。对任意 value $V$，记
$$
\delta^V_x:=r_x+\gamma V(s'_x)-V(s_x),
$$
其中 $r_x$ 为即时奖励，$s'_x$ 为执行动作 $a_x$ 后的下一状态；特别地，$\delta^\pi_x:=\delta^{V^\pi}_x$，$\hat\delta_x:=\delta^{\hat V}_x$。设 $x'\sim\pi_\theta(\cdot\mid x)$ 为环境转移与策略共同诱导的下一个状态-动作节点。链式 GAE 递推为
$$
A^{\mathrm{GAE}}_x
=
\delta^\pi_x+\gamma\lambda A^{\mathrm{GAE}}_{x'}.
$$
若下游递推在期望上给出真实优势，则结合条件期望的传递性与 $V^\pi$ 的性质 $\mathbb{E}[\delta^\pi_x\mid x]=A_x$、$\mathbb{E}_{x'\sim \pi_\theta(\cdot\mid x)}[A_{x'}]=0$，链式 GAE 在期望上保持为真实优势：
$$
\mathbb{E}[A^{\mathrm{GAE}}_x\mid x]
=
A_x+\gamma\lambda\,
\mathbb{E}_{x'\sim\pi_\theta(\cdot\mid x)}[A_{x'}]
=A_x.
$$

若树节点 $x$ 有 $m(x)$ 条 continuation，首节点为 $x'_1,\ldots,x'_{m(x)}$，它们条件于 $x$ 是同一诱导分布的 i.i.d. 样本。用其平均替代链式递推中的唯一下游项，有
$$
\mathbb{E}\!\left[
\delta^\pi_x+\gamma\lambda
\frac{1}{m(x)}\sum_{j=1}^{m(x)} A^{\mathrm{GAE}}_{x'_j}
\middle|x\right]
=
\mathbb{E}\!\left[
\delta^\pi_x+\gamma\lambda A^{\mathrm{GAE}}_{x'}
\middle|x\right]
=A_x.
$$
因此，算术平均是保持 GAE 递推期望一致的树结构替代项。

有限树轨迹上，叶节点递推终止于当前 TD 残差；非叶节点按上述平均方式回传。将 $V^\pi$ 替换为 $\hat V$，得到 TreeGAE：

**定理 2（TreeGAE 递推公式）** 假设任意非叶节点 $x$ 的 continuation 首节点 $x'_1,\ldots,x'_{m(x)}$ 条件于 $x$ 是诱导下一个节点分布中的 i.i.d. 样本，则
$$
\hat A^{\mathrm{TreeGAE}}_x=
\begin{cases}
\hat\delta_x, & x\text{ 为叶节点},\\[2mm]
\hat\delta_x+\gamma\lambda\,\dfrac{1}{m(x)}
\sum_{j=1}^{m(x)}\hat A^{\mathrm{TreeGAE}}_{x'_j},
& x\text{ 不是叶节点}.
\end{cases}
$$

#### 3.1.3 TTPO Objective

令 $\mathcal{B}$ 为当前 batch 的树节点集合，并使用 $1/W(x)$ 作为节点 $x$ 的分支校正权重。记 $\hat A_x=\hat A^{\mathrm{TreeGAE}}_x$，PPO ratio 为
$$
r(\theta)=\frac{\pi_\theta(a_x\mid s_x)}{\pi_{\theta_{\mathrm{old}}}(a_x\mid s_x)},
$$
TTPO actor 目标为
$$
\mathcal{L}_{\pi}=\frac{
\sum_{x\in\mathcal{B}}\frac{1}{W(x)}
\min\!\bigl(
r(\theta)\hat A_x,\;
\operatorname{clip}(r(\theta),1-\epsilon,1+\epsilon)\hat A_x
\bigr)
}{
\sum_{x\in\mathcal{B}}\frac{1}{W(x)}
}.
$$
critic 目标同样使用该权重：
$$
\mathcal{L}_{V}=\frac{
\sum_{x\in\mathcal{B}}\frac{1}{W(x)}
\max\!\bigl[
(\hat V(s_x)-\hat R_x)^2,\;
(\hat V^{\mathrm{clip}}(s_x)-\hat R_x)^2
\bigr]
}{
\sum_{x\in\mathcal{B}}\frac{1}{W(x)}
}.
$$
Advantage whitening 也使用同一校正测度；公式见附录 A。

### 3.2 On-policy Parallel Tree Search

TTPO 解决“如何从树样本更新策略”，OPTS 解决“树应在哪里继续生长”。

#### 3.2.1 贪心参考路径

对每棵树，从根节点出发，若当前节点存在多个 continuation，则选择首节点 TreeGAE 最大的 continuation，直到叶节点，得到贪心参考路径。随后只在该路径上选择重分支位置。

#### 3.2.2 OTRC 的开发项

设当前贪心参考路径为
$$
\tau^\star=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots,s_{n-1},a_{n-1},r_{n-1},s_n),
$$
其中位置 $t$ 对应树节点 $x_t$。若在位置 $k$ 后继续搜索，我们关心从 $s_k$ 重新产生后缀轨迹的期望回报，相比当前贪心后缀能改进多少。定义后缀期望改进量
$$
\Delta_k^\pi:=V^\pi(s_k)-G_k,
\qquad
G_k=\sum_{t=k}^{n-1}\gamma^{t-k}r_t,
$$
其中 $G_k$ 是当前贪心后缀的实际 return，$V^\pi(s_k)$ 是从 $s_k$ 出发按策略产生后缀的期望 return。假设参考路径有限，且 $V^\pi(s_n)=0$。令
$$
\delta^\pi_t
=
r_t+\gamma V^\pi(s_{t+1})-V^\pi(s_t).
$$
则
$$
\begin{aligned}
-\sum_{t=k}^{n-1}\gamma^{t-k}\delta^\pi_t
&= -G_k + V^\pi(s_k)-\gamma^{n-k}V^\pi(s_n)\\
&= \Delta_k^\pi.
\end{aligned}
$$
于是得到：

**引理 1（后缀期望改进量的残差分解）** 对有限贪心参考路径，若 $V^\pi(s_n)=0$，则
$$
\Delta_k^\pi
=
-\sum_{t=k}^{n-1}\gamma^{t-k}\delta^\pi_t.
$$

该分解把“后缀期望改进”转化为沿参考路径估计 TD 残差。路径采样完成后，$r_t,s_t,s_{t+1}$ 均固定；若 $V^\pi$ 已知，$\delta^\pi_t$ 也是确定量。直接用 $\hat\delta_t=r_t+\gamma\hat V(s_{t+1})-\hat V(s_t)$ 估计它没有额外采样方差，但继承 value 估计偏差。另一种做法是用 TreeGAE 引入下游 continuation 随机性：条件于当前一步转移，若 $\hat V=V^\pi$，则第 3.1.2 节给出
$$
\mathbb{E}\!\left[
\hat A^{\mathrm{TreeGAE}}_t
\middle|
s_t,a_t,r_t,s_{t+1}
\right]
=
\delta^\pi_t.
$$
实际使用 $\hat V$ 时，偏差主要来自 value 估计，$\lambda$ 控制引入多少下游轨迹信息。

据此，令 $\hat A_t:=\hat A^{\mathrm{TreeGAE}}_t$，定义 OTRC 的开发项为
$$
E_k:=-\sum_{t=k}^{n-1}\gamma^{t-k}\hat A_t.
$$
在理想 value 情形下，固定每个一步转移并仅对后续 on-policy continuation 取期望，则
$$
\mathbb{E}_{\mathrm{cont}}\!\left[E_k\right]
=
-\sum_{t=k}^{n-1}\gamma^{t-k}\delta^\pi_t
=
\Delta_k^\pi.
$$
因此，$E_k$ 是后缀期望改进量 $\Delta_k^\pi$ 的 TreeGAE 代理，为“在哪里继续分支”提供开发信号。

#### 3.2.3 OTRC 的探索项与重分支分数

开发项 $E_k$ 衡量从 $x_k$ 继续搜索的潜在收益。为避免反复扩展同一区域，引入 sibling-count 探索项
$$
U_k=(B_k-1)\max_j |E_j|,
$$
其中 $B_k$ 为 $x_k$ 父节点已有的 continuation 数。尺度因子 $\max_j|E_j|$ 使探索项与开发项量级接近，降低 $c$ 对奖励尺度的敏感性；该选择无需额外统计窗口，可在每条参考路径内直接计算。

于是，同策略轨迹重分支准则定义为
$$
\mathrm{OTRC}_k = E_k - c\,U_k,
$$
其中 $c>0$。对每棵树，在当前贪心参考路径上选择
$$
k^\star=\arg\max_k \mathrm{OTRC}_k
$$
作为重分支位置；随后从 $\operatorname{parent}(x_{k^\star})$ 重新采样动作并生成新的 continuation，即重采样到达 $x_{k^\star}$ 前的最后一个决策。

### 3.3 OPTS-TTPO Training Procedure

综合 TTPO 与 OPTS，单次训练 step 如下。

1. 进行 $R$ 轮搜索 rollout。第一轮采样根轨迹；后续轮次对仍继续搜索的树，根据 OTRC 选出的 $\operatorname{parent}(x_{k^\star})$ 并行生成新 continuation。
2. 将新节点并入树结构，更新 parent-child 关系、分支位置和 continuation 数。
3. 自底向上计算 TreeGAE；非分支节点执行标准 GAE，分支节点平均所有子 continuation 首节点优势后回传。
4. 根据第 3.1.1 节计算分支放大因子 $W(x)$，并使用其倒数 $1/W(x)$ 作为分支校正权重，按第 3.1.3 节构造 TTPO actor/critic 损失。
5. 使用 $\mathcal{L}_V^{\mathrm{TTPO}}$ 更新 critic，再使用 $\mathcal{L}_\pi^{\mathrm{TTPO}}$ 更新 actor。

跨域预算对齐与长度惩罚的实现细节见附录 B。

OPTS 利用当前树中的后验样本信息选择扩展位置，不再严格满足第 3.1.1 节的 i.i.d. 采样假设，因此 OPTS-TTPO 不继承给定同策略树轨迹下的严格无偏性。我们将其视为 bias-variance trade-off：搜索可能引入样本选择偏差，但会把预算集中到梯度估计误差更低的后缀区域。第 4.4 节的 MuJoCo 诊断显示，不同任务与 batch size 下，OPTS-TTPO 的策略梯度估计误差始终低于 matched-budget PPO。
