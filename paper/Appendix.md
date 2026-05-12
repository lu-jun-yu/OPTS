## 附录

### A. 分支校正的优势归一化

TTPO 中的树节点不能简单按节点数均匀计数。若节点 $x$ 的分支放大因子为 $W(x)$，则它在链式 on-policy rollout 测度下对应的有效权重为
$$
w_x=\frac{1}{W(x)}.
$$
因此，actor 目标、critic 目标以及 advantage whitening 的批内统计都应使用同一个分支校正经验测度。给定树节点集合 $\mathcal{T}_{\pi_\theta}$ 与 TreeGAE 优势 $\hat A_x$，令
$$
Z_{\mathcal{T}_{\pi_\theta}}=\sum_{x\in\mathcal{T}_{\pi_\theta}} w_x.
$$
优势归一化采用
$$
\mu_A=\frac{1}{Z_{\mathcal{T}_{\pi_\theta}}}\sum_{x\in\mathcal{T}_{\pi_\theta}} w_x\hat A_x, \qquad
\sigma_A^2=\frac{1}{Z_{\mathcal{T}_{\pi_\theta}}}\sum_{x\in\mathcal{T}_{\pi_\theta}} w_x(\hat A_x-\mu_A)^2.
$$
标准化优势为
$$
\tilde A_x=\frac{\hat A_x-\mu_A}{\sqrt{\sigma_A^2+\varepsilon}}.
$$
训练时，$\tilde A_x$ 用于替换 actor 损失中的 $\hat A_x$；TreeGAE 递推、return target 以及 critic 的监督目标本身不因 whitening 改变，只是在 critic loss 聚合时继续使用同样的 $w_x$。若使用未校正的均匀均值与方差，高分支区域会因为包含更多节点而在中心化和缩放阶段再次支配 batch 统计。上述加权归一化使 advantage whitening 与 TTPO 的 actor/critic 加权目标保持在同一经验测度下。

### B. 跨域预算对齐与长度惩罚

OPTS 的重分支分数需要与具体任务的预算单位对齐。对于 LLM 推理，训练或推理预算通常按完整回答、完整 episode 或采样次数计量；在这种计量下，从不同前缀继续生成都被视为一次 continuation，因而主文中的开发项
$$
E_k=-\sum_{t=k}^{n-1}\gamma^{t-k}\hat A_{x_t}
$$
可直接使用，不额外加入长度惩罚。

对于 Atari 和 MuJoCo 这类 action-level budget 任务，重分支位置越靠近根节点，新增 continuation 通常会消耗越多环境交互步数。若仍直接比较未归一化的 $E_k$，靠前位置可能因为后缀更长而获得更大的绝对改进分数，但这并不一定意味着其单位交互预算收益更高。为使不同位置的得分更接近“单位预算收益”，我们使用长度归一化的开发项
$$
E_k^{(\tau)}=
\frac{-\sum_{t=k}^{n-1}\gamma^{t-k}\hat A_{x_t}}{(n-k)^\tau},
\qquad \tau\ge 0.
$$
对应的 OTRC 分数同步写为
$$
\mathrm{OTRC}_k^{(\tau)}
=E_k^{(\tau)}-c\,U_k^{(\tau)},\qquad
U_k^{(\tau)}=\bigl[m(x_k)-1\bigr]\max_{0\le t<n}\left|E_t^{(\tau)}\right|.
$$
其中 $\tau=0$ 对应 episode-level budget，不进行长度惩罚；$\tau=1$ 近似按剩余动作数计算单位步收益；实践中可取 $0<\tau<1$ 作为较温和的长度校正，因为真实 continuation 长度、早期动作的长期影响与价值估计误差并不严格线性。本文在 LLM 实例中使用 $\tau=0$，在 Atari 与 MuJoCo 实例中使用 $\tau=0.7$ 的 action-level budget 版本。该项只是对 OTRC 开发项的预算化延拓，用于改变重分支位置的排序；TreeGAE、分支校正权重以及 TTPO actor/critic 目标保持不变。

### C. Atari-57 完整学习曲线

主文只报告 Atari-57 的任务级胜场统计，以避免在正文中放置过密的 57 任务曲线。图 A1 给出 PPO 与 OPTS-TTPO 在全部 Atari 任务上的完整可视化，便于检查收益与退化的任务分布。每个子图对应一个 Atari 游戏，横轴为环境交互步数，纵轴为该游戏的 raw mean return；同一算法的不同随机种子以同色细线展示，并仅做可视化平滑。由于 Atari 各任务的回报尺度差异很大，图 A1 主要用于任务内比较与观察稳定性，跨任务汇总结论仍以正文中的任务级胜场统计为准。

![Figure A1: Atari-57 上 PPO 与 OPTS-TTPO 的完整学习曲线。](../Atari_MuJoCo/visual/all_tasks_atari.png)

**图 A1：** Atari-57 上 PPO 与 OPTS-TTPO 的完整学习曲线。每个面板对应一个任务，曲线展示不同随机种子的 smoothed raw mean return。

### D. 实验设置

表 A1 汇总本文各组实验的主要设置。

**表 A1：主要实验设置。**

| 实验 | 设置 |
| --- | --- |
| LLM 训练时搜索 | `Qwen3-1.7B`；训练集为 `math12k` 与 `NuminaMath-1.5-RL-Verifiable` 竞赛子集；测试集为 `math12k` test、`minervamath`、`amc23`、`aime25`；最大 prompt 长度 `1024`，最大响应长度 `2048`；验证每 `20` step 采样 `n` 个回答并记录 `avg@n/pass@n/cons@n`。 |
| LLM 测试时搜索 | 比较 i.i.d. sampling、self-consistency、reward-guided OPTS 与 value-guided OPTS；推理预算 $k\in\{8,16,32,64,128\}$。 |
| Atari-57 | CleanRL 风格 PPO CNN actor-critic；每任务 `10M` 环境步；`8` 个并行环境；`128` 步 rollout；`3` 个随机种子；OPTS-TTPO 使用 action-level OTRC，$\tau=0.7$、`s=4`。 |
| MuJoCo | 任务为 `Hopper-v4`、`Walker2d-v4`、`HalfCheetah-v4`、`Ant-v4`、`Humanoid-v4`；PPO 与 OPTS-TTPO 使用 matched rollout/update 预算。 |
| 方差缩减实验 | 比较高/低优势样本以及 PPO/OPTS 样本的 bootstrap 策略梯度方差代理；参考梯度 $g^\star$ 由大样本池近似；OPTS 样本使用训练一致的 $1/W$ 分支校正。 |
