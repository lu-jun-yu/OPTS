本节只报告当前仓库中可以直接核实的实验设定与结果。对于已经完整跑通、但尚未汇总成最终表格的部分，我们明确写为“待整理”，而不提前做不可验证的总结。

## 4.1 统一实验原则

我们在三类场景中都遵循同一条原则：`搜索不能白拿额外预算`。

- 在 Atari 与 MuJoCo 中，预算按环境 action 计，因此更深的搜索会真实占用更多交互步数；TUCT 采用带长度惩罚的开发项。
- 在 LLM 中，预算按完整 episode 或完整回答计，因此 test-time 搜索需与 `pass@k` 或重复采样成本对齐；TUCT 不再使用长度惩罚。

在训练目标上，凡是进入更新的数据都使用 TTPO 的 branch-weight 修正，而不是把树上的所有 token 或 time step 当作等权样本。

## 4.2 LLM 训练时搜索

### 4.2.1 设定

LLM 训练使用 VeRL/Ray 实现，入口为 `LLM/trainer/main_opts_ttpo.py`，训练器为 `LLM/trainer/opts_ttpo/ray_trainer.py`。根据现有日志，当前主实验配置具有以下特征：

- 模型：`Qwen3-1.7B`。
- 任务：可验证数学推理。
- 训练数据：`math12k` 与 `NuminaMath-1.5-RL-Verifiable` 的竞赛子集。
- 测试数据：`math12k` test (`MATH500`)、`minervamath`、`amc23`、`aime25`。
- 响应长度上限：`2048`。
- prompt 长度上限：`1024`。
- 一个训练 step 内最终合并 batch 大小约为 `2048`。

与 Atari/MuJoCo 不同，LLM 这里的预算是 `episode-level`。因此 TUCT 的开发项使用原始折扣累计优势，不再除以剩余长度。同时，LLM 版代码只允许在 `</think>` 之前继续分支，以保证搜索主要作用于推理阶段。

### 4.2.2 当前可核实的验证结果

下表来自仓库中现有日志的最终验证点，指标均为 `acc/mean@1`。这些数字不是最终投稿版结论，而是本地可复现实验的当前状态。`DAPO-n8` 表示仓库中的一个现成参考运行，其 rollout 宽度与本文其余方法未必完全预算匹配，因此这里主要把它当作参考上界而不是严格对照。

| 方法 | math12k | minervamath | amc23 | aime25 |
| --- | ---: | ---: | ---: | ---: |
| REINFORCE++ | 56.16 | 15.88 | 45.00 | 12.00 |
| GPG | 71.42 | 27.24 | 61.50 | 18.33 |
| PPO | 72.74 | 27.57 | 62.50 | 27.67 |
| DAPO-n8 | 73.00 | 29.34 | 65.25 | 24.33 |
| OPTS-TTPO | 73.92 | 30.77 | 54.00 | 18.00 |

这个结果非常重要，因为它说明了两点。

第一，`OPTS-TTPO` 在 `math12k` 与 `minervamath` 上已经超过 PPO 和 GPG，说明同策略树搜索确实能改善中等难度、可验证推理任务中的训练信号质量。

第二，`OPTS-TTPO` 在 `amc23` 与 `aime25` 上尚未超过 PPO/DAPO，说明“把搜索引入训练”并不自动等价于“所有分布都更强”。当前版本更像是在提升样本利用效率和中等难度推理，而不是彻底解决极难竞赛题的泛化问题。这也是本文必须诚实面对的地方。

### 4.2.3 如何解读这些结果

我们认为上述现象至少有三种可能解释。

1. 当前 TUCT 更擅长在已有可行解附近做局部修正，对特别困难题目所需的长程探索仍不够强。
2. LLM 侧增量优势回传与搜索超参数还未完全稳定，尤其是深树情况下的 credit assignment 仍有优化空间。
3. 不同 baseline 的 rollout 宽度、温度与训练 recipe 并未完全对齐，正式版必须做更严格的 matched-budget 比较。

因此，LLM 部分当前更适合被表述为“一个有前景但尚未完全收敛的方向”，而不是简单宣布全面超越。

## 4.3 LLM 测试时搜索

除了训练时搜索，仓库还实现了推理时搜索入口 `LLM/trainer/main_opts_generation.py`。其关键特点是：

- 总推理预算严格对齐 `pass@k` 成本。
- 支持 `reward-guided` 与 `value-guided` 两种 TUCT 指导。
- 输出 `sample_index`，可直接离线计算 `avg@k`、`pass@k`、`cons@k`。

这部分代码已经形成完整实验框架，但当前仓库尚未把最终曲线和表格整理到论文目录中。因此在本文当前版本中，它更适合作为“test-time scaling 实验框架已打通”的证据，而不是 headline result。NeurIPS 正式版应重点补充：

1. OPTS 与重复采样、best-of-`n`、majority voting 的同预算比较。
2. `reward-guided` 与 `value-guided` 的差异。
3. 搜索深度对一致性指标 `cons@k` 的影响。

## 4.4 跨域验证：Atari 与 MuJoCo

在给出 LLM 主结果之后，我们再报告经典强化学习场景中的实现与结果状态。这样安排的目的不是淡化 Atari/MuJoCo 的重要性，而是强调本文首先是一篇面向 LLM 搜索训练的论文；与此同时，跨域结果说明 TTPO/OPTS 并不是只对语言模型工程特化的技巧。

### 4.4.1 Atari

Atari 实现基于 CleanRL 单文件 `ppo_atari.py` 改写而来。当前 `opts_ttpo_atari.py` 的默认设置为：

- 环境：ALE Atari，代码默认 `BreakoutNoFrameskip-v4`，结果目录已覆盖 57 个任务。
- 网络：Nature CNN actor-critic。
- 训练长度：`total_timesteps = 10M`。
- 并行度：`num_envs = 8`，`num_steps = 128`。
- PPO 超参数：`clip_coef = 0.1`，`update_epochs = 4`，`num_minibatches = 4`。
- 搜索超参数：代码默认 `tau = 0.6`、`max_search_per_tree = 6`、`c = 1.0`；仓库中也保留了 `tau = 0.7, s = 8` 的运行结果目录。

工程上，Atari 版本额外实现了对 ALE state、FrameStack、MaxAndSkip、EpisodicLife、TimeLimit 和 RecordEpisodeStatistics 的完整快照恢复，这是树搜索能够正确复用中间状态的前提。

仓库中已经存在完整的 ALE-57 结果目录，例如：

- `Atari_MuJoCo/results/1_4096/opts_ttpo_atari_20260315/`
- `Atari_MuJoCo/results/1_4096/ppo_atari_20260315/`
- `Atari_MuJoCo/results/8_128/opts_ttpo_atari_tau0.7_s8_20260413/`

这说明 Atari 端的全量跑通已完成。以 `BerzerkNoFrameskip-v4` 的日志式 JSON 为例，可以清楚看到回报曲线在训练早期已出现明显上升，说明树搜索采样、日志聚合与 TTPO 更新这条工程链路已经稳定工作。  

本稿暂不直接给出最终 human-normalized score 汇总，原因不是结果缺失，而是当前仓库尚未把 57 个任务的多种配置统一整理成论文表格。NeurIPS 正式版应补上：

1. 57 个任务的 HNS 均值与中位数。
2. `tau` 与 `max_search_per_tree` 的消融。
3. 搜索成本与性能收益的 Pareto 曲线。

### 4.4.2 MuJoCo

MuJoCo 实现基于 CleanRL `ppo_continuous_action.py` 改写而来。与旧稿不同，当前代码中的 actor-critic 网络并不是 `256 x 256`，而是：

- 两层 `64` 维 MLP。
- 激活函数为 `Tanh`。
- 连续动作高斯策略，独立学习 `actor_logstd`。

其他关键设定如下：

- 任务：`Hopper-v4`、`Walker2d-v4`、`HalfCheetah-v4`、`Ant-v4`、`Humanoid-v4`。
- 代码默认：`num_envs = 1`，`num_steps = 4096`。
- 实际结果目录中常用设置：`num_envs = 1`，`num_steps = 2048`。
- 搜索超参数扫描：`max_search_per_tree` 从 `1` 到 `4` 的多组运行都已存在。

MuJoCo 版最关键的工程点在于状态恢复。代码保存了物理状态、派生量、目标、随机数状态、归一化统计量以及 wrapper 内部累计回报，从而保证“回到旧状态继续采样”是真正等价的环境续跑。

仓库中已经保存了较系统的搜索强度扫描结果，例如：

- `Atari_MuJoCo/results/1_2048/ppo_continuous_action_20260301/`
- `Atari_MuJoCo/results/1_2048/opts_ttpo_search1_continuous_action_20260301/`
- `Atari_MuJoCo/results/1_2048/opts_ttpo_search2_continuous_action_20260301/`
- `Atari_MuJoCo/results/1_2048/opts_ttpo_search3_continuous_action_20260301/`
- `Atari_MuJoCo/results/1_2048/opts_ttpo_search4_continuous_action_20260301/`

从研究设计上看，MuJoCo 是最能检验“action-level 预算下长度惩罚是否必要”的场景，因为更深的分支会真实增加交互步数。正式论文应重点回答两个问题：

1. `tau > 0` 是否稳定优于 `tau = 0`。
2. 搜索深度增加后，收益主要来自更好地定位失败前缀，还是来自更大的有效 rollout 数。

目前这些结果已经具备整理成表的基础，但本稿仍以设定说明为主。

## 4.5 MuJoCo 方差缩减实验

主任务结果之外，仓库还在 `Atari_MuJoCo/experiments/` 下实现了两组专门用于验证“搜索是否降低策略梯度方差”的 MuJoCo 机制实验。这部分并不直接比较 episode return，而是直接估计策略梯度误差

$$
\mathbb{E}\left[\|\hat g_B - g^\star\|_2^2\right],
$$

其中 `\hat g_B` 表示由 `B` 个 step 构成的 bootstrap mini-batch 所估计的 actor 梯度，`g^\star` 表示由更大样本池近似得到的参考梯度。五个任务 `HalfCheetah-v4`、`Walker2d-v4`、`Hopper-v4`、`Ant-v4`、`Humanoid-v4` 都使用同一套评估协议：先加载每个任务一个训练至 `1M` step 的 checkpoint，再只改变数据收集 seed 做重复估计。

### 4.5.1 高优势样本与低优势样本

第一组实验对应 `verify_pg_variance.py`，检验“高优势样本是否天然具有更低的梯度方差”。与早期文档按 episode return 切分不同，当前代码实际上按 `step-level` GAE 优势排序：在 `1M` 个 PPO step 上计算全量优势后，取 top-`\alpha` 与 bottom-`\alpha` 的 step 子集，其中运行脚本使用 `\alpha = 0.42`。随后分别在 `B \in \{256,512,1024,2048,4096\}` 上做 `100` 次 bootstrap，估计

$$
\mathbb{E}\left[\|\hat g_B^{\text{pos}} - g^\star\|_2^2\right]
\quad \text{与} \quad
\mathbb{E}\left[\|\hat g_B^{\text{neg}} - g^\star\|_2^2\right].
$$

这组实验的汇总图已经保存在 `Atari_MuJoCo/visual/pg_variance_verify1.png`。因此，MuJoCo 部分不仅比较“搜索后的最终表现”，还单独检验了 TTPO/OPTS 的核心动机：如果搜索能更集中地收集高优势片段，那么由这些片段构成的策略梯度估计应当更稳定。

### 4.5.2 PPO vs OPTS 的 Batch-Size 缩放

第二组实验对应 `verify_scaling_variance.py`，检验“OPTS 是否能让梯度方差随 batch size 更快下降”。代码先从 PPO 收集 `1M` step，计算参考梯度 `g^\star`；随后在同一 checkpoint 上分别收集 PPO 与 OPTS 的样本池，其中运行脚本采用 `num_steps = 10000`、`batch_sizes = 64,128,256,512,1024,2048,4096`、`num_bootstrap = 100`、`max_search_per_tree = 2`。对 PPO，`g_B` 直接由前 `B` 个 rollout step 估计；对 OPTS，则对 tree-search 样本使用与训练一致的 `branch_weight` 逆概率加权，

$$
\mathcal{L}_{\pi}
=
\frac{\sum_i \ell_i / w_i}{\sum_i 1 / w_i},
$$

再比较 `\|\hat g_B - g^\star\|_2^2` 随 `B` 增大时的收敛速度。相应汇总图保存在 `Atari_MuJoCo/visual/scaling_variance_verify2.png`，仓库中还保留了多张同主题可视化版本。由于当前仓库未同步保留用于绘图的原始 JSON，本稿暂不从图中手工抄录数值表；但从实验设计上看，这部分正是对 TTPO 中 branch-weight 修正和 OPTS 中高价值分支采样的直接机制验证。

## 4.6 消融与讨论

从当前能核实的实验看，`OPTS-TTPO` 已经具备四个层面的证据：

1. LLM 端的训练时搜索已经产生了可见收益，但收益集中在部分数据集上，说明该方向有潜力但尚未完全收敛。
2. LLM 的 test-time search 框架已经实现完备的预算对齐与指标输出，但最终曲线和表格仍待整理。
3. Atari 与 MuJoCo 端，算法和工程闭环已经跑通，并且结果目录足够支持正式版表格整理。
4. MuJoCo 端还补充了面向梯度估计误差的方差缩减实验，因此本文并非只依赖最终 return 曲线来论证 OPTS-TTPO 的有效性。

正式版仍需补充三类关键消融。

1. LLM 侧更严格的 matched-budget baseline，以及训练时搜索与 test-time search 的系统比较。
2. Atari 侧的 HNS 汇总、`tau` 与 `max_search_per_tree` 消融，以及搜索成本与收益的 Pareto 曲线。
3. MuJoCo 侧关于 `tau > 0` 是否优于 `tau = 0`、以及搜索深度如何影响有效样本数与失败前缀定位能力的分析。

整体上，这些结果共同支持本文的主张：`搜索能否帮助策略梯度，不取决于“是否加入搜索”本身，而取决于搜索是否与预算、目标和并行实现方式严格对齐。`
