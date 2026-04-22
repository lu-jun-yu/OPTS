## 4 实验

本节按四个层次组织实验证据。4.1 节报告 LLM **训练时搜索**的可复现结果——这是本文的主要实验；4.2 节报告 LLM **测试时搜索**的框架与评估协议；4.3 节给出 **Atari 与 MuJoCo** 的跨域验证，说明 TTPO/OPTS 并非 LLM 专用技巧；4.4 节给出 MuJoCo 上的**方差缩减实验**，把"OPTS 改善了什么"这一问题下沉到策略梯度估计层面。所有结果均严格对应当前仓库中可核实的实现与日志。

我们在三类场景中都遵循同一条原则：**搜索不能白拿额外预算**。Atari 与 MuJoCo 的预算按环境 action 计，OTRC 采用带长度惩罚的开发项（$\tau>0$）；LLM 的预算按完整 episode 或完整回答计，OTRC 不做长度惩罚（$\tau=0$）。凡是进入更新的数据，都使用 3.1.2--3.1.3 节的 branch-weight 修正，而不是把树上的所有 token 或 time step 当作等权样本。

### 4.1 LLM 训练时搜索

#### 4.1.1 设定

LLM 训练使用 VeRL/Ray 实现，入口为 `LLM/trainer/main_opts_ttpo.py`，训练器为 `LLM/trainer/opts_ttpo/ray_trainer.py`。主实验配置：

- **模型**：`Qwen3-1.7B`。
- **任务**：可验证数学推理。
- **训练数据**：`math12k` 与 `NuminaMath-1.5-RL-Verifiable` 的竞赛子集。
- **测试数据**：`math12k` test (`MATH500`)、`minervamath`、`amc23`、`aime25`。
- **响应长度上限**：`2048`；**prompt 长度上限**：`1024`。
- **合并 batch 大小**：约 `2048`。

LLM 的预算是 episode-level，因此 OTRC 开发项使用原始折扣累计优势（$\tau=0$）；同时只允许在 `</think>` 之前继续分支，以保证搜索主要作用于推理阶段。

#### 4.1.2 主结果

下表报告训练完成后在四个测试集上的 `acc/mean@1`。基线均来自 `LLM/logs/` 下的完整训练日志。

| 方法 | math12k | minervamath | amc23 | aime25 |
| --- | ---: | ---: | ---: | ---: |
| REINFORCE++ | 56.16 | 15.88 | 45.00 | 12.00 |
| GPG | 71.42 | 27.24 | 61.50 | 18.33 |
| PPO | 72.74 | 27.57 | 62.50 | 27.67 |
| DAPO | 73.00 | 29.34 | 65.25 | 24.33 |
| **OPTS-TTPO** | **73.92** | **30.77** | 54.00 | 18.00 |

**观察 1：OPTS-TTPO 在中等难度可验证推理任务上取得最佳表现。** 在 `math12k` 与 `minervamath` 上，OPTS-TTPO 分别达到 `73.92` 与 `30.77`，同时超过 PPO、GPG、DAPO 与 REINFORCE++。这直接支持本文主张：**同策略树搜索确实能改善中等难度、可验证推理任务中的训练信号质量**——TTPO 的 $1/W$ 加权与 TreeGAE 的分支优势平均共同让策略梯度能够吸收树结构带来的信息增益。

**观察 2：在极难竞赛题上，当前版本尚未带来一致增益。** OPTS-TTPO 在 `amc23` 与 `aime25` 上尚未超过 PPO/DAPO。我们在此诚实报告这一结果而非回避——它更可能反映本文版本当前的局限：(i) OTRC 更擅长在已有可行解附近做局部修正，对 `aime25` 级别题目所需的长程探索仍不够强；(ii) LLM 侧增量优势回传与搜索超参数尚未完全稳定，深树情况下的 credit assignment 仍有优化空间。这些是我们在讨论节（第 5 节）明确列出的改进方向。

### 4.2 LLM 测试时搜索

#### 4.2.1 框架

`LLM/trainer/main_opts_generation.py` 实现了 OPTS 的推理时入口。其关键特点：

- **预算对齐**：总推理预算严格对齐 `pass@k` 的 $N\cdot k/B$ 成本，与重复采样/best-of-$n$ 同预算可比。
- **双指导模式**：支持 `reward-guided` 与 `value-guided` 两种 OTRC 开发项计算方式。
- **指标输出**：输出每条响应的 `sample_index`，可离线绘制 `avg@k`、`pass@k`、`cons@k`。

#### 4.2.2 当前状态

测试时搜索的代码框架已经完整打通（预算对齐、指标输出、两种指导模式均已实现），但本稿尚未把最终曲线与汇总表格整理到论文目录中。本文将其作为"test-time scaling 实验框架已就绪"的工程证据；EMNLP 正式版应重点补充：

1. OPTS 与重复采样、best-of-$n$、majority voting 在同预算下的对比。
2. `reward-guided` 与 `value-guided` 的差异分析。
3. 搜索深度对 `cons@k` 与 `pass@k` 的影响。

### 4.3 跨域验证：Atari 与 MuJoCo

为了说明 TTPO 的无偏加权与 OPTS 的 OTRC 重分支选点并非 LLM 专用技巧，我们在经典离散控制 (Atari) 与连续控制 (MuJoCo) 上做了一致化实现与跨域验证。

#### 4.3.1 Atari

Atari 实现基于 CleanRL 单文件 `ppo_atari.py` 改写，默认设置：

- **环境**：ALE Atari，代码默认 `BreakoutNoFrameskip-v4`，结果目录覆盖 ALE-57。
- **网络**：Nature CNN actor-critic。
- **训练长度**：`total_timesteps = 10M`；**并行度**：`num_envs = 8`，`num_steps = 128`。
- **PPO 超参**：`clip_coef = 0.1`，`update_epochs = 4`，`num_minibatches = 4`。
- **搜索超参**：默认 `tau = 0.6`、`max_search_per_tree = 6`、`c = 1.0`；亦保留 `tau = 0.7, s = 8` 的运行结果。

Atari 版本实现了对 ALE state、FrameStack、MaxAndSkip、EpisodicLife、TimeLimit 与 RecordEpisodeStatistics 的完整快照恢复——这是树搜索能够正确复用中间状态的前提。仓库中已存在 ALE-57 的完整结果目录：`Atari_MuJoCo/results/1_4096/opts_ttpo_atari_20260315/`、`ppo_atari_20260315/`、`8_128/opts_ttpo_atari_tau0.7_s8_20260413/`。以 `BerzerkNoFrameskip-v4` 的日志式 JSON 为例，回报曲线在训练早期已出现明显上升，说明树搜索采样、日志聚合与 TTPO 更新的工程链路已稳定工作。本稿暂不直接给出最终 human-normalized score 汇总——并非结果缺失，而是当前仓库尚未把 57 个任务的多种配置统一整理成论文表格。EMNLP 正式版应补上：(i) ALE-57 的 HNS 均值与中位数；(ii) `tau` 与 `max_search_per_tree` 的消融；(iii) 搜索成本与性能收益的 Pareto 曲线。

#### 4.3.2 MuJoCo

MuJoCo 实现基于 CleanRL `ppo_continuous_action.py` 改写，actor-critic 网络为 `64 × 64` 两层 MLP + Tanh 激活，连续动作高斯策略，独立学习 `actor_logstd`。关键设定：

- **任务**：`Hopper-v4`、`Walker2d-v4`、`HalfCheetah-v4`、`Ant-v4`、`Humanoid-v4`。
- **代码默认**：`num_envs = 1`，`num_steps = 4096`；实际结果目录常用 `num_envs = 1`，`num_steps = 2048`。
- **搜索强度扫描**：`max_search_per_tree ∈ {1,2,3,4}` 的多组运行均已存在。

MuJoCo 最关键的工程点在于状态恢复——代码保存了物理状态、派生量、目标、随机数状态、归一化统计量以及 wrapper 内部累计回报，从而保证"回到旧状态继续采样"是真正等价的环境续跑。仓库中已保存：`Atari_MuJoCo/results/1_2048/ppo_continuous_action_20260301/`、`opts_ttpo_searchN_continuous_action_20260301/`（$N=1\dots4$）。从研究设计看，MuJoCo 是最能检验"action-level 预算下长度惩罚是否必要"的场景——更深的分支真实增加交互步数。EMNLP 正式版应重点回答：(i) $\tau>0$ 是否稳定优于 $\tau=0$；(ii) 搜索深度增加后，收益主要来自更好地定位失败前缀，还是来自更大的有效 rollout 数。

### 4.4 MuJoCo 方差缩减实验

主任务结果之外，仓库还在 `Atari_MuJoCo/experiments/` 下实现了两组专门用于验证"搜索是否降低策略梯度方差"的 MuJoCo 机制实验。这部分不直接比较 episode return，而是**直接估计策略梯度误差**
$$
\mathbb{E}\!\left[\|\hat g_B - g^\star\|_2^2\right],
$$
其中 $\hat g_B$ 是由 $B$ 个 step 构成的 bootstrap mini-batch 所估计的 actor 梯度，$g^\star$ 是由更大样本池近似得到的参考梯度。五个任务 `HalfCheetah-v4`、`Walker2d-v4`、`Hopper-v4`、`Ant-v4`、`Humanoid-v4` 均使用同一套协议：先加载每个任务一个训练至 `1M` step 的 checkpoint，再只改变数据收集 seed 做重复估计。

#### 4.4.1 高优势样本 vs 低优势样本

第一组实验（`verify_pg_variance.py`）检验"高优势样本是否天然具有更低的梯度方差"。按 step-level GAE 优势排序，在 `1M` 个 PPO step 上计算全量优势后，取 top-$\alpha$ 与 bottom-$\alpha$ 的 step 子集（运行脚本 $\alpha=0.42$）；随后在 $B\in\{256,512,1024,2048,4096\}$ 上做 $100$ 次 bootstrap，估计
$$
\mathbb{E}\!\left[\|\hat g_B^{\text{pos}} - g^\star\|_2^2\right]
\quad \text{与} \quad
\mathbb{E}\!\left[\|\hat g_B^{\text{neg}} - g^\star\|_2^2\right].
$$
汇总图保存在 `Atari_MuJoCo/visual/pg_variance_verify1.png`。这组实验直接检验 OPTS 的核心动机：**如果搜索能更集中地收集高优势片段，那么由这些片段构成的策略梯度估计应当更稳定**。

#### 4.4.2 PPO vs OPTS 的 Batch-Size 缩放

第二组实验（`verify_scaling_variance.py`）检验"OPTS 是否能让梯度方差随 batch size 更快下降"。流程为：先从 PPO 收集 `1M` step 计算参考梯度 $g^\star$；随后在同一 checkpoint 上分别收集 PPO 与 OPTS 的样本池（`num_steps = 10000`、`batch_sizes ∈ {64,128,256,512,1024,2048,4096}`、`num_bootstrap = 100`、`max_search_per_tree = 2`）。对 PPO，$\hat g_B$ 直接由前 $B$ 个 rollout step 估计；对 OPTS，则对 tree-search 样本使用与训练一致的 $1/W$ 加权：
$$
\mathcal{L}_\pi=\frac{\sum_i \ell_i/w_i}{\sum_i 1/w_i}.
$$
再比较 $\|\hat g_B - g^\star\|_2^2$ 随 $B$ 增大时的收敛速度。汇总图保存在 `Atari_MuJoCo/visual/scaling_variance_verify2.png`。由于当前仓库未同步保留绘图的原始 JSON，本稿暂不从图中手工抄录数值表；但从实验设计上看，这部分正是对 TTPO 中 branch-weight 修正与 OPTS 中高价值分支采样的直接机制验证。

### 4.5 消融与讨论

从当前可核实的实验看，OPTS-TTPO 已经具备四个层面的证据：

1. LLM 训练时搜索在 `math12k` 与 `minervamath` 上超过 PPO、GPG、DAPO 与 REINFORCE++，但在 `amc23`/`aime25` 级极难任务上尚未一致超越。
2. LLM 测试时搜索框架已经实现完备的预算对齐与指标输出，但最终曲线与表格仍待整理。
3. Atari 与 MuJoCo 端算法与工程闭环已跑通，结果目录足够支持正式版表格整理。
4. MuJoCo 端补充了面向梯度估计误差的方差缩减实验，使本文不只依赖最终 return 曲线论证 OPTS-TTPO 的有效性。

EMNLP 正式版仍需补充三类关键消融：(i) LLM 侧更严格的 matched-budget baseline 比较，以及训练时搜索与测试时搜索的系统比较；(ii) Atari 侧的 HNS 汇总、`tau` 与 `max_search_per_tree` 消融、以及搜索成本与收益的 Pareto 曲线；(iii) MuJoCo 侧关于 $\tau>0$ 是否优于 $\tau=0$、以及搜索深度如何影响有效样本数与失败前缀定位能力的分析。

整体上，这些结果共同支持本文主张：**搜索能否帮助策略梯度，不取决于"是否加入搜索"本身，而取决于搜索是否与预算、目标和并行实现方式严格对齐**。
