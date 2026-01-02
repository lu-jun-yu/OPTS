# OPTS_TTPO 详细设计文档

> **代码实现状态**：已完成。代码位于 `LLM/trainer/opts_ttpo/` 目录下。

## 1 算法概述

OPTS_TTPO（On-policy Parallel Tree Search + Tree Trajectory Policy Optimization）是一种将树搜索与策略梯度优化相结合的强化学习新范式，由 Junyu Lu 设计。

### 1.1 核心思想

该算法由两个紧密配合的部分构成：

**1. OPTS（同策略并行树搜索）**
- 不同于传统 MCTS 的完全扩展方式，OPTS 采用采样实例树的形式
- 每轮并行采样 `n` 条轨迹，通过 `g` 轮循环构建树结构
- 使用 TUCT（Trajectory-level Upper Confidence bound for Trees）选择下一轮扩展的最优状态
- 支持回溯到早期状态进行重新扩展，这在语言场景中是有益的

**2. TTPO（树轨迹策略优化）**
- 将 PPO 的优势估计扩展至树轨迹（TreeGAE）
- 将策略梯度扩展至树轨迹，引入分支权重因子进行梯度校正
- 保证在树结构上的策略梯度估计是无偏的

### 1.2 与 MCTS 的主要区别

| 特性 | MCTS | OPTS |
|------|------|------|
| 扩展方式 | 完全扩展所有可能动作 | 采样扩展，每轮 n 条轨迹 |
| 策略类型 | 异策略（树策略 vs 默认策略） | 同策略（采样策略即优化策略） |
| 状态选择 | 仅选择子节点 | 可选择树中任意状态 |
| 回溯机制 | 每次模拟后回溯更新 | 支持回溯到早期状态重新扩展 |


## 2 参数配置

OPTS_TTPO 在 PPO 的基础上，新增以下参数：

```yaml
actor_rollout_ref:
  rollout:
    g: 8          # 循环采样的轮数（总采样数 = n * g）
    search: opts  # 搜索算法：null 为标准 PPO，"opts" 为 On-Policy Parallel Tree Search

algorithm:
  c: 1.0          # TUCT 探索常数，控制探索与利用的平衡
```

**参数说明：**
- `g`：总共进行的采样轮数，决定树的深度和广度
- `search`：搜索算法类型，设为 "opts" 启用 OPTS_TTPO 模式
- `c`：探索常数，较大的值鼓励探索未充分访问的状态


## 3 数据结构详解

### 3.1 核心数据容器 DataProto

#### 3.1.1 batch（张量数据）

**原有键：**
| 键名 | 形状 | 说明 |
|------|------|------|
| input_ids | (bs, seq_len) | 输入 token ID |
| attention_mask | (bs, seq_len) | 注意力掩码 |
| position_ids | (bs, seq_len) | 位置 ID |
| responses | (bs, response_len) | 生成的响应 |
| prompts | (bs, prompt_len) | 提示 |
| response_mask | (bs, response_len) | 响应掩码（EOS后为0） |
| old_log_probs | (bs, response_len) | 旧策略 log 概率 |
| ref_log_prob | (bs, response_len) | 参考策略 log 概率 |
| values | (bs, response_len) | Critic 预测的状态价值 |
| token_level_scores | (bs, response_len) | token 级别分数 |
| token_level_rewards | (bs, response_len) | token 级别奖励 |
| advantages | (bs, response_len) | 优势函数值 |
| returns | (bs, response_len) | 回报 |

**OPTS_TTPO 新增键：**
| 键名 | 形状 | 说明 |
|------|------|------|
| advantages_mean | (bs, response_len-1) | 分支节点处所有动作的平均优势 |
| gamma_t | (bs, response_len) | gamma 的累乘：γ^t |
| lam_t | (bs, response_len) | lambda 的累乘：λ^t |
| trajectory_reward | (bs, response_len) | 累计折扣奖励 |
| state_branches | (bs, response_len) | 每个状态的分支数 |
| subtree_branches | (bs, response_len) | 子树的轨迹数 |
| branch_weight_factor | (bs, response_len) | 策略梯度权重因子（更新阶段计算） |

#### 3.1.2 non_tensor_batch（非张量数据）

**原有键：**
- `data_source`：数据来源标识
- `reward_model`：奖励模型配置
- `uid`：prompt 的唯一标识符
- `extra_info`：额外信息

**OPTS_TTPO 新增键：**
- `rid`：每条 response 的唯一标识
- `pid`：父轨迹的 rid（第一轮为 None）
- `cid`：子轨迹映射，ndarray of OrderedDict `{位置索引: [子轨迹rid列表]}`
- `branch_pos`：在父轨迹中的分支位置（第一轮为 -1）
- `new_sample_indices`: 新样本的索引列表

### 3.2 变量详解

#### 3.2.1 轨迹标识符

```
uid (Unique ID)
├── 标识原始 prompt
├── 同一 uid 下的所有轨迹共享同一个树结构
└── 用于按 prompt 分组进行状态选择

rid (Response ID)
├── 每条 response 轨迹的唯一标识
└── 用于建立父子关系

pid (Parent ID)
├── 指向父轨迹的 rid
├── 第一轮采样的轨迹 pid = None
└── 用于反向传播优势值和构建树结构

cid (Children ID)
├── 有序字典 OrderedDict[int, List[str]]
├── key: 分支发生的 token 位置索引
├── value: 从该位置出发的子轨迹 rid 列表
└── 用于前向遍历树结构和计算分支权重

branch_pos (Branch Position)
├── 当前轨迹在父轨迹中的分支位置
├── 第一轮采样的轨迹 branch_pos = -1
├── 用于：
│   ├── 1. compute_forward_values：从父轨迹继承 gamma_t、lam_t、trajectory_reward
│   └── 2. compute_branch_weight_factors：计算祖先轨迹的 state_branches 累乘
└── 由 set_opts_ttpo_info 函数生成并保存

new_sample_indices
├── 标识新样本的索引
└── 用于在计算 TreeGAE 时快速定位新样本

next_states（函数参数，不存储在 non_tensor_batch 中）
├── 字典 Dict[str, Tuple[int, int]]
├── key: uid（子轨迹的 uid，与父轨迹 uid 相同）
├── value: (parent_index, branch_pos)
│   ├── parent_index: 父轨迹在全局batch中的索引
│   └── branch_pos: 父轨迹中被选中状态的 token 位置索引
├── 用于：
│   ├── 1. set_opts_ttpo_info：通过 uid 直接查找父节点信息，设置 pid 和 branch_pos
│   ├── 2. compute_treegae_advantage_return：将子轨迹的优势值传播回父轨迹
│   └── 3. 构建下一轮的局部batch
└── 每轮选择后更新，标识本轮被选中进行扩展的状态
```

**示例：**
```
假设有如下树结构：
       root (prompt)
      /    \
   traj_0  traj_1    (第1轮，从 prompt 出发，无 pid)
     |
   traj_2            (第2轮，从 traj_0 的位置 5 出发)
    / \
traj_3 traj_4        (第3轮，从 traj_2 的位置 8 出发)

则：
- traj_0.cid = {5: ["traj_2"]}
- traj_2.cid = {8: ["traj_3", "traj_4"]}
- traj_2.pid = "traj_0"
- traj_3.pid = "traj_2"
```

#### 3.2.2 时间累积量

**gamma_t[t]**：γ^t，用于折扣未来奖励
- t=0 时：gamma_t[0] = 1
- t>0 时：gamma_t[t] = gamma_t[t-1] * gamma
- 若有父轨迹：从父轨迹所选状态的 gamma_t 继续累乘

**lam_t[t]**：λ^t，用于 TUCT 中的公平比较
- t=0 时：lam_t[0] = 1
- t>0 时：lam_t[t] = lam_t[t-1] * lam
- 若有父轨迹：从父轨迹所选状态的 lam_t 继续累乘

**trajectory_reward[t]**：到达状态 t 时的累计折扣奖励
- t=0 时：trajectory_reward[0] = token_level_rewards[0]
- t>0 时：trajectory_reward[t] = trajectory_reward[t-1] + gamma_t[t] * reward[t]
- 若有父轨迹：从父轨迹所选状态的 trajectory_reward 继续累加

#### 3.2.3 分支相关计数

**state_branches[t]**：状态 t 处的分支数
- 初始化：全部为 1
- 当状态 t 被选中进行扩展时：state_branches[t] = n + 1（+1 是因为原轨迹本身也算一个分支）
- 用于计算 branch_weight_factor

**subtree_branches[t]**：经过状态 t 的轨迹总数
- 初始化：全部为 1
- 当某个下游状态被选中扩展时：祖先轨迹的 subtree_branches += n
- 用于 TUCT 探索项的分母

**partree_branches[t]**：父分支点的 subtree_branches（临时计算量）
- 父分支点：状态 t 上游最近的分支节点
- 用于 TUCT 探索项的分子
- 计算方式：
  - 若当前 response 无状态 t 的上游分支节点：
    - 若存在父轨迹，使用 prompt 对应的父轨迹状态的 subtree_branches
    - 若不存在父轨迹，使用根状态的 subtree_branches：(i + 1) * n
  - 若当前 response 有状态 t 的上游分支节点：使用该分支节点的 subtree_branches

#### 3.2.4 价值估计相关

**gve[t]**：广义状态价值估计 (Generalized Value Estimate)
- 公式：gve[t] = values[t+1] + lam * advantages_mean[t]
- 维度：(bs, response_len - 1)，对应位置 0 到 response_len-2
- 用于估计从状态 t 出发的期望价值

**expected_trajectory_reward[t]**：期望轨迹累计奖励
- 公式：trajectory_reward[t] + gamma_t[t] * gve[t]
- 含义：到达状态 t 的实际奖励 + 未来期望奖励
- 用于 TUCT 的利用项

**advantages_mean[t]**：分支节点处的平均优势
- 非分支节点：advantages_mean[t] = advantages[t+1]
- 分支节点：所有分支第一个 token 的 advantage 的平均值
- 用于 TreeGAE 计算

#### 3.2.5 TUCT 和权重因子

**tuct[t]**：Tree UCT 值
- 公式：expected_trajectory_reward[t] * lam_t[t+1] + c * sqrt(log(partree_branches[t])) / subtree_branches[t]
- 利用项：expected_trajectory_reward * lam_t（lam_t 用于公平比较不同深度的状态）
- 探索项：c * sqrt(log(N_parent)) / N_child（UCB1 风格）

**branch_weight_factor[t]**：策略梯度权重因子
- t=0 时：branch_weight_factor[0] = 1
- t>0 时：branch_weight_factor[t] = branch_weight_factor[t-1] * state_branches[t-1]
- 含义：祖先轨迹分支数的累乘
- 用于校正策略梯度，保证无偏估计


## 4 训练流程详解

### 4.1 整体流程图

```
┌─────────────────────────────────────────────────────────────┐
│                      for epoch in epochs:                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  for batch in batches:                 │  │
│  │                                                        │  │
│  │  ┌──────────────────────────────────────────────────┐ │  │
│  │  │  1. 数据准备                                      │ │  │
│  │  └──────────────────────────────────────────────────┘ │  │
│  │                          ↓                             │  │
│  │  ┌──────────────────────────────────────────────────┐ │  │
│  │  │  2. 采样循环：for i in range(g):                  │ │  │
│  │  │     a. 前向：生成轨迹，计算各种值                  │ │  │
│  │  │     b. 反向：TreeGAE / GAE 计算优势               │ │  │
│  │  │     c. 选择：TUCT 选择下一轮扩展状态（仅OPTS_TTPO）│ │  │
│  │  └──────────────────────────────────────────────────┘ │  │
│  │                          ↓                             │  │
│  │  ┌──────────────────────────────────────────────────┐ │  │
│  │  │  3. 更新：更新 Actor 和 Critic                    │ │  │
│  │  └──────────────────────────────────────────────────┘ │  │
│  │                                                        │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 训练流程伪代码

```
for epoch in ...:

    for batch in ...:

        ======== 1. 数据准备 ========

        if opts_ttpo:
            - 全局batch：保存每一轮的局部batch，用于生成下一轮的局部batch
            - 局部batch：每轮动态变化，由"prompt+部分已采样结果"构建后重复n遍
        else (PPO模式):
            - 无全局/局部batch区分，直接使用原始batch
            - g = 1

        ======== 2. 采样循环 ========

        for i in range(g):

            -------- a. 前向 --------

            - (局部batch) 重复样本
            - (局部batch) 生成序列
            - (局部batch) 计算奖励
            - (局部batch) 计算旧策略log概率
            - (局部batch) 计算参考策略log概率
            - (局部batch) 计算values

            if opts_ttpo:
                - _set_opts_ttpo_info：建立树结构关系
                    - (局部batch) 为每条轨迹生成唯一 rid
                    - (局部batch) 根据 next_states 设置 pid 和 branch_pos：
                        - 第一轮：pid=None, branch_pos=-1
                        - 后续轮：通过 uid 查询 next_states 获取父节点信息
                    - (局部batch) 初始化 cid 为空 OrderedDict
                    - (全局batch) 在父轨迹的 cid 中注册当前轨迹为子节点
                    - 返回 new_sample_indices
                - (局部batch) 初始化 state_branches、subtree_branches 为全1
                - (局部batch) 初始化 advantages、advantages_mean、returns 为全0
                - (局部batch) compute_forward_values：计算时间累积量
                    - gamma_t[t] = γ^t（若有父节点则从父节点位置继承后继续累乘）
                    - lam_t[t] = λ^t（同上）
                    - trajectory_reward[t] = 累计折扣奖励（同上）
                - (局部batch) 合并到全局batch（_merge_batches）

            -------- b. 反向 --------

            - 使用原有的 compute_advantage 函数，通过 adv_estimator 参数选择计算方式：
                - if adv_estimator == AdvantageEstimator.TreeGAE:
                    - 调用 compute_treegae_advantage_return：
                        - 第一个循环：对新样本执行标准 GAE（从后向前计算优势值）
                        - 第二个循环：向祖先轨迹传播优势值（使用线程池并行）
                            - 非分支位置：正常 GAE 传播
                            - 分支位置：取所有子分支首 token 优势的均值再传播
                - elif adv_estimator == AdvantageEstimator.GAE:
                    - 调用原有的 compute_gae_advantage_return

            -------- c. 选择 --------

            if opts_ttpo and i < g - 1:  # 仅在非最后一轮执行
                - (全局batch) select_next_states：用 TUCT 选择下一轮扩展的状态
                    1) 计算各状态的期望轨迹奖励：
                       - gve = values[1:] + lam * advantages_mean[:]
                       - expected_trajectory_reward = trajectory_reward[:-1] + gamma_t[:-1] * gve

                    2) 计算 partree_branches（父分支点的 subtree_branches）

                    3) 计算 TUCT：
                       - tuct = expected_trajectory_reward * lam_t[1:] + c * sqrt(log(partree_branches + 1)) / subtree_branches[:-1]

                    4) 为每个 uid 选择 TUCT 最高的状态：
                       - 同时考虑"根状态"（从 prompt 重新开始），与树中状态竞争
                       - 返回 selected_states = [(rid, pos), ...]

                    5) 更新 subtree_branches：沿祖先链向上传播（+= n）

                - (全局batch) 更新 state_branches：被选中状态的分支数 += n

                - 构建 next_states 字典供下一轮使用：
                    - next_states = {uid: (parent_idx, branch_pos) for 每个选中状态}

                - _prepare_next_round_input：构建下一轮输入
                    - 提取 prompt + 部分响应（到选中位置）作为新的 input_ids
                    - 重新构建 left-padded 的输入张量
                    - 从根节点继承 data_source、reward_model、extra_info 等元信息

        ======== 3. 更新 ========

        if opts_ttpo:
            - 使用全局 batch 进行更新（包含完整的树）
            - 对优势进行白化（whiten）
            - compute_branch_weight_factors：计算策略梯度的权重校正因子
                - W_t = 祖先轨迹所有分支数的累乘（保证树结构上的无偏梯度估计）

        - 更新 Critic：Loss 计算与 PPO 一致

        - 更新 Actor：
            - 若存在 branch_weight_factor：
                - pg_loss = pg_loss / W_t
                - 使用 "weighted-token-mean" 聚合模式：
                    loss = masked_sum(loss_mat) / masked_sum(1/W_t) * dp_size
            - 否则使用标准 PPO loss
```


## 5 关键公式汇总

### 5.1 TreeGAE

$$
\hat{A}_t = \begin{cases}
\delta_t + \gamma \lambda \hat{A}_{t+1} & \text{if } t \notin \text{branch\_nodes} \\
\delta_t + \gamma \lambda \cdot \frac{1}{|B_t|} \sum_{b \in B_t} \hat{A}_b^{(0)} & \text{if } t \in \text{branch\_nodes}
\end{cases}
$$

其中：
- $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 为 TD 误差
- $B_t$ 为从状态 $t$ 分出的所有子轨迹集合
- $\hat{A}_b^{(0)}$ 为子轨迹 $b$ 的第一个 token 的优势值

### 5.2 TUCT

$$
\text{TUCT}(s_t) = \underbrace{\bigg[R(\tau_{0:t}) + \gamma^t \cdot \text{GVE}(s_t)\bigg]}_{\text{利用项}} \cdot \lambda^t + \underbrace{c \cdot \frac{\sqrt{\log (N_{\text{parent}} + 1)}}{N_{\text{child}}}}_{\text{探索项}}
$$

其中：
- $R(\tau_{0:t})$ 为到达状态 $t$ 的累计折扣奖励
- $\text{GVE}(s_t) = V(s_{t+1}) + \lambda \cdot \bar{A}(s_{t+1})$ 为广义状态价值估计
- $\lambda^t$ 用于公平比较不同深度的状态
- $N_{\text{parent}}$ 为父分支点的 subtree_branches
- $N_{\text{child}}$ 为当前状态的 subtree_branches

### 5.3 TTPO 策略梯度

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_t \frac{1}{W_t} \nabla_\theta \log \pi_\theta(a_t|s_t) \hat{A}_t \right]
$$

其中 $W_t = \prod_{i=0}^{t-1} |B_i|$ 为 branch_weight_factor。


## 6 实现要求

1. **代码位置**：所有改动不应出现在 `LLM/verl` 中，需要修改的文件复制到 `LLM/` 下并保持目录结构
2. **代码风格**：与原文件风格保持一致
3. **日志输出**：不需要打印调试信息
4. **优化建议**：如有更好的实现方式，优先采用
5. **最小化改动原则**：
   - 优先通过注册机制扩展（如 `@register_adv_est`）而非创建独立函数
   - 优先在现有函数中添加条件分支而非创建新的并行函数
   - 新增参数应设置合理默认值，确保对 PPO 模式的兼容性


## 7 附录：核心函数清单

### 7.1 代码文件结构

```
LLM/trainer/opts_ttpo/
├── main_opts_ttpo.py     # 入口文件
├── ray_trainer.py        # RayOPTSTTPOTrainer 类
├── core_algos.py         # 核心算法函数
└── opts_ttpo_v1.md       # 本文档
```

### 7.2 核心函数列表

| 函数名 | 所在文件 | 类型 | 说明 |
|--------|----------|------|------|
| `_set_opts_ttpo_info` | ray_trainer.py | RayOPTSTTPOTrainer 方法 | 设置树结构信息：rid, pid, branch_pos, cid |
| `_prepare_next_round_input` | ray_trainer.py | RayOPTSTTPOTrainer 方法 | 构建下一轮采样的输入 batch |
| `_merge_batches` | ray_trainer.py | RayOPTSTTPOTrainer 方法 | 合并局部 batch 到全局 batch |
| `compute_forward_values` | core_algos.py | 新函数 | 计算 gamma_t, lam_t, trajectory_reward |
| `compute_treegae_advantage_return` | core_algos.py | 注册函数 | 通过 `@register_adv_est(AdvantageEstimator.TreeGAE)` 注册 |
| `select_next_states` | core_algos.py | 新函数 | TUCT 状态选择 |
| `compute_partree_branches` | core_algos.py | 新函数 | 计算父分支点的 subtree_branches |
| `compute_branch_weight_factors` | core_algos.py | 新函数 | 计算分支权重因子 |
| `agg_loss` | core_algos.py | 修改 | 新增 "weighted-token-mean" 模式 |
| `compute_policy_loss_vanilla` | core_algos.py | 修改 | 新增 branch_weight_factor 参数 |
| `AdvantageEstimator` | core_algos.py | 枚举扩展 | 新增 `TreeGAE = "treegae"` |

### 7.3 配置参数

在 verl 配置文件中新增以下参数：

```yaml
actor_rollout_ref:
  rollout:
    g: 8          # 循环采样的轮数（总采样数 = n * g）
    search: opts  # 搜索算法："opts" 启用 OPTS_TTPO

algorithm:
  adv_estimator: treegae  # 使用 TreeGAE 优势估计
  c: 1.0                  # TUCT 探索常数
  gamma: 1.0              # 折扣因子
  lam: 0.95               # GAE lambda
```

### 7.4 verl 框架修改

为支持 OPTS_TTPO，需要对 verl 框架进行以下修改：

#### 7.4.1 配置文件修改

**`LLM/verl/verl/trainer/config/rollout/rollout.yaml`**：
```yaml
# This controls the third loop inside fit() for TTPO algorithm
g: 1

# OPTS parameters
c: 1.0

# Search algorithm: null for standard PPO, "opts" for OPTS
search: null
```

**`LLM/verl/verl/workers/config/rollout.py`**：
```python
@dataclass
class RolloutConfig(BaseConfig):
    # ...
    g: int = 1
    c: float = 1.0
    search: str = None
```

#### 7.4.2 Agent Loop 续写模式

OPTS_TTPO 需要从已有的 `input_ids` 续写生成，而不是从 `raw_prompt` 重新生成。

**`LLM/verl/verl/experimental/agent_loop/agent_loop.py`**：

1. **`AgentLoopBase` 基类**：新增 `run_from_input_ids` 抽象方法，定义续写模式接口，接收 `prompt_ids` 参数而非从 `raw_prompt` 解析

2. **`generate_sequences` 方法**：添加续写模式分支，当 `batch.meta_info["round_idx"] >= 1` 时调用 `_generate_from_input_ids`

3. **`_run_agent_loop_from_input_ids` 方法**（新增）：仿照 `_run_agent_loop` 实现，创建 agent loop 实例后调用其 `run_from_input_ids` 方法，最后通过 `_agent_loop_postprocess` 进行后处理

4. **`_generate_from_input_ids` 方法**：仿照 `generate_sequences` 的完整模式实现
   - 设置 sampling_params、agent_name、index、traced_indices 等
   - 提取有效 token（移除 left padding）
   - 为每个样本创建异步任务调用 `_run_agent_loop_from_input_ids`
   - 使用 `_postprocess(outputs)` 进行后处理，与原有流程对齐

**`LLM/verl/verl/experimental/agent_loop/single_turn_agent_loop.py`**：

5. **`SingleTurnAgentLoop` 类**：实现 `run_from_input_ids` 方法，接收 `prompt_ids` 直接调用 `server_manager.generate` 生成新 token，返回 `AgentLoopOutput`

**关键改动说明**：

- 通过 `run_from_input_ids` 抽象方法，各 AgentLoop 子类可以自定义续写逻辑
- `_generate_from_input_ids` 直接复用 `_postprocess` 和 `_agent_loop_postprocess`，保证与原有流程输出格式完全一致

#### 7.4.3 Actor 策略损失修改

**`LLM/verl/verl/workers/actor/dp_actor.py`**：

1. **导入本地 `core_algos`**：优先使用 OPTS_TTPO 的 `core_algos` 以支持 `branch_weight_factor`
   ```python
   try:
       from trainer.opts_ttpo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
   except ImportError:
       from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
   ```

2. **提取并传递 `branch_weight_factor`**：
   ```python
   # Include branch_weight_factor for OPTS_TTPO gradient correction
   if "branch_weight_factor" in data.batch.keys():
       select_keys.append("branch_weight_factor")

   # Extract branch_weight_factor for OPTS_TTPO gradient correction
   branch_weight_factor = model_inputs.get("branch_weight_factor", None)

   # Compute policy loss with branch_weight_factor
   if branch_weight_factor is not None:
       pg_loss, pg_metrics = policy_loss_fn(..., branch_weight_factor=branch_weight_factor)
   ```

#### 7.4.4 Critic Value Head 激活函数

为支持 0-1 范围的规则奖励，在 verl 的 critic 配置中新增 `value_head_activation` 参数，可选值为 `none`、`sigmoid`、`tanh`。设为 `sigmoid` 时，critic 输出将被限制到 (0, 1) 范围，与规则奖励的范围对齐。

修改文件：
- `verl/workers/config/critic.py`：新增 `value_head_activation` 配置项
- `verl/trainer/config/critic/critic.yaml`：新增对应的 yaml 配置
- `verl/workers/critic/dp_critic.py`：在 `_forward_micro_batch` 中根据配置应用激活函数

#### 7.4.5 注意事项

1. **Tensor 布尔判断**：使用 `if tensor is not None:` 而非 `if tensor:`，避免多元素 Tensor 的歧义错误
2. **`non_tensor_batch` 类型约束**：`DataProto.non_tensor_batch` 的所有值必须是 `np.ndarray` 类型
   - `cid` 存储为 `np.array([OrderedDict() for ...], dtype=object)`（正确）
   - `next_states` 是 `Dict[str, Tuple[int, int]]`，不能存入 `non_tensor_batch`，应作为函数参数传递

