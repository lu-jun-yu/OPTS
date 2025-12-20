# OPTS_TTPO 详细设计文档

> 本文档基于 `opts_ttpo.md` 需求文档，对 OPTS_TTPO 算法进行详细的技术阐述。

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
- `cid`：子轨迹映射，有序字典 `{位置索引: [子轨迹rid列表]}`
- `branch_pos`：在父轨迹中的分支位置（第一轮为 -1）
- `new_sample_indices`: 新样本的索引列表
- `next_states`: 本轮循环中被选中的状态索引

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

next_states
├── 字典 Dict[str, Tuple[int, int]]
├── key: pid（父轨迹的 rid）
├── value: (parent_index, branch_pos)
│   ├── parent_index: 父轨迹在全局batch中的索引
│   └── branch_pos: 父轨迹中被选中状态的 token 位置索引
├── 用于：
│   ├── 1. compute_treegae_advantage_return：将子轨迹的优势值传播回父轨迹
│   └── 2. 构建下一轮的局部batch
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
                - set_opts_ttpo_info【新函数】：
                    - (局部batch) 生成rid：所有response都有rid、uid，但第一轮的response没有pid
                    - (全局batch) 在父轨迹上插入新的cid键值
                - (局部batch) 初始化state_branches、subtree_branches为全1，初始化advantages、advantages_mean、returns为全0【新行】
                - (局部batch) compute_forward_values【新函数】：
                    - gamma_t = []
                    - lam_t = []
                    - trajectory_reward = []
                    - last_gamma_t = torch.where(pid存在, 全局batch的gamma_t[next_states[pid]], 1 / gamma)
                    - last_lam_t = torch.where(pid存在, 全局batch的lam_t[next_states[pid]], 1 / lam)
                    - last_trajectory_reward = torch.where(pid存在, 全局batch的trajectory_reward[next_states[pid]], 0)
                    - for ...:
                        - last_gamma_t *= gamma
                        - last_lam_t *= lam
                        - last_trajectory_reward += last_gamma_t * token_level_rewards[t]
                        - gamma_t.append(last_gamma_t)
                        - lam_t.append(last_lam_t)
                        - trajectory_reward.append(last_trajectory_reward)
                    - gamma_t = torch.stack(gamma_t, dim=1)
                    - lam_t = torch.stack(lam_t, dim=1)
                    - trajectory_reward = torch.stack(trajectory_reward, dim=1)
                    - return gamma_t, lam_t, trajectory_reward
                - 将上述函数返回的结果赋值到局部batch
                - 局部batch添加至全局batch【新行】
                    - 得到 new_sample_indices

            -------- b. 反向 --------

            - 使用原有的compute_advantage函数，通过adv_estimator参数选择计算方式：
                - if adv_estimator == AdvantageEstimator.TreeGAE:
                    - 调用compute_treegae_advantage_return【新注册函数】：

                        - 第一个循环（对新样本进行标准GAE）：
                            - 只处理 new_sample_indices 对应的样本
                            - for t in reversed(range(gen_len)):
                                - delta = rewards[:, t] + gamma * nextvalues - values[:, t]
                                - lastgaelam = delta + gamma * lam * lastgaelam
                            - 保存到 advantages[new_sample_indices] 和 advantages_mean[new_sample_indices]

                        - 第二个循环（使用next_states向祖先轨迹传播优势值）：
                            - current_level = next_states.values()
                            - while current_level:
                                - for 每个父轨迹 (p_idx, branch_pos):
                                    - Step 1: 处理 branch_pos（分支节点）
                                        - child_indices = [rid2idx[c_rid] for c_rid in parent_cid[branch_pos]]
                                        - lastgaelam_mean = (advantages[child_indices, 0].sum() + advantages[p_idx, branch_pos + 1]) / state_branches
                                        - 计算并保存 advantages_mean 和 advantages
                                    - Step 2: 从 branch_pos - 1 反向遍历到 0
                                        - if 分支节点: 计算 lastgaelam_mean 并更新
                                        - else: 标准 GAE 更新
                                    - 若父轨迹有祖父轨迹，加入 next_level
                                - current_level = next_level
                - elif adv_estimator == AdvantageEstimator.GAE:
                    - 调用原有的compute_gae_advantage_return

            -------- c. 选择 --------

            if opts_ttpo:
                - (全局batch) select_next_states【新函数】：

                    1) 计算各状态的评估值：
                       - gve = values[1:] + lam * advantages_mean[:]
                       - expected_trajectory_reward = trajectory_reward[:-1] + gamma_t[:-1] * gve

                    2) 计算partree_branches【新函数：compute_partree_branches】：
                       - 初始化：与subtree_branches形状一样的全零Tensor
                       - if cid is None：
                           - partree_branches[:] = 父轨迹所选状态的subtree_branches if pid is not None else (i+1) * n
                       - else：
                           - partree_branches[:cid.keys()[0]+1] = 父轨迹所选状态的subtree_branches (if pid存在)
                                                   else (i+1) * n
                           - for j in range(len(cid.keys())-1):
                               - partree_branches[cid.keys()[j]+1: cid.keys()[j+1]+1] = subtree_branches[cid.keys()[j]]
                           - partree_branches[cid.keys()[-1]+1:] = subtree_branches[cid.keys()[-1]]

                    3) 计算tuct：
                       - tuct = expected_trajectory_reward * lam_t[1:] + c * sqrt(log(partree_branches)) / subtree_branches[:-1]

                    4) 为每个uid选择最优状态——最大tuct的状态索引：next_states = {pid: (parent_index, branch_pos)}
                       - 计算根状态的tuct：
                           - root.gve = values[0] + lam * mean(advantages[相同uid且无pid的response][0])
                           - root.tuct = root.gve + c * 1
                       - 比较最大tuct状态与根状态，取tuct较大者

                    5) 更新分支计数：
                       - 被选中状态的state_branches = n + 1
                       - 从被选中状态向上更新祖先轨迹的subtree_branches += n

                    6) 构建新的局部batch：
                        - 使用next_states构建完整状态作为输入
                        - 设置：pid、uid、branch_pos及其他信息
                        - branch_pos = next_states中的位置索引（第一轮为-1）

        ======== 3. 更新 ========

        if opts_ttpo:
            - compute_branch_weight_factors【新函数】：
                - 输入：state_branches, pid, rid, branch_pos

                - Step 1: 计算每个样本的初始权重（祖先链累乘）
                    - init_weights = torch.ones(batch_size)
                    - for 每个样本 i:
                        - current_idx, current_bp = i, branch_pos[i]
                        - while pid[current_idx] 存在且在 rid2idx 中:
                            - parent_idx = rid2idx[pid[current_idx]]
                            - init_weights[i] *= state_branches[parent_idx, :current_bp+1].prod()
                            - current_idx, current_bp = parent_idx, branch_pos[parent_idx]

                - Step 2: 前向累乘（使用 cumprod）
                    - padded = concat([ones(bs,1), state_branches[:, :-1]], dim=1)
                    - cumulative = cumprod(padded, dim=1)
                    - branch_weight_factor = init_weights.unsqueeze(1) * cumulative

                - return branch_weight_factor

        - 更新critic：Critic Loss计算与先前保持一致

        - 更新actor（在原有policy loss函数基础上修改）：
            - agg_loss函数新增"weighted-token-mean"分支【修改agg_loss】：
                - elif loss_agg_mode == "weighted-token-mean":
                    - loss = masked_sum(loss_mat) / masked_sum(1/branch_weight_factor) * dp_size
            - compute_policy_loss_vanilla等函数新增branch_weight_factor参数【修改现有函数签名】：
                - if branch_weight_factor is not None:
                    - pg_losses = pg_losses / branch_weight_factor【新行】
                    - 使用"weighted-token-mean"模式调用agg_loss
                - else:
                    - 使用原有的loss计算逻辑
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
\text{TUCT}(s_t) = \underbrace{\bigg[R(\tau_{0:t}) + \gamma^t \cdot \text{GVE}(s_t)\bigg]}_{\text{利用项}} \cdot \lambda^t + \underbrace{c \cdot \frac{\sqrt{\log N_{\text{parent}}}}{N_{\text{child}}}}_{\text{探索项}}
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

| 函数名 | 类型 | 说明 |
|--------|------|------|
| `set_opts_ttpo_info` | 新函数 | 设置 rid, pid, uid, cid 等树结构信息 |
| `compute_forward_values` | 新函数 | 计算 gamma_t, lam_t, trajectory_reward |
| `compute_treegae_advantage_return` | 新注册函数 | 通过 `@register_adv_est(AdvantageEstimator.TreeGAE)` 注册，由 `compute_advantage` 调用 |
| `select_next_states` | 新函数 | TUCT 状态选择 |
| `compute_partree_branches` | 新函数 | 计算父分支点的 subtree_branches |
| `compute_branch_weight_factors` | 新函数 | 计算分支权重因子（祖先链累乘 + cumprod） |
| `agg_loss` | 修改 | 新增 "weighted-token-mean" 模式，需传入 branch_weight_factor |
| `compute_policy_loss_vanilla` 等 | 修改 | 新增 branch_weight_factor 参数，条件判断是否应用权重校正 |
| `AdvantageEstimator` | 修改 | 新增 `TreeGAE = "treegae"` 枚举值 |
