<!-- Copyright 2025 Junyu Lu (Julian Lou). All rights reserved. -->

# OPTS_TTPO 详细设计文档

> **代码实现状态**：已完成。代码位于 `LLM/trainer/opts_ttpo/` 目录下。

## 1 算法概述

OPTS_TTPO（On-policy Parallel Tree Search + Tree Trajectory Policy Optimization）是一种将树搜索与策略梯度优化相结合的强化学习新范式，由 Junyu Lu 设计。

### 1.1 核心思想

该算法由两个紧密配合的部分构成：

**1. OPTS（同策略并行树搜索）**
- 不同于传统 MCTS 的完全扩展方式，OPTS 采用采样实例树的形式
- 每步（step）进行 `n_rounds` 轮循环采样，每轮生成一个 batch 的轨迹，逐步构建树结构
- 使用 OTRC（On-policy Trajectory Rebranching Criterion）选择下一轮扩展的最优状态
- 支持回溯到早期状态进行重新扩展，这在语言场景中是有益的

**2. TTPO（树轨迹策略优化）**
- 将 PPO 的优势估计扩展至树轨迹（TreeGAE）
- 将策略梯度扩展至树轨迹，引入分支权重因子进行梯度校正
- 保证在树结构上的策略梯度估计是无偏的

### 1.2 与 MCTS 的主要区别

| 特性 | MCTS | OPTS |
|------|------|------|
| 扩展方式 | 完全扩展所有可能动作 | 采样扩展，每轮一个 batch 的轨迹 |
| 策略类型 | 异策略（树策略 vs 默认策略） | 同策略（采样策略即优化策略） |
| 状态选择 | 仅选择子节点 | 可选择树中任意状态 |
| 回溯机制 | 每次模拟后回溯更新 | 支持回溯到早期状态重新扩展 |


## 2 参数配置

OPTS_TTPO 在 PPO 的基础上，新增以下参数：

```yaml
actor_rollout_ref:
  rollout:
    n: 4                    # 循环采样的轮数
    max_search_per_tree: 4  # 每棵树每个训练迭代最大搜索次数
    c: 1.0                  # OTRC 探索项系数
```

**参数说明：**
- `n`（n_rounds）：总共进行的采样轮数，决定树的深度和广度
- `max_search_per_tree`：每棵树（uid）在一个训练 step 内允许的最大搜索次数，达到上限后该树不再被 OTRC 选中
- `c`：OTRC 公式中 exploration 项的系数，控制利用与探索的平衡


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
| state_branches | (bs, response_len) | 每个状态的分支数 |
| branch_weight | (bs, response_len) | 策略梯度权重因子（更新阶段计算） |

#### 3.1.2 non_tensor_batch（非张量数据）

**原有键：**
- `data_source`：数据来源标识
- `reward_model`：奖励模型配置
- `uid`：prompt 的唯一标识符，同时作为树标识符（tree ID）
- `extra_info`：额外信息
- `raw_prompt_len`：原始 prompt 长度

**OPTS_TTPO 新增键：**
- `rid`：每条 response 的唯一标识，格式为 `r{round_idx}_{batch_idx}`
- `pid`：父轨迹的 rid（第一轮为 None）
- `cid`：子轨迹映射，ndarray of OrderedDict `{位置索引: [子轨迹rid列表]}`
- `branch_pos`：在父轨迹中的分支位置（第一轮为 -1）
- `new_sample_indices`：新样本在全局 batch 中的索引
- `episodic_returns`：每条轨迹的完整 episodic return（含祖先奖励）

### 3.2 变量详解

#### 3.2.1 轨迹标识符

```
uid (Unique ID / Tree ID)
├── 标识原始 prompt，同时作为树标识符
├── 同一 uid 下的所有轨迹共享同一个树结构
├── 第一轮采样时由 uuid.uuid4() 生成
└── 用于按 prompt 分组进行 OTRC 状态选择

rid (Response ID)
├── 每条 response 轨迹的唯一标识
├── 格式：r{round_idx}_{batch_idx}
└── 用于建立父子关系

pid (Parent ID)
├── 指向父轨迹的 rid
├── 第一轮采样的轨迹 pid = None
└── 用于反向传播优势值和构建树结构

cid (Children ID)
├── 有序字典 OrderedDict[int, List[str]]
├── key: 分支发生的 token 位置索引
├── value: 从该位置出发的子轨迹 rid 列表
└── 由 set_opts_ttpo_info 在父轨迹上注册

branch_pos (Branch Position)
├── 当前轨迹在父轨迹中的分支位置
├── 第一轮采样的轨迹 branch_pos = -1
└── 由 selected_to_branch_points 转换得到

episodic_returns
├── 每条轨迹的完整 episodic return
├── 沿祖先链累加奖励：own_rewards + ancestor_rewards[:branch_pos+1] + ...
├── 由 compute_episodic_returns 计算
└── 用于日志与 aggregated return 统计

next_states（函数参数，不存储在 non_tensor_batch 中）
├── 字典 Dict[str, Tuple[int, int]]
├── key: uid
├── value: (parent_index, branch_pos)
│   ├── parent_index: 父轨迹在全局 batch 中的索引
│   └── branch_pos: 父轨迹中被选中状态的 token 位置索引
├── 由 selected_to_branch_points 从 OTRC 选择结果转换得到
└── 用于 set_opts_ttpo_info 和 prepare_next_round_input
```

**示例：**
```
假设有如下树结构：
       root (prompt, uid="abc")
      /    \
   traj_0  traj_1    (第1轮，从 prompt 出发，pid=None)
     |
   traj_2            (第2轮，从 traj_0 的位置 5 出发)
    / \
traj_3 traj_4        (第3轮，从 traj_2 的位置 8 出发)

则：
- traj_0.cid = {5: ["traj_2"]}
- traj_2.cid = {8: ["traj_3", "traj_4"]}
- traj_2.pid = "traj_0", traj_2.branch_pos = 5
- traj_3.pid = "traj_2", traj_3.branch_pos = 8
- 所有轨迹的 uid = "abc"
```

#### 3.2.2 分支相关计数

**state_branches[i, t]**：轨迹 i 的状态 t 处的分支数
- 初始化：全部为 1
- 当 OTRC 选择从状态 (i, t) 分支时：`state_branches[i, t] += 1`
- 更新在 `selected_to_branch_points` 中完成
- 用于计算 branch_weight

#### 3.2.3 权重因子

**branch_weight[i, t]**：轨迹 i 位置 t 的策略梯度权重因子
- 计算过程分两步：
  1. **初始权重**：沿祖先链从当前轨迹追溯到根，累乘每段祖先轨迹上 `state_branches[:branch_pos+1]` 的乘积，最后乘以根处同 uid 的根轨迹数
  2. **轨迹内传播**：`weight[t] = init_weight * cumprod(state_branches[0:t])`
- 含义：从根到当前节点路径上所有祖先分支数的累乘
- 用于校正策略梯度，保证无偏估计


## 4 训练流程详解

### 4.1 整体流程图

```
┌─────────────────────────────────────────────────────────────┐
│                      for epoch in epochs:                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  for step in steps:                     │  │
│  │                                                        │  │
│  │  ┌──────────────────────────────────────────────────┐ │  │
│  │  │  采样循环：for round_idx in range(n_rounds):      │ │  │
│  │  │     a. 构建本轮 batch                             │ │  │
│  │  │     b. 前向：生成轨迹，计算各种值                  │ │  │
│  │  │     c. 设置树结构信息，合并到全局 batch             │ │  │
│  │  │     d. 反向：TreeGAE 计算优势                     │ │  │
│  │  │     e. 选择：OTRC 选择下一轮扩展状态（非最后一轮）  │ │  │
│  │  └──────────────────────────────────────────────────┘ │  │
│  │                          ↓                             │  │
│  │  ┌──────────────────────────────────────────────────┐ │  │
│  │  │  后处理：计算 branch_weight，进行加权白化优势        │ │  │
│  │  │  计算 aggregated_returns，更新 step_mean_return     │ │  │
│  │  └──────────────────────────────────────────────────┘ │  │
│  │                          ↓                             │  │
│  │  ┌──────────────────────────────────────────────────┐ │  │
│  │  │  更新：更新 Critic 和 Actor                       │ │  │
│  │  └──────────────────────────────────────────────────┘ │  │
│  │                                                        │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 训练流程伪代码

```
prompt_buffer = PromptBuffer(train_dataloader)

for epoch in ...:
    for step in ...:
        初始化 global_batch、next_states、search_count

        ======== 采样循环 ========

        for round_idx in range(n_rounds):

            -------- a. 构建本轮 batch --------

            if 第一轮:
                从 prompt_buffer 中抽取 batch_size 个样本
                为每个样本分配新的 uid
            else:
                k = 继续搜索的树数量（next_states 的大小）
                若 k > 0，用 prepare_next_round_input 从 global_batch 构建续写输入
                若 k < batch_size，从 prompt_buffer 补充新 prompt 并分配新 uid
                合并为本轮 batch

            -------- b. 前向 --------

            生成轨迹（支持续写模式），计算 reward、old_log_probs、ref_log_prob、values

            -------- c. 设置树结构，合并到全局 batch --------

            set_opts_ttpo_info：为本轮 batch 设置树结构信息
              - 生成 rid
              - 根据 next_states 设置 pid 和 branch_pos
              - 在父轨迹的 cid 中注册子节点

            compute_episodic_returns：沿祖先链累加奖励，得到完整 episodic return

            初始化 state_branches 为全 1，advantages 和 returns 为全 0
            将本轮 batch 合并到 global_batch

            -------- d. 反向：TreeGAE --------

            compute_treegae_advantage_return：在 global_batch 上计算 TreeGAE 优势
              - 第一个循环：对新样本执行标准 GAE
              - 第二个循环：向祖先轨迹传播优势值（线程池并行）
                - 非分支位置：正常 GAE 传播
                - 分支位置：取所有子分支首 token 优势的均值再传播

            -------- e. 选择（非最后一轮） --------

            if 非最后一轮:
                select_next_states：用 OTRC 选择下一轮扩展状态
                  - 跳过搜索次数已达上限的树
                  - 沿最优路径计算 OTRC，选择 argmax
                  - 记录各树的 max_exploitations[uid] = exploitation[k]
                  - 用 max_exploitations 的均值做门控（仅保留 exploitation[k] 更大的候选）
                  - 应用 prompt 长度约束和 </think> 位置掩码
                  - 全局排序，取 top batch_size 个候选

                selected_to_branch_points：将选中节点转换为其父节点作为分支点，更新 state_branches

        ======== 后处理 ========

        compute_branch_weight：沿祖先链追溯到根，累乘 state_branches，根节点乘以同 uid 的根轨迹数
        weighted_masked_whiten：对 global_batch 的 advantages 做全局加权白化
          - 仅统计 response_mask=1 的 token
          - 权重使用 1 / branch_weight
          - 与 masked_whiten 一致，使用 (adv-mean)/sqrt(var+eps)

        计算 aggregated_returns：按 uid 分组，组内用 weight 倒数加权平均 episodic_returns
        step_mean_return = mean(各 uid 的 aggregated_return)

        ======== 更新 ========

        更新 Critic：
          - branch_weight 存在时使用 "weighted-token-mean" 聚合（与 Actor 一致）
          - 校正 value 梯度，避免 value 随训练逐渐偏低

        更新 Actor：
          - branch_weight 存在时使用 "weighted-token-mean" 聚合：
            loss = masked_sum(loss_mat / W) / masked_sum(1 / W) * dp_size


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
- $B_t$ 为从状态 $t$ 分出的所有子轨迹集合（包括当前轨迹自身的延续）
- $\hat{A}_b^{(0)}$ 为子轨迹 $b$ 的第一个 token 的优势值
- 分支节点的均值计算：`(sum(child_advantages[:, 0]) + continuation_advantage) / state_branches[t]`

**实现细节**：
1. 第一个循环对新样本执行标准 GAE（从后向前）
2. 第二个循环从 next_states 指定的父轨迹开始，从 branch_pos 向前传播优势值
3. 传播过程中，遇到分支位置时取所有子分支首 token 优势的均值
4. 使用 child_first_adv_sum 缓存子分支首 token 的 advantage 和
5. 在统一 response 坐标上做全局逆序扫描，对受影响子图做增量更新

### 5.2 OTRC

#### 5.2.1 最优路径追踪

对每棵树（uid），从 advantage 最大的根轨迹出发，沿树贪心行走：
- 在每个位置检查是否存在子分支（通过 cid）
- 若有子分支，比较子分支首 token 的 advantage 与当前轨迹下一个 token 的 advantage
- 选择 advantage 更大的方向继续

#### 5.2.2 exploitation（期望改善量）

沿最优路径从后向前累积：

$$
\text{exploitation}[k] = -\hat{A}_k + \gamma \cdot \text{exploitation}[k+1]
$$

即 $\text{exploitation}[k] = -\sum_{t=k}^{T} \gamma^{t-k} \hat{A}_t$。正值表示从节点 k 分支有改善空间。

**数学推导**：

定义 TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。通过 telescoping 恒等式：

$$
V^{\pi}(s_k) - G_k = -\sum_{t=k}^{T} \gamma^{t-k} \delta_t
$$

此式对任意 V 成立。用 GAE 优势估计器逐项替代 $\delta_t$：

$$
V^{\pi}(s_k) - G_k \approx -\sum_{t=k}^{T} \gamma^{t-k} \hat{A}_t^{GAE}
$$

exploitation 为正值表示原轨迹的实际回报低于策略期望，有改善空间。

**注意**：与 Atari/MuJoCo 版本不同，LLM 版本不除以路径长度 $(n-k)$。

#### 5.2.3 exploration（搜索惩罚）

$$
\text{exploration}[k] = (\text{sibling\_count}[k] - 1) \cdot \max_{j}|\text{exploitation}[j]|
$$

其中：
- $\text{sibling\_count}[k]$ = `state_branches[path[k-1]]`，即路径上前一个节点的分支数（第一个节点的 sibling_count 为 1）
- $\max|\text{exploitation}|$ 是整条路径上 exploitation 绝对值的最大值（若为 0 则设为 1.0），用于将 exploration 项标准化到与 exploitation 同一量级

#### 5.2.4 OTRC 与选择

$$
\text{OTRC}[k] = \text{exploitation}[k] - c \cdot \text{exploration}[k]
$$

选择流程：
1. 跳过 `search_count >= max_search_per_tree` 的树
2. 对每棵树沿最优路径计算 OTRC，取 argmax
3. 记录 `max_exploitations[uid] = exploitation[argmax]`（LLM 中使用原始 exploitation，不做长度归一化）
4. 用 `max_exploitations` 的正值均值做门控，仅保留 `exploitation[argmax]` 超过均值的候选
5. 应用掩码：prompt 长度约束 + `</think>` 位置约束（确保分支在思考阶段内）
6. 跨所有树全局排序，取 top batch_size 个候选
7. 通过 `selected_to_branch_points` 将选中节点转换为其父节点作为分支点

**step_mean_return 更新机制**：
- 每个 step 结束时，step_mean_return 更新为该 step 内所有 uid 的 aggregated_return 的均值
- 该值仅作为 `opts_ttpo/step_mean_return` 监控指标，不参与 OTRC 选择或 checkpoint 恢复

### 5.3 TTPO 策略梯度

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_t \frac{1}{W_t} \nabla_\theta \log \pi_\theta(a_t|s_t) \hat{A}_t \right]
$$

**branch_weight 的计算**：

$$
W_{i,t} = \text{init\_weight}_i \cdot \prod_{j=0}^{t-1} \text{state\_branches}[i, j]
$$

其中 $\text{init\_weight}_i$ 通过沿祖先链追溯计算：从当前轨迹开始，依次找到父轨迹、祖父轨迹直到根轨迹，将每段祖先轨迹上从位置 0 到 branch_pos 的所有 state_branches 值相乘累积到 weight 中，最后再乘以同 uid 下根轨迹的数量。

**Loss 聚合**：当 branch_weight 存在时，自动切换为 "weighted-token-mean" 模式。policy loss 和 value loss 均除以 branch_weight 后加权求和，而非简单均值：

$$
\text{loss} = \frac{\sum_t \text{loss}_t / W_t \cdot \text{mask}_t}{\sum_t (1/W_t) \cdot \text{mask}_t} \cdot \text{dp\_size}
$$

当 branch_weight 不存在时（非 OPTS_TTPO 模式），退化为标准聚合模式。

### 5.4 Weighted Advantage Whitening

OPTS_TTPO 在 step 后处理阶段使用 `weighted_masked_whiten` 对优势做全局加权白化：

$$
\mu = \frac{\sum_t \hat{A}_t \cdot (1/W_t)\cdot m_t}{\sum_t (1/W_t)\cdot m_t},
\quad
\sigma^2 = \frac{\sum_t (\hat{A}_t-\mu)^2 \cdot (1/W_t)\cdot m_t}{\sum_t (1/W_t)\cdot m_t}
$$

$$
\hat{A}'_t = \frac{\hat{A}_t - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

其中 $m_t$ 是 `response_mask`。实现里使用 `torch.rsqrt(var + eps)`，等价于除以 `sqrt(var + eps)`。

### 5.5 Aggregated Returns

每个 step 结束后，计算 aggregated_returns：

1. 取每条轨迹最后一个有效 token 位置的 branch_weight 作为权重
2. 按 uid 分组，对组内 episodic_returns 用 weight 倒数加权平均，得到每个 uid 的 aggregated_return
3. 计算各 uid 的 aggregated_return 的均值，更新 step_mean_return（仅监控指标）

### 5.6 Episodic Returns

每条轨迹的 episodic return 由 `compute_episodic_returns` 计算，沿祖先链累加：

$$
\text{episodic\_return}_i = \sum_t r_{i,t} \cdot m_{i,t} + \sum_t r_{p,t} \cdot m_{p,t} \cdot \mathbb{1}[t \leq \text{branch\_pos}_i] + \cdots
$$

即当前轨迹的全部奖励 + 父轨迹分支点之前的奖励 + 祖父轨迹分支点之前的奖励 + ... 一直追溯到根。


## 6 实现要点

### 6.1 代码文件结构

```
LLM/trainer/opts_ttpo/
├── __init__.py
├── ../main_opts_ttpo.py  # 入口文件
├── ray_trainer.py        # RayOPTSTTPOTrainer 类及辅助函数
├── core_algos.py         # 核心算法函数（TreeGAE、branch_weight、loss 聚合等）
└── README.md          # 本文档
```

### 6.2 核心函数列表

| 函数名 | 所在文件 | 说明 |
|--------|----------|------|
| `set_opts_ttpo_info` | ray_trainer.py | 设置树结构信息：rid, pid, branch_pos, cid；在父轨迹 cid 中注册子节点 |
| `compute_episodic_returns` | ray_trainer.py | 沿祖先链累加奖励，计算完整 episodic return |
| `select_next_states` | ray_trainer.py | OTRC 状态选择：最优路径追踪、exploitation/exploration 计算、全局排序 |
| `selected_to_branch_points` | ray_trainer.py | 将 OTRC 选中节点转换为父节点作为分支点，更新 state_branches |
| `prepare_next_round_input` | ray_trainer.py | 构建下一轮采样的输入：提取 prompt + 部分响应，重新 left-pad |
| `merge_batches` | ray_trainer.py | 合并两个 DataProto batch |
| `compute_aggregated_returns` | ray_trainer.py | 按 uid 分组计算 weight 加权平均 episodic return |
| `PromptBuffer` | ray_trainer.py | 从 dataloader 按需抽取 prompt 的缓冲区 |
| `compute_treegae_advantage_return` | core_algos.py | TreeGAE 优势估计：新样本 GAE + 祖先传播 |
| `compute_branch_weight` | core_algos.py | 计算分支权重因子：祖先链追溯 + 轨迹内 cumprod |
| `agg_loss` | core_algos.py | Loss 聚合，新增 "weighted-token-mean" 模式 |
| `compute_value_loss` | core_algos.py | PPO value loss，新增 branch_weight 参数 |
| `compute_policy_loss_vanilla` | core_algos.py | PPO policy loss，新增 branch_weight 参数 |
| `AdvantageEstimator` | core_algos.py | 枚举扩展：新增 `TreeGAE = "treegae"` |
| `compute_advantage` | ray_trainer.py | 优势计算入口，根据 adv_estimator 参数分发到 TreeGAE 或 GAE |

### 6.3 配置参数

在 verl 配置文件中新增以下参数：

```yaml
actor_rollout_ref:
  rollout:
    n: 4                    # 循环采样的轮数
    max_search_per_tree: 4  # 每棵树每迭代最大搜索次数
    c: 1.0                  # OTRC 探索系数

algorithm:
  adv_estimator: treegae    # 使用 TreeGAE 优势估计
  gamma: 1.0                # 折扣因子
  lam: 0.95                 # GAE lambda
```

### 6.4 verl 框架修改

#### 6.4.1 Agent Loop 续写模式

OPTS_TTPO 需要从已有的 `input_ids` 续写生成，而不是从 `raw_prompt` 重新生成。

**`LLM/verl/verl/experimental/agent_loop/agent_loop.py`**：

1. **`AgentLoopBase` 基类**：新增 `run_from_input_ids` 抽象方法，定义续写模式接口
2. **`generate_sequences` 方法**：当 `batch.meta_info["round_idx"] >= 1` 时调用 `_generate_from_input_ids`
3. **`_run_agent_loop_from_input_ids` 方法**（新增）：创建 agent loop 实例后调用其 `run_from_input_ids`
4. **`_generate_from_input_ids` 方法**：仿照 `generate_sequences` 实现续写逻辑

**`LLM/verl/verl/experimental/agent_loop/single_turn_agent_loop.py`**：

5. **`SingleTurnAgentLoop` 类**：实现 `run_from_input_ids` 方法，接收 `prompt_ids` 直接调用 `server_manager.generate`

#### 6.4.2 Actor 策略损失修改

**`LLM/verl/verl/workers/actor/dp_actor.py`**：

1. **导入本地 `core_algos`**：优先使用 OPTS_TTPO 的 `core_algos` 以支持 `branch_weight`
2. **提取并传递 `branch_weight`**：当 batch 中存在 `branch_weight` 时传递给 policy loss 函数
3. **自动切换聚合模式**：`agg_loss` 检测到 `branch_weight is not None` 时自动使用 `"weighted-token-mean"` 模式

#### 6.4.3 Critic 价值损失修改

**`LLM/verl/verl/workers/critic/dp_critic.py`**：

1. **导入本地 `core_algos`**：优先使用 OPTS_TTPO 的 `core_algos` 以支持 `branch_weight`
2. **`select_keys` 扩展**：当 batch 中存在 `branch_weight` 时将其加入 `select_keys`，确保数据在 mini-batch 分割时被保留
3. **提取并传递 `branch_weight`**：从 `model_inputs` 中取出 `branch_weight` 并传递给 `compute_value_loss`
4. **自动切换聚合模式**：与 Actor 一致，`branch_weight` 存在时使用 `"weighted-token-mean"` 模式

#### 6.4.4 注意事项

1. **Tensor 布尔判断**：使用 `if tensor is not None:` 而非 `if tensor:`，避免多元素 Tensor 的歧义错误
2. **`non_tensor_batch` 类型约束**：`DataProto.non_tensor_batch` 的所有值必须是 `np.ndarray` 类型
   - `cid` 存储为 `np.array([OrderedDict() for ...], dtype=object)`
   - `next_states` 是 `Dict[str, Tuple[int, int]]`，不能存入 `non_tensor_batch`，应作为函数参数传递
3. **step_mean_return 更新粒度**：在 step 级别更新，每个 step 结束时将各 uid 的 aggregated_return 取均值，仅用于监控
4. **OTRC 分支点转换**：`select_next_states` 返回的是 OTRC 选中的节点本身，需要通过 `selected_to_branch_points` 转换为其父节点，因为分支是从父节点重新采样新动作
5. **`</think>` 掩码**：OTRC 选择时掩盖 `</think>` 之后的位置，确保分支发生在思考阶段内
