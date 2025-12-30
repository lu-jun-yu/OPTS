# C51 (Categorical DQN) - Atari 实现

## 算法概述

C51 是**分布式强化学习**的开创性工作，它学习完整的回报分布而不仅仅是期望值。名称来源于使用 **51 个原子**来离散化回报分布。

**源文件**: `cleanrl/c51_atari.py`

## 核心思想

传统 DQN 学习 $Q(s, a) = \mathbb{E}[G_t]$，而 C51 学习完整分布 $Z(s, a)$，其中 $Q(s, a) = \mathbb{E}[Z(s, a)]$。

**优势**:
1. 捕获回报的不确定性
2. 更丰富的监督信号
3. 更稳定的训练

## 网络结构

```python
class QNetwork(nn.Module):
    def __init__(self, env, n_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        self.env = env
        self.n_atoms = n_atoms
        # 注册支撑点 (support)
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.n = env.single_action_space.n

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.n * n_atoms),  # 输出: 动作数 × 原子数
        )

    def get_action(self, x, action=None):
        logits = self.network(x / 255.0)
        # 重塑为 [batch, actions, atoms]
        # 对每个动作计算概率质量函数 (PMF)
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        # Q 值 = 分布的期望
        q_values = (pmfs * self.atoms).sum(2)

        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]
```

**关键点**:
- 输出维度: `num_actions × n_atoms`
- 使用 softmax 将 logits 转换为概率分布
- Q 值通过期望计算: $Q(s,a) = \sum_i z_i \cdot p_i(s,a)$

## 分布表示

使用固定支撑点的离散分布：

```
支撑点 (atoms): z_0, z_1, ..., z_50
其中 z_i = v_min + i * Δz, Δz = (v_max - v_min) / (n_atoms - 1)

默认参数:
- n_atoms = 51
- v_min = -10
- v_max = 10
- Δz = 0.4
```

## 算法流程

### 1. 动作选择

```python
epsilon = linear_schedule(args.start_e, args.end_e,
                          args.exploration_fraction * args.total_timesteps, global_step)

if random.random() < epsilon:
    actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
else:
    actions, pmf = q_network.get_action(torch.Tensor(obs).to(device))
    actions = actions.cpu().numpy()
```

### 2. 分布投影 (核心)

这是 C51 最关键的部分——将贝尔曼更新后的分布投影回固定支撑点：

```python
with torch.no_grad():
    _, next_pmfs = target_network.get_action(data.next_observations)

    # 计算转移后的原子位置: r + γ * z
    next_atoms = data.rewards + args.gamma * target_network.atoms * (1 - data.dones)

    # 裁剪到 [v_min, v_max]
    delta_z = target_network.atoms[1] - target_network.atoms[0]
    tz = next_atoms.clamp(args.v_min, args.v_max)

    # 计算投影位置
    b = (tz - args.v_min) / delta_z  # 连续索引
    l = b.floor().clamp(0, args.n_atoms - 1)  # 下界
    u = b.ceil().clamp(0, args.n_atoms - 1)   # 上界

    # 按比例分配概率到相邻原子
    # 处理 b 恰好为整数的边界情况
    d_m_l = (u + (l == u).float() - b) * next_pmfs  # 分配给下界的概率
    d_m_u = (b - l) * next_pmfs                      # 分配给上界的概率

    # 构建目标分布
    target_pmfs = torch.zeros_like(next_pmfs)
    for i in range(target_pmfs.size(0)):
        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])
```

### 3. 损失计算

使用交叉熵损失（KL 散度）：

```python
_, old_pmfs = q_network.get_action(data.observations, data.actions.flatten())

# 交叉熵损失
loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 投影算法详解

投影的目标是将连续的贝尔曼更新 $T\hat{Z}$ 映射到离散支撑点：

```
原始分布 Z(s', a*) 的支撑点: {z_0, z_1, ..., z_N}
贝尔曼更新后: {r + γz_0, r + γz_1, ..., r + γz_N}

投影步骤:
1. 对每个更新后的原子 tz_j = r + γz_j:
   - 计算连续索引 b_j = (tz_j - v_min) / Δz
   - 找到相邻整数索引 l_j = floor(b_j), u_j = ceil(b_j)

2. 将概率 p_j 按比例分配:
   - 分配 (u_j - b_j) * p_j 到索引 l_j
   - 分配 (b_j - l_j) * p_j 到索引 u_j
```

## 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 2.5e-4 | 学习率 |
| `n_atoms` | 51 | 分布支撑点数量 |
| `v_min` | -10 | 回报下界 |
| `v_max` | 10 | 回报上界 |
| `buffer_size` | 1,000,000 | 经验回放缓冲区大小 |
| `gamma` | 0.99 | 折扣因子 |
| `target_network_frequency` | 10000 | 目标网络更新频率 |
| `batch_size` | 32 | 批量大小 |
| `learning_starts` | 80000 | 开始学习的时间步 |
| `train_frequency` | 4 | 训练频率 |

## 与 DQN 的对比

| 特性 | DQN | C51 |
|------|-----|-----|
| 输出 | Q 值 (标量) | 分布 (概率向量) |
| 损失函数 | MSE | 交叉熵/KL 散度 |
| 信息量 | 只有期望 | 完整分布 |
| 网络输出维度 | `num_actions` | `num_actions × n_atoms` |
| 目标计算 | 直接 TD | 分布投影 |

## 关键公式

### 分布贝尔曼方程
$$Z(s, a) \stackrel{D}{=} R(s, a) + \gamma Z(S', A')$$

### 投影操作
$$(\Phi \hat{T} Z_\theta)(s, a) = \sum_{j=0}^{N-1} \left[ p_j(s', a^*) \right]_{l_j}^{u_j}$$

### 损失函数 (KL 散度)
$$L(\theta) = D_{KL}(\Phi \hat{T} Z_{\theta^-} || Z_\theta) = -\sum_i m_i \log p_i(s, a; \theta)$$

## 使用示例

```bash
python cleanrl/c51_atari.py \
    --env-id BreakoutNoFrameskip-v4 \
    --total-timesteps 10000000 \
    --n-atoms 51 \
    --v-min -10 \
    --v-max 10 \
    --save-model
```

## 分布可视化

```
v_min=-10                                           v_max=10
   |----|----|----|----|----|----|----|----|----|----|
  z_0  z_1  z_2  ...                           ... z_50
   ↓    ↓    ↓
  p_0  p_1  p_2  ...  概率质量函数 (PMF)

Q(s,a) = Σ z_i * p_i  (期望值)
```

## 实现注意事项

1. **数值稳定性**: 在 log 操作前添加 `clamp(min=1e-5, max=1-1e-5)`
2. **支撑点选择**: v_min 和 v_max 需要覆盖可能的回报范围
3. **原子数量**: 更多原子提供更精细的分布表示，但计算成本更高
