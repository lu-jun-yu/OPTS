# PQN (Parallelized Q-Network) - Atari 实现

## 算法概述

PQN (Parallelized Q-Network) 是一种结合了 DQN 和 PPO 风格训练的值函数方法。它使用 **Q(λ) 回报**进行训练，采用类似 PPO 的 **on-policy rollout** 方式收集数据，而不是传统 DQN 的经验回放缓冲区。

**源文件**: `cleanrl/pqn_atari_envpool.py`

## 核心特点

1. **无经验回放缓冲区**: 使用 on-policy 数据收集，类似 PPO
2. **Q(λ) 目标**: 结合 TD(λ) 思想计算 Q 值目标
3. **LayerNorm**: 网络中使用 LayerNorm 提高稳定性
4. **并行环境**: 使用 envpool 进行高效的并行环境采样

## 网络结构

```python
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.LayerNorm([32, 20, 20]),  # LayerNorm 而非 BatchNorm
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LayerNorm([64, 9, 9]),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.LayerNorm([64, 7, 7]),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, 512)),
            nn.LayerNorm(512),
            nn.ReLU(),
            layer_init(nn.Linear(512, env.single_action_space.n)),
        )

    def forward(self, x):
        return self.network(x / 255.0)  # 输入归一化
```

**关键设计**:
- 使用正交初始化 (`orthogonal_`)
- 每个卷积层后添加 LayerNorm
- 输入图像归一化到 [0, 1]

## 算法流程

### 1. 数据收集阶段

```python
for step in range(0, args.num_steps):
    # ε-greedy 探索
    epsilon = linear_schedule(args.start_e, args.end_e,
                              args.exploration_fraction * args.total_timesteps, global_step)

    random_actions = torch.randint(0, envs.single_action_space.n, (args.num_envs,))
    with torch.no_grad():
        q_values = q_network(next_obs)
        max_actions = torch.argmax(q_values, dim=1)
        values[step] = q_values[torch.arange(args.num_envs), max_actions].flatten()

    # 以 ε 概率随机探索
    explore = torch.rand((args.num_envs,)) < epsilon
    action = torch.where(explore, random_actions, max_actions)
```

### 2. Q(λ) 目标计算

这是 PQN 的核心创新点：

```python
with torch.no_grad():
    returns = torch.zeros_like(rewards)
    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            next_value, _ = torch.max(q_network(next_obs), dim=-1)
            nextnonterminal = 1.0 - next_done
            returns[t] = rewards[t] + args.gamma * next_value * nextnonterminal
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]
            # Q(λ) 目标: 混合 TD(0) 和 MC 回报
            returns[t] = (
                rewards[t]
                + args.gamma * (args.q_lambda * returns[t + 1]
                               + (1 - args.q_lambda) * next_value) * nextnonterminal
            )
```

**Q(λ) 公式**:
$$G_t^{Q(\lambda)} = r_t + \gamma \cdot [(1-\lambda) \cdot Q(s_{t+1}, a^*) + \lambda \cdot G_{t+1}^{Q(\lambda)}]$$

其中 `q_lambda=0.65` 是默认值，平衡了偏差和方差。

### 3. 网络更新

```python
b_inds = np.arange(args.batch_size)
for epoch in range(args.update_epochs):  # 默认 4 个 epoch
    np.random.shuffle(b_inds)
    for start in range(0, args.batch_size, args.minibatch_size):
        mb_inds = b_inds[start:end]

        # 获取当前 Q 值
        old_val = q_network(b_obs[mb_inds]).gather(
            1, b_actions[mb_inds].unsqueeze(-1).long()
        ).squeeze()

        # MSE 损失
        loss = F.mse_loss(b_returns[mb_inds], old_val)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
        optimizer.step()
```

## 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 2.5e-4 | 学习率 |
| `num_envs` | 8 | 并行环境数量 |
| `num_steps` | 128 | 每次 rollout 的步数 |
| `gamma` | 0.99 | 折扣因子 |
| `q_lambda` | 0.65 | Q(λ) 的 λ 参数 |
| `num_minibatches` | 4 | mini-batch 数量 |
| `update_epochs` | 4 | 每次更新的 epoch 数 |
| `max_grad_norm` | 10.0 | 梯度裁剪阈值 |
| `start_e` | 1.0 | 初始探索率 |
| `end_e` | 0.01 | 最终探索率 |
| `exploration_fraction` | 0.10 | 探索衰减比例 |

## 探索策略

使用线性衰减的 ε-greedy:

```python
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
```

探索率从 1.0 线性衰减到 0.01，持续 10% 的总训练步数。

## 与 DQN 的主要区别

| 特性 | DQN | PQN |
|------|-----|-----|
| 数据收集 | 经验回放 | On-policy rollout |
| 目标计算 | TD(0) + 目标网络 | Q(λ) |
| 网络更新 | 单步更新 | 多 epoch 批量更新 |
| 归一化 | 无 | LayerNorm |
| 优化器 | Adam | RAdam |

## 优势

1. **更稳定**: LayerNorm + Q(λ) 提供更稳定的训练
2. **更高效**: 无需维护大型经验回放缓冲区
3. **更简单**: 不需要目标网络
4. **样本效率**: 多 epoch 更新提高样本利用率

## 使用示例

```bash
python cleanrl/pqn_atari_envpool.py \
    --env-id Breakout-v5 \
    --total-timesteps 10000000 \
    --learning-rate 2.5e-4 \
    --q-lambda 0.65
```
