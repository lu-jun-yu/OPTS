# TD3 (Twin Delayed DDPG) - MuJoCo 实现

## 算法概述

TD3 (Twin Delayed Deep Deterministic Policy Gradient) 是 DDPG 的改进版本，通过三个关键技术解决 DDPG 中的 Q 值过估计问题：**双 Q 网络**、**延迟策略更新**和**目标策略平滑**。

**源文件**: `cleanrl/td3_continuous_action.py`

## 核心改进

1. **Twin (双 Q 网络)**: 使用两个独立的 Q 网络，取最小值减少过估计
2. **Delayed (延迟策略更新)**: 策略更新频率低于 Q 网络
3. **Target Policy Smoothing (目标策略平滑)**: 在目标动作中添加裁剪噪声

## 网络结构

### QNetwork (双 Critic)

```python
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape), 256
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### Actor

```python
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))

        # 动作缩放
        self.register_buffer("action_scale", torch.tensor(
            (env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32
        ))
        self.register_buffer("action_bias", torch.tensor(
            (env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32
        ))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias
```

## 算法流程

### 1. 动作选择 (带探索噪声)

```python
if global_step < args.learning_starts:
    actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
else:
    with torch.no_grad():
        actions = actor(torch.Tensor(obs).to(device))
        # 添加探索噪声
        actions += torch.normal(0, actor.action_scale * args.exploration_noise)
        actions = actions.cpu().numpy().clip(
            envs.single_action_space.low, envs.single_action_space.high
        )
```

### 2. Critic 更新 (TD3 核心)

```python
if global_step > args.learning_starts:
    data = rb.sample(args.batch_size)

    with torch.no_grad():
        # 目标策略平滑 (Target Policy Smoothing)
        clipped_noise = (
            torch.randn_like(data.actions, device=device) * args.policy_noise
        ).clamp(-args.noise_clip, args.noise_clip) * target_actor.action_scale

        # 目标动作 = 目标策略 + 裁剪噪声
        next_state_actions = (
            target_actor(data.next_observations) + clipped_noise
        ).clamp(envs.single_action_space.low[0], envs.single_action_space.high[0])

        # 双 Q 目标 (Clipped Double Q)
        qf1_next_target = qf1_target(data.next_observations, next_state_actions)
        qf2_next_target = qf2_target(data.next_observations, next_state_actions)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)

        # TD 目标
        next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target.view(-1)

    # 更新两个 Q 网络
    qf1_a_values = qf1(data.observations, data.actions).view(-1)
    qf2_a_values = qf2(data.observations, data.actions).view(-1)
    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
    qf_loss = qf1_loss + qf2_loss

    q_optimizer.zero_grad()
    qf_loss.backward()
    q_optimizer.step()
```

### 3. 延迟策略更新 (Delayed Policy Update)

```python
if global_step % args.policy_frequency == 0:
    # Actor 损失: 只使用 qf1
    actor_loss = -qf1(data.observations, actor(data.observations)).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # 软更新目标网络
    for param, target_param in zip(actor.parameters(), target_actor.parameters()):
        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
```

## TD3 三大技巧详解

### 1. Clipped Double Q-Learning

使用两个独立的 Q 网络，取最小值计算目标：

```python
min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
```

**作用**: 减少 Q 值过估计偏差

### 2. Delayed Policy Updates

策略更新频率低于 Q 网络（默认 2:1）：

```python
if global_step % args.policy_frequency == 0:  # policy_frequency = 2
    # 更新 Actor
```

**作用**: 让 Q 网络先收敛，提供更准确的梯度

### 3. Target Policy Smoothing

在目标动作中添加裁剪噪声：

```python
clipped_noise = (torch.randn_like(data.actions) * args.policy_noise).clamp(
    -args.noise_clip, args.noise_clip
)
next_state_actions = target_actor(next_obs) + clipped_noise
```

**作用**: 相当于对 Q 函数进行正则化，使其对动作扰动更平滑

## 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 3e-4 | 学习率 |
| `buffer_size` | 1,000,000 | 经验回放缓冲区大小 |
| `gamma` | 0.99 | 折扣因子 |
| `tau` | 0.005 | 软更新系数 |
| `batch_size` | 256 | 批量大小 |
| `policy_noise` | 0.2 | 目标策略噪声标准差 |
| `exploration_noise` | 0.1 | 探索噪声标准差 |
| `noise_clip` | 0.5 | 目标噪声裁剪范围 |
| `learning_starts` | 25000 | 开始学习的时间步 |
| `policy_frequency` | 2 | 策略更新频率 |

## 关键公式

### 目标 Q 值
$$y = r + \gamma \min_{i=1,2} Q_{\theta'_i}(s', \tilde{a}')$$

其中 $\tilde{a}' = \text{clip}(\mu_{\phi'}(s') + \text{clip}(\epsilon, -c, c), a_{low}, a_{high})$，$\epsilon \sim \mathcal{N}(0, \sigma)$

### Actor 损失
$$L(\phi) = -\mathbb{E}_{s \sim \mathcal{D}} \left[ Q_{\theta_1}(s, \mu_\phi(s)) \right]$$

注意：Actor 损失只使用 Q1，不使用 min(Q1, Q2)

### 更新频率
- Critic: 每步更新
- Actor: 每 `policy_frequency` 步更新
- 目标网络: 与 Actor 同步更新

## 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                     TD3 训练循环                            │
├─────────────────────────────────────────────────────────────┤
│  for global_step in range(total_timesteps):                 │
│    1. 选择动作: a = μ(s) + ε_explore                        │
│    2. 执行动作，存储 (s, a, r, s', done)                    │
│    3. 每步更新 Critic:                                      │
│       a) 目标动作 = μ'(s') + clip(ε_target)                 │
│       b) y = r + γ * min(Q1', Q2')                          │
│       c) 更新 Q1, Q2                                        │
│    4. 每 policy_frequency 步:                               │
│       a) 更新 Actor: max Q1(s, μ(s))                        │
│       b) 软更新目标网络                                      │
└─────────────────────────────────────────────────────────────┘
```

## TD3 vs DDPG 对比

| 特性 | DDPG | TD3 |
|------|------|-----|
| Q 网络数量 | 1 | 2 |
| 目标计算 | Q_target | min(Q1_target, Q2_target) |
| 策略更新 | 每步 | 延迟 (每 2 步) |
| 目标策略 | 无噪声 | 带裁剪噪声 |
| 过估计 | 严重 | 缓解 |

## 噪声对比

| 噪声类型 | 位置 | 目的 | 参数 |
|----------|------|------|------|
| 探索噪声 | 动作选择 | 探索环境 | `exploration_noise=0.1` |
| 目标噪声 | TD 目标计算 | 平滑 Q 函数 | `policy_noise=0.2` |
| 噪声裁剪 | 目标噪声 | 防止极端动作 | `noise_clip=0.5` |

## 使用示例

```bash
python cleanrl/td3_continuous_action.py \
    --env-id Hopper-v4 \
    --total-timesteps 1000000 \
    --policy-noise 0.2 \
    --noise-clip 0.5 \
    --policy-frequency 2
```

## 实现注意事项

1. **双 Q 更新**: 两个 Q 网络共享优化器，同时更新
2. **Actor 梯度**: 只通过 Q1 反传，不使用 min(Q1, Q2)
3. **噪声缩放**: 目标噪声按 action_scale 缩放
4. **目标网络**: 仅在策略更新时更新（与 Actor 同步）
