# DDPG (Deep Deterministic Policy Gradient) - MuJoCo 实现

## 算法概述

DDPG 是一种**确定性策略梯度**算法，结合了 DQN 的技术（经验回放、目标网络）和 Actor-Critic 架构，专门用于**连续动作空间**任务。

**源文件**: `cleanrl/ddpg_continuous_action.py`

## 核心特点

1. **确定性策略**: 直接输出动作，而非动作分布
2. **Actor-Critic 架构**: Actor 选择动作，Critic 评估动作
3. **经验回放**: 提高样本效率，打破数据相关性
4. **目标网络**: 使用软更新稳定训练
5. **探索噪声**: 添加高斯噪声进行探索

## 网络结构

### QNetwork (Critic)

```python
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # 输入: 状态 + 动作
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape), 256
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)  # 拼接状态和动作
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

        # 动作缩放参数
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))  # 输出 [-1, 1]
        return x * self.action_scale + self.action_bias  # 缩放到实际动作范围
```

**动作缩放**:
$$a = \tanh(\mu(s)) \cdot \text{scale} + \text{bias}$$

其中 scale = (high - low) / 2, bias = (high + low) / 2

## 算法流程

### 1. 动作选择

```python
if global_step < args.learning_starts:
    # 预热：随机动作
    actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
else:
    with torch.no_grad():
        actions = actor(torch.Tensor(obs).to(device))
        # 添加探索噪声
        actions += torch.normal(0, actor.action_scale * args.exploration_noise)
        # 裁剪到动作范围
        actions = actions.cpu().numpy().clip(
            envs.single_action_space.low, envs.single_action_space.high
        )
```

### 2. 经验存储

```python
# 处理 truncation
real_next_obs = next_obs.copy()
for idx, trunc in enumerate(truncations):
    if trunc:
        real_next_obs[idx] = infos["final_observation"][idx]

rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
obs = next_obs
```

### 3. Critic 更新

```python
if global_step > args.learning_starts:
    data = rb.sample(args.batch_size)

    with torch.no_grad():
        # 使用目标 Actor 计算下一状态的动作
        next_state_actions = target_actor(data.next_observations)
        # 使用目标 Critic 计算 Q 值
        qf1_next_target = qf1_target(data.next_observations, next_state_actions)
        # TD 目标
        next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * qf1_next_target.view(-1)

    # 当前 Q 值
    qf1_a_values = qf1(data.observations, data.actions).view(-1)
    # Critic 损失
    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

    q_optimizer.zero_grad()
    qf1_loss.backward()
    q_optimizer.step()
```

### 4. Actor 更新 (延迟更新)

```python
if global_step % args.policy_frequency == 0:
    # Actor 损失 = -Q(s, π(s))
    actor_loss = -qf1(data.observations, actor(data.observations)).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # 软更新目标网络
    for param, target_param in zip(actor.parameters(), target_actor.parameters()):
        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
```

## 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 3e-4 | 学习率 (Actor 和 Critic 共用) |
| `buffer_size` | 1,000,000 | 经验回放缓冲区大小 |
| `gamma` | 0.99 | 折扣因子 |
| `tau` | 0.005 | 软更新系数 |
| `batch_size` | 256 | 批量大小 |
| `exploration_noise` | 0.1 | 探索噪声标准差 |
| `learning_starts` | 25000 | 开始学习的时间步 |
| `policy_frequency` | 2 | 策略更新频率 (延迟更新) |

## 关键公式

### 确定性策略梯度
$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \nabla_\theta \mu_\theta(s) \nabla_a Q_\phi(s, a)|_{a=\mu_\theta(s)} \right]$$

### Critic 损失
$$L(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ (Q_\phi(s, a) - y)^2 \right]$$

其中 $y = r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s'))$

### 软更新
$$\theta' \leftarrow \tau \theta + (1 - \tau) \theta'$$

## 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                     DDPG 训练循环                           │
├─────────────────────────────────────────────────────────────┤
│  预热阶段 (learning_starts 步):                             │
│    - 随机动作填充回放缓冲区                                  │
│                                                             │
│  训练阶段:                                                   │
│  for global_step in range(total_timesteps):                 │
│    1. 选择动作: a = μ(s) + ε, ε ~ N(0, σ)                   │
│    2. 执行动作，存储 (s, a, r, s', done)                    │
│    3. 从缓冲区采样 mini-batch                                │
│    4. 更新 Critic:                                          │
│       - 计算 TD 目标 (使用目标网络)                          │
│       - 最小化 MSE 损失                                      │
│    5. 每 policy_frequency 步:                               │
│       - 更新 Actor: 最大化 Q(s, μ(s))                        │
│       - 软更新目标网络                                       │
└─────────────────────────────────────────────────────────────┘
```

## 探索策略

DDPG 使用加性高斯噪声进行探索：

```python
# 探索噪声
noise = torch.normal(0, actor.action_scale * args.exploration_noise)
actions = actor(obs) + noise
actions = actions.clip(action_low, action_high)
```

原始论文使用 Ornstein-Uhlenbeck 过程，但高斯噪声通常效果相当。

## DDPG vs DQN

| 特性 | DQN | DDPG |
|------|-----|------|
| 动作空间 | 离散 | 连续 |
| 策略类型 | ε-greedy (隐式) | 确定性 |
| 网络数量 | 1 个 Q 网络 | Actor + Critic |
| 动作选择 | argmax Q | 直接输出 |
| 探索方式 | ε-greedy | 高斯噪声 |

## 使用示例

```bash
python cleanrl/ddpg_continuous_action.py \
    --env-id Hopper-v4 \
    --total-timesteps 1000000 \
    --learning-rate 3e-4 \
    --exploration-noise 0.1
```

## 注意事项

1. **动作缩放**: 确保动作输出在环境的合法范围内
2. **目标网络**: 使用较小的 τ (0.005) 保证稳定性
3. **探索噪声**: 噪声大小影响探索-利用平衡
4. **延迟更新**: Actor 更新频率低于 Critic
