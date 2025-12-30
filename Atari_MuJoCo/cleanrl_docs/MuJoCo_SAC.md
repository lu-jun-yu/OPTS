# SAC (Soft Actor-Critic) - MuJoCo 实现

## 算法概述

SAC (Soft Actor-Critic) 是一种**最大熵强化学习**算法，通过最大化累积奖励和策略熵的组合目标来学习。它是目前连续控制任务中最有效的算法之一。

**源文件**: `cleanrl/sac_continuous_action.py`

## 核心特点

1. **最大熵框架**: 同时最大化奖励和策略熵
2. **随机策略**: 输出动作分布而非确定性动作
3. **双 Q 网络**: 缓解 Q 值过估计
4. **自动温度调节**: 自适应调整熵系数 α
5. **重参数化技巧**: 使随机策略可微分

## 网络结构

### SoftQNetwork (Critic)

```python
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # 输入维度 = 状态维度 + 动作维度
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

### Actor (Squashed Gaussian Policy)

```python
LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))

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
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        # 使用 tanh 裁剪 log_std 到合理范围
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        # 重参数化采样
        x_t = normal.rsample()  # mean + std * N(0,1)
        # Squash 到 [-1, 1]
        y_t = torch.tanh(x_t)
        # 缩放到动作范围
        action = y_t * self.action_scale + self.action_bias

        # 计算对数概率 (考虑 tanh 变换)
        log_prob = normal.log_prob(x_t)
        # Jacobian 修正
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
```

**Squashed Gaussian**:
1. 从 Normal(μ, σ) 采样 x
2. 应用 tanh: y = tanh(x)
3. 缩放到动作范围: a = y * scale + bias
4. 修正对数概率: log π(a|s) = log N(x|μ,σ) - log(1 - y²)

## 算法流程

### 1. 动作选择

```python
if global_step < args.learning_starts:
    actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
else:
    actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
    actions = actions.detach().cpu().numpy()
```

### 2. Critic 更新

```python
if global_step > args.learning_starts:
    data = rb.sample(args.batch_size)

    with torch.no_grad():
        # 从当前策略采样下一动作
        next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
        # 双 Q 目标
        qf1_next_target = qf1_target(data.next_observations, next_state_actions)
        qf2_next_target = qf2_target(data.next_observations, next_state_actions)
        # 取最小 Q - 熵项
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
        # TD 目标
        next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target.view(-1)

    # 当前 Q 值
    qf1_a_values = qf1(data.observations, data.actions).view(-1)
    qf2_a_values = qf2(data.observations, data.actions).view(-1)

    # Critic 损失
    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
    qf_loss = qf1_loss + qf2_loss

    q_optimizer.zero_grad()
    qf_loss.backward()
    q_optimizer.step()
```

### 3. Actor 更新

```python
if global_step % args.policy_frequency == 0:
    for _ in range(args.policy_frequency):  # 补偿延迟
        # 重新采样动作
        pi, log_pi, _ = actor.get_action(data.observations)
        qf1_pi = qf1(data.observations, pi)
        qf2_pi = qf2(data.observations, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Actor 损失: 最大化 Q - α * log π
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # 自动温度调节
        if args.autotune:
            with torch.no_grad():
                _, log_pi, _ = actor.get_action(data.observations)
            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

            a_optimizer.zero_grad()
            alpha_loss.backward()
            a_optimizer.step()
            alpha = log_alpha.exp().item()
```

### 4. 目标网络更新

```python
if global_step % args.target_network_frequency == 0:
    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
```

## 自动温度调节

```python
# 初始化
if args.autotune:
    # 目标熵 = -dim(A)
    target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
else:
    alpha = args.alpha
```

**目标熵**: 对于连续动作，通常设为 -dim(A)，即动作空间的负维度。

## 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `policy_lr` | 3e-4 | Actor 学习率 |
| `q_lr` | 1e-3 | Critic 学习率 |
| `buffer_size` | 1,000,000 | 经验回放缓冲区大小 |
| `gamma` | 0.99 | 折扣因子 |
| `tau` | 0.005 | 软更新系数 |
| `batch_size` | 256 | 批量大小 |
| `learning_starts` | 5000 | 开始学习的时间步 |
| `policy_frequency` | 2 | 策略更新频率 |
| `target_network_frequency` | 1 | 目标网络更新频率 |
| `alpha` | 0.2 | 固定熵系数 |
| `autotune` | True | 是否自动调节 α |

## 关键公式

### 最大熵目标
$$J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$

### 软 Q 函数
$$Q(s_t, a_t) = r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1}} \left[ V(s_{t+1}) \right]$$

$$V(s_t) = \mathbb{E}_{a_t \sim \pi} \left[ Q(s_t, a_t) - \alpha \log \pi(a_t|s_t) \right]$$

### Actor 损失
$$L_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \mathbb{E}_{a \sim \pi_\theta} \left[ \alpha \log \pi_\theta(a|s) - Q_\phi(s, a) \right] \right]$$

### 温度损失
$$L(\alpha) = \mathbb{E}_{a_t \sim \pi_t} \left[ -\alpha \left( \log \pi_t(a_t|s_t) + \bar{\mathcal{H}} \right) \right]$$

## 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                     SAC 训练循环                            │
├─────────────────────────────────────────────────────────────┤
│  for global_step in range(total_timesteps):                 │
│    1. 从策略采样: a ~ π(·|s)                                 │
│    2. 执行动作，存储 (s, a, r, s', done)                    │
│    3. 从缓冲区采样                                          │
│    4. 更新双 Critic:                                        │
│       y = r + γ(min Q_target(s', a') - α log π(a'|s'))      │
│       L_Q = MSE(Q(s,a), y)                                  │
│    5. 每 policy_frequency 步更新 Actor:                     │
│       L_π = E[α log π(a|s) - min Q(s, a)]                   │
│    6. 如果 autotune, 更新 α:                                │
│       L_α = -α (log π(a|s) + target_entropy)                │
│    7. 软更新目标网络                                        │
└─────────────────────────────────────────────────────────────┘
```

## SAC vs DDPG vs TD3

| 特性 | DDPG | TD3 | SAC |
|------|------|-----|-----|
| 策略类型 | 确定性 | 确定性 | 随机 |
| Q 网络数 | 1 | 2 | 2 |
| 探索方式 | 外部噪声 | 目标策略噪声 | 策略熵 |
| 目标噪声 | 无 | 有 | 无 |
| 熵正则化 | 无 | 无 | 有 |

## 使用示例

```bash
python cleanrl/sac_continuous_action.py \
    --env-id Hopper-v4 \
    --total-timesteps 1000000 \
    --autotune True \
    --policy-lr 3e-4 \
    --q-lr 1e-3
```

## 实现细节

1. **Log-std 范围**: 使用 tanh 将 log_std 限制在 [-5, 2]
2. **数值稳定性**: 在对数概率计算中添加 1e-6
3. **延迟更新**: Actor 每 2 步更新一次
4. **梯度流**: 使用 rsample() 而非 sample() 进行重参数化
