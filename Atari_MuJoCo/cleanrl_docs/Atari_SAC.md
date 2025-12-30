# SAC (Soft Actor-Critic) - Atari 实现

## 算法概述

SAC 是一种**最大熵强化学习**算法，它在最大化累积奖励的同时也最大化策略的熵。这个 Atari 版本是 SAC 的**离散动作空间**变体，适用于像 Atari 这样的离散控制任务。

**源文件**: `cleanrl/sac_atari.py`

## 核心特点

1. **最大熵框架**: 鼓励探索，增加策略鲁棒性
2. **离散动作**: 使用 Categorical 分布处理离散动作
3. **双 Q 网络**: 缓解 Q 值过估计
4. **自动温度调节**: 自适应调整熵系数 α
5. **分离的 Actor-Critic**: Actor 和 Critic 使用独立的网络

## 网络结构

### SoftQNetwork (Critic)

```python
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_q = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self, x):
        x = F.relu(self.conv(x / 255.0))
        x = F.relu(self.fc1(x))
        q_vals = self.fc_q(x)  # 输出每个动作的 Q 值
        return q_vals
```

### Actor (Policy Network)

```python
class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)
        return logits

    def get_action(self, x):
        logits = self(x / 255.0)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action_probs = policy_dist.probs        # 动作概率
        log_prob = F.log_softmax(logits, dim=1)  # 对数概率
        return action, log_prob, action_probs
```

**注意**: Actor 和 Critic 使用独立的 CNN 编码器，避免 Actor 梯度干扰 Critic 的表示学习。

## 算法流程

### 1. 动作选择

```python
if global_step < args.learning_starts:
    # 预热阶段：随机动作
    actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
else:
    # 从策略采样
    actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
    actions = actions.detach().cpu().numpy()
```

### 2. Critic 更新

离散 SAC 使用期望形式计算目标 Q 值：

```python
with torch.no_grad():
    _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
    qf1_next_target = qf1_target(data.next_observations)
    qf2_next_target = qf2_target(data.next_observations)

    # 使用动作概率计算期望，而非 MC 采样
    min_qf_next_target = next_state_action_probs * (
        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
    )
    # 对所有动作求和得到期望
    min_qf_next_target = min_qf_next_target.sum(dim=1)

    # TD 目标
    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target

# 计算 Critic 损失
qf1_values = qf1(data.observations)
qf2_values = qf2(data.observations)
qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
qf_loss = qf1_loss + qf2_loss

q_optimizer.zero_grad()
qf_loss.backward()
q_optimizer.step()
```

### 3. Actor 更新

```python
_, log_pi, action_probs = actor.get_action(data.observations)
with torch.no_grad():
    qf1_values = qf1(data.observations)
    qf2_values = qf2(data.observations)
    min_qf_values = torch.min(qf1_values, qf2_values)

# 离散动作的 Actor 损失（期望形式，无需重参数化）
actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

actor_optimizer.zero_grad()
actor_loss.backward()
actor_optimizer.step()
```

### 4. 自动温度调节

```python
if args.autotune:
    # 目标熵 = -scale * log(1/|A|) = scale * log(|A|)
    target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))

    # α 损失
    alpha_loss = (action_probs.detach() * (
        -log_alpha.exp() * (log_pi + target_entropy).detach()
    )).mean()

    a_optimizer.zero_grad()
    alpha_loss.backward()
    a_optimizer.step()
    alpha = log_alpha.exp().item()
```

### 5. 目标网络更新

```python
if global_step % args.target_network_frequency == 0:
    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
```

## 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `policy_lr` | 3e-4 | Actor 学习率 |
| `q_lr` | 3e-4 | Critic 学习率 |
| `buffer_size` | 1,000,000 | 经验回放缓冲区大小 |
| `gamma` | 0.99 | 折扣因子 |
| `tau` | 1.0 | 目标网络软更新系数 |
| `batch_size` | 64 | 批量大小 |
| `learning_starts` | 20000 | 开始学习的时间步 |
| `update_frequency` | 4 | 更新频率 |
| `target_network_frequency` | 8000 | 目标网络更新频率 |
| `alpha` | 0.2 | 熵正则化系数（固定时） |
| `autotune` | True | 是否自动调节 α |
| `target_entropy_scale` | 0.89 | 目标熵缩放系数 |

## 关键公式

### 最大熵目标
$$J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$

### 离散 SAC 的软 Q 函数目标
$$y = r + \gamma \sum_{a'} \pi(a'|s') \left[ \min_{i=1,2} Q_{\theta_i'}(s', a') - \alpha \log \pi(a'|s') \right]$$

### Actor 损失（离散动作）
$$L_\pi = \mathbb{E}_s \left[ \sum_a \pi(a|s) \left( \alpha \log \pi(a|s) - \min_{i=1,2} Q_{\theta_i}(s, a) \right) \right]$$

### 温度损失
$$L_\alpha = \mathbb{E}_{a \sim \pi} \left[ -\alpha \left( \log \pi(a|s) + \bar{\mathcal{H}} \right) \right]$$

## 离散 vs 连续 SAC 的区别

| 特性 | 离散 SAC (Atari) | 连续 SAC (MuJoCo) |
|------|------------------|-------------------|
| 动作分布 | Categorical | Squashed Gaussian |
| 期望计算 | 精确求和 | MC 采样 |
| 重参数化 | 不需要 | 需要 (rsample) |
| 输出 | 动作概率 | 均值 + 方差 |
| 目标熵 | ~0.89 * log(|A|) | -dim(A) |

## 使用示例

```bash
python cleanrl/sac_atari.py \
    --env-id BeamRiderNoFrameskip-v4 \
    --total-timesteps 5000000 \
    --autotune True \
    --target-entropy-scale 0.89
```

## 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                     SAC 训练循环                            │
├─────────────────────────────────────────────────────────────┤
│  预热阶段 (learning_starts 步):                             │
│    - 随机探索，填充回放缓冲区                                 │
│                                                             │
│  训练阶段:                                                   │
│  1. 从策略采样动作 a ~ π(·|s)                                │
│  2. 执行动作，存储 (s, a, r, s', done)                       │
│  3. 每 update_frequency 步:                                 │
│     a) 更新双 Q 网络 (Critic)                                │
│     b) 更新策略网络 (Actor)                                  │
│     c) 如果 autotune: 更新温度 α                             │
│  4. 每 target_network_frequency 步:                         │
│     - 软更新目标 Q 网络                                       │
└─────────────────────────────────────────────────────────────┘
```
