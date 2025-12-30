# PPO (Proximal Policy Optimization) - MuJoCo 实现

## 算法概述

PPO 是目前最流行的策略梯度算法之一。这个 MuJoCo 版本专门用于**连续动作空间**，使用高斯分布对动作进行参数化。

**源文件**: `cleanrl/ppo_continuous_action.py`

## 核心特点

1. **裁剪目标函数**: 限制策略更新幅度
2. **高斯策略**: 输出动作均值，学习方差
3. **GAE**: 平衡偏差和方差的优势估计
4. **观测和奖励标准化**: 提高训练稳定性

## 环境预处理

MuJoCo 环境使用多重包装器：

```python
def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)      # 展平字典观测
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)              # 裁剪动作到合法范围
        env = gym.wrappers.NormalizeObservation(env)    # 在线观测标准化
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)  # 在线奖励标准化
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk
```

**重要**: 观测和奖励标准化是 MuJoCo 任务成功的关键。

## 网络结构

```python
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Critic 网络
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Actor 均值网络
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        # 可学习的对数标准差 (状态无关)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
```

**设计选择**:
- 使用 Tanh 激活函数（比 ReLU 更适合 MuJoCo）
- 较小的网络（64 隐藏单元）
- Actor 输出层使用小标准差初始化（0.01）
- 对数标准差作为独立参数学习

## 算法流程

### 1. 数据收集

```python
for step in range(0, args.num_steps):
    global_step += args.num_envs
    obs[step] = next_obs
    dones[step] = next_done

    with torch.no_grad():
        action, logprob, _, value = agent.get_action_and_value(next_obs)
        values[step] = value.flatten()
    actions[step] = action
    logprobs[step] = logprob

    next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
    next_done = np.logical_or(terminations, truncations)
    rewards[step] = torch.tensor(reward).to(device).view(-1)
```

### 2. GAE 优势估计

```python
with torch.no_grad():
    next_value = agent.get_value(next_obs).reshape(1, -1)
    advantages = torch.zeros_like(rewards).to(device)
    lastgaelam = 0

    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]

        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

    returns = advantages + values
```

### 3. 策略和价值更新

```python
b_inds = np.arange(args.batch_size)
clipfracs = []

for epoch in range(args.update_epochs):
    np.random.shuffle(b_inds)

    for start in range(0, args.batch_size, args.minibatch_size):
        end = start + args.minibatch_size
        mb_inds = b_inds[start:end]

        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
            b_obs[mb_inds], b_actions[mb_inds]
        )

        logratio = newlogprob - b_logprobs[mb_inds]
        ratio = logratio.exp()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

        # 优势标准化
        mb_advantages = b_advantages[mb_inds]
        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # PPO-Clip 策略损失
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # 价值损失
        newvalue = newvalue.view(-1)
        if args.clip_vloss:
            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
            v_clipped = b_values[mb_inds] + torch.clamp(
                newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
            )
            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

        # 熵损失
        entropy_loss = entropy.mean()

        # 总损失
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

    # 早停
    if args.target_kl is not None and approx_kl > args.target_kl:
        break
```

## 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 3e-4 | 学习率 |
| `num_envs` | 1 | 并行环境数量 |
| `num_steps` | 2048 | 每次 rollout 的步数 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE 的 λ 参数 |
| `num_minibatches` | 32 | mini-batch 数量 |
| `update_epochs` | 10 | 每次更新的 epoch 数 |
| `clip_coef` | 0.2 | PPO 裁剪系数 |
| `clip_vloss` | True | 是否裁剪价值损失 |
| `ent_coef` | 0.0 | 熵损失系数 (**MuJoCo 通常为 0**) |
| `vf_coef` | 0.5 | 价值损失系数 |
| `max_grad_norm` | 0.5 | 梯度裁剪阈值 |
| `norm_adv` | True | 是否标准化优势 |
| `anneal_lr` | True | 是否线性衰减学习率 |

**注意**: MuJoCo 任务中熵系数通常设为 0，因为连续动作的策略熵本身就很高。

## Atari vs MuJoCo PPO 对比

| 特性 | Atari PPO | MuJoCo PPO |
|------|-----------|------------|
| 动作空间 | 离散 | 连续 |
| 策略分布 | Categorical | Normal (Gaussian) |
| 网络结构 | CNN | MLP |
| 隐藏层 | 512 (ReLU) | 64 (Tanh) |
| 熵系数 | 0.01 | 0.0 |
| 观测处理 | 帧堆叠、灰度化 | 标准化 |
| 奖励处理 | 裁剪到 [-1, 1] | 标准化 |
| num_envs | 8 | 1 |
| num_steps | 128 | 2048 |
| clip_coef | 0.1 | 0.2 |
| update_epochs | 4 | 10 |

## 关键公式

### 高斯策略
$$\pi(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta^2)$$

$$\log \pi(a|s) = -\frac{1}{2} \sum_i \left[ \frac{(a_i - \mu_i)^2}{\sigma_i^2} + \log \sigma_i^2 + \log 2\pi \right]$$

### PPO-Clip 目标
$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$

### GAE
$$\hat{A}_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

## 批量计算

```python
args.batch_size = int(args.num_envs * args.num_steps)      # 1 * 2048 = 2048
args.minibatch_size = int(args.batch_size // args.num_minibatches)  # 2048 / 32 = 64
args.num_iterations = args.total_timesteps // args.batch_size
```

## 使用示例

```bash
python cleanrl/ppo_continuous_action.py \
    --env-id HalfCheetah-v4 \
    --total-timesteps 1000000 \
    --learning-rate 3e-4 \
    --num-steps 2048 \
    --update-epochs 10
```

## 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                 PPO (MuJoCo) 训练循环                       │
├─────────────────────────────────────────────────────────────┤
│  for iteration in range(num_iterations):                    │
│    1. 学习率退火                                            │
│    2. 收集 num_steps 步轨迹:                                │
│       - 采样动作 a ~ N(μ(s), σ)                             │
│       - 存储 (obs, action, reward, done, logprob, value)   │
│    3. GAE 优势估计                                          │
│    4. 多 epoch 更新:                                        │
│       for epoch in range(update_epochs):                   │
│         - 随机打乱数据                                       │
│         - 对每个 minibatch:                                 │
│           · 计算新的 log π 和 ratio                         │
│           · PPO-Clip 策略损失                               │
│           · 裁剪价值损失                                     │
│           · 更新网络                                         │
│         - 如果 KL > target_kl: break                       │
└─────────────────────────────────────────────────────────────┘
```

## 实现细节

1. **状态无关方差**: log_std 是独立参数，不依赖状态
2. **观测标准化**: 使用运行均值和方差在线标准化
3. **奖励标准化**: 同样使用在线统计量标准化
4. **正交初始化**: 所有层使用正交初始化
5. **梯度裁剪**: 使用全局范数裁剪 (max_grad_norm=0.5)
