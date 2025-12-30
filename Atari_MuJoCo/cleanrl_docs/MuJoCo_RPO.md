# RPO (Robust Policy Optimization) - MuJoCo 实现

## 算法概述

RPO (Robust Policy Optimization) 是 PPO 的一个变体，通过在策略更新时向动作均值添加**随机扰动**来提高策略的鲁棒性。这种简单的修改可以带来更稳定和更好的性能。

**源文件**: `cleanrl/rpo_continuous_action.py`

## 核心思想

RPO 的关键创新是在计算新策略的对数概率时，向动作均值添加均匀分布的噪声：

```python
# RPO 的核心修改
z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha)
action_mean = action_mean + z
```

这使得策略对输入扰动更加鲁棒。

## 网络结构

```python
class Agent(nn.Module):
    def __init__(self, envs, rpo_alpha):
        super().__init__()
        self.rpo_alpha = rpo_alpha

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

        # 可学习的对数标准差
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
        else:
            # RPO 核心：在更新时添加随机扰动
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
```

**关键设计**:
- 使用 Tanh 激活函数
- 对数标准差作为可学习参数
- 仅在训练更新时（action 不为 None）添加扰动

## 环境预处理

MuJoCo 环境使用标准化和裁剪：

```python
def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)     # 展平观测
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)             # 裁剪动作到合法范围
        env = gym.wrappers.NormalizeObservation(env)   # 观测标准化
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)  # 奖励标准化
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk
```

## 算法流程

### 1. 数据收集

```python
for step in range(0, args.num_steps):
    global_step += 1 * args.num_envs
    obs[step] = next_obs
    dones[step] = next_done

    with torch.no_grad():
        # 采样时不添加扰动 (action=None)
        action, logprob, _, value = agent.get_action_and_value(next_obs)
        values[step] = value.flatten()
    actions[step] = action
    logprobs[step] = logprob

    next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
```

### 2. GAE 优势估计

与标准 PPO 相同：

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

### 3. 策略更新 (RPO 核心)

```python
for epoch in range(args.update_epochs):
    np.random.shuffle(b_inds)
    for start in range(0, args.batch_size, args.minibatch_size):
        end = start + args.minibatch_size
        mb_inds = b_inds[start:end]

        # 传入 action 触发 RPO 扰动
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
            b_obs[mb_inds], b_actions[mb_inds]
        )

        logratio = newlogprob - b_logprobs[mb_inds]
        ratio = logratio.exp()

        # PPO 裁剪目标
        mb_advantages = b_advantages[mb_inds]
        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

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

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

    if args.target_kl is not None:
        if approx_kl > args.target_kl:
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
| `ent_coef` | 0.0 | 熵损失系数 (MuJoCo 通常为 0) |
| `vf_coef` | 0.5 | 价值损失系数 |
| `max_grad_norm` | 0.5 | 梯度裁剪阈值 |
| **`rpo_alpha`** | **0.5** | **RPO 扰动范围 [-α, α]** |

## RPO vs PPO

| 特性 | PPO | RPO |
|------|-----|-----|
| 策略更新 | 标准计算 | 添加均匀噪声扰动 |
| 鲁棒性 | 一般 | 更强 |
| 实现复杂度 | 低 | 仅增加 2 行代码 |
| 超参数 | 标准 | 新增 `rpo_alpha` |

## 理论动机

RPO 的扰动可以被理解为：
1. **数据增强**: 在状态空间中隐式增加训练数据多样性
2. **正则化**: 防止策略过拟合到特定动作均值
3. **鲁棒性**: 使策略对输入噪声更加稳健

## 使用示例

```bash
python cleanrl/rpo_continuous_action.py \
    --env-id HalfCheetah-v4 \
    --total-timesteps 8000000 \
    --rpo-alpha 0.5 \
    --learning-rate 3e-4
```

## 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                      RPO 训练循环                           │
├─────────────────────────────────────────────────────────────┤
│  for update in range(num_updates):                          │
│    1. 数据收集 (无扰动):                                     │
│       - action = π(s) 无扰动采样                             │
│       - 记录 (obs, action, reward, done, logprob, value)    │
│    2. GAE 优势估计                                          │
│    3. 策略更新 (有扰动):                                     │
│       for epoch in range(update_epochs):                    │
│         - 重新计算 logprob 时添加扰动: μ' = μ + z           │
│         - 使用 PPO-Clip 目标                                 │
│         - 计算并优化总损失                                   │
└─────────────────────────────────────────────────────────────┘
```

## 代码对比

**PPO (标准)**:
```python
def get_action_and_value(self, x, action=None):
    action_mean = self.actor_mean(x)
    action_std = torch.exp(self.actor_logstd)
    probs = Normal(action_mean, action_std)
    if action is None:
        action = probs.sample()
    return action, probs.log_prob(action).sum(1), ...
```

**RPO (扰动)**:
```python
def get_action_and_value(self, x, action=None):
    action_mean = self.actor_mean(x)
    action_std = torch.exp(self.actor_logstd)
    probs = Normal(action_mean, action_std)
    if action is None:
        action = probs.sample()
    else:
        # RPO: 添加扰动
        z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha)
        action_mean = action_mean + z
        probs = Normal(action_mean, action_std)
    return action, probs.log_prob(action).sum(1), ...
```
