# PPO (Proximal Policy Optimization) - Atari 实现

## 算法概述

PPO 是目前最流行的策略梯度算法之一，它通过**裁剪目标函数**来限制策略更新幅度，在保持简单性的同时实现了稳定高效的训练。

**源文件**: `cleanrl/ppo_atari.py`

## 核心特点

1. **裁剪目标函数**: 防止过大的策略更新
2. **GAE (Generalized Advantage Estimation)**: 平衡偏差和方差
3. **多环境并行**: 使用向量化环境加速数据收集
4. **共享网络**: Actor 和 Critic 共享特征提取层
5. **多 epoch 更新**: 提高样本效率

## 网络结构

```python
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # 共享的特征提取网络
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        # Actor 头 (策略)
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        # Critic 头 (价值函数)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
```

**初始化技巧**:
- 使用正交初始化
- Actor 输出层使用较小的标准差 (0.01)
- Critic 输出层使用标准差 1

## 算法流程

### 1. 数据收集 (Rollout)

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

    # 执行动作
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

        # TD 误差
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
        # GAE 递推
        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

    returns = advantages + values  # 回报 = 优势 + 价值
```

**GAE 公式**:
$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

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
            b_obs[mb_inds], b_actions.long()[mb_inds]
        )

        # 计算比率
        logratio = newlogprob - b_logprobs[mb_inds]
        ratio = logratio.exp()

        # 近似 KL 散度 (用于监控)
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

        # 优势标准化
        mb_advantages = b_advantages[mb_inds]
        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # PPO 裁剪目标 (策略损失)
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # 价值损失 (可选裁剪)
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

        # 熵损失 (鼓励探索)
        entropy_loss = entropy.mean()

        # 总损失
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

    # 早停 (可选)
    if args.target_kl is not None and approx_kl > args.target_kl:
        break
```

## 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 2.5e-4 | 学习率 |
| `num_envs` | 8 | 并行环境数量 |
| `num_steps` | 128 | 每次 rollout 的步数 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE 的 λ 参数 |
| `num_minibatches` | 4 | mini-batch 数量 |
| `update_epochs` | 4 | 每次更新的 epoch 数 |
| `clip_coef` | 0.1 | PPO 裁剪系数 |
| `clip_vloss` | True | 是否裁剪价值损失 |
| `ent_coef` | 0.01 | 熵损失系数 |
| `vf_coef` | 0.5 | 价值损失系数 |
| `max_grad_norm` | 0.5 | 梯度裁剪阈值 |
| `norm_adv` | True | 是否标准化优势 |
| `anneal_lr` | True | 是否线性衰减学习率 |

## 关键公式

### PPO-Clip 目标
$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

### 价值损失
$$L^V = \frac{1}{2} \mathbb{E}_t \left[ (V_\theta(s_t) - R_t)^2 \right]$$

### 熵损失
$$L^S = \mathbb{E}_t \left[ \mathcal{H}(\pi_\theta(\cdot|s_t)) \right]$$

### 总损失
$$L = L^{CLIP} - c_1 L^V + c_2 L^S$$

## 学习率退火

```python
if args.anneal_lr:
    frac = 1.0 - (iteration - 1.0) / args.num_iterations
    lrnow = frac * args.learning_rate
    optimizer.param_groups[0]["lr"] = lrnow
```

## 训练流程图

```
┌─────────────────────────────────────────────────────────────┐
│                      PPO 训练循环                           │
├─────────────────────────────────────────────────────────────┤
│  for iteration in range(num_iterations):                    │
│    1. 学习率退火 (可选)                                      │
│    2. 数据收集:                                              │
│       for step in range(num_steps):                         │
│         - 使用当前策略采样动作                                │
│         - 执行动作，存储 (obs, action, reward, done, value)  │
│    3. 计算 GAE 优势估计                                      │
│    4. 策略和价值更新:                                        │
│       for epoch in range(update_epochs):                    │
│         for minibatch in minibatches:                       │
│           - 计算策略损失 (PPO-Clip)                          │
│           - 计算价值损失                                     │
│           - 计算熵损失                                       │
│           - 反向传播，梯度裁剪，参数更新                      │
│         - 如果 KL > target_kl: break                        │
└─────────────────────────────────────────────────────────────┘
```

## 批量计算

```python
# 运行时计算
args.batch_size = int(args.num_envs * args.num_steps)      # 8 * 128 = 1024
args.minibatch_size = int(args.batch_size // args.num_minibatches)  # 1024 / 4 = 256
args.num_iterations = args.total_timesteps // args.batch_size  # 总迭代数
```

## 监控指标

- `approx_kl`: 近似 KL 散度，监控策略变化
- `clipfracs`: 被裁剪的比率，衡量裁剪程度
- `explained_variance`: 解释方差，衡量价值函数质量
- `entropy`: 策略熵，衡量探索程度

## 使用示例

```bash
python cleanrl/ppo_atari.py \
    --env-id BreakoutNoFrameskip-v4 \
    --total-timesteps 10000000 \
    --learning-rate 2.5e-4 \
    --num-envs 8 \
    --clip-coef 0.1
```

## PPO 的优势

1. **稳定性**: 裁剪机制防止灾难性更新
2. **简单性**: 比 TRPO 简单得多，无需二阶优化
3. **样本效率**: 多 epoch 更新提高样本利用率
4. **并行化**: 易于使用向量化环境
5. **通用性**: 适用于离散和连续动作空间
