# DQN (Deep Q-Network) - Atari 实现

## 算法概述

DQN 是深度强化学习的里程碑算法，首次成功将深度学习应用于复杂的视觉强化学习任务。它通过**经验回放**和**目标网络**两个关键技术解决了训练不稳定的问题。

**源文件**: `cleanrl/dqn_atari.py`

## 核心特点

1. **经验回放缓冲区**: 打破数据相关性，提高样本效率
2. **目标网络**: 稳定 TD 目标，防止训练发散
3. **ε-greedy 探索**: 平衡探索与利用
4. **CNN 特征提取**: 直接从像素学习

## 环境预处理

Atari 环境使用标准预处理流程：

```python
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Atari 特定包装器
        env = NoopResetEnv(env, noop_max=30)      # 随机 no-op 开始
        env = MaxAndSkipEnv(env, skip=4)          # 跳帧 (动作重复)
        env = EpisodicLifeEnv(env)                # 生命结束 = episode 结束
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)               # 自动按 FIRE 开始
        env = ClipRewardEnv(env)                  # 奖励裁剪到 [-1, 1]
        env = gym.wrappers.ResizeObservation(env, (84, 84))  # 缩放到 84x84
        env = gym.wrappers.GrayScaleObservation(env)         # 灰度化
        env = gym.wrappers.FrameStack(env, 4)                # 堆叠 4 帧

        return env
    return thunk
```

## 网络结构

经典的 Nature DQN 网络：

```python
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),   # 输入: 4x84x84 -> 32x20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),  # -> 64x9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),  # -> 64x7x7
            nn.ReLU(),
            nn.Flatten(),                     # -> 3136
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),  # 输出: 动作数量
        )

    def forward(self, x):
        return self.network(x / 255.0)  # 归一化到 [0, 1]
```

## 算法流程

### 1. 动作选择 (ε-greedy)

```python
epsilon = linear_schedule(args.start_e, args.end_e,
                          args.exploration_fraction * args.total_timesteps, global_step)

if random.random() < epsilon:
    # 随机探索
    actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
else:
    # 贪婪选择
    q_values = q_network(torch.Tensor(obs).to(device))
    actions = torch.argmax(q_values, dim=1).cpu().numpy()
```

### 2. 经验存储

```python
# 处理 truncation (时间限制导致的终止)
real_next_obs = next_obs.copy()
for idx, trunc in enumerate(truncations):
    if trunc:
        real_next_obs[idx] = infos["final_observation"][idx]

# 存入回放缓冲区
rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
```

### 3. TD 目标计算与更新

```python
if global_step > args.learning_starts:
    if global_step % args.train_frequency == 0:
        data = rb.sample(args.batch_size)

        with torch.no_grad():
            # 使用目标网络计算下一状态的最大 Q 值
            target_max, _ = target_network(data.next_observations).max(dim=1)
            # TD 目标
            td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())

        # 当前 Q 值
        old_val = q_network(data.observations).gather(1, data.actions).squeeze()

        # MSE 损失
        loss = F.mse_loss(td_target, old_val)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 4. 目标网络更新

```python
if global_step % args.target_network_frequency == 0:
    for target_network_param, q_network_param in zip(
        target_network.parameters(), q_network.parameters()
    ):
        # 软更新 (Polyak averaging)
        target_network_param.data.copy_(
            args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
        )
```

当 `tau=1.0` 时，等价于硬更新（完全复制参数）。

## 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 1e-4 | 学习率 |
| `buffer_size` | 1,000,000 | 经验回放缓冲区大小 |
| `gamma` | 0.99 | 折扣因子 |
| `tau` | 1.0 | 目标网络更新系数 |
| `target_network_frequency` | 1000 | 目标网络更新频率 |
| `batch_size` | 32 | 批量大小 |
| `start_e` | 1.0 | 初始探索率 |
| `end_e` | 0.01 | 最终探索率 |
| `exploration_fraction` | 0.10 | 探索衰减比例 |
| `learning_starts` | 80000 | 开始学习的时间步 |
| `train_frequency` | 4 | 训练频率 |

## 经验回放缓冲区

使用 `cleanrl_utils.buffers.ReplayBuffer`:

```python
rb = ReplayBuffer(
    args.buffer_size,
    envs.single_observation_space,
    envs.single_action_space,
    device,
    optimize_memory_usage=True,     # 节省内存
    handle_timeout_termination=False,
)
```

## 关键公式

### TD 目标
$$y_t = r_t + \gamma \cdot \max_{a'} Q_{target}(s_{t+1}, a')$$

### 损失函数
$$L(\theta) = \mathbb{E}[(y_t - Q(s_t, a_t; \theta))^2]$$

### ε 衰减
$$\epsilon_t = \max(\epsilon_{end}, \epsilon_{start} + \frac{\epsilon_{end} - \epsilon_{start}}{T} \cdot t)$$

## 训练流程图

```
┌─────────────────────────────────────────────────────────────┐
│                        训练循环                              │
├─────────────────────────────────────────────────────────────┤
│  1. ε-greedy 选择动作                                        │
│  2. 执行动作，获取 (s', r, done)                             │
│  3. 存储到经验回放缓冲区                                      │
│  4. 每 train_frequency 步:                                   │
│     - 从缓冲区采样 mini-batch                                 │
│     - 计算 TD 目标 (使用目标网络)                             │
│     - 更新 Q 网络                                            │
│  5. 每 target_network_frequency 步:                          │
│     - 更新目标网络                                            │
└─────────────────────────────────────────────────────────────┘
```

## 使用示例

```bash
python cleanrl/dqn_atari.py \
    --env-id BreakoutNoFrameskip-v4 \
    --total-timesteps 10000000 \
    --learning-rate 1e-4 \
    --buffer-size 1000000 \
    --save-model
```

## 与其他变体的关系

| 算法 | 主要改进 |
|------|----------|
| Double DQN | 解耦动作选择和评估 |
| Dueling DQN | 分离状态价值和优势函数 |
| Prioritized DQN | 优先经验回放 |
| Rainbow | 集成多种改进 |
