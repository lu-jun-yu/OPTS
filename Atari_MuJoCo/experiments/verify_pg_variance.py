# 验证1：正轨迹的策略梯度方差 < 负轨迹的策略梯度方差
import argparse
import json
import os
import sys

import gymnasium as gym
import numpy as np
import torch
from torch.distributions.normal import Normal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cleanrl', 'cleanrl'))
from opts_ttpo_continuous_action import Agent, MuJoCoStateSnapshotWrapper


def make_eval_env(env_id, gamma):
    env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    env = MuJoCoStateSnapshotWrapper(env)
    return env


def restore_normalization(env, checkpoint):
    current = env
    while current is not None:
        if isinstance(current, gym.wrappers.NormalizeObservation):
            if "obs_rms_mean" in checkpoint:
                current.obs_rms.mean = checkpoint["obs_rms_mean"]
                current.obs_rms.var = checkpoint["obs_rms_var"]
                current.obs_rms.count = checkpoint["obs_rms_count"]
            current.update_running_mean = False
        if isinstance(current, gym.wrappers.NormalizeReward):
            if "ret_rms_mean" in checkpoint:
                current.return_rms.mean = checkpoint["ret_rms_mean"]
                current.return_rms.var = checkpoint["ret_rms_var"]
                current.return_rms.count = checkpoint["ret_rms_count"]
            current.update_running_mean = False
        current = getattr(current, 'env', None)


def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    num_steps = len(rewards)
    advantages = torch.zeros(num_steps)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        nextnonterminal = 1.0 - dones[t]
        nextvalues = next_value if t == num_steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    return advantages


def collect_steps(env, agent, num_steps, device, resume_state=None):
    """跑 rollout，返回原始数据 dict。若提供 resume_state 则从该状态精确续跑。"""
    obs_buf, act_buf, rew_buf, val_buf, done_buf = [], [], [], [], []
    episode_info = []  # (start, end, return)

    if resume_state is not None:
        env.reset()  # 初始化 wrapper 内部状态（episode_returns 等）
        env.restore_state(resume_state['env_state'])
        obs_t = resume_state['last_obs_t'].to(device)
        ep_start = resume_state['ep_start']
        np.random.set_state(resume_state['np_rng'])
        torch.random.set_rng_state(resume_state['torch_rng'])
    else:
        obs_np, _ = env.reset()
        obs_t = torch.Tensor(obs_np).to(device)
        ep_start = 0

    for step in range(num_steps):
        obs_buf.append(obs_t)
        with torch.no_grad():
            action, _, _, value = agent.get_action_and_value(obs_t.unsqueeze(0))
        action = action.squeeze(0)
        act_buf.append(action)
        val_buf.append(value.item())

        obs_np, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        done = terminated or truncated
        rew_buf.append(reward)
        done_buf.append(float(done))

        if done and "episode" in info:
            ep_r = info['episode']['r']
            episode_info.append((ep_start, step, float(ep_r[0] if hasattr(ep_r, '__getitem__') else ep_r)))
            ep_start = step + 1

        if done:
            obs_np, _ = env.reset()
        obs_t = torch.Tensor(obs_np).to(device)

    with torch.no_grad():
        next_value = agent.get_value(obs_t.unsqueeze(0)).item()

    return {
        'obs': torch.stack(obs_buf),
        'actions': torch.stack(act_buf),
        'rewards': torch.tensor(rew_buf, dtype=torch.float32),
        'values': torch.tensor(val_buf, dtype=torch.float32),
        'dones': torch.tensor(done_buf, dtype=torch.float32),
        'episode_info': episode_info,
        'next_value': next_value,
        'env_state': env.clone_state(),
        'last_obs_t': obs_t.cpu(),
        'ep_start': ep_start,
        'np_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }


def compute_pg_gradient(agent, obs, actions, advantages, device):
    """REINFORCE 梯度，返回拼接后的 flat 向量"""
    agent.zero_grad()
    obs, actions, advantages = obs.to(device), actions.to(device), advantages.to(device)

    action_mean = agent.actor_mean(obs)
    action_logstd = agent.actor_logstd.expand_as(action_mean)
    dist = Normal(action_mean, torch.exp(action_logstd))
    log_probs = dist.log_prob(actions).sum(-1)

    pg_loss = -(log_probs * advantages.detach()).mean()
    pg_loss.backward()

    return torch.cat([p.grad.detach().flatten() if p.grad is not None
                      else torch.zeros(p.numel(), device=device) for p in agent.parameters()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--num-steps", type=int, default=32768)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/variance/verify1")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="proportion of top/bottom episodes as positive/negative samples")
    args = parser.parse_args()

    device = torch.device("cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu"))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 加载 checkpoint
    ckpt_path = os.path.join(args.checkpoint_dir, args.env_id, "seed1.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    env = make_eval_env(args.env_id, gamma=args.gamma)
    restore_normalization(env, checkpoint)

    import types
    env_spec = types.SimpleNamespace(
        single_observation_space=env.observation_space,
        single_action_space=env.action_space,
    )
    agent = Agent(env_spec).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    # ---- 缓存逻辑：加载 / 续跑 / 从头跑 ----
    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{args.env_id}_seed{args.seed}.pt")

    cache = None
    cached_steps = 0
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location=device, weights_only=False)
        cached_steps = len(cache['obs'])

    if cached_steps >= args.num_steps:
        # 缓存足够，直接截取前 num_steps
        print(f"Loaded cache ({cached_steps} steps), using first {args.num_steps}")
        obs = cache['obs'][:args.num_steps]
        actions = cache['actions'][:args.num_steps]
        rewards = cache['rewards'][:args.num_steps]
        values = cache['values'][:args.num_steps]
        dones = cache['dones'][:args.num_steps]
        all_episodes = [(s, e, r) for s, e, r in cache['episode_info'] if e < args.num_steps]
        if args.num_steps < cached_steps:
            with torch.no_grad():
                next_value = agent.get_value(cache['obs'][args.num_steps].unsqueeze(0).to(device)).item()
        else:
            next_value = cache['next_value']

    elif cached_steps > 0:
        # 缓存不够，恢复环境状态后从断点精确续跑
        additional = args.num_steps - cached_steps
        print(f"Loaded cache ({cached_steps} steps), continuing {additional} more from saved env state...")

        resume_state = {
            'env_state': cache['env_state'],
            'last_obs_t': cache['last_obs_t'],
            'ep_start': 0,  # 新 chunk 内部从 0 开始编号
            'np_rng': cache['np_rng'],
            'torch_rng': cache['torch_rng'],
        }
        new = collect_steps(env, agent, additional, device, resume_state=resume_state)

        obs = torch.cat([cache['obs'], new['obs']])
        actions = torch.cat([cache['actions'], new['actions']])
        rewards = torch.cat([cache['rewards'], new['rewards']])
        values = torch.cat([cache['values'], new['values']])
        dones = torch.cat([cache['dones'], new['dones']])

        # 合并 episode_info（新 chunk 的索引偏移 cached_steps）
        all_episodes = list(cache['episode_info'])
        # 处理跨 chunk 的 episode：如果旧数据最后处于 episode 中间
        old_ep_start = cache['ep_start']
        for s, e, r in new['episode_info']:
            if s == 0 and old_ep_start < cached_steps:
                # 第一个 episode 起始于旧 chunk 中
                all_episodes.append((old_ep_start, e + cached_steps, r))
            else:
                all_episodes.append((s + cached_steps, e + cached_steps, r))
        next_value = new['next_value']

        # 保存扩展后的缓存
        torch.save({
            'obs': obs, 'actions': actions,
            'rewards': rewards, 'values': values, 'dones': dones,
            'episode_info': all_episodes,
            'next_value': next_value,
            'env_state': new['env_state'],
            'last_obs_t': new['last_obs_t'],
            'ep_start': new['ep_start'] + cached_steps,
            'np_rng': new['np_rng'],
            'torch_rng': new['torch_rng'],
        }, cache_path)
        print(f"Saved extended cache ({args.num_steps} steps)")

    else:
        # 无缓存，从头跑
        print(f"No cache found, collecting {args.num_steps} steps...")
        data = collect_steps(env, agent, args.num_steps, device)
        obs, actions = data['obs'], data['actions']
        rewards, values, dones = data['rewards'], data['values'], data['dones']
        all_episodes = data['episode_info']
        next_value = data['next_value']

        torch.save(data, cache_path)
        print(f"Saved cache ({args.num_steps} steps)")

    # ---- 计算 GAE ----
    advantages = compute_gae(rewards, values, dones, next_value, args.gamma, args.gae_lambda)

    if not all_episodes:
        print(f"WARNING: No complete episodes for {args.env_id} seed {args.seed}")
        sys.exit(1)

    # 按 episode return 降序排列，取前 alpha 比例为正样本，后 alpha 比例为负样本
    sorted_episodes = sorted(all_episodes, key=lambda ep: ep[2], reverse=True)
    num_eps = len(sorted_episodes)
    n_select = max(1, int(num_eps * args.alpha))
    top_episodes = sorted_episodes[:n_select]
    bottom_episodes = sorted_episodes[-n_select:]
    print(f"{args.env_id} seed{args.seed}: {num_eps} episodes in {args.num_steps} steps, "
          f"alpha={args.alpha}, top {n_select} eps (ret >= {top_episodes[-1][2]:.2f}), "
          f"bottom {n_select} eps (ret <= {bottom_episodes[0][2]:.2f})")

    pos_idx, neg_idx = [], []
    for start, end, ret in top_episodes:
        pos_idx.extend(range(start, end + 1))
    for start, end, ret in bottom_episodes:
        neg_idx.extend(range(start, end + 1))

    N = min(len(pos_idx), len(neg_idx))
    if N == 0:
        print(f"WARNING: Cannot split pos/neg for {args.env_id} seed {args.seed}")
        sys.exit(1)
    pos_idx, neg_idx = pos_idx[:N], neg_idx[:N]

    # 计算梯度
    g_pos = compute_pg_gradient(agent, obs[pos_idx], actions[pos_idx], advantages[pos_idx], device)
    g_neg = compute_pg_gradient(agent, obs[neg_idx], actions[neg_idx], advantages[neg_idx], device)
    g_expected = compute_pg_gradient(agent, obs, actions, advantages, device)

    # 方差 = mean_i((g_i - g_expected_i)^2)
    var_pos = ((g_pos - g_expected) ** 2).mean().item()
    var_neg = ((g_neg - g_expected) ** 2).mean().item()

    print(f"  positive_variance={var_pos:.6e}, negative_variance={var_neg:.6e}")

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        "env_id": args.env_id,
        "seed": args.seed,
        "alpha": args.alpha,
        "num_steps": args.num_steps,
        "num_episodes": num_eps,
        "num_selected_episodes": n_select,
        "balanced_N": N,
        "positive_variance": var_pos,
        "negative_variance": var_neg,
    }
    out_path = os.path.join(args.output_dir, f"{args.env_id}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"  saved to {out_path}")
    env.close()
