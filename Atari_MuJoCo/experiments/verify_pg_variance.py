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
from opts_ttpo_continuous_action import Agent


def make_eval_env(env_id, gamma):
    env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
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


def collect_rollout(env, agent, num_steps, device, gamma=0.99, gae_lambda=0.95):
    """跑一次 rollout，返回 obs, actions, advantages, episode_info"""
    obs_buf, act_buf, rew_buf, val_buf, done_buf = [], [], [], [], []
    episode_info = []  # (start, end, return)

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

    advantages = compute_gae(
        torch.tensor(rew_buf), torch.tensor(val_buf), torch.tensor(done_buf),
        next_value, gamma, gae_lambda)
    return torch.stack(obs_buf), torch.stack(act_buf), advantages, episode_info


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
    parser.add_argument("--num-steps", type=int, default=4096)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/variance/verify1")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    args = parser.parse_args()

    device = torch.device("cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu"))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 加载 checkpoint
    ckpt_path = os.path.join(args.checkpoint_dir, args.env_id, "seed1.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    env = make_eval_env(args.env_id, gamma=args.gamma)
    restore_normalization(env, checkpoint)

    agent = Agent([env]).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    # 收集多次 rollout
    all_obs, all_act, all_adv = [], [], []
    first_episodes = None

    for r in range(args.num_rollouts):
        obs, actions, advantages, episodes = collect_rollout(
            env, agent, args.num_steps, device, args.gamma, args.gae_lambda)
        all_obs.append(obs)
        all_act.append(actions)
        all_adv.append(advantages)
        if r == 0:
            first_episodes = episodes

    if not first_episodes:
        print(f"WARNING: No complete episodes for {args.env_id} seed {args.seed}")
        sys.exit(1)

    # 按 episode return 分正负
    ep_returns = [ep[2] for ep in first_episodes]
    mean_return = np.mean(ep_returns)
    print(f"{args.env_id} seed{args.seed}: {len(first_episodes)} episodes, mean_return={mean_return:.2f}")

    pos_idx, neg_idx = [], []
    for start, end, ret in first_episodes:
        indices = list(range(start, end + 1))
        (pos_idx if ret > mean_return else neg_idx).extend(indices)

    N = min(len(pos_idx), len(neg_idx))
    if N == 0:
        print(f"WARNING: Cannot split pos/neg for {args.env_id} seed {args.seed}")
        sys.exit(1)
    pos_idx, neg_idx = pos_idx[:N], neg_idx[:N]

    # 计算梯度
    first_obs, first_act, first_adv = all_obs[0], all_act[0], all_adv[0]
    full_obs, full_act, full_adv = torch.cat(all_obs), torch.cat(all_act), torch.cat(all_adv)

    g_pos = compute_pg_gradient(agent, first_obs[pos_idx], first_act[pos_idx], first_adv[pos_idx], device)
    g_neg = compute_pg_gradient(agent, first_obs[neg_idx], first_act[neg_idx], first_adv[neg_idx], device)
    g_expected = compute_pg_gradient(agent, full_obs, full_act, full_adv, device)

    # 方差 = mean_i((g_i - g_expected_i)^2)
    var_pos = ((g_pos - g_expected) ** 2).mean().item()
    var_neg = ((g_neg - g_expected) ** 2).mean().item()

    print(f"  positive_variance={var_pos:.6e}, negative_variance={var_neg:.6e}")

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        "env_id": args.env_id,
        "seed": args.seed,
        "num_episodes": len(first_episodes),
        "mean_episode_return": float(mean_return),
        "balanced_N": N,
        "positive_variance": var_pos,
        "negative_variance": var_neg,
    }
    out_path = os.path.join(args.output_dir, f"{args.env_id}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"  saved to {out_path}")
    env.close()
