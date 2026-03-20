# 验证2：OPTS 加速策略梯度方差随 batch_size 的收敛速度
import argparse
import json
import os
import sys

import gymnasium as gym
import numpy as np
import torch
from torch.distributions.normal import Normal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cleanrl', 'cleanrl'))
from opts_ttpo_continuous_action import (
    Agent, MuJoCoStateSnapshotWrapper,
    compute_tree_gae, select_next_states, compute_branch_weight,
)


def make_eval_env(env_id, gamma, with_snapshot=False):
    env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    if with_snapshot:
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


def collect_ppo_rollout(env, agent, num_steps, device, gamma=0.99, gae_lambda=0.95):
    obs_buf, act_buf, rew_buf, val_buf, done_buf = [], [], [], [], []

    obs_np, _ = env.reset()
    obs_t = torch.Tensor(obs_np).to(device)

    for step in range(num_steps):
        obs_buf.append(obs_t)
        with torch.no_grad():
            action, _, _, value = agent.get_action_and_value(obs_t.unsqueeze(0))
        action = action.squeeze(0)
        act_buf.append(action)
        val_buf.append(value.item())

        obs_np, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        rew_buf.append(reward)
        done_buf.append(float(terminated or truncated))

        if terminated or truncated:
            obs_np, _ = env.reset()
        obs_t = torch.Tensor(obs_np).to(device)

    with torch.no_grad():
        next_value = agent.get_value(obs_t.unsqueeze(0)).item()

    advantages = compute_gae(
        torch.tensor(rew_buf), torch.tensor(val_buf), torch.tensor(done_buf),
        next_value, gamma, gae_lambda)
    return torch.stack(obs_buf), torch.stack(act_buf), advantages


def collect_opts_rollout(env, agent, num_steps, device, max_search, c,
                         gamma=0.99, gae_lambda=0.95):
    """OPTS 树搜索 rollout（单环境）"""
    env_idx = 0
    num_envs = 1

    obs = torch.zeros((num_steps,) + env.observation_space.shape).to(device)
    actions = torch.zeros((num_steps,) + env.action_space.shape).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)
    parent_indices = -torch.ones((num_steps, num_envs), dtype=torch.long).to(device)
    tree_indices = torch.zeros((num_steps, num_envs), dtype=torch.long).to(device)
    state_branches = torch.ones((num_steps, num_envs), dtype=torch.long).to(device)
    advantages = torch.zeros((num_steps, num_envs)).to(device)

    env_states = [None] * num_steps
    root_branch_counts = [{}]
    search_count = [{}]
    tree_max_returns = [{}]

    obs_np, _ = env.reset()
    next_obs = torch.Tensor(obs_np).to(device)
    root_states = [env.clone_state()]
    current_parent = -1
    episodic_returns = []
    prev_mean_return = None

    for step in range(num_steps):
        obs[step] = next_obs
        with torch.no_grad():
            action, _, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
        action = action.squeeze(0)
        actions[step] = action
        values[step, env_idx] = value.item()

        parent_indices[step, env_idx] = current_parent
        tree_indices[step, env_idx] = current_parent if current_parent < 0 else tree_indices[current_parent, env_idx]
        if current_parent < 0:
            root_branch_counts[0][current_parent] = root_branch_counts[0].get(current_parent, 0) + 1
        current_parent = step

        obs_np, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        done = terminated or truncated
        rewards[step, env_idx] = reward
        dones[step, env_idx] = float(done)
        env_states[step] = env.clone_state()

        if done and "episode" in info:
            ep_r = info['episode']['r']
            ep_return = float(ep_r[0] if hasattr(ep_r, '__getitem__') else ep_r)
            episodic_returns.append(ep_return)
            tid = tree_indices[step, env_idx].item()
            if ep_return > tree_max_returns[0].get(tid, float('-inf')):
                tree_max_returns[0][tid] = ep_return

        next_obs = torch.Tensor(obs_np).to(device)

        if done:
            compute_tree_gae(step, env_idx, rewards, values, dones,
                             parent_indices, advantages, gamma, gae_lambda)

            if episodic_returns:
                prev_mean_return = np.mean(episodic_returns)

            selected = select_next_states(
                terminated_envs=[env_idx], current_step=step, num_steps=num_steps,
                advantages=advantages, parent_indices=parent_indices,
                tree_indices=tree_indices, skip_search=(step >= num_steps - 1),
                search_count=search_count, max_search=max_search, c=c,
                tree_max_returns=tree_max_returns, return_threshold=prev_mean_return,
                gamma=gamma,
            )

            sel = selected[0]
            if sel < 0:
                obs_np, _ = env.reset()
                next_obs = torch.Tensor(obs_np).to(device)
                root_states.insert(0, env.clone_state())
                current_parent = -len(root_states)
            else:
                parent = parent_indices[sel, env_idx].item()
                if parent < 0:
                    env.restore_state(root_states[abs(parent) - 1])
                else:
                    env.restore_state(env_states[parent])
                    state_branches[parent, env_idx] += 1
                next_obs = obs[sel]
                current_parent = parent

    # Bootstrap
    with torch.no_grad():
        next_value = agent.get_value(next_obs.unsqueeze(0)).item()
    if current_parent >= 0:
        compute_tree_gae(current_parent, env_idx, rewards, values, dones,
                         parent_indices, advantages, gamma, gae_lambda, next_value)

    return obs, actions, advantages[:, env_idx]


def compute_pg_gradient(agent, obs, actions, advantages, device):
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


def compute_chunk_variance(agent, obs, actions, advantages, batch_sizes, g_expected, device):
    """对每个 batch_size，将数据分 chunk 计算梯度方差"""
    num_steps = obs.shape[0]
    variances = {}
    for B in batch_sizes:
        num_chunks = num_steps // B
        if num_chunks < 2:
            continue
        chunk_grads = []
        for k in range(num_chunks):
            g_k = compute_pg_gradient(agent, obs[k*B:(k+1)*B], actions[k*B:(k+1)*B],
                                       advantages[k*B:(k+1)*B], device)
            chunk_grads.append(g_k)
        chunk_grads = torch.stack(chunk_grads)
        var_per_param = ((chunk_grads - g_expected.unsqueeze(0)) ** 2).mean(dim=0)
        variances[B] = var_per_param.mean().item()
    return variances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--num-steps", type=int, default=4096)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/variance/verify2")
    parser.add_argument("--max-search-per-tree", type=int, default=4)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    args = parser.parse_args()

    device = torch.device("cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu"))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = os.path.join(args.checkpoint_dir, args.env_id, "seed1.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    batch_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    # === PPO ===
    print(f"PPO rollouts: {args.env_id} seed {args.seed}")
    env_ppo = make_eval_env(args.env_id, gamma=args.gamma)
    restore_normalization(env_ppo, checkpoint)

    agent = Agent([env_ppo]).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    ppo_all_obs, ppo_all_act, ppo_all_adv = [], [], []
    for r in range(args.num_rollouts):
        o, a, adv = collect_ppo_rollout(env_ppo, agent, args.num_steps, device,
                                         args.gamma, args.gae_lambda)
        ppo_all_obs.append(o); ppo_all_act.append(a); ppo_all_adv.append(adv)

    g_expected_ppo = compute_pg_gradient(
        agent, torch.cat(ppo_all_obs), torch.cat(ppo_all_act), torch.cat(ppo_all_adv), device)
    ppo_variances = compute_chunk_variance(
        agent, ppo_all_obs[0], ppo_all_act[0], ppo_all_adv[0], batch_sizes, g_expected_ppo, device)
    print(f"  PPO variances: {ppo_variances}")
    env_ppo.close()

    # === OPTS ===
    print(f"OPTS rollouts: {args.env_id} seed {args.seed}")
    opts_all_obs, opts_all_act, opts_all_adv = [], [], []
    for r in range(args.num_rollouts):
        env_opts = make_eval_env(args.env_id, gamma=args.gamma, with_snapshot=True)
        restore_normalization(env_opts, checkpoint)
        np.random.seed(args.seed * 100 + r)
        torch.manual_seed(args.seed * 100 + r)

        o, a, adv = collect_opts_rollout(
            env_opts, agent, args.num_steps, device, args.max_search_per_tree, args.c,
            args.gamma, args.gae_lambda)
        opts_all_obs.append(o); opts_all_act.append(a); opts_all_adv.append(adv)
        env_opts.close()

    g_expected_opts = compute_pg_gradient(
        agent, torch.cat(opts_all_obs), torch.cat(opts_all_act), torch.cat(opts_all_adv), device)
    opts_variances = compute_chunk_variance(
        agent, opts_all_obs[0], opts_all_act[0], opts_all_adv[0], batch_sizes, g_expected_opts, device)
    print(f"  OPTS variances: {opts_variances}")

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        "env_id": args.env_id,
        "seed": args.seed,
        "batch_sizes": sorted(set(ppo_variances.keys()) | set(opts_variances.keys())),
        "ppo_variance": {str(k): v for k, v in sorted(ppo_variances.items())},
        "opts_variance": {str(k): v for k, v in sorted(opts_variances.items())},
    }
    out_path = os.path.join(args.output_dir, f"{args.env_id}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"  saved to {out_path}")
