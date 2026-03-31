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
    compute_tree_gae, compute_branch_weight,
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


def select_next_states_v2(
    terminated_envs, current_step, num_steps, advantages, parent_indices,
    tree_indices, skip_search, search_count, max_search, max_exploitations,
    c=1.0, gamma=0.99,
):
    """Modified select_next_states for variance verification:
    - No return_threshold filtering; filter by whether selected position is last in path
    - Use raw weighted cumulative advantage (no division by remaining steps)
    """
    selected = []
    n_steps = current_step + 1

    for env_idx in terminated_envs:
        env_tree_ids = torch.unique(tree_indices[:n_steps, env_idx]).tolist()
        num_env_trees = len(env_tree_ids)

        if skip_search:
            selected.append(-(num_env_trees + 1))
            continue

        best_tuct_val = float('-inf')
        best_step_overall = None
        best_tree_id = None
        best_depth = None
        best_path_len = None

        for tid in env_tree_ids:
            current_count = search_count[env_idx].get(tid, 0)
            if current_count >= max_search:
                continue

            tree_mask = tree_indices[:n_steps, env_idx] == tid
            tree_node_steps = tree_mask.nonzero(as_tuple=True)[0]
            tree_advs = advantages[tree_node_steps, env_idx].clone()
            tree_parents = parent_indices[tree_node_steps, env_idx]

            root_local_mask = tree_parents < 0
            root_steps = tree_node_steps[root_local_mask]
            root_advs = advantages[root_steps, env_idx]
            best_root_local = root_advs.argmax().item()
            current_node = root_steps[best_root_local].item()

            path = [current_node]
            while True:
                children_of_node = (parent_indices[:n_steps, env_idx] == current_node) & tree_mask
                children_steps = children_of_node.nonzero(as_tuple=True)[0]
                if len(children_steps) == 0:
                    break
                child_advs = advantages[children_steps, env_idx]
                best_child = child_advs.argmax().item()
                current_node = children_steps[best_child].item()
                path.append(current_node)

            path_t = torch.tensor(path, device=advantages.device)
            path_local_mask = torch.isin(tree_node_steps, path_t)
            path_advs = tree_advs[path_local_mask]
            path_steps = tree_node_steps[path_local_mask]

            n = len(path_advs)
            exploitation = torch.zeros_like(path_advs)
            mean_exploitation = torch.zeros_like(path_advs)
            discounted_sum = 0.0
            for k in range(n - 1, -1, -1):
                discounted_sum = -path_advs[k].item() + gamma * discounted_sum
                exploitation[k] = discounted_sum  # 直接使用加权累加，不除以 (n-k)
                mean_exploitation[k] = exploitation[k] / (n - k)

            path_parents_vals = tree_parents[path_local_mask]
            sibling_counts = torch.zeros(len(path_steps), device=advantages.device)
            for i in range(len(path_steps)):
                sibling_counts[i] = (tree_parents == path_parents_vals[i]).sum()

            max_abs_exploitation = exploitation.abs().max().item()
            if max_abs_exploitation == 0:
                max_abs_exploitation = 1.0

            exploration = (sibling_counts - 1) * max_abs_exploitation
            tuct = exploitation - c * exploration

            max_path_idx = tuct.argmax().item()
            max_exploitations[env_idx][tid] = mean_exploitation[max_path_idx].item()

            # 过滤：所选位置是最后一个位置 → 跳过该树
            max_exploitation_values = [v for d in max_exploitations for v in d.values() if v > 0]
            mean_max_exploitations = np.mean(max_exploitation_values) if len(max_exploitation_values) > 0 else 0.0
            std_max_exploitations = np.std(max_exploitation_values) if len(max_exploitation_values) > 1 else 0.0
            if mean_exploitation[max_path_idx] <= mean_max_exploitations + 1.0 * std_max_exploitations:
                continue

            max_tuct_val = tuct[max_path_idx].item()

            if max_tuct_val > best_tuct_val:
                best_tuct_val = max_tuct_val
                best_step_overall = path_steps[max_path_idx].item()
                best_tree_id = tid
                best_depth = max_path_idx
                best_path_len = len(path)

        if best_step_overall is None:
            selected.append(-(num_env_trees + 1))
        else:
            search_count[env_idx][best_tree_id] = search_count[env_idx].get(best_tree_id, 0) + 1
            selected.append(best_step_overall)

    return selected


def collect_ppo_steps(env, agent, num_steps, device, resume_state=None):
    """跑 PPO rollout，返回原始数据 dict。若提供 resume_state 则从该状态精确续跑。"""
    obs_buf, act_buf, rew_buf, val_buf, done_buf = [], [], [], [], []

    if resume_state is not None:
        env.reset()
        env.restore_state(resume_state['env_state'])
        obs_t = resume_state['last_obs_t'].to(device)
        np.random.set_state(resume_state['np_rng'])
        torch.random.set_rng_state(resume_state['torch_rng'])
    else:
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
        done = terminated or truncated
        rew_buf.append(reward)
        done_buf.append(float(done))

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
        'next_value': next_value,
        'env_state': env.clone_state(),
        'last_obs_t': obs_t.cpu(),
        'np_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }


def collect_opts_steps(env, agent, num_steps, device, max_search, c,
                       gamma=0.99, gae_lambda=0.95):
    """OPTS 树搜索 rollout（单环境）。
    不使用 return_threshold，改为按所选位置是否是最后一个位置过滤。
    """
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
    max_exploitations = [{}]

    obs_np, _ = env.reset()
    next_obs = torch.Tensor(obs_np).to(device)
    root_states = [env.clone_state()]
    current_parent = -1
    episode_count = 0

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
            episode_count += 1

        next_obs = torch.Tensor(obs_np).to(device)

        if done:
            compute_tree_gae(
                terminal_step=step,
                env_idx=env_idx,
                rewards=rewards,
                values=values,
                dones=dones,
                parent_indices=parent_indices,
                advantages=advantages,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )

            skip_search = step >= num_steps - 1

            selected = select_next_states_v2(
                terminated_envs=[env_idx],
                current_step=step,
                num_steps=num_steps,
                advantages=advantages,
                parent_indices=parent_indices,
                tree_indices=tree_indices,
                skip_search=skip_search,
                search_count=search_count,
                max_search=max_search,
                max_exploitations=max_exploitations,
                c=c,
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
                    env.restore_state(root_states[parent])
                else:
                    env.restore_state(env_states[parent])
                    state_branches[parent, env_idx] += 1
                next_obs = obs[sel]
                current_parent = parent

    # Bootstrap 非终止叶节点
    with torch.no_grad():
        next_value = agent.get_value(next_obs.unsqueeze(0)).item()
    if current_parent >= 0:
        compute_tree_gae(
            terminal_step=current_parent,
            env_idx=env_idx,
            rewards=rewards,
            values=values,
            dones=dones,
            parent_indices=parent_indices,
            advantages=advantages,
            gamma=gamma,
            gae_lambda=gae_lambda,
            next_value=next_value,
        )

    weights = compute_branch_weight(
        num_steps=num_steps,
        parent_indices=parent_indices,
        state_branches=state_branches,
        env_indices=[env_idx],
        root_branch_counts=root_branch_counts,
    )

    print(f"  OPTS: {episode_count} episodes completed")

    return {
        'obs': obs.cpu(),
        'actions': actions.cpu(),
        'advantages': advantages[:, env_idx].cpu(),
        'weights': weights[:, 0].cpu(),
    }


def load_or_collect_ppo(env, agent, num_steps, device, cache_path, gamma, gae_lambda):
    """加载/续跑 PPO rollout 缓存"""
    cache = None
    cached_steps = 0
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location=device, weights_only=False)
        cached_steps = len(cache['obs'])

    if cached_steps >= num_steps:
        print(f"  PPO: loaded cache ({cached_steps} steps), using first {num_steps}")
        obs = cache['obs'][:num_steps]
        act = cache['actions'][:num_steps]
        rew = cache['rewards'][:num_steps]
        val = cache['values'][:num_steps]
        don = cache['dones'][:num_steps]
        if num_steps < cached_steps:
            with torch.no_grad():
                nv = agent.get_value(cache['obs'][num_steps].unsqueeze(0).to(device)).item()
        else:
            nv = cache['next_value']
        adv = compute_gae(rew, val, don, nv, gamma, gae_lambda)
        return obs, act, adv

    elif cached_steps > 0:
        additional = num_steps - cached_steps
        print(f"  PPO: loaded cache ({cached_steps} steps), continuing {additional} more...")
        resume_state = {
            'env_state': cache['env_state'],
            'last_obs_t': cache['last_obs_t'],
            'np_rng': cache['np_rng'],
            'torch_rng': cache['torch_rng'],
        }
        new = collect_ppo_steps(env, agent, additional, device, resume_state=resume_state)
        merged = {
            'obs': torch.cat([cache['obs'], new['obs']]),
            'actions': torch.cat([cache['actions'], new['actions']]),
            'rewards': torch.cat([cache['rewards'], new['rewards']]),
            'values': torch.cat([cache['values'], new['values']]),
            'dones': torch.cat([cache['dones'], new['dones']]),
            'next_value': new['next_value'],
            'env_state': new['env_state'],
            'last_obs_t': new['last_obs_t'],
            'np_rng': new['np_rng'],
            'torch_rng': new['torch_rng'],
        }
        torch.save(merged, cache_path)
        print(f"  PPO: saved extended cache ({num_steps} steps)")
        adv = compute_gae(merged['rewards'], merged['values'], merged['dones'],
                          merged['next_value'], gamma, gae_lambda)
        return merged['obs'], merged['actions'], adv

    else:
        print(f"  PPO: no cache found, collecting {num_steps} steps...")
        data = collect_ppo_steps(env, agent, num_steps, device)
        torch.save(data, cache_path)
        print(f"  PPO: saved cache ({num_steps} steps)")
        adv = compute_gae(data['rewards'], data['values'], data['dones'],
                          data['next_value'], gamma, gae_lambda)
        return data['obs'], data['actions'], adv


def load_or_collect_opts(env, agent, num_steps, device, cache_path,
                         max_search, c, gamma, gae_lambda):
    """加载/收集 OPTS rollout 缓存。
    缓存够用则截取，不够则重新从头跑。
    """
    # if os.path.exists(cache_path):
    #     cache = torch.load(cache_path, map_location=device, weights_only=False)
    #     cached_steps = len(cache['obs'])
    #     if cached_steps >= num_steps:
    #         print(f"  OPTS: loaded cache ({cached_steps} steps), using first {num_steps}")
    #         return cache['obs'][:num_steps], cache['actions'][:num_steps], \
    #                cache['advantages'][:num_steps], cache['weights'][:num_steps]
    #     else:
    #         print(f"  OPTS: cache has {cached_steps} steps but need {num_steps}, re-collecting...")

    print(f"  OPTS: collecting {num_steps} steps...")
    data = collect_opts_steps(env, agent, num_steps, device, max_search, c,
                              gamma, gae_lambda)
    torch.save(data, cache_path)
    print(f"  OPTS: saved cache ({num_steps} steps)")
    return data['obs'], data['actions'], data['advantages'], data['weights']


def compute_pg_gradient(agent, obs, actions, advantages, device, weights=None):
    """计算策略梯度（仅 actor 参数）。weights 非 None 时使用 OPTS IPW 加权方式"""
    agent.zero_grad()
    obs, actions, advantages = obs.to(device), actions.to(device), advantages.to(device)
    action_mean = agent.actor_mean(obs)
    action_logstd = agent.actor_logstd.expand_as(action_mean)
    dist = Normal(action_mean, torch.exp(action_logstd))
    log_probs = dist.log_prob(actions).sum(-1)
    pg_loss_per_sample = -(log_probs * advantages.detach())
    if weights is not None:
        weights = weights.to(device)
        pg_loss = (pg_loss_per_sample / weights).sum() / (1.0 / weights).sum()
    else:
        pg_loss = pg_loss_per_sample.mean()
    pg_loss.backward()
    actor_params = list(agent.actor_mean.parameters()) + [agent.actor_logstd]
    return torch.cat([p.grad.detach().flatten() if p.grad is not None
                      else torch.zeros(p.numel(), device=device) for p in actor_params])


def estimate_scaling_variance(agent, obs, actions, advantages, batch_sizes,
                               g_star, device, num_bootstrap=200, weights=None):
    """对每个 batch_size，用 bootstrap 有放回采样估计策略梯度总方差 E[||ĝ - g*||²]。

    对每个 B:
        重复 num_bootstrap 次:
            从 pool 中有放回采样 B 个 step
            计算梯度 ĝ_B（OPTS 使用对应 weights 做 IPW 加权）
            记录总方差 ||ĝ_B - g*||² = Σ_i (ĝ_B,i - g*_i)²
        Var(B) = mean, Std(B) = std

    Returns:
        dict: {B: {"mean": float, "std": float}} for each valid batch_size
    """
    N = len(obs)
    results = {}
    for B in batch_sizes:
        if B > N:
            print(f"    batch_size={B} exceeds available steps ({N}), skipping")
            continue
        sq_diffs = []
        for _ in range(num_bootstrap):
            idx = np.random.choice(N, size=B, replace=True)
            w = weights[idx] if weights is not None else None
            g_B = compute_pg_gradient(agent, obs[idx], actions[idx], advantages[idx], device, w)
            sq_diffs.append(((g_B - g_star) ** 2).sum().item())
        mean_var = float(np.mean(sq_diffs))
        std_var = float(np.std(sq_diffs))
        print(f"    B={B}: var={mean_var:.6e} (±{std_var:.2e})")
        results[B] = {"mean": mean_var, "std": std_var}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--total-steps", type=int, default=1000000,
                        help="total PPO steps to collect (for computing g*)")
    parser.add_argument("--num-steps", type=int, default=100000,
                        help="steps for bootstrap sampling (PPO uses first num-steps, OPTS uses all num-steps)")
    parser.add_argument("--batch-sizes", type=str, default="256,512,1024,2048,4096,8192,16384",
                        help="comma-separated list of batch sizes")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/variance/verify2")
    parser.add_argument("--max-search-per-tree", type=int, default=4)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-bootstrap", type=int, default=200,
                        help="number of bootstrap resamples per batch_size")
    args = parser.parse_args()

    batch_sizes = sorted([int(x) for x in args.batch_sizes.split(",")])
    max_bs = max(batch_sizes)
    if args.num_steps <= max_bs:
        print(f"ERROR: --num-steps ({args.num_steps}) must be greater than max batch_size ({max_bs})")
        sys.exit(1)
    if args.total_steps < args.num_steps:
        print(f"ERROR: --total-steps ({args.total_steps}) must be >= --num-steps ({args.num_steps})")
        sys.exit(1)

    device = torch.device("cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu"))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = os.path.join(args.checkpoint_dir, args.env_id, "seed1.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 缓存目录
    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # === PPO ===
    print(f"PPO: {args.env_id} seed {args.seed}")
    env_ppo = make_eval_env(args.env_id, gamma=args.gamma, with_snapshot=True)
    restore_normalization(env_ppo, checkpoint)

    import types
    env_spec = types.SimpleNamespace(
        single_observation_space=env_ppo.observation_space,
        single_action_space=env_ppo.action_space,
    )
    agent = Agent(env_spec).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    # PPO 跑 total-steps（用于计算 g*），bootstrap 从前 num-steps 中采样
    ppo_cache_path = os.path.join(cache_dir, f"{args.env_id}_seed{args.seed}_ppo.pt")
    ppo_obs, ppo_act, ppo_adv = load_or_collect_ppo(
        env_ppo, agent, args.total_steps, device, ppo_cache_path,
        args.gamma, args.gae_lambda)

    print(f"Computing g* from full PPO buffer ({args.total_steps} steps)...")
    g_star = compute_pg_gradient(agent, ppo_obs, ppo_act, ppo_adv, device)

    # PPO bootstrap 仅使用前 num-steps
    print(f"PPO bootstrap variance estimation (from first {args.num_steps} steps, "
          f"{args.num_bootstrap} resamples per B)...")
    ppo_variances = estimate_scaling_variance(
        agent, ppo_obs[:args.num_steps], ppo_act[:args.num_steps],
        ppo_adv[:args.num_steps], batch_sizes, g_star, device,
        num_bootstrap=args.num_bootstrap)
    env_ppo.close()

    # === OPTS ===
    print(f"OPTS: {args.env_id} seed {args.seed}")
    env_opts = make_eval_env(args.env_id, gamma=args.gamma, with_snapshot=True)
    restore_normalization(env_opts, checkpoint)

    # OPTS 只跑 num-steps，bootstrap 使用全部数据
    opts_cache_path = os.path.join(cache_dir, f"{args.env_id}_seed{args.seed}_opts_ms{args.max_search_per_tree}.pt")
    opts_obs, opts_act, opts_adv, opts_weights = load_or_collect_opts(
        env_opts, agent, args.num_steps, device, opts_cache_path,
        args.max_search_per_tree, args.c, args.gamma, args.gae_lambda)
    env_opts.close()

    print(f"OPTS bootstrap variance estimation (from all {args.num_steps} steps, "
          f"{args.num_bootstrap} resamples per B)...")
    opts_variances = estimate_scaling_variance(
        agent, opts_obs, opts_act, opts_adv, batch_sizes, g_star, device,
        num_bootstrap=args.num_bootstrap, weights=opts_weights)

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        "env_id": args.env_id,
        "seed": args.seed,
        "total_steps": args.total_steps,
        "num_steps": args.num_steps,
        "num_bootstrap": args.num_bootstrap,
        "batch_sizes": sorted(set(ppo_variances.keys()) | set(opts_variances.keys())),
        "ppo_variance": {str(k): v for k, v in sorted(ppo_variances.items())},
        "opts_variance": {str(k): v for k, v in sorted(opts_variances.items())},
    }
    out_path = os.path.join(args.output_dir, f"{args.env_id}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"  saved to {out_path}")
