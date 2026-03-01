"""
环境状态恢复验证测试脚本
测试 MuJoCoStateSnapshotWrapper 的 clone_state / restore_state 是否正确工作
"""
import gymnasium as gym
import numpy as np
import torch
import sys
sys.path.insert(0, '../cleanrl/cleanrl')
from opts_ttpo_continuous_action import MuJoCoStateSnapshotWrapper

def make_test_env(env_id, gamma=0.99):
    """创建带完整 wrapper 链的测试环境"""
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

def test_basic_restore(env_id):
    """测试1: 基本状态恢复 - 恢复后物理状态是否一致"""
    print(f"\n{'='*60}")
    print(f"测试1: 基本状态恢复 - {env_id}")
    print('='*60)
    
    env = make_test_env(env_id)
    env.reset(seed=42)
    
    # 执行几步
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    
    # 保存状态
    saved_state = env.clone_state()
    saved_qpos = env.unwrapped.data.qpos.copy()
    saved_qvel = env.unwrapped.data.qvel.copy()
    saved_goal = env.unwrapped.goal.copy() if hasattr(env.unwrapped, 'goal') else None
    
    print(f"保存时 qpos: {saved_qpos[:4]}...")
    print(f"保存时 qvel: {saved_qvel[:4]}...")
    if saved_goal is not None:
        print(f"保存时 goal: {saved_goal}")
    
    # 继续执行更多步，改变状态
    for _ in range(20):
        action = env.action_space.sample()
        env.step(action)
    
    print(f"\n执行20步后 qpos: {env.unwrapped.data.qpos[:4]}...")
    print(f"执行20步后 qvel: {env.unwrapped.data.qvel[:4]}...")
    
    # 恢复状态
    env.restore_state(saved_state)
    
    restored_qpos = env.unwrapped.data.qpos.copy()
    restored_qvel = env.unwrapped.data.qvel.copy()
    restored_goal = env.unwrapped.goal.copy() if hasattr(env.unwrapped, 'goal') else None
    
    print(f"\n恢复后 qpos: {restored_qpos[:4]}...")
    print(f"恢复后 qvel: {restored_qvel[:4]}...")
    if restored_goal is not None:
        print(f"恢复后 goal: {restored_goal}")
    
    # 验证
    qpos_match = np.allclose(saved_qpos, restored_qpos)
    qvel_match = np.allclose(saved_qvel, restored_qvel)
    goal_match = saved_goal is None or np.allclose(saved_goal, restored_goal)
    
    print(f"\n✓ qpos 恢复正确: {qpos_match}")
    print(f"✓ qvel 恢复正确: {qvel_match}")
    print(f"✓ goal 恢复正确: {goal_match}")
    
    env.close()
    return qpos_match and qvel_match and goal_match

def test_observation_consistency(env_id):
    """测试2: 观测一致性 - 恢复后获取的观测是否和保存时一致"""
    print(f"\n{'='*60}")
    print(f"测试2: 观测一致性 - {env_id}")
    print('='*60)
    
    env = make_test_env(env_id)
    env.reset(seed=42)
    
    # 执行几步
    for _ in range(10):
        action = env.action_space.sample()
        obs_after_step, _, _, _, _ = env.step(action)
    
    # 保存状态和观测
    saved_state = env.clone_state()
    saved_obs = obs_after_step.copy()
    
    # 获取 NormalizeObservation 的统计量
    norm_obs_wrapper = env._normalize_obs_wrapper
    saved_obs_mean = norm_obs_wrapper.obs_rms.mean.copy()
    saved_obs_var = norm_obs_wrapper.obs_rms.var.copy()
    
    print(f"保存时观测: {saved_obs[:5]}...")
    print(f"保存时 obs_rms.mean: {saved_obs_mean[:5]}...")
    print(f"保存时 obs_rms.var: {saved_obs_var[:5]}...")
    
    # 继续执行更多步
    for _ in range(50):
        action = env.action_space.sample()
        env.step(action)
    
    print(f"\n执行50步后 obs_rms.mean: {norm_obs_wrapper.obs_rms.mean[:5]}...")
    print(f"执行50步后 obs_rms.var: {norm_obs_wrapper.obs_rms.var[:5]}...")
    
    # 恢复状态
    env.restore_state(saved_state)
    
    restored_obs_mean = norm_obs_wrapper.obs_rms.mean.copy()
    restored_obs_var = norm_obs_wrapper.obs_rms.var.copy()
    
    print(f"\n恢复后 obs_rms.mean: {restored_obs_mean[:5]}...")
    print(f"恢复后 obs_rms.var: {restored_obs_var[:5]}...")
    
    # 从恢复后的状态重新计算观测
    raw_obs = env.unwrapped._get_obs()
    # 手动应用归一化（不更新统计量）
    # 使用 NormalizeObservation 的实际 epsilon
    epsilon = norm_obs_wrapper.epsilon
    recomputed_obs = (raw_obs - restored_obs_mean) / np.sqrt(restored_obs_var + epsilon)
    recomputed_obs = np.clip(recomputed_obs, -10, 10)
    
    print(f"\n恢复后重算观测: {recomputed_obs[:5]}...")
    print(f"保存时的观测:   {saved_obs[:5]}...")
    
    # 验证
    obs_rms_mean_match = np.allclose(saved_obs_mean, restored_obs_mean)
    obs_rms_var_match = np.allclose(saved_obs_var, restored_obs_var)
    obs_match = np.allclose(saved_obs, recomputed_obs, atol=1e-5)
    
    print(f"\n✓ obs_rms.mean 恢复正确: {obs_rms_mean_match}")
    print(f"✓ obs_rms.var 恢复正确: {obs_rms_var_match}")
    print(f"✓ 观测值一致: {obs_match}")
    
    if not obs_match:
        print(f"  观测差异: {np.abs(saved_obs - recomputed_obs).max()}")
    
    env.close()
    return obs_rms_mean_match and obs_rms_var_match and obs_match

def test_deterministic_rollout(env_id):
    """测试3: 确定性轨迹 - 从相同状态执行相同动作序列，结果是否一致"""
    print(f"\n{'='*60}")
    print(f"测试3: 确定性轨迹 - {env_id}")
    print('='*60)
    
    env = make_test_env(env_id)
    env.reset(seed=42)
    
    # 执行几步到达某个状态
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    
    # 保存状态
    saved_state = env.clone_state()
    
    # 生成固定的动作序列
    np.random.seed(123)
    action_sequence = [env.action_space.sample() for _ in range(20)]
    
    # 第一次执行动作序列
    obs_list_1 = []
    reward_list_1 = []
    for action in action_sequence:
        obs, reward, terminated, truncated, _ = env.step(action)
        obs_list_1.append(obs.copy())
        reward_list_1.append(reward)
        if terminated or truncated:
            break
    
    print(f"第一次执行: {len(obs_list_1)} 步")
    print(f"  最终观测: {obs_list_1[-1][:5]}...")
    print(f"  累计奖励: {sum(reward_list_1):.4f}")
    
    # 恢复状态
    env.restore_state(saved_state)
    
    # 第二次执行相同的动作序列
    obs_list_2 = []
    reward_list_2 = []
    for i, action in enumerate(action_sequence):
        if i >= len(obs_list_1):
            break
        obs, reward, terminated, truncated, _ = env.step(action)
        obs_list_2.append(obs.copy())
        reward_list_2.append(reward)
        if terminated or truncated:
            break
    
    print(f"\n第二次执行: {len(obs_list_2)} 步")
    print(f"  最终观测: {obs_list_2[-1][:5]}...")
    print(f"  累计奖励: {sum(reward_list_2):.4f}")
    
    # 验证
    length_match = len(obs_list_1) == len(obs_list_2)
    
    if length_match:
        obs_diffs = [np.abs(o1 - o2).max() for o1, o2 in zip(obs_list_1, obs_list_2)]
        reward_diffs = [abs(r1 - r2) for r1, r2 in zip(reward_list_1, reward_list_2)]
        
        max_obs_diff = max(obs_diffs)
        max_reward_diff = max(reward_diffs)
        
        obs_match = max_obs_diff < 1e-5
        reward_match = max_reward_diff < 1e-5
        
        print(f"\n✓ 轨迹长度一致: {length_match}")
        print(f"✓ 观测序列一致: {obs_match} (最大差异: {max_obs_diff:.2e})")
        print(f"✓ 奖励序列一致: {reward_match} (最大差异: {max_reward_diff:.2e})")
        
        if not obs_match or not reward_match:
            # 找出第一个不一致的位置
            for i, (o1, o2, r1, r2) in enumerate(zip(obs_list_1, obs_list_2, reward_list_1, reward_list_2)):
                if np.abs(o1 - o2).max() > 1e-5 or abs(r1 - r2) > 1e-5:
                    print(f"\n  ⚠ 第 {i} 步开始出现不一致:")
                    print(f"    obs diff: {np.abs(o1 - o2).max():.2e}")
                    print(f"    reward diff: {abs(r1 - r2):.2e}")
                    break
    else:
        obs_match = False
        reward_match = False
        print(f"\n✗ 轨迹长度不一致: {len(obs_list_1)} vs {len(obs_list_2)}")
    
    env.close()
    return length_match and obs_match and reward_match

def test_normalize_reward_restore(env_id):
    """测试4: NormalizeReward 状态恢复"""
    print(f"\n{'='*60}")
    print(f"测试4: NormalizeReward 状态恢复 - {env_id}")
    print('='*60)
    
    env = make_test_env(env_id)
    env.reset(seed=42)
    
    # 执行几步
    for _ in range(30):
        action = env.action_space.sample()
        env.step(action)
    
    # 保存状态
    saved_state = env.clone_state()
    norm_reward_wrapper = env._normalize_reward_wrapper
    
    saved_return_rms_mean = norm_reward_wrapper.return_rms.mean.copy()
    saved_return_rms_var = norm_reward_wrapper.return_rms.var.copy()
    saved_returns = float(norm_reward_wrapper.returns) if hasattr(norm_reward_wrapper.returns, '__float__') else norm_reward_wrapper.returns.copy()
    
    print(f"保存时 return_rms.mean: {saved_return_rms_mean}")
    print(f"保存时 return_rms.var: {saved_return_rms_var}")
    print(f"保存时 returns: {saved_returns}")
    
    # 继续执行
    for _ in range(50):
        action = env.action_space.sample()
        env.step(action)
    
    print(f"\n执行50步后 return_rms.mean: {norm_reward_wrapper.return_rms.mean}")
    print(f"执行50步后 returns: {norm_reward_wrapper.returns}")
    
    # 恢复
    env.restore_state(saved_state)
    
    restored_return_rms_mean = norm_reward_wrapper.return_rms.mean.copy()
    restored_return_rms_var = norm_reward_wrapper.return_rms.var.copy()
    restored_returns = float(norm_reward_wrapper.returns) if hasattr(norm_reward_wrapper.returns, '__float__') else norm_reward_wrapper.returns.copy()
    
    print(f"\n恢复后 return_rms.mean: {restored_return_rms_mean}")
    print(f"恢复后 return_rms.var: {restored_return_rms_var}")
    print(f"恢复后 returns: {restored_returns}")
    
    # 验证
    mean_match = np.allclose(saved_return_rms_mean, restored_return_rms_mean)
    var_match = np.allclose(saved_return_rms_var, restored_return_rms_var)
    returns_match = np.allclose(saved_returns, restored_returns)
    
    print(f"\n✓ return_rms.mean 恢复正确: {mean_match}")
    print(f"✓ return_rms.var 恢复正确: {var_match}")
    print(f"✓ returns 恢复正确: {returns_match}")
    
    env.close()
    return mean_match and var_match and returns_match

def test_branch_consistency(env_id):
    """测试5: 分支一致性 - 模拟树搜索中的分支场景"""
    print(f"\n{'='*60}")
    print(f"测试5: 分支一致性（模拟树搜索）- {env_id}")
    print('='*60)
    
    env = make_test_env(env_id)
    env.reset(seed=42)
    
    # 执行到某个状态作为分支点
    for _ in range(10):
        action = env.action_space.sample()
        obs_at_branch, _, _, _, _ = env.step(action)
    
    # 保存分支点状态
    branch_state = env.clone_state()
    branch_obs = obs_at_branch.copy()
    
    print(f"分支点观测: {branch_obs[:5]}...")
    
    # 分支1: 执行动作序列 A
    np.random.seed(100)
    actions_A = [env.action_space.sample() for _ in range(10)]
    results_A = []
    for action in actions_A:
        obs, reward, term, trunc, _ = env.step(action)
        results_A.append((obs.copy(), reward))
        if term or trunc:
            break
    
    print(f"\n分支1: 执行 {len(results_A)} 步, 累计奖励: {sum(r for _, r in results_A):.4f}")
    
    # 恢复到分支点
    env.restore_state(branch_state)
    restored_obs = env.unwrapped._get_obs()
    # 手动归一化 - 使用 NormalizeObservation 的实际 epsilon
    norm_obs_wrapper = env._normalize_obs_wrapper
    obs_rms = norm_obs_wrapper.obs_rms
    epsilon = norm_obs_wrapper.epsilon
    restored_obs_normalized = (restored_obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon)
    restored_obs_normalized = np.clip(restored_obs_normalized, -10, 10)
    
    print(f"\n恢复后观测: {restored_obs_normalized[:5]}...")
    print(f"分支点观测: {branch_obs[:5]}...")
    obs_restored_match = np.allclose(branch_obs, restored_obs_normalized, atol=1e-5)
    print(f"✓ 恢复后观测与分支点一致: {obs_restored_match}")
    
    # 分支2: 执行动作序列 B
    np.random.seed(200)
    actions_B = [env.action_space.sample() for _ in range(10)]
    results_B = []
    for action in actions_B:
        obs, reward, term, trunc, _ = env.step(action)
        results_B.append((obs.copy(), reward))
        if term or trunc:
            break
    
    print(f"分支2: 执行 {len(results_B)} 步, 累计奖励: {sum(r for _, r in results_B):.4f}")
    
    # 再次恢复并重新执行分支1，验证确定性
    env.restore_state(branch_state)
    results_A_repeat = []
    for action in actions_A:
        if len(results_A_repeat) >= len(results_A):
            break
        obs, reward, term, trunc, _ = env.step(action)
        results_A_repeat.append((obs.copy(), reward))
        if term or trunc:
            break
    
    # 验证分支1的重复执行结果一致
    if len(results_A) == len(results_A_repeat):
        obs_diffs = [np.abs(r1[0] - r2[0]).max() for r1, r2 in zip(results_A, results_A_repeat)]
        reward_diffs = [abs(r1[1] - r2[1]) for r1, r2 in zip(results_A, results_A_repeat)]
        branch_deterministic = max(obs_diffs) < 1e-5 and max(reward_diffs) < 1e-5
    else:
        branch_deterministic = False
    
    print(f"\n✓ 分支重复执行确定性: {branch_deterministic}")
    
    env.close()
    return obs_restored_match and branch_deterministic

def get_env_max_steps(env_id):
    """获取环境的最大 episode 步数"""
    # MuJoCo 环境的默认 TimeLimit
    max_steps_map = {
        'Reacher-v4': 50,
        'HalfCheetah-v4': 1000,
        'Walker2d-v4': 1000,
        'Hopper-v4': 1000,
        'Ant-v4': 1000,
        'Humanoid-v4': 1000,
        'Swimmer-v4': 1000,
        'InvertedPendulum-v4': 1000,
        'InvertedDoublePendulum-v4': 1000,
        'Pusher-v4': 100,
    }
    return max_steps_map.get(env_id, 1000)


def test_deep_restore(env_id):
    """测试6: 深度恢复 - 恢复到不同深度的状态是否正确"""
    print(f"\n{'='*60}")
    print(f"测试6: 深度恢复（模拟树搜索回溯到较远状态）- {env_id}")
    print('='*60)
    
    env = make_test_env(env_id)
    env.reset(seed=42)
    
    # 根据环境的最大 episode 长度调整测试深度
    max_steps = get_env_max_steps(env_id)
    # 只测试不超过 episode 长度的深度，避免 reset 带来的混淆
    all_depths = [10, 20, 40, 80, 100]
    depths = [d for d in all_depths if d <= max_steps - 5]  # 留一些余量
    
    print(f"环境最大步数: {max_steps}, 测试深度: {depths}")
    
    saved_states = {}
    saved_obs = {}
    saved_obs_rms = {}
    saved_reward_rms = {}
    
    norm_obs_wrapper = env._normalize_obs_wrapper
    norm_reward_wrapper = env._normalize_reward_wrapper
    
    step_count = 0
    episode_steps = 0  # 当前 episode 内的步数
    for depth in depths:
        # 执行到目标深度（在单个 episode 内）
        while step_count < depth:
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            step_count += 1
            episode_steps += 1
            # 如果 episode 终止，跳过后续深度（因为涉及 reset）
            if terminated or truncated:
                print(f"  ⚠ Episode 在深度 {step_count} 终止 (terminated={terminated}, truncated={truncated})")
                # 只保留已完成的深度
                depths = [d for d in depths if d < step_count]
                break
        
        if step_count < depth:
            break
            
        # 保存状态
        saved_states[depth] = env.clone_state()
        saved_obs[depth] = obs.copy()
        saved_obs_rms[depth] = {
            'mean': norm_obs_wrapper.obs_rms.mean.copy(),
            'var': norm_obs_wrapper.obs_rms.var.copy(),
            'count': norm_obs_wrapper.obs_rms.count,
        }
        saved_reward_rms[depth] = {
            'mean': norm_reward_wrapper.return_rms.mean.copy() if hasattr(norm_reward_wrapper.return_rms.mean, 'copy') else float(norm_reward_wrapper.return_rms.mean),
            'var': norm_reward_wrapper.return_rms.var.copy() if hasattr(norm_reward_wrapper.return_rms.var, 'copy') else float(norm_reward_wrapper.return_rms.var),
        }
        print(f"深度 {depth}: obs_rms.count={norm_obs_wrapper.obs_rms.count:.0f}, obs_mean[:3]={norm_obs_wrapper.obs_rms.mean[:3]}")
    
    if len(depths) == 0:
        print("没有可测试的深度（episode 太短）")
        env.close()
        return True
    
    # 继续执行更多步，改变状态（但不要触发 reset）
    max_continue_steps = min(100, max_steps - step_count - 5)
    if max_continue_steps > 0:
        print(f"\n继续执行 {max_continue_steps} 步（不触发 reset）...")
        for _ in range(max_continue_steps):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                print(f"  ⚠ 在继续执行中 episode 终止")
                break
    
    print(f"执行后 obs_rms.count={norm_obs_wrapper.obs_rms.count:.0f}")
    
    # 测试从不同深度恢复
    all_passed = True
    print(f"\n{'='*40}")
    print("测试恢复到不同深度的状态...")
    print('='*40)
    
    for depth in reversed(depths):  # 从最远的开始恢复
        print(f"\n--- 恢复到深度 {depth} ---")
        env.restore_state(saved_states[depth])
        
        # 验证 obs_rms 是否正确恢复
        obs_rms_match = (
            np.allclose(norm_obs_wrapper.obs_rms.mean, saved_obs_rms[depth]['mean']) and
            np.allclose(norm_obs_wrapper.obs_rms.var, saved_obs_rms[depth]['var']) and
            abs(norm_obs_wrapper.obs_rms.count - saved_obs_rms[depth]['count']) < 1
        )
        
        # 验证 reward_rms 是否正确恢复
        reward_rms_match = (
            np.allclose(norm_reward_wrapper.return_rms.mean, saved_reward_rms[depth]['mean']) and
            np.allclose(norm_reward_wrapper.return_rms.var, saved_reward_rms[depth]['var'])
        )
        
        # 从恢复的状态重新计算观测
        raw_obs = env.unwrapped._get_obs()
        epsilon = norm_obs_wrapper.epsilon
        recomputed_obs = (raw_obs - norm_obs_wrapper.obs_rms.mean) / np.sqrt(norm_obs_wrapper.obs_rms.var + epsilon)
        recomputed_obs = np.clip(recomputed_obs, -10, 10)
        obs_match = np.allclose(saved_obs[depth], recomputed_obs, atol=1e-4)
        
        print(f"  obs_rms 恢复: {'✓' if obs_rms_match else '✗'} (count: {norm_obs_wrapper.obs_rms.count:.0f} vs {saved_obs_rms[depth]['count']:.0f})")
        print(f"  reward_rms 恢复: {'✓' if reward_rms_match else '✗'}")
        print(f"  观测一致: {'✓' if obs_match else '✗'}")
        
        if not obs_match:
            diff = np.abs(saved_obs[depth] - recomputed_obs).max()
            print(f"    观测差异: {diff:.2e}")
        
        # 执行确定性测试：从恢复状态执行固定动作序列
        np.random.seed(42 + depth)
        test_actions = [env.action_space.sample() for _ in range(20)]
        
        results_1 = []
        for action in test_actions:
            obs, reward, term, trunc, _ = env.step(action)
            results_1.append((obs.copy(), reward))
            if term or trunc:
                break
        
        # 再次恢复并执行相同动作
        env.restore_state(saved_states[depth])
        results_2 = []
        for i, action in enumerate(test_actions):
            if i >= len(results_1):
                break
            obs, reward, term, trunc, _ = env.step(action)
            results_2.append((obs.copy(), reward))
            if term or trunc:
                break
        
        if len(results_1) == len(results_2):
            obs_diffs = [np.abs(r1[0] - r2[0]).max() for r1, r2 in zip(results_1, results_2)]
            reward_diffs = [abs(r1[1] - r2[1]) for r1, r2 in zip(results_1, results_2)]
            deterministic = max(obs_diffs) < 1e-5 and max(reward_diffs) < 1e-5
            print(f"  确定性执行: {'✓' if deterministic else '✗'} (obs_diff: {max(obs_diffs):.2e}, reward_diff: {max(reward_diffs):.2e})")
        else:
            deterministic = False
            print(f"  确定性执行: ✗ (轨迹长度不一致)")
        
        if not (obs_rms_match and reward_rms_match and obs_match and deterministic):
            all_passed = False
    
    env.close()
    return all_passed


def test_multiple_restore_cycles(env_id):
    """测试7: 多次恢复循环 - 模拟树搜索中频繁的保存/恢复"""
    print(f"\n{'='*60}")
    print(f"测试7: 多次恢复循环（模拟树搜索频繁回溯）- {env_id}")
    print('='*60)
    
    env = make_test_env(env_id)
    env.reset(seed=42)
    
    norm_obs_wrapper = env._normalize_obs_wrapper
    norm_reward_wrapper = env._normalize_reward_wrapper
    
    # 预先生成固定的动作数组（重要：不能用 env.action_space.sample()，因为它的随机性不受 np.random.seed 控制）
    action_shape = env.action_space.shape
    action_low = env.action_space.low
    action_high = env.action_space.high
    
    rng = np.random.RandomState(999)  # 使用固定种子的随机数生成器
    fixed_actions = [rng.uniform(action_low, action_high).astype(np.float32) for _ in range(30)]
    explore_actions = [rng.uniform(action_low, action_high).astype(np.float32) for _ in range(50)]
    
    # 执行一些步骤到达基准状态（不会提前终止的较短步数）
    for _ in range(15):
        action = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            env.reset(seed=42)
    
    # 保存基准状态
    base_state = env.clone_state()
    base_obs_rms_count = norm_obs_wrapper.obs_rms.count
    
    print(f"基准状态 obs_rms.count: {base_obs_rms_count:.0f}")
    
    # 执行多次恢复循环
    num_cycles = 10
    cycle_results = []
    
    for cycle in range(num_cycles):
        # 恢复到基准状态
        env.restore_state(base_state)
        
        # 验证 obs_rms.count 是否正确
        count_after_restore = norm_obs_wrapper.obs_rms.count
        count_match = abs(count_after_restore - base_obs_rms_count) < 1
        
        # 执行固定动作序列（使用预生成的固定动作）
        total_reward = 0
        final_obs = None
        step_count = 0
        for action in fixed_actions:
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            final_obs = obs.copy()
            step_count += 1
            if term or trunc:
                break
        
        cycle_results.append({
            'count_match': count_match,
            'count_after_restore': count_after_restore,
            'total_reward': total_reward,
            'final_obs_mean': final_obs.mean() if final_obs is not None else None,
            'step_count': step_count,
        })
        
        # 继续执行更多步骤（模拟树搜索中的分支探索）
        # 使用不同的随机动作来改变归一化统计量
        for i, action in enumerate(explore_actions):
            if i >= 50:
                break
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                env.reset()
                break
    
    # 分析结果
    print(f"\n多次恢复循环结果:")
    all_counts_match = all(r['count_match'] for r in cycle_results)
    rewards = [r['total_reward'] for r in cycle_results]
    obs_means = [r['final_obs_mean'] for r in cycle_results if r['final_obs_mean'] is not None]
    step_counts = [r['step_count'] for r in cycle_results]
    
    reward_consistent = max(rewards) - min(rewards) < 1e-5
    obs_consistent = max(obs_means) - min(obs_means) < 1e-5 if obs_means else True
    steps_consistent = len(set(step_counts)) == 1
    
    print(f"  obs_rms.count 一致: {'✓' if all_counts_match else '✗'}")
    print(f"  执行步数一致: {'✓' if steps_consistent else '✗'} ({step_counts})")
    print(f"  奖励一致: {'✓' if reward_consistent else '✗'} (范围: {min(rewards):.6f} ~ {max(rewards):.6f})")
    print(f"  观测均值一致: {'✓' if obs_consistent else '✗'}")
    
    if not all_counts_match:
        counts = [r['count_after_restore'] for r in cycle_results]
        print(f"    obs_rms.count 值: {counts}")
    
    if not reward_consistent:
        print(f"    奖励差异: {max(rewards) - min(rewards):.2e}")
    
    env.close()
    return all_counts_match and reward_consistent and obs_consistent and steps_consistent


def run_all_tests(env_id="Reacher-v4"):
    """运行所有测试"""
    print(f"\n{'#'*60}")
    print(f"环境恢复测试 - {env_id}")
    print('#'*60)
    
    results = {}
    
    try:
        results['基本状态恢复'] = test_basic_restore(env_id)
    except Exception as e:
        print(f"测试1失败: {e}")
        import traceback
        traceback.print_exc()
        results['基本状态恢复'] = False
    
    try:
        results['观测一致性'] = test_observation_consistency(env_id)
    except Exception as e:
        print(f"测试2失败: {e}")
        import traceback
        traceback.print_exc()
        results['观测一致性'] = False
    
    try:
        results['确定性轨迹'] = test_deterministic_rollout(env_id)
    except Exception as e:
        print(f"测试3失败: {e}")
        import traceback
        traceback.print_exc()
        results['确定性轨迹'] = False
    
    try:
        results['NormalizeReward恢复'] = test_normalize_reward_restore(env_id)
    except Exception as e:
        print(f"测试4失败: {e}")
        import traceback
        traceback.print_exc()
        results['NormalizeReward恢复'] = False
    
    try:
        results['分支一致性'] = test_branch_consistency(env_id)
    except Exception as e:
        print(f"测试5失败: {e}")
        import traceback
        traceback.print_exc()
        results['分支一致性'] = False
    
    try:
        results['深度恢复'] = test_deep_restore(env_id)
    except Exception as e:
        print(f"测试6失败: {e}")
        import traceback
        traceback.print_exc()
        results['深度恢复'] = False
    
    try:
        results['多次恢复循环'] = test_multiple_restore_cycles(env_id)
    except Exception as e:
        print(f"测试7失败: {e}")
        import traceback
        traceback.print_exc()
        results['多次恢复循环'] = False
    
    # 总结
    print(f"\n{'='*60}")
    print("测试总结")
    print('='*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 所有测试通过！环境恢复功能正常。")
    else:
        print("⚠️  部分测试失败！环境恢复可能存在问题。")
    print('='*60)
    
    return all_passed

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="Reacher-v4")
    args = parser.parse_args()
    
    # 测试指定环境
    run_all_tests(args.env_id)
    
    # 如果是 Reacher，也测试其他几个环境
    if args.env_id == "Reacher-v4":
        print("\n\n" + "="*60)
        print("额外测试其他环境...")
        print("="*60)
        
        for env_id in ["HalfCheetah-v4", "Ant-v4", "Walker2d-v4", "Hopper-v4", "Humanoid-v4"]:
            try:
                run_all_tests(env_id)
            except Exception as e:
                print(f"\n{env_id} 测试失败: {e}")
