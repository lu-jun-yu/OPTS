"""
Atari 环境状态恢复验证测试脚本
测试 AtariStateSnapshotWrapper 的 clone_state / restore_state 是否正确工作
"""
import gymnasium as gym
import numpy as np
import sys
sys.path.insert(0, '../cleanrl/cleanrl')
from opts_ttpo_atari import AtariStateSnapshotWrapper

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def make_test_env(env_id):
    """创建带完整 wrapper 链的测试环境（与 opts_ttpo_atari.py 相同）"""
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env = AtariStateSnapshotWrapper(env)
    return env


def test_basic_restore(env_id):
    """测试1: 基本状态恢复 - 恢复后 ALE RAM 是否一致"""
    print(f"\n{'='*60}")
    print(f"测试1: 基本状态恢复 - {env_id}")
    print('='*60)
    
    env = make_test_env(env_id)
    env.reset(seed=42)
    
    # 执行几步
    for _ in range(50):
        action = env.action_space.sample()
        env.step(action)
    
    # 保存状态
    saved_state = env.clone_state()
    saved_ram = env.ale.getRAM().copy()
    saved_lives = env.ale.lives()
    
    print(f"保存时 RAM (前20字节): {saved_ram[:20]}...")
    print(f"保存时 lives: {saved_lives}")
    
    # 继续执行更多步，改变状态
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
    
    print(f"\n执行100步后 RAM (前20字节): {env.ale.getRAM()[:20]}...")
    print(f"执行100步后 lives: {env.ale.lives()}")
    
    # 恢复状态
    env.restore_state(saved_state)
    
    restored_ram = env.ale.getRAM().copy()
    restored_lives = env.ale.lives()
    
    print(f"\n恢复后 RAM (前20字节): {restored_ram[:20]}...")
    print(f"恢复后 lives: {restored_lives}")
    
    # 验证
    ram_match = np.array_equal(saved_ram, restored_ram)
    lives_match = saved_lives == restored_lives
    
    print(f"\n✓ RAM 恢复正确: {ram_match}")
    print(f"✓ lives 恢复正确: {lives_match}")
    
    env.close()
    return ram_match and lives_match


def test_framestack_restore(env_id):
    """测试2: FrameStack 缓冲区恢复 - 恢复后观测是否一致"""
    print(f"\n{'='*60}")
    print(f"测试2: FrameStack 观测一致性 - {env_id}")
    print('='*60)
    
    env = make_test_env(env_id)
    env.reset(seed=42)
    
    # 执行几步获取有意义的帧历史
    for _ in range(50):
        action = env.action_space.sample()
        obs_after_step, _, _, _, _ = env.step(action)
    
    # 保存状态和观测
    saved_state = env.clone_state()
    saved_obs = np.array(obs_after_step, copy=True)
    
    # 获取 FrameStack 的帧缓冲区
    framestack_wrapper = env._framestack_wrapper
    saved_frames = [np.array(f, copy=True) for f in framestack_wrapper.frames]
    
    print(f"保存时观测 shape: {saved_obs.shape}")
    print(f"保存时观测 mean: {saved_obs.mean():.4f}")
    print(f"保存时 FrameStack 帧数: {len(saved_frames)}")
    
    # 继续执行更多步
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
    
    print(f"\n执行100步后观测 mean: {np.array(obs).mean():.4f}")
    
    # 恢复状态
    env.restore_state(saved_state)
    
    # 获取恢复后的帧缓冲区
    restored_frames = [np.array(f, copy=True) for f in framestack_wrapper.frames]
    
    # 从恢复后的帧缓冲区构建观测
    # FrameStack 的观测是 LazyFrames，需要转换为数组
    restored_obs = np.array(framestack_wrapper.frames)
    
    print(f"\n恢复后观测 mean: {restored_obs.mean():.4f}")
    
    # 验证每一帧
    frames_match = all(
        np.array_equal(saved_frames[i], restored_frames[i]) 
        for i in range(len(saved_frames))
    )
    obs_match = np.array_equal(saved_obs, restored_obs)
    
    print(f"\n✓ FrameStack 帧缓冲区恢复正确: {frames_match}")
    print(f"✓ 观测值一致: {obs_match}")
    
    if not obs_match:
        diff = np.abs(saved_obs.astype(float) - restored_obs.astype(float)).max()
        print(f"  观测最大差异: {diff}")
    
    env.close()
    return frames_match and obs_match


def test_deterministic_rollout(env_id):
    """测试3: 确定性轨迹 - 从相同状态执行相同动作序列，结果是否一致"""
    print(f"\n{'='*60}")
    print(f"测试3: 确定性轨迹 - {env_id}")
    print('='*60)
    
    env = make_test_env(env_id)
    env.reset(seed=42)
    
    # 执行几步到达某个状态
    for _ in range(50):
        action = env.action_space.sample()
        env.step(action)
    
    # 保存状态
    saved_state = env.clone_state()
    
    # 生成固定的动作序列
    np.random.seed(123)
    action_sequence = [env.action_space.sample() for _ in range(100)]
    
    # 第一次执行动作序列
    obs_list_1 = []
    reward_list_1 = []
    for action in action_sequence:
        obs, reward, terminated, truncated, _ = env.step(action)
        obs_list_1.append(np.array(obs, copy=True))
        reward_list_1.append(reward)
        if terminated or truncated:
            break
    
    print(f"第一次执行: {len(obs_list_1)} 步")
    print(f"  最终观测 mean: {obs_list_1[-1].mean():.4f}")
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
        obs_list_2.append(np.array(obs, copy=True))
        reward_list_2.append(reward)
        if terminated or truncated:
            break
    
    print(f"\n第二次执行: {len(obs_list_2)} 步")
    print(f"  最终观测 mean: {obs_list_2[-1].mean():.4f}")
    print(f"  累计奖励: {sum(reward_list_2):.4f}")
    
    # 验证
    length_match = len(obs_list_1) == len(obs_list_2)
    
    if length_match:
        obs_diffs = [np.abs(o1.astype(float) - o2.astype(float)).max() for o1, o2 in zip(obs_list_1, obs_list_2)]
        reward_diffs = [abs(r1 - r2) for r1, r2 in zip(reward_list_1, reward_list_2)]
        
        max_obs_diff = max(obs_diffs)
        max_reward_diff = max(reward_diffs)
        
        obs_match = max_obs_diff == 0
        reward_match = max_reward_diff == 0
        
        print(f"\n✓ 轨迹长度一致: {length_match}")
        print(f"✓ 观测序列一致: {obs_match} (最大差异: {max_obs_diff:.2e})")
        print(f"✓ 奖励序列一致: {reward_match} (最大差异: {max_reward_diff:.2e})")
        
        if not obs_match or not reward_match:
            # 找出第一个不一致的位置
            for i, (o1, o2, r1, r2) in enumerate(zip(obs_list_1, obs_list_2, reward_list_1, reward_list_2)):
                if np.abs(o1.astype(float) - o2.astype(float)).max() > 0 or abs(r1 - r2) > 0:
                    print(f"\n  ⚠ 第 {i} 步开始出现不一致:")
                    print(f"    obs diff: {np.abs(o1.astype(float) - o2.astype(float)).max():.2e}")
                    print(f"    reward diff: {abs(r1 - r2):.2e}")
                    break
    else:
        obs_match = False
        reward_match = False
        print(f"\n✗ 轨迹长度不一致: {len(obs_list_1)} vs {len(obs_list_2)}")
    
    env.close()
    return length_match and obs_match and reward_match


def test_episodiclife_restore(env_id):
    """测试4: EpisodicLifeEnv 状态恢复"""
    print(f"\n{'='*60}")
    print(f"测试4: EpisodicLifeEnv 状态恢复 - {env_id}")
    print('='*60)
    
    env = make_test_env(env_id)
    env.reset(seed=42)
    
    # 执行一些步骤
    for _ in range(50):
        action = env.action_space.sample()
        env.step(action)
    
    # 保存状态
    saved_state = env.clone_state()
    episodiclife_wrapper = env._episodiclife_wrapper
    
    if episodiclife_wrapper is None:
        print("⚠ 此环境没有 EpisodicLifeEnv wrapper，跳过测试")
        env.close()
        return True
    
    saved_lives = episodiclife_wrapper.lives
    saved_was_real_done = episodiclife_wrapper.was_real_done
    
    print(f"保存时 lives: {saved_lives}")
    print(f"保存时 was_real_done: {saved_was_real_done}")
    
    # 继续执行，可能会失去生命
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
    
    print(f"\n执行后 lives: {episodiclife_wrapper.lives}")
    print(f"执行后 was_real_done: {episodiclife_wrapper.was_real_done}")
    
    # 恢复
    env.restore_state(saved_state)
    
    restored_lives = episodiclife_wrapper.lives
    restored_was_real_done = episodiclife_wrapper.was_real_done
    
    print(f"\n恢复后 lives: {restored_lives}")
    print(f"恢复后 was_real_done: {restored_was_real_done}")
    
    # 验证
    lives_match = saved_lives == restored_lives
    was_real_done_match = saved_was_real_done == restored_was_real_done
    
    print(f"\n✓ lives 恢复正确: {lives_match}")
    print(f"✓ was_real_done 恢复正确: {was_real_done_match}")
    
    env.close()
    return lives_match and was_real_done_match


def test_branch_consistency(env_id):
    """测试5: 分支一致性 - 模拟树搜索中的分支场景"""
    print(f"\n{'='*60}")
    print(f"测试5: 分支一致性（模拟树搜索）- {env_id}")
    print('='*60)
    
    env = make_test_env(env_id)
    env.reset(seed=42)
    
    # 执行到某个状态作为分支点
    for _ in range(50):
        action = env.action_space.sample()
        obs_at_branch, _, _, _, _ = env.step(action)
    
    # 保存分支点状态
    branch_state = env.clone_state()
    branch_obs = np.array(obs_at_branch, copy=True)
    
    print(f"分支点观测 mean: {branch_obs.mean():.4f}")
    
    # 分支1: 执行动作序列 A
    np.random.seed(100)
    actions_A = [env.action_space.sample() for _ in range(50)]
    results_A = []
    for action in actions_A:
        obs, reward, term, trunc, _ = env.step(action)
        results_A.append((np.array(obs, copy=True), reward))
        if term or trunc:
            break
    
    print(f"\n分支1: 执行 {len(results_A)} 步, 累计奖励: {sum(r for _, r in results_A):.4f}")
    
    # 恢复到分支点
    env.restore_state(branch_state)
    
    # 获取恢复后的观测
    restored_obs = np.array(env._framestack_wrapper.frames)
    
    print(f"\n恢复后观测 mean: {restored_obs.mean():.4f}")
    print(f"分支点观测 mean: {branch_obs.mean():.4f}")
    obs_restored_match = np.array_equal(branch_obs, restored_obs)
    print(f"✓ 恢复后观测与分支点一致: {obs_restored_match}")
    
    # 分支2: 执行动作序列 B
    np.random.seed(200)
    actions_B = [env.action_space.sample() for _ in range(50)]
    results_B = []
    for action in actions_B:
        obs, reward, term, trunc, _ = env.step(action)
        results_B.append((np.array(obs, copy=True), reward))
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
        results_A_repeat.append((np.array(obs, copy=True), reward))
        if term or trunc:
            break
    
    # 验证分支1的重复执行结果一致
    if len(results_A) == len(results_A_repeat):
        obs_diffs = [np.abs(r1[0].astype(float) - r2[0].astype(float)).max() for r1, r2 in zip(results_A, results_A_repeat)]
        reward_diffs = [abs(r1[1] - r2[1]) for r1, r2 in zip(results_A, results_A_repeat)]
        branch_deterministic = max(obs_diffs) == 0 and max(reward_diffs) == 0
    else:
        branch_deterministic = False
    
    print(f"\n✓ 分支重复执行确定性: {branch_deterministic}")
    
    if not branch_deterministic and len(results_A) == len(results_A_repeat):
        print(f"  最大观测差异: {max(obs_diffs):.2e}")
        print(f"  最大奖励差异: {max(reward_diffs):.2e}")
    
    env.close()
    return obs_restored_match and branch_deterministic


def test_record_episode_stats(env_id):
    """测试6: RecordEpisodeStatistics 状态恢复"""
    print(f"\n{'='*60}")
    print(f"测试6: RecordEpisodeStatistics 状态恢复 - {env_id}")
    print('='*60)
    
    env = make_test_env(env_id)
    env.reset(seed=42)
    
    # 执行一些步骤累积奖励
    for _ in range(50):
        action = env.action_space.sample()
        env.step(action)
    
    # 保存状态
    saved_state = env.clone_state()
    record_stats_wrapper = env._record_stats_wrapper
    
    saved_return = float(record_stats_wrapper.episode_returns[0])
    saved_length = int(record_stats_wrapper.episode_lengths[0])
    
    print(f"保存时 episode_returns: {saved_return:.4f}")
    print(f"保存时 episode_lengths: {saved_length}")
    
    # 继续执行
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
    
    print(f"\n执行后 episode_returns: {float(record_stats_wrapper.episode_returns[0]):.4f}")
    print(f"执行后 episode_lengths: {int(record_stats_wrapper.episode_lengths[0])}")
    
    # 恢复
    env.restore_state(saved_state)
    
    restored_return = float(record_stats_wrapper.episode_returns[0])
    restored_length = int(record_stats_wrapper.episode_lengths[0])
    
    print(f"\n恢复后 episode_returns: {restored_return:.4f}")
    print(f"恢复后 episode_lengths: {restored_length}")
    
    # 验证
    return_match = abs(saved_return - restored_return) < 1e-6
    length_match = saved_length == restored_length
    
    print(f"\n✓ episode_returns 恢复正确: {return_match}")
    print(f"✓ episode_lengths 恢复正确: {length_match}")
    
    env.close()
    return return_match and length_match


def run_all_tests(env_id="BreakoutNoFrameskip-v4"):
    """运行所有测试"""
    print(f"\n{'#'*60}")
    print(f"Atari 环境恢复测试 - {env_id}")
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
        results['FrameStack观测一致性'] = test_framestack_restore(env_id)
    except Exception as e:
        print(f"测试2失败: {e}")
        import traceback
        traceback.print_exc()
        results['FrameStack观测一致性'] = False
    
    try:
        results['确定性轨迹'] = test_deterministic_rollout(env_id)
    except Exception as e:
        print(f"测试3失败: {e}")
        import traceback
        traceback.print_exc()
        results['确定性轨迹'] = False
    
    try:
        results['EpisodicLifeEnv恢复'] = test_episodiclife_restore(env_id)
    except Exception as e:
        print(f"测试4失败: {e}")
        import traceback
        traceback.print_exc()
        results['EpisodicLifeEnv恢复'] = False
    
    try:
        results['分支一致性'] = test_branch_consistency(env_id)
    except Exception as e:
        print(f"测试5失败: {e}")
        import traceback
        traceback.print_exc()
        results['分支一致性'] = False
    
    try:
        results['RecordEpisodeStatistics恢复'] = test_record_episode_stats(env_id)
    except Exception as e:
        print(f"测试6失败: {e}")
        import traceback
        traceback.print_exc()
        results['RecordEpisodeStatistics恢复'] = False
    
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
        print("🎉 所有测试通过！Atari 环境恢复功能正常。")
    else:
        print("⚠️  部分测试失败！环境恢复可能存在问题。")
    print('='*60)
    
    return all_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
                        help="Atari environment ID (default: BreakoutNoFrameskip-v4)")
    args = parser.parse_args()
    
    # 测试指定环境
    run_all_tests(args.env_id)
    
    # 如果是 Breakout，也测试其他几个常用的 Atari 环境
    if args.env_id == "BreakoutNoFrameskip-v4":
        print("\n\n" + "="*60)
        print("额外测试其他 Atari 环境...")
        print("="*60)
        
        other_envs = ["PongNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4"]
        for env_id in other_envs:
            try:
                run_all_tests(env_id)
            except Exception as e:
                print(f"\n{env_id} 测试失败: {e}")
