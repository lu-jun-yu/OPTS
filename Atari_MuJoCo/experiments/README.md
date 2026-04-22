# OPTS_TTPO 方差缩减验证实验

## 实验目的

策略梯度方法的核心瓶颈在于梯度估计的高方差。OPTS_TTPO 通过树搜索机制选择性地探索高回报轨迹，理论上能够降低策略梯度的方差。本实验从两个角度验证这一点。

## 实验设置

- **任务**: HalfCheetah-v4, Walker2d-v4, Hopper-v4, Ant-v4, Humanoid-v4
- **预训练模型**: 每个任务使用 OPTS_TTPO 训练 1M step (seed=1)，保存完整 checkpoint（含 normalization 统计量）
- **验证 seed**: 1-5，使用相同的预训练模型，仅改变数据收集时的随机种子

## 验证1: 正轨迹的策略梯度方差 < 负轨迹的策略梯度方差

**命题**: 挖掘正轨迹（高回报轨迹）能降低策略梯度方差。

**方法**:

1. 加载预训练模型，按 PPO 方式收集 `num_steps` 个 step
2. 对全部 step 计算标准 GAE advantage，按 step-level advantage 排序，取前 α 比例的 step 作为正样本，后 α 比例的 step 作为负样本（脚本 `run_verify_pg_variance.sh` 使用 `alpha=0.42`）
3. 分别计算正/负样本的策略梯度（全部 step 平均 pg_loss，一次反向传播）
4. 用全部采样 step 计算期望策略梯度 g_expected
5. 计算方差：对每个参数 i，variance_i = (g_sub_i - g_expected_i)^2，对所有参数取均值

**预期结果**: 正轨迹的方差 < 负轨迹的方差。即用正轨迹估计的策略梯度更接近期望策略梯度。

**脚本**:
```bash
# 运行
bash scripts/run_verify_pg_variance.sh

# 可视化
python visual/plot_pg_variance.py results/variance/verify1 visual
```

## 验证2: OPTS 加速策略梯度方差的收敛

**命题**: 相比于 PPO 的纯 rollout，OPTS 的策略梯度方差随 batch_size 增大能更快地逼近真实策略梯度。

**方法**:

对每个 batch_size B ∈ [64, 128, 256, 512, 1024, 2048, 4096]：

- **PPO 基线**: 取 PPO rollout 的前 B 个 step，计算策略梯度 g_B，衡量 (g_B - g_expected)^2 的参数均值
- **OPTS_TTPO**: 跑 `num_steps` 个 tree-search step，取前 B 个 step，使用 branch_weight 做 IPW 加权计算策略梯度 g_B，衡量 (g_B - g_expected)^2 的参数均值

g_expected 统一使用 PPO `total_steps` 全量梯度作为基准。

**为什么 OPTS 需要 IPW 加权**: OPTS 的树搜索会从同一状态分支出多条轨迹，导致采样分布偏离 on-policy 分布。branch_weight 记录了每个 step 所在路径的分支数量，除以 branch_weight 等价于逆概率加权（IPW），将偏离的采样分布校正回 on-policy 分布，确保梯度估计无偏。加权公式与训练代码一致：

```
pg_loss = (loss_per_sample / weights).sum() / (1 / weights).sum()
```

**预期结果**: 在相同 batch_size 下，OPTS 的方差显著低于 PPO。即 OPTS 用更少的 step 就能达到 PPO 需要更多 step 才能达到的梯度估计精度。

**脚本**:
```bash
# 运行
bash scripts/run_verify_scaling_variance.sh

# 可视化
python visual/plot_scaling_variance.py results/variance/verify2 visual
```

## 文件结构

```
experiments/
  verify_pg_variance.py       # 验证1: 正负轨迹方差对比
  verify_scaling_variance.py   # 验证2: PPO vs OPTS 方差收敛速度

scripts/
  run_train_save_model.sh      # 训练并保存模型
  run_verify_pg_variance.sh    # 运行验证1（5 task x 5 seed）
  run_verify_scaling_variance.sh  # 运行验证2（5 task x 5 seed）

visual/
  plot_pg_variance.py          # 验证1 五宫格图（x=seed, y=variance, 正/负两条曲线）
  plot_scaling_variance.py     # 验证2 五宫格图（x=batch_size, y=variance, PPO/OPTS mean±std）
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-steps` | `1000000`（验证1）/ `100000`（验证2 默认） | 单次采样的 step 数；`run_verify_scaling_variance.sh` 传入 `10000` |
| `--total-steps` | `1000000` | 验证2 中用于计算 `g_expected` 的 PPO 全量 step 数 |
| `--alpha` | 0.3 | 正/负样本的 step 比例（验证1）；批量脚本传入 `0.42` |
| `--max-search-per-tree` | 4 | OPTS 每棵树的最大搜索次数 |
| `--c` | 1.0 | OPTS OTRC 探索系数 |
| `--gamma` | 0.99 | 折扣因子 |
| `--gae-lambda` | 0.95 | GAE lambda |
