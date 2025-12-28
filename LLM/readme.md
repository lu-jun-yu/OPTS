# LLM 模块

本模块包含基于 [verl](https://github.com/volcengine/verl) 框架实现的 LLM 强化学习训练代码，包括自研的 OPTS_TTPO 算法。

## 目录结构

```
LLM/
├── trainer/                    # 训练代码
│   ├── opts_ttpo/              # OPTS_TTPO 算法实现
│   │   ├── main_opts_ttpo.py   # 入口文件
│   │   ├── ray_trainer.py      # RayOPTSTTPOTrainer 类
│   │   ├── core_algos.py       # 核心算法函数
│   │   ├── opts_ttpo_v0.md     # 需求文档
│   │   └── opts_ttpo_v1.md     # 详细设计文档
│   └── ...
├── verl/                       # verl 框架（submodule，指向 fork 仓库）
│   ├── verl/                   # verl 核心代码
│   ├── examples/               # 示例代码
│   ├── recipe/                 # 各类算法配方
│   └── ...
└── verl_guide.md               # verl submodule 管理指南
```

## OPTS_TTPO 算法

OPTS_TTPO（On-policy Parallel Tree Search + Tree Trajectory Policy Optimization）是一种将树搜索与策略梯度优化相结合的强化学习新范式。

### 核心思想

1. **OPTS（同策略并行树搜索）**
   - 采样实例树的形式，每轮并行采样 `n` 条轨迹
   - 通过 `g` 轮循环构建树结构
   - 使用 TUCT（Trajectory-level Upper Confidence bound for Trees）选择下一轮扩展的最优状态
   - 支持回溯到早期状态进行重新扩展

2. **TTPO（树轨迹策略优化）**
   - 将 PPO 的优势估计扩展至树轨迹（TreeGAE）
   - 将策略梯度扩展至树轨迹，引入分支权重因子进行梯度校正
   - 保证在树结构上的策略梯度估计是无偏的

### 配置参数

```yaml
actor_rollout_ref:
  rollout:
    n: 8          # 每轮采样的轨迹数
    g: 4          # 循环采样的轮数
    search: opts  # 搜索算法："opts" 启用 OPTS_TTPO

algorithm:
  adv_estimator: treegae  # 使用 TreeGAE 优势估计
  c: 1.0                  # TUCT 探索常数
  gamma: 1.0              # 折扣因子
  lam: 0.95               # GAE lambda
```

### 文档

详细设计文档请参阅 [opts_ttpo_v1.md](trainer/opts_ttpo/opts_ttpo_v1.md)。

## verl 框架

本模块使用 verl 作为底层框架。verl 是一个灵活、高效的分布式强化学习框架，支持多种 RL 算法（PPO、GRPO、RLOO 等）。

### Submodule 管理

verl 以 git submodule 形式管理，指向 fork 仓库以便进行自定义修改。

**常用操作：**

```bash
# 克隆项目时初始化 submodule
git clone --recurse-submodules <repo_url>

# 或者克隆后手动初始化
git submodule init
git submodule update

# 拉取 verl 更新
cd LLM/verl
git checkout main
git pull origin main

# 拉取原仓库更新
git fetch upstream
git merge upstream/main
```

详细操作指南请参阅 [verl_guide.md](verl_guide.md)。

## 运行训练

```bash
# 进入训练目录
cd LLM/trainer/opts_ttpo

# 运行 OPTS_TTPO 训练
python main_opts_ttpo.py --config <config_file>
```

## 依赖

- Python >= 3.10
- PyTorch >= 2.0
- verl 框架及其依赖
- flash-attn（推荐）

## 参考

- verl 框架：https://github.com/volcengine/verl
- OPTS_TTPO 算法设计：[opts_ttpo_v1.md](trainer/opts_ttpo/opts_ttpo_v1.md)
