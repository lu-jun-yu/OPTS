# OPTS Generation 详细设计文档

> **代码实现状态**：已完成。代码位于 `LLM/trainer/main_opts_generation.py`。

## 1 功能概述

OPTS Generation 是一个基于 OPTS（On-policy Parallel Tree Search）算法的推理生成脚本，用于对给定的 prompt 数据集进行多轮树搜索式生成，并计算每个状态的优势值（advantage）。

### 1.1 核心功能

1. **多轮树搜索生成**：通过 OPTS 算法进行多轮采样，每轮基于 TUCT 选择下一轮扩展的状态
2. **优势值计算**：使用 TreeGAE 计算树结构上的优势值，用于评估生成质量
3. **分布式推理**：基于 Ray 框架进行分布式计算，支持多 GPU 并行生成
4. **结果持久化**：将生成的响应保存到 parquet 文件

### 1.2 与训练模式的区别

| 特性 | 训练模式 (OPTS_TTPO) | 生成模式 (OPTS Generation) |
|------|---------------------|---------------------------|
| 目标 | 策略优化 | 生成响应 |
| 梯度计算 | 需要 | 不需要 |
| Actor 更新 | 是 | 否 |
| Critic 更新 | 是 | 否 |
| 输出 | 更新后的模型 | 生成的响应文件 |


## 2 参数配置

### 2.1 配置文件结构

配置文件基于 Hydra 框架，主配置路径为 `verl.trainer.config`，配置名称为 `generation`。

### 2.2 核心参数

```yaml
# 模型配置
model:
  path: "path/to/model"  # 模型路径

# 数据配置
data:
  path: "path/to/dataset.parquet"  # 输入数据集路径
  output_path: "path/to/output.parquet"  # 输出文件路径
  prompt_key: "prompt"  # prompt 列名
  batch_size: 32  # 批次大小
  n_samples: 4  # 每轮每个 prompt 采样的响应数量
  n_rounds: 8  # 树搜索的轮数（对应 OPTS_TTPO 中的 g）
  trust_remote_code: false  # 是否信任远程代码

# 生成配置
rollout:
  temperature: 0.7  # 采样温度（为0时 n_samples 必须为1）
  prompt_length: 2048  # 最大 prompt 长度
  root_tuct: 0.5  # 根状态的 TUCT 常数值

# 算法配置
algorithm:
  gamma: 1.0  # 折扣因子
  lam: 0.95  # GAE lambda

# Critic 配置
critic:
  # Critic 模型相关配置

# Trainer 配置
trainer:
  n_gpus_per_node: 8  # 每节点 GPU 数量
  nnodes: 1  # 节点数量
  device: "cuda"  # 设备类型

# Ray 配置
ray_kwargs:
  ray_init:
    runtime_env: {}
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `data.n_samples` | int | 每轮每个状态采样的轨迹数，对应 OPTS 中的 n |
| `data.n_rounds` | int | 树搜索的轮数，对应 OPTS 中的 g |
| `rollout.root_tuct` | float | 根状态的 TUCT 常数值，用于与树中状态竞争 |
| `algorithm.gamma` | float | 折扣因子，用于 TreeGAE 计算 |
| `algorithm.lam` | float | GAE lambda，控制偏差-方差权衡 |


## 3 数据结构

### 3.1 输入数据格式

输入数据为 parquet 格式，需包含 prompt 列（列名由 `config.data.prompt_key` 指定）。

**示例：**
```python
# 输入 DataFrame 结构
pd.DataFrame({
    "prompt": [
        [{"role": "user", "content": "问题1"}],
        [{"role": "user", "content": "问题2"}],
        ...
    ]
})
```

### 3.2 输出数据格式

输出数据在输入基础上新增 `responses` 列，包含每个 prompt 的所有生成响应。

**示例：**
```python
# 输出 DataFrame 结构
pd.DataFrame({
    "prompt": [...],  # 原有列
    "responses": [
        ["响应1-1", "响应1-2", ...],  # prompt 1 的所有响应
        ["响应2-1", "响应2-2", ...],  # prompt 2 的所有响应
        ...
    ]
})
```

### 3.3 内部数据结构 (DataProto)

生成过程中使用的核心数据容器，与 OPTS_TTPO 一致：

#### 3.3.1 batch（张量数据）

| 键名 | 形状 | 说明 |
|------|------|------|
| input_ids | (bs, seq_len) | 输入 token ID |
| attention_mask | (bs, seq_len) | 注意力掩码 |
| position_ids | (bs, seq_len) | 位置 ID |
| responses | (bs, response_len) | 生成的响应 |
| response_mask | (bs, response_len) | 响应掩码（EOS后为0） |
| values | (bs, response_len) | Critic 预测的状态价值 |
| token_level_rewards | (bs, response_len) | token 级别奖励 |
| state_branches | (bs, response_len) | 每个状态的分支数 |
| advantages | (bs, response_len) | 优势函数值 |
| returns | (bs, response_len) | 回报 |

#### 3.3.2 non_tensor_batch（非张量数据）

| 键名 | 类型 | 说明 |
|------|------|------|
| raw_prompt_len | np.ndarray | 原始 prompt 的长度 |
| uid | np.ndarray | prompt 的唯一标识符 |
| rid | np.ndarray | response 的唯一标识符 |
| pid | np.ndarray | 父轨迹的 rid |
| cid | np.ndarray | 子轨迹映射（OrderedDict） |
| branch_pos | np.ndarray | 在父轨迹中的分支位置 |
| tid | np.ndarray | 树标识符 |
| new_sample_indices | np.ndarray | 新样本的索引列表 |


## 4 执行流程

### 4.1 整体流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        run_generation                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  1. 初始化 Ray 集群                                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  2. 启动 main_task（远程任务）                             │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          main_task                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  1. 加载模型和 tokenizer                                   │  │
│  │  2. 读取数据集                                             │  │
│  │  3. 创建分布式 worker 资源池                               │  │
│  │  4. 初始化 Actor 和 Critic 模型                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  for batch in batches:                                     │  │
│  │    ┌─────────────────────────────────────────────────────┐│  │
│  │    │  for round_idx in range(n_rounds):                  ││  │
│  │    │    a. 重复样本 n_samples 次                          ││  │
│  │    │    b. 生成序列 (wg.generate_sequences)               ││  │
│  │    │    c. 计算 values (critic_wg.compute_values)         ││  │
│  │    │    d. 计算 token_level_rewards                       ││  │
│  │    │    e. 设置树结构信息 (set_opts_ttpo_info)            ││  │
│  │    │    f. 合并到全局 batch                               ││  │
│  │    │    g. 计算 TreeGAE 优势值                            ││  │
│  │    │    h. 提取响应文本                                   ││  │
│  │    │    i. 选择下一轮状态 (select_next_states)            ││  │
│  │    │    j. 准备下一轮输入 (prepare_next_round_input)      ││  │
│  │    └─────────────────────────────────────────────────────┘│  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  保存结果到 parquet 文件                                   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 详细流程说明

#### 4.2.1 初始化阶段

1. **Ray 初始化**：设置环境变量，初始化 Ray 集群
2. **模型加载**：加载 tokenizer，设置 padding 配置
3. **数据读取**：从 parquet 文件读取 prompt 数据
4. **Worker 创建**：
   - 创建 `RayResourcePool` 资源池
   - 初始化 `ActorRolloutRefWorker` 和 `CriticWorker`
   - 使用 colocated worker 模式实现模型共置

#### 4.2.2 生成循环

对于每个 batch：

```python
for round_idx in range(n_rounds):
    # 1. 样本重复
    batch_repeated = batch.repeat(repeat_times=n_samples, interleave=True)

    # 2. 序列生成
    output = wg.generate_sequences(batch_repeated)

    # 3. Critic 计算
    values_output = critic_wg.compute_values(output)

    # 4. 奖励计算（使用最后有效位置的 value 作为奖励）
    token_level_rewards[batch_indices, last_pos] = values[batch_indices, last_pos]

    # 5. 树结构信息设置
    new_sample_indices = set_opts_ttpo_info(output, global_batch, next_states, round_idx)

    # 6. 合并到全局 batch
    global_batch = merge_batches(global_batch, output)

    # 7. TreeGAE 优势计算
    advantages, returns = compute_treegae_advantage_return(...)

    # 8. 提取响应文本
    for i in range(batch_size):
        response_str = tokenizer.decode(valid_response_ids)
        output_lst[idx].append(response_str)

    # 9. 准备下一轮（非最后一轮）
    if round_idx < n_rounds - 1:
        next_states = select_next_states(...)
        batch = prepare_next_round_input(...)
```

#### 4.2.3 奖励计算逻辑

生成模式使用 Critic 的 value 预测作为奖励信号：

```python
# 获取每个响应的最后有效位置
last_pos = (response_mask.sum(dim=1) - 1).clamp(min=0)

# 将该位置的 value 作为 token 级别奖励
token_level_rewards[batch_indices, last_pos] = values[batch_indices, last_pos]
```

这种设计使得：
- 奖励集中在响应的最后一个 token
- 利用 Critic 模型评估生成质量
- 无需外部奖励模型


## 5 关键函数

### 5.1 函数调用关系

```
main()
└── run_generation()
    └── main_task.remote()
        ├── wg.generate_sequences()       # 序列生成
        ├── critic_wg.compute_values()    # 价值计算
        ├── set_opts_ttpo_info()          # 树结构设置
        ├── merge_batches()               # batch 合并
        ├── compute_treegae_advantage_return()  # 优势计算
        ├── select_next_states()          # 状态选择
        └── prepare_next_round_input()    # 输入准备
```

### 5.2 核心函数列表

| 函数名 | 来源 | 说明 |
|--------|------|------|
| `run_generation` | main_opts_generation.py | 入口函数，初始化 Ray 并启动任务 |
| `main_task` | main_opts_generation.py | 主任务函数，包含完整生成流程 |
| `set_opts_ttpo_info` | ray_trainer.py | 设置树结构信息 |
| `merge_batches` | ray_trainer.py | 合并局部 batch 到全局 batch |
| `select_next_states` | ray_trainer.py | TUCT 状态选择 |
| `prepare_next_round_input` | ray_trainer.py | 构建下一轮输入 |
| `compute_treegae_advantage_return` | core_algos.py | TreeGAE 优势计算 |
| `compute_response_mask` | ray_trainer.py | 计算响应掩码 |

### 5.3 Worker 类型

| Worker 类 | 功能 |
|-----------|------|
| ActorRolloutRefWorker | 序列生成 |
| CriticWorker | 状态价值估计 |


## 6 使用示例

### 6.1 命令行调用

```bash
python -m trainer.main_opts_generation \
    model.path=/path/to/model \
    data.path=/path/to/dataset.parquet \
    data.output_path=/path/to/output.parquet \
    data.n_samples=4 \
    data.n_rounds=8 \
    rollout.temperature=0.7 \
    rollout.root_tuct=0.5
```

### 6.2 配置文件调用

```bash
python -m trainer.main_opts_generation --config-name=my_generation_config
```

### 6.3 脚本调用

```python
from omegaconf import OmegaConf
from trainer.main_opts_generation import run_generation

config = OmegaConf.create({
    "model": {"path": "/path/to/model"},
    "data": {
        "path": "/path/to/dataset.parquet",
        "output_path": "/path/to/output.parquet",
        "prompt_key": "prompt",
        "batch_size": 32,
        "n_samples": 4,
        "n_rounds": 8,
    },
    "rollout": {
        "temperature": 0.7,
        "prompt_length": 2048,
        "root_tuct": 0.5,
    },
    "algorithm": {
        "gamma": 1.0,
        "lam": 0.95,
    },
    # ... 其他配置
})

run_generation(config)
```


## 7 注意事项

### 7.1 温度设置

- 当 `temperature=0.0` 时，`n_samples` 必须为 1（贪婪解码无随机性）
- 建议使用 `temperature > 0` 以获得多样化的生成结果

### 7.2 内存管理

- `n_samples * n_rounds` 决定了每个 prompt 的总生成数量
- 较大的 `batch_size` 和 `n_samples` 会增加 GPU 内存消耗
- 建议根据 GPU 内存调整这些参数

### 7.3 依赖模块

本模块依赖以下 OPTS_TTPO 组件：

```python
from .opts_ttpo.core_algos import compute_treegae_advantage_return
from .opts_ttpo.ray_trainer import (
    compute_response_mask,
    merge_batches,
    prepare_next_round_input,
    set_opts_ttpo_info,
    select_next_states,
)
```

确保 `opts_ttpo` 模块正确安装和配置。


## 8 与 OPTS_TTPO 训练的对比

| 阶段 | OPTS_TTPO 训练 | OPTS Generation |
|------|---------------|-----------------|
| 数据准备 | 从 DataLoader 获取 | 从 parquet 文件读取 |
| 序列生成 | 相同（wg.generate_sequences） | 相同 |
| 奖励计算 | 外部奖励模型 | Critic value 作为奖励 |
| 优势估计 | TreeGAE | TreeGAE |
| 状态选择 | TUCT 选择 | TUCT 选择 |
| 模型更新 | Actor + Critic 更新 | 无更新 |
| 输出 | 更新后的模型权重 | 生成的响应文件 |

生成模式复用了 OPTS_TTPO 的核心搜索算法，但跳过了模型更新步骤，专注于高质量响应的生成和评估。
