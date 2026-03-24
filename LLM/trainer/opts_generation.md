<!-- Copyright 2025 Junyu Lu (Julian Lou). All rights reserved. -->

# OPTS 推理时扩展（Inference-Time Scaling）

> 代码位于 `LLM/trainer/main_opts_generation.py`，配置文件为 `LLM/verl/verl/trainer/config/generation.yaml`。

## 1 目标

在推理阶段复用 OPTS 训练的树搜索算法，通过多轮 TUCT 引导的树结构采样生成高质量响应。与 pass@k 保持相同的推理成本，但利用树搜索的优势引导来分配计算资源，从而在同等预算下获得更好的结果。

支持两种奖励引导模式：
- **reward-guided**（`reward_mode="reward"`）：使用实际奖励函数（如数学正确性评分）作为搜索引导信号，流程与训练完全一致。
- **value-guided**（`reward_mode="value"`）：使用 Critic 模型在最后有效位置的 value 预测作为奖励，无需外部奖励模型。


## 2 推理预算

总推理步数 = `ceil(dataset_size * n_samples / batch_size)`，与 pass@k（每个 prompt 采样 k 次）的总推理量完全一致。每一步内部进行 `n_samples` 轮树结构采样，每轮生成 `batch_size` 条响应。


## 3 参数配置

### 3.1 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data.n_samples` | 5 | 每步的采样轮数，同时决定总推理预算 |
| `data.reward_mode` | `"value"` | 奖励模式：`"reward"`（奖励引导）或 `"value"`（价值引导） |
| `data.batch_size` | 128 | 每轮生成的批次大小 |
| `rollout.c` | 1.0 | TUCT 探索系数，控制搜索的探索-利用平衡 |
| `rollout.max_search_per_tree` | 1 | 每步每棵树的最大搜索次数 |
| `algorithm.gamma` | 1.0 | 折扣因子 |
| `algorithm.lam` | 1.0 | GAE lambda |

### 3.2 其他参数

| 参数 | 说明 |
|------|------|
| `model.path` | Actor/Critic 模型路径 |
| `data.path` | 输入数据集路径（parquet 格式） |
| `data.output_path` | 输出文件路径 |
| `data.prompt_key` | 数据集中 prompt 列的列名 |
| `rollout.temperature` | 采样温度（为 0 时 n_samples 必须为 1） |
| `rollout.prompt_length` | 最大 prompt 长度 |
| `rollout.response_length` | 最大响应长度 |
| `trainer.n_gpus_per_node` | 每节点 GPU 数量 |
| `trainer.nnodes` | 节点数量 |


## 4 执行流程

### 4.1 初始化

1. 初始化 Ray 集群，加载模型和 tokenizer
2. 读取 parquet 数据集，构建 `InferencePromptBuffer`
3. 创建分布式 Actor（负责序列生成）和 Critic（负责价值估计）worker
4. 若为 reward-guided 模式，初始化 `NaiveRewardManager`（硬编码使用 `utils/reward_fn.py` 中的 `compute_score`）

### 4.2 主循环

```
for step in range(total_steps):                         # 总步数 = ceil(数据集大小 × n_samples / batch_size)
    global_batch = None                                  # 当前步的全局树结构
    next_states = {}                                     # TUCT 选中的续写分支点

    for round in range(n_samples):                       # 每步 n_samples 轮采样
        ┌─ 构建本轮输入 ──────────────────────────────┐
        │  round 0: 从 PromptBuffer 取 batch_size 条    │
        │  round > 0: 续写部分(next_states) + 新 prompt │
        └──────────────────────────────────────────────┘
                            ↓
        ┌─ 策略采样 ─────────────────────────────────────┐
        │  wg.generate_sequences(batch)                   │
        └─────────────────────────────────────────────────┘
                            ↓
        ┌─ 价值估计 ─────────────────────────────────────┐
        │  critic_wg.compute_values(output)               │
        └─────────────────────────────────────────────────┘
                            ↓
        ┌─ 奖励计算 ─────────────────────────────────────┐
        │  reward 模式: 解码响应 → 奖励函数打分           │
        │  value 模式:  取最后位置的 value 作为奖励        │
        └─────────────────────────────────────────────────┘
                            ↓
        ┌─ 树结构记账 ───────────────────────────────────┐
        │  set_opts_ttpo_info → 合并至 global_batch       │
        │  compute_treegae_advantage_return               │
        └─────────────────────────────────────────────────┘
                            ↓
        ┌─ 收集响应 ─────────────────────────────────────┐
        │  解码响应文本，按数据集行号记录 sample_index     │
        └─────────────────────────────────────────────────┘
                            ↓
        ┌─ TUCT 搜索选择（非最后一轮） ──────────────────┐
        │  select_next_states → selected_to_branch_points │
        │  确定下一轮的续写分支点                          │
        └─────────────────────────────────────────────────┘

    # 步结束：计算 branch_weight 和聚合回报，更新 return_threshold
```

### 4.3 InferencePromptBuffer

自定义的数据缓冲区，按顺序循环遍历数据集中的所有 prompt。每次 `draw(n)` 取出 n 条 prompt 并为每条分配一个全新的 UUID（同一 prompt 在不同 draw 中获得不同的 uid）。通过 `uid_to_idx` 字典记录每个 uid 对应的原始数据集行号，用于最终的结果汇聚。

当 cursor 回绕（所有 prompt 被取过一次）时，设置 `just_cycled` 标志。主循环检测到此标志后，用已有的聚合回报计算 `prev_mean_return`，正式作为 TUCT 搜索的 `return_threshold`。

### 4.4 return_threshold 的生命周期

1. 初始为 `None`：TUCT 的 `select_next_states` 不会选中任何续写状态，每轮都是全新 prompt，相当于普通的 pass@k 采样
2. PromptBuffer 首次循环完毕后：使用所有已有树的加权聚合回报的均值作为 threshold
3. 之后：持续更新，threshold 以上的树不再被搜索，资源分配给回报较低的树

### 4.5 奖励计算

**value-guided 模式**：将 Critic 在响应最后有效位置的 value 预测作为唯一的 token 级别奖励，其余位置为零。优点是无需外部奖励模型，缺点是完全依赖 Critic 的估计质量。

**reward-guided 模式**：解码完整响应文本，调用 `NaiveRewardManager` 获取实际奖励分数（如数学正确性），放置在对应的 token 位置上。流程与 OPTS TTPO 训练时一致。

### 4.6 raw_prompt_len 与完整响应解码

`raw_prompt_len` 始终是**原始 prompt 的长度**，即使在续写分支中也不会被 `prepare_next_round_input` 修改。续写 prompt 的 `input_ids` 布局为 `[pad | 原始prompt + 前缀response | 新response]`，`set_full_response_str` 从 `pad_len + raw_prompt_len`（原始 prompt 结束位置）开始截取 `response_length` 长度，得到的是「前缀 response + 新生成片段」，即从原始 prompt 之后的完整响应。这保证了无论是新 prompt 还是续写分支，奖励函数都能拿到完整的响应文本进行评分。


## 5 输出格式

输出为 parquet 文件，在原始数据集基础上新增两列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `responses` | `List[str]` | 该 prompt 的所有生成响应 |
| `sample_indices` | `List[int]` | 每条响应的采样轮次编号（1-based，按数据集行号递增） |

### 5.1 推理时扩展分析

`sample_indices` 的设计支持单次大规模实验即可得出不同 n_samples 预算下的结果：

设 n_samples=128 运行一次，然后：
- 取 `sample_indices <= 4` 的响应 → 相当于 n_samples=4 的结果
- 取 `sample_indices <= 16` 的响应 → 相当于 n_samples=16 的结果
- 取 `sample_indices <= 64` 的响应 → 相当于 n_samples=64 的结果

以此类推，可直接绘制推理时扩展曲线，对比 OPTS 树搜索与普通 pass@k 在不同预算下的性能差异。


## 6 输入数据格式

输入 parquet 需包含以下列：

| 列名 | 必需 | 说明 |
|------|------|------|
| prompt 列（由 `prompt_key` 指定） | 是 | chat 格式的 prompt，如 `[{"role": "user", "content": "..."}]` |
| `data_source` | 否 | 数据来源标识，reward 模式下用于奖励函数调度 |
| `reward_model` | 否 | 奖励模型标识 |


## 7 与训练模式的区别

| 特性 | OPTS TTPO 训练 | OPTS 推理时扩展 |
|------|---------------|-----------------|
| 目标 | 策略优化 | 生成高质量响应 |
| 梯度/模型更新 | Actor + Critic 更新 | 无 |
| 数据来源 | DataLoader 按 epoch 遍历 | InferencePromptBuffer 循环遍历 |
| 奖励来源 | 外部奖励函数 | value（默认）或 reward 两种模式 |
| 预算控制 | 由训练步数决定 | 与 pass@k 相同：dataset_size × n_samples / batch_size |
| 输出 | 更新后的模型权重 | parquet 文件（响应 + sample_index） |


## 8 使用示例

```bash
python3 -m verl.trainer.main_opts_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=data/test.parquet \
    data.prompt_key=prompt \
    data.batch_size=512 \
    data.n_samples=32 \
    data.reward_mode=value \
    data.output_path=output/test_opts.parquet \
    model.path=models/Qwen3-1.7B \
    critic.model.path=models/Qwen3-1.7B \
    rollout.c=1.0 \
    rollout.max_search_per_tree=1 \
    rollout.temperature=1.0 \
    rollout.prompt_length=1024 \
    rollout.response_length=1024 \
    algorithm.lam=0.995
```


## 9 依赖模块

复用自 `opts_ttpo` 的核心组件：

```
opts_ttpo/core_algos.py
  ├── compute_treegae_advantage_return    TreeGAE 优势计算
  └── compute_branch_weight               树结构分支权重

opts_ttpo/ray_trainer.py
  ├── set_opts_ttpo_info                  树结构信息初始化（uid/rid/pid/cid/branch_pos）
  ├── compute_episodic_returns            计算 episode 回报
  ├── compute_aggregated_returns          按 uid 加权聚合回报
  ├── select_next_states                  TUCT 搜索选择
  ├── selected_to_branch_points           将选中状态转为分支点
  ├── prepare_next_round_input            构建续写输入
  ├── merge_batches                       DataProto 批次合并
  └── compute_response_mask               响应掩码计算

verl/trainer/ppo/reward.py
  └── compute_reward                      奖励计算（reward 模式）
```
