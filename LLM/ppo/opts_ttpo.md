# OPTS_TTPO
该算法是由Junyu Lu设计的强化学习新范式：搜索 & 基于策略梯度的策略优化方法。

由两部分构成：
1. 同策略并行树搜索 On-policy Parallel Search (OPTS)
    - 与MCTS较为不同
2. 树轨迹策略优化 Tree Trajectory Policy Optimization (TTPO)
    - 由PPO扩展至树轨迹而来，其中包括优势估计扩展至树轨迹、策略梯度扩展至树轨迹


## 1 参数层面

OPTS_TTPO 在 PPO 的基础上，多了两个参数：
- actor_rollout_ref.rollout.g: 循环采样的次数
- algorithm.c: 一个决定探索水平的常数


## 2 数据结构层面

```python
class DataProto:
    """
    verl的核心数据容器

    属性:
        batch: TensorDict - 包含所有张量数据
            常见键:
            - input_ids: (bs, seq_len) - 输入token ID
            - attention_mask: (bs, seq_len) - 注意力掩码
            - position_ids: (bs, seq_len) - 位置ID
            - responses: (bs, response_len) - 生成的响应
            - prompts: (bs, prompt_len) - 提示
            - response_mask: (bs, response_len) - 响应掩码
            - old_log_probs: (bs, response_len) - 旧策略log概率
            - ref_log_prob: (bs, response_len) - 参考策略log概率
            - values: (bs, response_len) - Critic预测值
            - token_level_scores: (bs, response_len) - token级别分数
            - token_level_rewards: (bs, response_len) - token级别奖励
            - advantages: (bs, response_len) - 优势
            - returns: (bs, response_len) - 回报
            - sum_of_advantages: (bs, response_len) - 所有动作的优势和，对应位置前移一位 （新增键）
            - gamma_t: (bs, response_len) - gamma累乘 （新增键）
            - lam_t: (bs, response_len) - lambda累乘 （新增键）
            - trajectory_reward: (bs, response_len) - 当前已经获得的累计折扣奖励 （新增键）
            - gve: (bs, response_len) - 广义状态价值估计：values + lam * sum_of_advantages / state_branches （新增键）
            - expected_trajectory_reward: (bs, response_len) - 期望轨迹累计折扣奖励：trajectory_reward + gamma_t * gve （新增键）
            - state_branches: (bs, response_len) - 每个状态的分支数 （新增键）
            - subtree_branches: (bs, response_len) - 每个状态对应的子树的分支数之和 （新增键）
            - partree_branches: (bs, response_len) - 每个状态对应的父树的分支数之和 （新增键）
            - tuct: (bs, response_len) - 每个状态的tuct值：expected_trajectory_reward * lam_t + c * \sqrt{\frac{\log{partree_branches}}{subtree_branches}} （新增键）
            - branch_weight_factor: (bs, response_len) - 每个状态-动作对的策略梯度权重因子 （新增键）

        non_tensor_batch: dict - 非张量数据
            常见键:
            - data_source: 数据来源标识
            - reward_model: 奖励模型配置
            - uid: 唯一标识符
            - rid: 每条response的唯一标识 （新增键）
            - pid: 每条response的父轨迹的rid （新增键）
            - cid: 有序字典：{token所在的位置: [从该token出发产生的分支轨迹的rid列表]} （新增键）
            - extra_info: 额外信息

        meta_info: dict - 元信息
            常见键:
            - temperature: 采样温度
            - eos_token_id: EOS token ID
            - pad_token_id: PAD token ID
            - global_steps: 当前全局步数

    常用方法:
        from_single_dict(dict): 从字典创建
        from_dict(tensors): 从张量字典创建
        repeat(n, interleave): 重复数据
        union(other): 合并另一个DataProto
        pop(batch_keys, non_tensor_batch_keys): 弹出指定键
        reorder(indices): 重新排序
    """
```


## 3 训练主循环的改变——主要是fit()

### 3.1 总体变化

当 epoch==0 时，需要为critic进行初始化训练，这一轮循环与PPO训练没有区别，但使用更低的 actor model 损失权重或者学习率，以及更高的 critic model 损失权重或者学习率

当 epoch>0 时，才开始 OPTS_TTPO 训练

### 3.2 OPTS_TTPO 训练流程

for epoch in ...:
    for batch in ...:
        1. 数据准备：batch区分全局的batch和局部的batch
            - 全局的batch：保存每一轮的局部batch，同时用于生成下一轮局部batch的输入；最后更新策略和价值模型使用的是全局的batch
            - 局部的batch：每一轮采样时的输入都动态变化，由全局的batch截取部分"prompt+部分已采样结果"构建成新的prompt，然后重复self.config.actor_rollout_ref.rollout.n遍得到。初始时是来自原始的batch，第一次采样后的局部batch的prompt才开始由"prompt+部分已采样结果"构建
        2. 采样：for i in range(self.config.actor_rollout_ref.rollout.g):
            a. 前向：
                - 重复样本，生成rid
                - 在父轨迹上插入新的cid键值；在当前轨迹上设置pid
                - 生成序列
                - 计算奖励
                - 新函数1（根据保存的rid找到本轮生成的新样本，在全局batch上操作）：
                    - 前向计算gamma_t、lam_t、trajectory_reward
                    - 初始化state_branches为全1，初始化branch_weight_factor
                    - 前向计算partree_branches
                - 计算旧策略log概率
                - 合并生成结果
                - 新函数2：反向计算subtree_branches
            b. 反向：
                - 计算优势 & 保存sum_of_advantages：若存在父轨迹，则继续传播到父轨迹上
            c. 选择：相同的prompt中 TUCT 最大的状态，构建local_batch
                - 计算gve和expected_trajectory_reward
                - 计算tuct
                - 在全局batch上argmax最大tuct的状态
        3. 更新：
            - 新函数3：前向更新branch_weight_factor
            - 更新critic：Critic Loss计算与先前保持一致
            - 更新actor：Policy Loss计算：(掩码不变)
                - 在原来的基础上，pg_losses除以branch_weight_factor
                - agg_loss增加"weighted-token-mean"的分支，当使用opts_ttpo算法时，默认是"weighted-token-mean"：需要先对branch_weight_factor的掩码结果进行求和，然后使用pg_losses除以求和结果