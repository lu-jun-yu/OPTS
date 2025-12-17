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
            - advantages_mean: (bs, response_len) - 所有动作的优势均值 （新增键）
            - gamma_t: (bs, response_len) - gamma累乘 （新增键）
            - lam_t: (bs, response_len) - lambda累乘 （新增键）
            - trajectory_reward: (bs, response_len) - 当前已经获得的累计折扣奖励 （新增键）
            - gve: (bs, response_len - 1) - 广义状态价值估计：values[1:] + lam * advantages_mean[1:] （新增键）
            - expected_trajectory_reward: (bs, response_len - 1) - 期望轨迹累计折扣奖励：trajectory_reward + gamma_t * gve （新增键）
            - state_branches: (bs, response_len) - 每个状态的分支数 （新增键）
            - subtree_branches: (bs, response_len) - 每个状态对应的下游子树的分支总数 （新增键）
            - tuct: (bs, response_len - 1) - 每个状态的tuct值：expected_trajectory_reward * lam_t + c * \frac{\sqrt{\log{partree_branches}}}{subtree_branches} （新增键）
            - branch_weight_factor: (bs, response_len) - 每个状态-动作对的策略梯度权重因子 = 当前状态的祖先轨迹的分支数的累乘 （新增键）

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

### 3.1 OPTS_TTPO 训练流程伪代码

for epoch in ...:

    for batch in ...:

        1. 数据准备：batch区分全局batch和局部batch
            - 全局batch：保存每一轮的局部batch，同时用于生成下一轮的局部batch；最后更新策略和价值模型使用的是全局batch
            - 局部batch：每一轮采样都动态变化，由全局batch截取部分"prompt+部分已采样结果"构建，然后重复 self.config.actor_rollout_ref.rollout.n 遍得到。初始时是来自原始的batch，第一次采样后的局部batch才开始由"prompt+部分已采样结果"构建

        2. 采样：for i in range(self.config.actor_rollout_ref.rollout.g):
            a. 前向：
                - (局部batch) 重复样本
                - (局部batch)(新行) 生成rid，设置pid、uid及其他信息
                    - 所有response都有rid、uid，但第一轮的response没有pid
                - (全局batch)(新行) 父轨迹上插入新的cid键值
                - (局部batch) 生成序列
                - (局部batch)(新行) 初始化state_branches为 全1，初始化subtree_branches为 全1
                - (局部batch) 计算奖励
                - (局部batch)(新函数) 前向计算
                    - gamma_t：
                        - 若无父轨迹，则从gamma_t = 1(对应response第一个token)开始，每前进一步乘一次gamma
                        - 若有父轨迹，取父轨迹所选状态的gamma_t，每前进一步乘一次gamma
                    - lam_t：
                        - 若无父轨迹，则从lam_t = 1(对应response第一个token)开始，每前进一步乘一次lam
                        - 若有父轨迹，取父轨迹所选状态的lam_t，每前进一步乘一次lam
                    - trajectory_reward：上一步的trajectory_reward + gamma_t * 当前步的reward
                        - 若无父轨迹，则从trajectory_reward = token_level_rewards[0](对应response第一个token)开始
                        - 若有父轨迹，取父轨迹所选状态的trajectory_reward作为上一步的trajectory_reward
                    - 父轨迹所选状态：根据pid找到父轨迹，根据当前轨迹的prompt_len，确定父轨迹所选状态的sentence长度及对应最后一个token的位置
                - (局部batch) 计算旧策略log概率
                - (局部batch) 计算参考策略log概率
                - (局部batch) 计算values
                - (新行) 局部batch添加至全局batch
            b. 反向：
                - (全局batch)(新函数：compute_treegae_advantage_return) 更新advantages & 保存advantages_mean：
                    - 反向更新：若存在父轨迹(pid)，则继续传播到父轨迹上
                    - if 当前位置不在cid.keys():
                        - 常规的GAE更新方式：lastgaelam_ = delta + gamma * lam * lastgaelam
                    - elif 当前位置在cid.keys(): (说明是分支节点)
                        - lastgaelam_mean = (sum(advantages[cid.values()][0]) + lastgaelam) / state_branches
                        - lastgaelam_ = delta + gamma * lam * lastgaelam_mean
                    - advantages 和 advantages_mean 都保存
            c. 选择(全局batch)(新函数)：
                - 计算 gve = values[1:] + lam * advantages_mean[1:]
                - 计算 expected_trajectory_reward = trajectory_reward[:-1] + gamma_t[:-1] * gve
                - 计算 partree_branches （上游分支节点的分支总数）：
                    - 初始化：与subtree_branches形状一样的全零Tensor
                    - if cid不存在：partree_branches[:] = 父轨迹所选状态对应的最后一个token位置的subtree_branches if pid存在 else (i + 1) * self.config.actor_rollout_ref.rollout.n
                    - elif cid存在：范围赋值（是否有更简洁的赋值函数或方法？）：
                        - partree_branches[: cid.keys()[0]+1] = 父轨迹所选状态对应的最后一个token位置的subtree_branches if pid存在 else (i + 1) * self.config.actor_rollout_ref.rollout.n
                        - for j in range(len(cid.keys()) - 1):
                            - partree_branches[cid.keys()[j]+1: cid.keys()[j+1]+1] = subtree_branches[cid.keys()[j]]
                        - partree_branches[cid.keys()[-1]+1: ] = subtree_branches[cid.keys()[-1]]
                - 计算tuct：expected_trajectory_reward * lam_t[1:] + self.config.algorithm.c * sqrt(log(partree_branches[:-1])) / subtree_branches[:-1]
                - 为每个uid选择一个最优的下一轮采样初始状态：
                    - 选择最大tuct的状态索引 (rid, 部分input_ids构成的状态的最后一个token的索引)
                    - 对于每个uid：计算根状态(prompt)的tuct值：（每个uid只有一个root）
                        - root.expected_trajectory_reward = root.gve = values[0] + lam * sum(advantages[相同uid、没有pid的所有response][0]) / 满足条件的response数量 （注：values[0]从任意response取都行，因为相同uid、无pid的response的第一个value是相同的）
                        - root.tuct = root.expected_trajectory_reward + self.config.algorithm.c * 1
                    - 将上述的最大tuct状态与根状态比较，取tuct较大者
                - 更新被选中状态的state_branches为 (self.config.actor_rollout_ref.rollout.n + 1)
                - 根据被选择的状态索引，构建新的局部batch（以被选中的状态作为输入）
                - 反向更新：从被选中的状态(包括被选中的状态)出发对应的祖先轨迹：
                    - subtree_branches += self.config.actor_rollout_ref.rollout.n

        3. 更新：
            - (新函数) 深度优先遍历（从无pid的样本出发，通过cid遍历所有response）：
                - 对于response第一个token：branch_weight_factor = 1
                - 对于后续token：branch_weight_factor = 上一步的branch_weight_factor * 上一步的state_branches
            - 更新critic：Critic Loss计算与先前保持一致
            - 更新actor：
                - Policy Loss 新函数：(新增一行) weighted_pg_losses = pg_losses / branch_weight_factor
                - agg_loss 函数：新增"weighted-token-mean"的分支，当使用opts_ttpo算法时，必须使用"weighted-token-mean"：
                    - loss = masked_sum(weighted_pg_losses) / masked_sum(1 / branch_weight_factor) * dp_size


注：
1. "新行"：表示在原代码的基础上增加新代码，适用于简单的改动
2. "新函数"：表示在原代码的基础上新增一个函数和该函数的调用，适用于复杂的改动

要求：
1. 所有改动都不应该出现在"LLM/verl"中，若需要改动某个代码文件，请将其复制到"LLM/"下，并保持其目录结构与代码文件所在的"LLM/verl/verl"下的目录结构类似或一致
2. 若存在更好的改动方式，请使用更好的
3. 不需要打印信息
4. 代码风格与原文件风格保持一致