# `core_algos.py` 文件详解

## 文件概述

`core_algos.py` 是 verl PPO 训练器的核心算法实现文件。该文件实现了 PPO（Proximal Policy Optimization）算法的核心功能，包括各种优势估计器（Advantage Estimator）、策略损失函数、KL 散度控制器等关键组件。

## 文件路径

```
LLM/verl/verl/trainer/ppo/core_algos.py
```

## 模块导出

```python
__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]
```

---

## 一、类型定义与全局注册表

### 1.1 PolicyLossFn 类型定义

```python
PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | ActorConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]
```

**功能说明**：
- 定义策略损失函数的类型签名
- 输入参数包括新旧对数概率、优势值、掩码、聚合模式和配置
- 返回值为损失张量和指标字典的元组

### 1.2 全局注册表

```python
POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}
ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}
```

- `POLICY_LOSS_REGISTRY`: 存储已注册的策略损失函数
- `ADV_ESTIMATOR_REGISTRY`: 存储已注册的优势估计器

---

## 二、注册器函数

### 2.1 register_policy_loss

```python
def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
```

**功能**：注册策略损失函数的装饰器

**参数**：
- `name`: 损失函数的注册名称

**返回值**：装饰器函数

**使用示例**：
```python
@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(...):
    ...
```

### 2.2 get_policy_loss_fn

```python
def get_policy_loss_fn(name):
```

**功能**：根据名称获取已注册的策略损失函数

**参数**：
- `name`: 损失函数名称

**返回值**：对应的策略损失函数

**异常**：若名称不存在，抛出 `ValueError`

### 2.3 register_adv_est

```python
def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
```

**功能**：注册优势估计器的装饰器

**参数**：
- `name_or_enum`: 估计器名称（字符串或枚举）

**返回值**：装饰器函数

**使用示例**：
```python
@register_adv_est(AdvantageEstimator.GAE)
def compute_gae_advantage_return(...):
    ...
```

### 2.4 get_adv_estimator_fn

```python
def get_adv_estimator_fn(name_or_enum):
```

**功能**：根据名称获取已注册的优势估计器函数

**参数**：
- `name_or_enum`: 估计器名称（字符串或枚举）

**返回值**：对应的优势估计器函数

---

## 三、枚举类

### 3.1 AdvantageEstimator

```python
class AdvantageEstimator(str, Enum):
    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
```

**功能**：定义所有支持的优势估计算法类型

| 枚举值 | 算法名称 | 描述 |
|--------|----------|------|
| GAE | Generalized Advantage Estimation | 广义优势估计 |
| GRPO | Group Relative Policy Optimization | 组相对策略优化 |
| REINFORCE_PLUS_PLUS | REINFORCE++ | 改进的 REINFORCE |
| REINFORCE_PLUS_PLUS_BASELINE | REINFORCE++ with Baseline | 带基线的 REINFORCE++ |
| REMAX | ReMax | ReMax 算法 |
| RLOO | Reinforcement Learning with Leave-One-Out | 留一法强化学习 |
| OPO | Output-length-weighted Policy Optimization | 输出长度加权策略优化 |
| GRPO_PASSK | GRPO Pass@k | Pass@k 变体的 GRPO |
| GPG | Group Policy Gradient | 组策略梯度 |
| RLOO_VECTORIZED | Vectorized RLOO | 向量化 RLOO |
| GRPO_VECTORIZED | Vectorized GRPO | 向量化 GRPO |

---

## 四、KL 控制器类

### 4.1 AdaptiveKLController

```python
class AdaptiveKLController:
    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult
```

**功能**：自适应 KL 控制器，根据当前 KL 散度动态调整 KL 惩罚系数

**参考论文**：[https://arxiv.org/pdf/1909.08593.pdf](https://arxiv.org/pdf/1909.08593.pdf)

**参数**：
- `init_kl_coef`: 初始 KL 系数
- `target_kl`: 目标 KL 值
- `horizon`: 调整时间范围

**方法**：
- `update(current_kl, n_steps)`: 根据当前 KL 和步数更新系数

**更新公式**：
```
proportional_error = clip(current_kl / target - 1, -0.2, 0.2)
mult = 1 + proportional_error * n_steps / horizon
value *= mult
```

### 4.2 FixedKLController

```python
class FixedKLController:
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass
```

**功能**：固定 KL 控制器，KL 系数保持不变

**参数**：
- `kl_coef`: 固定的 KL 系数

### 4.3 get_kl_controller

```python
def get_kl_controller(kl_ctrl):
```

**功能**：工厂函数，根据配置创建相应的 KL 控制器

**参数**：
- `kl_ctrl`: KL 控制器配置对象

**返回值**：`FixedKLController` 或 `AdaptiveKLController` 实例

---

## 五、优势估计函数

### 5.1 compute_gae_advantage_return (GAE)

```python
@register_adv_est(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
```

**功能**：使用广义优势估计（GAE）计算优势值和回报

**参考论文**：[GAE Paper](https://arxiv.org/abs/1506.02438)

**参数**：
| 参数 | 形状 | 描述 |
|------|------|------|
| `token_level_rewards` | (bs, response_length) | token 级奖励 |
| `values` | (bs, response_length) | 价值函数估计 |
| `response_mask` | (bs, response_length) | 响应掩码 |
| `gamma` | float | 折扣因子 |
| `lam` | float | GAE 的 λ 参数 |

**返回值**：
- `advantages`: 优势值，形状 (bs, response_length)
- `returns`: 回报值，形状 (bs, response_length)

**核心算法**：
```python
# 反向遍历计算 GAE
for t in reversed(range(gen_len)):
    delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
    lastgaelam_ = delta + gamma * lam * lastgaelam
    # 处理 observation tokens
    nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
    lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam
```

### 5.2 compute_grpo_outcome_advantage (GRPO)

```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
```

**功能**：计算 GRPO（Group Relative Policy Optimization）的优势值

**参数**：
| 参数 | 描述 |
|------|------|
| `token_level_rewards` | token 级奖励，形状 (bs, response_length) |
| `response_mask` | 响应掩码 |
| `index` | 分组索引数组 |
| `epsilon` | 防止除零的小值 |
| `norm_adv_by_std_in_grpo` | 是否用标准差归一化优势值 |

**算法流程**：
1. 计算每个样本的总分数：`scores = token_level_rewards.sum(dim=-1)`
2. 按 `index` 分组，计算每组的均值和标准差
3. 归一化分数：
   - 若 `norm_adv_by_std_in_grpo=True`：`(score - mean) / (std + epsilon)`
   - 若 `norm_adv_by_std_in_grpo=False`：`score - mean`（Dr.GRPO 变体）

**参考论文**：[Dr.GRPO](https://arxiv.org/abs/2503.20783)

### 5.3 compute_grpo_vectorized_outcome_advantage (GRPO 向量化版本)

```python
@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(...):
```

**功能**：GRPO 的向量化高效实现

**特点**：
- 使用 `torch.bincount` 进行向量化分组计算
- 比循环版本更高效

### 5.4 compute_grpo_passk_outcome_advantage (GRPO Pass@k)

```python
@register_adv_est(AdvantageEstimator.GRPO_PASSK)
def compute_grpo_passk_outcome_advantage(...):
```

**功能**：Pass@k 评估的 GRPO 变体

**参考论文**：[https://arxiv.org/abs/2503.19595](https://arxiv.org/abs/2503.19595)

**算法逻辑**：
- 每组只有最优响应获得非零优势值
- 优势值 = 最大奖励 - 次大奖励

### 5.5 compute_reinforce_plus_plus_baseline_outcome_advantage (RF++ Baseline)

```python
@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE)
def compute_reinforce_plus_plus_baseline_outcome_advantage(...):
```

**功能**：带基线的 REINFORCE++ 优势估计

**参考论文**：[https://arxiv.org/abs/2501.03262](https://arxiv.org/abs/2501.03262)

**算法特点**：
1. 计算组内均值作为基线
2. 减去基线后进行白化（whitening）

### 5.6 compute_rloo_outcome_advantage (RLOO)

```python
@register_adv_est(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(...):
```

**功能**：实现留一法（Leave-One-Out）强化学习

**参考论文**：[https://arxiv.org/abs/2402.14740](https://arxiv.org/abs/2402.14740)

**核心公式**：
```python
# 留一法估计
scores[i] = scores[i] * n / (n - 1) - mean * n / (n - 1)
```

### 5.7 compute_opo_outcome_advantage (OPO)

```python
@register_adv_est(AdvantageEstimator.OPO)
def compute_opo_outcome_advantage(...):
```

**功能**：输出长度加权策略优化

**参考论文**：[https://arxiv.org/pdf/2505.23585](https://arxiv.org/pdf/2505.23585)

**核心公式**：
```python
# 长度加权基线
baseline = (len_tensor * score_tensor).sum() / len_tensor.sum()
```

### 5.8 compute_reinforce_plus_plus_outcome_advantage (RF++)

```python
@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)
def compute_reinforce_plus_plus_outcome_advantage(...):
```

**功能**：REINFORCE++ 算法实现

**参考论文**：[https://arxiv.org/abs/2501.03262](https://arxiv.org/abs/2501.03262)

**算法特点**：
1. 反向累积折扣回报
2. 使用掩码白化优势值
3. EOS 后重置回报

### 5.9 compute_remax_outcome_advantage (ReMax)

```python
@register_adv_est(AdvantageEstimator.REMAX)
def compute_remax_outcome_advantage(...):
```

**功能**：ReMax 算法实现

**参考论文**：[https://arxiv.org/abs/2310.10505](https://arxiv.org/abs/2310.10505)

**算法公式**：
```python
returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
advantages = returns - reward_baselines.unsqueeze(-1) * response_mask
```

### 5.10 compute_gpg_outcome_advantage (GPG)

```python
@register_adv_est(AdvantageEstimator.GPG)
def compute_gpg_outcome_advantage(...):
```

**功能**：Group Policy Gradient 算法

**特点**：
- 使用自适应 alpha 系数
- 按 `f_norm` 归一化

### 5.11 compute_rloo_vectorized_outcome_advantage (RLOO 向量化版本)

```python
@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)
def compute_rloo_vectorized_outcome_advantage(...):
```

**功能**：RLOO 的向量化高效实现

---

## 六、损失聚合函数

### 6.1 agg_loss

```python
def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
```

**功能**：将损失矩阵聚合为标量，确保损失对分布式并行不变

**参数**：
| 参数 | 描述 |
|------|------|
| `loss_mat` | 微批次损失矩阵，形状 (bs, response_length) |
| `loss_mask` | 微批次损失掩码 |
| `loss_agg_mode` | 聚合模式 |
| `dp_size` | 数据并行大小 |

**支持的聚合模式**：

| 模式 | 公式 | 描述 |
|------|------|------|
| `token-mean` | `sum(loss * mask) / num_tokens * dp_size` | token 级均值 |
| `seq-mean-token-sum` | 先 token 求和，再序列均值 | 序列均值-token 求和 |
| `seq-mean-token-mean` | 先 token 均值，再序列均值 | 序列均值-token 均值 |
| `seq-mean-token-sum-norm` | token 求和后按 scale_factor 归一化 | 归一化序列损失 |

---

## 七、策略损失函数

### 7.1 compute_policy_loss (已弃用)

```python
@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(...):
```

**功能**：已弃用，建议使用 `compute_policy_loss_vanilla`

### 7.2 compute_policy_loss_vanilla (标准 PPO)

```python
@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
```

**功能**：标准 PPO 剪切目标函数实现

**参考论文**：[PPO Paper](https://arxiv.org/abs/1707.06347)

**核心算法**：
```python
# 计算概率比
ratio = torch.exp(log_prob - old_log_prob)

# PPO 剪切损失
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

# Dual-clip PPO（可选）
pg_losses3 = -advantages * clip_ratio_c
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

# 最终损失
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

**返回指标**：
- `actor/pg_clipfrac`: 被剪切的比例（上界）
- `actor/ppo_kl`: 近似 KL 散度
- `actor/pg_clipfrac_lower`: 被剪切的比例（下界）

### 7.3 compute_policy_loss_gspo (GSPO)

```python
@register_policy_loss("gspo")
def compute_policy_loss_gspo(...):
```

**功能**：Group Sequence Policy Optimization

**参考论文**：[https://arxiv.org/pdf/2507.18071](https://arxiv.org/pdf/2507.18071)

**核心创新**：
- 使用序列级重要性比率：`s_i(θ) = (π_θ/π_θold)^(1/|y_i|)`
- 结合 token 级梯度流

### 7.4 compute_policy_loss_gpg (GPG)

```python
@register_policy_loss("gpg")
def compute_policy_loss_gpg(...):
```

**功能**：Group Policy Gradient 损失

**参考代码**：[GPG Implementation](https://github.com/AMAP-ML/GPG)

**核心公式**：
```python
pg_losses = -log_prob * advantages
```

### 7.5 compute_policy_loss_clip_cov (Clip-Cov)

```python
@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(...):
```

**功能**：基于协方差的剪切策略损失

**参考代码**：[PRIME-RL](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL)

**算法特点**：
- 随机选择高协方差 token 进行剪切
- 结合 PPO 剪切和协方差剪切

### 7.6 compute_policy_loss_kl_cov (KL-Cov)

```python
@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(...):
```

**功能**：基于协方差的 KL 惩罚策略损失

**算法特点**：
- 对高协方差 token 添加 KL 惩罚项

### 7.7 compute_policy_loss_geo_mean (GMPO)

```python
@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(...):
```

**功能**：几何均值策略优化

**参考论文**：[GMPO Paper](https://arxiv.org/abs/2507.20673)

**核心创新**：
- 使用几何均值重要性比率
- token 级剪切，序列级聚合

### 7.8 compute_policy_loss_rollout_correction_wrapper

```python
@register_policy_loss("rollout_correction")
def compute_policy_loss_rollout_correction_wrapper(...):
```

**功能**：滚动校正损失的包装器

**用途**：当启用 `use_policy_gradient=True` 时使用

---

## 八、其他损失函数

### 8.1 compute_entropy_loss

```python
def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
```

**功能**：计算分类熵损失

**参数**：
- `logits`: 形状 (bs, response_length, vocab_size)
- `response_mask`: 形状 (bs, response_length)

**返回值**：标量熵损失

### 8.2 compute_value_loss

```python
def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
```

**功能**：计算 PPO 的剪切价值函数损失

**参考代码**：[TRL PPO Trainer](https://github.com/huggingface/trl)

**核心算法**：
```python
vpredclipped = clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
vf_losses1 = (vpreds - returns) ** 2
vf_losses2 = (vpredclipped - returns) ** 2
clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
vf_loss = 0.5 * agg_loss(clipped_vf_losses, ...)
```

---

## 九、KL 惩罚函数

### 9.1 kl_penalty

```python
def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
```

**功能**：计算 KL 惩罚，支持直通梯度估计

**参考博客**：[KL Approximation](http://joschu.net/blog/kl-approx.html)

### 9.2 kl_penalty_forward

```python
def kl_penalty_forward(logprob, ref_logprob, kl_penalty) -> torch.FloatTensor:
```

**功能**：KL 散度的前向计算

**支持的模式**：

| 模式 | 公式 | 描述 |
|------|------|------|
| `kl` / `k1` | `logprob - ref_logprob` | 直接估计 |
| `abs` | `|logprob - ref_logprob|` | 绝对值 |
| `mse` / `k2` | `0.5 * (logprob - ref_logprob)²` | 均方误差 |
| `low_var_kl` / `k3` | `exp(ref - log) - (ref - log) - 1` | 低方差估计 |

---

## 十、奖励计算

### 10.1 compute_rewards

```python
def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
```

**功能**：计算带 KL 惩罚的 token 级奖励

**公式**：
```python
kl = old_log_prob - ref_log_prob
return token_level_scores - kl * kl_ratio
```

---

## 十一、数据重采样

### 11.1 compute_pf_ppo_reweight_data

```python
def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
```

**功能**：基于分数的数据重要性采样

**支持的重采样方法**：

| 方法 | 权重计算 |
|------|----------|
| `pow` | `|score|^weight_pow` |
| `max_min` | 只保留最大和最小分数样本 |
| `max_random` | 最大分数权重 0.4，其他 0.1 |

---

## 十二、滚动校正策略损失

### 12.1 compute_policy_loss_with_rollout_correction

```python
def compute_policy_loss_with_rollout_correction(
    rollout_log_prob,
    log_prob,
    advantages,
    eos_mask,
    loss_agg_mode="seq-mean-token-sum",
    config: Optional[ActorConfig] = None,
    loss_scale_factor=1.0,
    rollout_is: Optional[str] = None,
    rollout_is_threshold: float = 2.0,
    rollout_rs: Optional[str] = None,
    rollout_rs_threshold: Optional[float] = None,
    rollout_rs_threshold_lower: Optional[float] = None,
    rollout_token_veto_threshold: Optional[float] = None,
    rollout_is_batch_normalize: bool = False,
):
```

**功能**：带滚动校正的纯策略梯度损失（无 PPO 剪切）

**数学公式**：
- 无 IS：`L = -E[log π(a|s) * A(s,a)]`
- 有 IS：`L = -E_π_rollout[w * log π(a|s) * A(s,a)]`，其中 `w = π_current / π_rollout`

**参数说明**：
| 参数 | 描述 |
|------|------|
| `rollout_is` | IS 聚合级别（"token"/"sequence"） |
| `rollout_is_threshold` | IS 权重上限阈值 |
| `rollout_rs` | 拒绝采样聚合级别 |
| `rollout_rs_threshold` | RS 上限阈值 |
| `rollout_rs_threshold_lower` | RS 下限阈值 |
| `rollout_token_veto_threshold` | 灾难性 token 否决阈值 |

---

## 总结

`core_algos.py` 是 verl PPO 训练器的核心，实现了：

1. **优势估计器**：支持 11 种不同的优势估计算法
2. **策略损失函数**：支持 8 种不同的策略损失计算方式
3. **KL 控制器**：自适应和固定两种 KL 系数控制
4. **损失聚合**：支持多种聚合模式，适配分布式训练
5. **滚动校正**：支持重要性采样和拒绝采样校正

该文件通过注册器模式实现了高度的可扩展性，用户可以方便地添加新的算法。
