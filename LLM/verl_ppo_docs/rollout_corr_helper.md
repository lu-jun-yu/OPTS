# `rollout_corr_helper.py` 文件详解

## 文件概述

`rollout_corr_helper.py` 是 verl PPO 训练器的滚动校正（Rollout Correction）辅助模块。该模块提供了完整的流水线来解决 RL 训练中的**离策略（off-policy）问题**，包括：
1. 策略不匹配（如 vLLM BFloat16 vs FSDP FP32）
2. 模型更新滞后（基于旧检查点的轨迹训练）
3. 数据收集和训练之间的分布偏移

## 文件路径

```
LLM/verl/verl/trainer/ppo/rollout_corr_helper.py
```

## 核心功能

1. **多粒度聚合**：
   - 重要性采样（IS）：Token 级和序列级
   - 拒绝采样（RS）：Token 级、序列级和几何均值

2. **灾难性异常值否决**：独立的 token 级否决机制

3. **内存高效设计**：
   - 对数空间计算避免溢出
   - 固定安全边界（exp(±20)）
   - 无大型中间张量的指标计算

4. **全面的指标追踪**：
   - IS/RS 统计量
   - 离策略诊断（KL、PPL、χ² 散度）

## 常量定义

```python
SAFETY_BOUND = 20.0  # 安全边界，防止数值溢出
# exp(20) ≈ 4.85亿（上限），exp(-20) ≈ 2e-9（下限）
```

---

## 一、拒绝采样函数

### 1.1 compute_rollout_rejection_mask

```python
def compute_rollout_rejection_mask(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_rs: str = "token",
    rollout_rs_threshold: Optional[float] = None,
    rollout_rs_threshold_lower: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
```

**功能**：计算用于离策略 RL 训练中异常值处理的拒绝掩码

**参数**：
| 参数 | 描述 |
|------|------|
| `log_ratio` | 对数比率 log(π_train / π_rollout)，形状 (batch_size, seq_length) |
| `response_mask` | 有效 token 的二值掩码（1=有效，0=填充） |
| `rollout_rs` | 拒绝采样聚合级别：`"token"`、`"sequence"`、`"geometric"` |
| `rollout_rs_threshold` | IS 权重上限阈值 |
| `rollout_rs_threshold_lower` | IS 权重下限阈值（默认为 1/上限阈值） |

**返回值**：
- `modified_response_mask`：应用拒绝后的响应掩码
- `metrics`：拒绝采样指标字典

**聚合级别说明**：

| 级别 | 计算方式 | 描述 |
|------|----------|------|
| `token` | `exp(log_ratio)` | 每 token 的异常值检测 |
| `sequence` | `exp(sum(log_ratio))` | 序列级（token 比率的乘积） |
| `geometric` | `exp(mean(log_ratio))` | 序列级几何均值 |

**核心实现**：

```python
# Token 级
log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
rollout_is_weights = torch.exp(log_ratio_safe)

# 序列级
log_ratio_sum = verl_F.masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(-1)
log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)

# 几何均值级
log_ratio_mean = verl_F.masked_mean(log_ratio, response_mask, axis=-1).unsqueeze(-1)
log_ratio_mean_safe = torch.clamp(log_ratio_mean, min=-SAFETY_BOUND, max=SAFETY_BOUND)
rollout_is_weights = torch.exp(log_ratio_mean_safe).expand_as(log_ratio)

# 生成异常值掩码
mask = (rollout_is_weights >= lower_threshold) & (rollout_is_weights <= upper_threshold)
```

### 1.2 compute_rs_metrics

```python
def compute_rs_metrics(
    rollout_is_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_rs: str,
    rollout_rs_threshold: float,
    rollout_rs_threshold_lower: float,
) -> dict[str, float]:
```

**功能**：计算拒绝采样的综合指标

**返回的指标**：
| 指标名 | 描述 |
|--------|------|
| `rollout_rs_mean` | IS 权重均值 |
| `rollout_rs_max` | IS 权重最大值 |
| `rollout_rs_min` | IS 权重最小值 |
| `rollout_rs_std` | IS 权重标准差 |
| `rollout_rs_ratio_fraction_high` | 超过上限阈值的比例 |
| `rollout_rs_ratio_fraction_low` | 低于下限阈值的比例 |
| `rollout_rs_eff_sample_size` | 有效样本量（ESS） |
| `rollout_rs_seq_mean` | 序列级权重均值 |
| `rollout_rs_seq_std` | 序列级权重标准差 |
| `rollout_rs_seq_max_deviation` | 序列偏离理想权重(1.0)的最大值 |

---

## 二、重要性采样函数

### 2.1 compute_rollout_correction_weights

```python
def compute_rollout_correction_weights(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: str = "token",
    rollout_is_threshold: float = 2.0,
    rollout_is_batch_normalize: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
```

**功能**：计算重要性采样权重以校正离策略分布偏移

**参数**：
| 参数 | 描述 |
|------|------|
| `log_ratio` | 对数比率 log(π_train / π_rollout) |
| `response_mask` | 有效 token 掩码 |
| `rollout_is` | IS 聚合级别：`"token"` 或 `"sequence"` |
| `rollout_is_threshold` | 截断极端权重的上限阈值（默认 2.0） |
| `rollout_is_batch_normalize` | 是否将权重归一化为均值=1.0 |

**返回值**：
- `rollout_is_weights`：截断后的 IS 权重
- `metrics`：IS 权重指标字典

**核心实现**：

```python
# Token 级
log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
rollout_is_weights = torch.exp(log_ratio_safe)

# 序列级
log_ratio_sum = verl_F.masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(-1)
log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)

# 掩码处理
rollout_is_weights = rollout_is_weights * response_mask

# 截断（TIS: Truncated Importance Sampling）
rollout_is_weights = rollout_is_weights.clamp(max=rollout_is_threshold)

# 分离梯度（IS 理论要求）
rollout_is_weights = rollout_is_weights.detach()

# 批次归一化（可选）
if rollout_is_batch_normalize:
    weights_mean = verl_F.masked_mean(rollout_is_weights, response_mask)
    if weights_mean > 1e-8:
        rollout_is_weights = rollout_is_weights / weights_mean
```

### 2.2 compute_is_metrics

```python
def compute_is_metrics(
    rollout_is_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: str,
    rollout_is_threshold: float,
) -> dict[str, float]:
```

**功能**：计算截断 IS 权重的综合指标

**返回的指标**：
| 指标名 | 描述 |
|--------|------|
| `rollout_is_mean` | IS 权重均值 |
| `rollout_is_max` | IS 权重最大值 |
| `rollout_is_min` | IS 权重最小值 |
| `rollout_is_std` | IS 权重标准差 |
| `rollout_is_ratio_fraction_high` | 超过上限阈值的比例 |
| `rollout_is_ratio_fraction_low` | 低于下限阈值的比例 |
| `rollout_is_eff_sample_size` | 有效样本量（ESS） |
| `rollout_is_seq_*` | 序列级统计量 |

---

## 三、统一接口函数

### 3.1 compute_rollout_correction_and_rejection_mask

```python
def compute_rollout_correction_and_rejection_mask(
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: Optional[str] = None,
    rollout_is_threshold: Optional[float] = 2.0,
    rollout_rs: Optional[str] = None,
    rollout_rs_threshold: Optional[float] = 2.0,
    rollout_rs_threshold_lower: Optional[float] = None,
    rollout_token_veto_threshold: Optional[float] = None,
    rollout_is_batch_normalize: bool = False,
) -> tuple[Optional[DataProto], torch.Tensor, dict[str, float]]:
```

**功能**：统一接口，计算 IS 权重和拒绝掩码

**参数**：
| 参数 | 描述 |
|------|------|
| `old_log_prob` | 训练策略的对数概率（如 FSDP FP32） |
| `rollout_log_prob` | 滚动策略的对数概率（如 vLLM BF16） |
| `response_mask` | 有效 token 掩码 |
| `rollout_is` | IS 聚合级别（None 禁用 IS） |
| `rollout_is_threshold` | IS 权重上限阈值 |
| `rollout_rs` | RS 聚合级别（None 禁用 RS） |
| `rollout_rs_threshold` | RS 上限阈值 |
| `rollout_rs_threshold_lower` | RS 下限阈值 |
| `rollout_token_veto_threshold` | 灾难性 token 否决阈值 |
| `rollout_is_batch_normalize` | 是否批次归一化 |

**返回值**：
- `rollout_is_weights_proto`：IS 权重的 DataProto（若禁用则为 None）
- `modified_response_mask`：应用 RS 和否决后的响应掩码
- `metrics_scalar`：所有指标（带 "rollout_corr/" 前缀）

**处理流程**：

```
Step 1: 计算对数比率
    log_ratio = old_log_prob - rollout_log_prob

Step 2: 计算 IS 权重（如果启用）
    rollout_is_weights, is_metrics = compute_rollout_correction_weights(...)

Step 3: 计算拒绝掩码（如果启用）
    modified_response_mask, rs_metrics = compute_rollout_rejection_mask(...)

Step 4: 应用 token 否决（如果启用）
    catastrophic_tokens = (log_ratio < log_veto_threshold) & response_mask
    has_catastrophic = catastrophic_tokens.any(dim=-1, keepdim=True)
    veto_mask = (~has_catastrophic).float()
    modified_response_mask = modified_response_mask * veto_mask

Step 5: 计算离策略指标
    offpolicy_metrics = compute_offpolicy_metrics(...)

Step 6: 添加指标前缀
    metrics_scalar = {f"rollout_corr/{k}": v for k, v in metrics.items()}
```

---

## 四、离策略诊断指标

### 4.1 compute_offpolicy_metrics

```python
def compute_offpolicy_metrics(
    old_log_prob: torch.Tensor,
    rollout_log_prob: Optional[torch.Tensor],
    response_mask: torch.Tensor,
) -> dict[str, Any]:
```

**功能**：计算离策略诊断指标

**返回的指标**：

| 指标名 | 公式 | 描述 |
|--------|------|------|
| `training_ppl` | `exp(-mean(log_prob))` | 训练策略困惑度 |
| `training_log_ppl` | `-mean(log_prob)` | 训练策略对数困惑度 |
| `rollout_ppl` | `exp(-mean(rollout_log_prob))` | 滚动策略困惑度 |
| `rollout_log_ppl` | `-mean(rollout_log_prob)` | 滚动策略对数困惑度 |
| `kl` | `E[log(π_rollout) - log(π_training)]` | KL 散度直接估计 |
| `k3_kl` | `E[exp(r) - r - 1]` | K3 KL 估计器（更稳定） |
| `log_ppl_diff` | `mean_log_prob_rollout - mean_log_prob_training` | 对数困惑度差异 |
| `ppl_ratio` | `exp(log_ppl_diff)` | 困惑度比率 |
| `chi2_token` | `E_token[ρ²] - 1` | Token 级 χ² 散度 |
| `chi2_seq` | `E_seq[(∏ρ_t)²] - 1` | 序列级 χ² 散度 |

**χ² 散度计算**：

```python
# Token 级 χ²: E[ρ²] - 1，其中 ρ = π_training / π_rollout
log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
rho_token = torch.exp(log_ratio_safe)
rho_squared_token = rho_token.square()
chi2_token = verl_F.masked_mean(rho_squared_token, response_mask) - 1.0

# 序列级 χ²: E[(∏ρ_t)²] - 1 = E[exp(2 * Σlog(ρ_t))] - 1
log_ratio_sum = verl_F.masked_sum(log_ratio, response_mask, axis=-1)
log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
rho_squared_seq = torch.exp(2.0 * log_ratio_sum_safe)
chi2_seq = rho_squared_seq.mean() - 1.0
```

---

## 五、批次集成函数

### 5.1 compute_rollout_correction_and_add_to_batch

```python
def compute_rollout_correction_and_add_to_batch(
    batch: DataProto, rollout_corr_config: RolloutCorrectionConfig
) -> tuple[DataProto, dict]:
```

**功能**：计算滚动校正权重并应用拒绝采样到批次

**行为**：
- `response_mask`：**始终**更新（应用拒绝）
- `rollout_is_weights`：**仅当**设置 `rollout_is` 参数时添加

**实现**：

```python
# 从配置获取参数
rollout_is = rollout_corr_config.get("rollout_is", None)
rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
rollout_rs = rollout_corr_config.get("rollout_rs", None)
# ...

# 计算 IS 权重和修改后的 response_mask
rollout_is_weights, modified_response_mask, rollout_corr_metrics = (
    compute_rollout_correction_and_rejection_mask(
        old_log_prob=batch.batch["old_log_probs"],
        rollout_log_prob=batch.batch["rollout_log_probs"],
        response_mask=batch.batch["response_mask"],
        rollout_is=rollout_is,
        # ...
    )
)

# 始终更新 response_mask
batch.batch["response_mask"] = modified_response_mask

# 如果计算了 IS 权重，添加到批次
if rollout_is_weights is not None:
    batch = batch.union(rollout_is_weights)

return batch, rollout_corr_metrics
```

### 5.2 compute_rollout_corr_metrics_from_logprobs

```python
def compute_rollout_corr_metrics_from_logprobs(
    log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
```

**功能**：从对数概率计算滚动校正指标（用于 actor 训练时）

**用途**：在训练过程中追踪**当前策略**与滚动策略之间的离策略差距

### 5.3 apply_rollout_correction

```python
def apply_rollout_correction(
    batch: DataProto,
    rollout_corr_config: Optional[RolloutCorrectionConfig] = None,
    policy_loss_config: PolicyLossConfig = None,
) -> None:
```

**功能**：**旁路模式（Bypass Mode）**下使用 `rollout_log_probs` 作为 `old_log_probs`

**两种子模式**：

| 模式 | use_policy_gradient | 描述 |
|------|---------------------|------|
| Bypass + PPO | False（默认） | 使用标准 PPO 损失，old_log_prob=rollout_log_prob |
| Bypass + PG | True | 使用策略梯度损失，无 PPO 剪切 |

**实现**：

```python
# 使用 rollout log probs 作为 old log probs（零成本替换）
batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]

# 将 rollout_correction 配置传递给 actor
policy_loss_config["rollout_correction"] = rollout_corr_config

# 如果启用策略梯度模式
use_policy_gradient = rollout_corr_config.get("use_policy_gradient", False)
if use_policy_gradient:
    policy_loss_config["loss_mode"] = "rollout_correction"
```

---

## 六、使用示例

### 6.1 配置示例

```yaml
algorithm:
  rollout_correction:
    # 旁路模式：跳过重新计算 old_log_prob
    bypass_mode: true

    # 策略梯度模式（无 PPO 剪切）
    use_policy_gradient: false

    # 重要性采样
    rollout_is: "token"  # 或 "sequence"
    rollout_is_threshold: 2.0
    rollout_is_batch_normalize: false

    # 拒绝采样
    rollout_rs: "token"  # 或 "sequence" 或 "geometric"
    rollout_rs_threshold: 2.0
    rollout_rs_threshold_lower: 0.5

    # 灾难性 token 否决
    rollout_token_veto_threshold: 0.1
```

### 6.2 代码使用

```python
from verl.trainer.ppo.rollout_corr_helper import (
    compute_rollout_correction_and_rejection_mask,
    compute_rollout_correction_and_add_to_batch,
)

# 方式 1：直接使用
rollout_is_weights, modified_mask, metrics = compute_rollout_correction_and_rejection_mask(
    old_log_prob=batch.batch["old_log_probs"],
    rollout_log_prob=batch.batch["rollout_log_probs"],
    response_mask=batch.batch["response_mask"],
    rollout_is="token",
    rollout_is_threshold=2.0,
    rollout_rs="token",
    rollout_rs_threshold=2.0,
)

# 方式 2：集成到批次
batch, metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
```

---

## 七、参考文献

- [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://richardli.xyz/rl-collapse)
- [Off-policy RL 理论基础](https://fengyao.notion.site/off-policy-rl)

---

## 总结

`rollout_corr_helper.py` 提供了完整的离策略校正工具集：

1. **重要性采样（IS）**：校正分布偏移，支持 token 和序列级聚合
2. **拒绝采样（RS）**：过滤异常值样本，支持多种聚合方式
3. **Token 否决**：处理灾难性异常值
4. **离策略诊断**：KL、PPL、χ² 散度等指标
5. **旁路模式**：跳过昂贵的 old_log_prob 重计算
6. **内存高效**：对数空间计算，固定安全边界

该模块是解决 RL 训练中策略不匹配问题的关键组件。
