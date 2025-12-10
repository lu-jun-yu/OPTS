# verl PPO 算法自定义扩展指南

本文档介绍如何在**最小侵入性**的前提下，对 verl 的 PPO 算法进行自定义修改，包括优势函数、损失函数和采样逻辑。

## 目录

- [核心发现](#核心发现)
- [推荐方案：创建独立扩展模块](#推荐方案创建独立扩展模块)
  - [1. 自定义优势函数](#1-自定义优势函数)
  - [2. 自定义损失函数](#2-自定义损失函数)
  - [3. 自定义采样逻辑](#3-自定义采样逻辑)
- [使用方法](#使用方法)
- [关键文件位置速查](#关键文件位置速查)
- [已有算法参考](#已有算法参考)

---

## 核心发现

verl 已经设计了良好的**注册机制**，可以在不修改原始代码的情况下扩展算法：

| 组件 | 注册装饰器 | 注册表 | 配置键 |
|------|-----------|--------|--------|
| 优势函数 | `@register_adv_est(name)` | `ADV_ESTIMATOR_REGISTRY` | `algorithm.adv_estimator` |
| 损失函数 | `@register_policy_loss(name)` | `POLICY_LOSS_REGISTRY` | `actor_rollout_ref.actor.policy_loss.loss_mode` |

**优势：**
- **零侵入性**：不修改 verl 源码
- **易于维护**：当 verl 更新时，只需确保注册接口兼容
- **可追溯**：自定义代码完全独立，便于版本控制

---

## 推荐方案：创建独立扩展模块

在你的项目中创建一个独立的扩展文件（如 `my_custom_ppo.py`），通过注册机制添加自定义算法。

### 1. 自定义优势函数

```python
# my_custom_ppo.py

import torch
import numpy as np
from typing import Optional
from collections import defaultdict

from verl.trainer.ppo.core_algos import register_adv_est
from verl.trainer.config import AlgoConfig


@register_adv_est("my_custom_advantage")  # 注册名称
def compute_my_custom_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    自定义优势估计算法

    Args:
        token_level_rewards: (bs, response_length) - token级别奖励
        response_mask: (bs, response_length) - 响应掩码，[EOS]后的token掩码为0
        index: (bs,) - 用于分组的uid数组（同一prompt的不同response共享相同uid）
        epsilon: 防止除零的小值
        config: AlgoConfig - 算法配置对象

    Returns:
        advantages: (bs, response_length) - 优势值
        returns: (bs, response_length) - 回报值
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    with torch.no_grad():
        bsz = scores.shape[0]

        # 按uid分组计算统计量
        id2score = defaultdict(list)
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        id2mean = {}
        id2std = {}
        for idx in id2score:
            scores_tensor = torch.stack(id2score[idx])
            id2mean[idx] = torch.mean(scores_tensor)
            id2std[idx] = torch.std(scores_tensor) if len(id2score[idx]) > 1 else torch.tensor(1.0)

        # === 在这里实现你的自定义优势计算逻辑 ===
        # 示例：组内标准化
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        # 扩展到token维度
        advantages = scores.unsqueeze(-1) * response_mask

    return advantages, advantages
```

**函数签名要求：**

```python
def your_advantage_fn(
    token_level_rewards: torch.Tensor,  # 必需
    response_mask: torch.Tensor,        # 必需
    index: np.ndarray = None,           # 可选，用于分组
    config: AlgoConfig = None,          # 可选，算法配置
    **kwargs,                           # 接收其他参数
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

---

### 2. 自定义损失函数

```python
# my_custom_ppo.py (续)

import torch
from typing import Optional, Any

from verl.trainer.ppo.core_algos import register_policy_loss, agg_loss
from verl.workers.config import ActorConfig
import verl.utils.torch_functional as verl_F


@register_policy_loss("my_custom_loss")  # 注册名称
def compute_my_custom_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    自定义策略损失函数

    Args:
        old_log_prob: (bs, response_length) - 旧策略的log概率
        log_prob: (bs, response_length) - 当前策略的log概率
        advantages: (bs, response_length) - 优势值
        response_mask: (bs, response_length) - 响应掩码
        loss_agg_mode: 损失聚合模式，可选值：
            - "token-mean": 按token平均
            - "seq-mean-token-sum": 序列内token求和，序列间平均
            - "seq-mean-token-mean": 序列内外都平均
            - "seq-mean-token-sum-norm": 序列内token求和并归一化
        config: ActorConfig - actor配置对象
        rollout_is_weights: 重要性采样权重（可选）

    Returns:
        loss: scalar tensor - 策略损失
        metrics: dict - 用于日志记录的指标
    """
    assert config is not None, "config is required"

    # 从config获取超参数
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    # 计算概率比
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)  # 数值稳定性
    ratio = torch.exp(negative_approx_kl)

    # === 在这里实现你的自定义损失计算逻辑 ===
    # 示例：标准PPO clipping
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # 应用重要性采样权重（如果提供）
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # 聚合损失
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info  # 包含dp_size, batch_num_tokens等
    )

    # 计算指标
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    metrics = {
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        # 添加你的自定义指标...
    }

    return pg_loss, metrics
```

**函数签名要求：**

```python
def your_policy_loss_fn(
    old_log_prob: torch.Tensor,         # 必需
    log_prob: torch.Tensor,             # 必需
    advantages: torch.Tensor,           # 必需
    response_mask: torch.Tensor,        # 必需
    loss_agg_mode: str,                 # 必需
    config: ActorConfig,                # 必需
    rollout_is_weights: torch.Tensor | None,  # 必需
) -> tuple[torch.Tensor, dict[str, Any]]:
    ...
```

---

### 3. 自定义采样逻辑

如果需要修改采样逻辑（如 rollout 生成、batch 处理），推荐**继承 RayPPOTrainer**：

```python
# my_custom_trainer.py

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage
from verl import DataProto


class MyCustomTrainer(RayPPOTrainer):
    """自定义PPO训练器"""

    def fit(self):
        """
        重写训练循环以修改采样逻辑

        关键修改点：
        - L1057-1059: batch.repeat() - 修改response重复逻辑
        - L1064-1068: generate_sequences() - 修改rollout生成
        - L1102: batch.repeat() - 修改batch重复策略
        - L1222-1230: compute_advantage() - 修改优势计算调用
        """
        # 复制原始fit()方法的代码
        # 在需要的地方添加你的自定义逻辑
        ...

    def _custom_sampling(self, batch: DataProto) -> DataProto:
        """自定义采样逻辑"""
        # 示例：修改采样数量
        custom_n = self.config.my_custom_config.n_samples
        return batch.repeat(repeat_times=custom_n, interleave=True)
```

**或者重写 `compute_advantage` 函数：**

```python
# my_custom_ppo.py (续)

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, get_adv_estimator_fn


def my_compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    config=None,
) -> DataProto:
    """
    自定义优势计算入口函数

    可以在这里添加前/后处理逻辑
    """
    # 前处理
    # ...

    # 调用注册的优势函数
    adv_fn = get_adv_estimator_fn(adv_estimator)
    advantages, returns = adv_fn(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=data.batch["response_mask"],
        index=data.non_tensor_batch.get("uid"),
        config=config,
    )

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns

    # 后处理
    # ...

    return data
```

---

## 使用方法

### 步骤1：在训练脚本启动时导入扩展模块

```python
# main.py 或 train.py

# 重要：必须在verl训练代码之前导入，以触发注册！
import my_custom_ppo

# 然后正常启动verl训练...
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
# ...
```

### 步骤2：修改配置文件

```yaml
# config.yaml

algorithm:
  adv_estimator: "my_custom_advantage"  # 使用自定义优势函数
  gamma: 1.0
  lam: 0.95
  # 其他算法参数...

actor_rollout_ref:
  actor:
    clip_ratio: 0.2
    clip_ratio_low: null   # 默认使用clip_ratio
    clip_ratio_high: null  # 默认使用clip_ratio
    loss_agg_mode: "token-mean"
    policy_loss:
      loss_mode: "my_custom_loss"  # 使用自定义损失函数
      # 其他损失函数参数...
```

### 步骤3：（可选）使用自定义Trainer

```python
# main.py

import my_custom_ppo
from my_custom_trainer import MyCustomTrainer

# 使用自定义训练器
trainer = MyCustomTrainer(
    config=config,
    tokenizer=tokenizer,
    # ...其他参数
)
trainer.init_workers()
trainer.fit()
```

---

## 关键文件位置速查

| 功能 | 文件路径 | 关键行号 |
|------|---------|---------|
| 优势函数注册机制 | `verl/trainer/ppo/core_algos.py` | L113-131 |
| 损失函数注册机制 | `verl/trainer/ppo/core_algos.py` | L53-85 |
| 优势函数调用入口 | `verl/trainer/ppo/ray_trainer.py` | L181-259 (`compute_advantage`) |
| 损失函数调用入口 | `verl/workers/utils/losses.py` | L95-155 (`ppo_loss`) |
| 训练主循环 | `verl/trainer/ppo/ray_trainer.py` | L977-1350 (`fit()`) |
| Actor更新逻辑 | `verl/workers/engine_workers.py` | L317-361 (`update_actor`) |
| KL惩罚计算 | `verl/trainer/ppo/ray_trainer.py` | L121-160 (`apply_kl_penalty`) |

---

## 已有算法参考

### 优势函数（`core_algos.py`）

| 名称 | 注册名 | 说明 |
|------|--------|------|
| GAE | `gae` | Generalized Advantage Estimation |
| GRPO | `grpo` | Group Relative Policy Optimization |
| REINFORCE++ | `reinforce_plus_plus` | REINFORCE with baseline |
| REINFORCE++ Baseline | `reinforce_plus_plus_baseline` | RF++ with learned baseline |
| RLOO | `rloo` | Leave-One-Out baseline |
| ReMax | `remax` | Reward maximization |
| OPO | `opo` | Length-weighted baseline |
| GPG | `gpg` | Grouped Policy Gradient |
| GRPO Pass@k | `grpo_passk` | Best-of-k variant |

### 损失函数（`core_algos.py`）

| 名称 | 注册名 | 说明 |
|------|--------|------|
| Vanilla PPO | `vanilla` | 标准PPO clipping |
| GSPO | `gspo` | Grouped Sequence-level PPO |
| GPG | `gpg` | Grouped Policy Gradient loss |
| Clip-Cov | `clip_cov` | Covariance-based clipping |
| KL-Cov | `kl_cov` | KL-regularized covariance |
| Geo-Mean | `geo_mean` | Geometric mean PPO (GMPO) |
| Rollout Correction | `rollout_correction` | IS-corrected policy gradient |

---

## 注意事项

1. **导入顺序**：确保在 verl 训练代码执行前导入你的扩展模块
2. **注册名唯一性**：自定义算法的注册名不能与已有算法冲突
3. **接口兼容性**：遵循函数签名要求，确保与 verl 的调用接口兼容
4. **配置传递**：通过 `config` 参数访问超参数，避免硬编码
5. **数值稳定性**：注意 log 概率和 ratio 的数值范围，必要时进行 clamp

---

## 完整示例文件

参见项目中的 `my_custom_ppo.py` 文件，包含完整的自定义优势函数和损失函数实现。
