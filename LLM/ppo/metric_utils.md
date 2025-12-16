# `metric_utils.py` 文件详解

## 文件概述

`metric_utils.py` 是 verl PPO 训练器的指标计算工具模块。该文件提供了训练过程中各种指标的计算功能，包括数据指标、时间指标、吞吐量指标以及验证指标的统计分析。

## 文件路径

```
LLM/verl/verl/trainer/ppo/metric_utils.py
```

## 模块导入

```python
from collections import defaultdict
from functools import partial
from typing import Any, Callable

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated
```

---

## 一、指标归约函数

### 1.1 reduce_metrics (已弃用)

```python
@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
```

**功能**：将指标列表归约为均值

**参数**：
- `metrics`: 字典，键为指标名称，值为指标值列表

**返回值**：字典，每个列表被替换为其均值

**使用示例**：
```python
>>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
>>> reduce_metrics(metrics)
{"loss": 2.0, "accuracy": 0.8}
```

**注意**：该函数已弃用，建议使用 `verl.utils.metric.reduce_metrics`

---

## 二、辅助函数

### 2.1 _compute_response_info

```python
def _compute_response_info(batch: DataProto) -> dict[str, Any]:
```

**功能**：从批次数据中提取提示和响应的相关信息

**参数**：
- `batch`: DataProto 对象，包含批次数据

**返回值**：字典，包含以下键：
| 键名 | 描述 | 形状 |
|------|------|------|
| `response_mask` | 响应 token 的注意力掩码 | (batch_size, response_length) |
| `prompt_length` | 每个样本的提示长度 | (batch_size,) |
| `response_length` | 每个样本的响应长度 | (batch_size,) |

**实现细节**：
```python
response_length = batch.batch["responses"].shape[-1]

# 分离提示和响应的掩码
prompt_mask = batch.batch["attention_mask"][:, :-response_length]
response_mask = batch.batch["attention_mask"][:, -response_length:]

# 计算长度
prompt_length = prompt_mask.sum(-1).float()
response_length = response_mask.sum(-1).float()
```

---

## 三、数据指标计算

### 3.1 compute_data_metrics

```python
def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
```

**功能**：计算 PPO 训练批次的各种数据指标

**参数**：
- `batch`: DataProto 对象，包含训练数据
- `use_critic`: 是否包含 critic 相关指标，默认 True

**返回值**：包含以下指标的字典：

#### 分数指标
| 指标名 | 描述 |
|--------|------|
| `critic/score/mean` | 序列分数均值 |
| `critic/score/max` | 序列分数最大值 |
| `critic/score/min` | 序列分数最小值 |

#### 奖励指标
| 指标名 | 描述 |
|--------|------|
| `critic/rewards/mean` | 序列奖励均值 |
| `critic/rewards/max` | 序列奖励最大值 |
| `critic/rewards/min` | 序列奖励最小值 |

#### 优势指标
| 指标名 | 描述 |
|--------|------|
| `critic/advantages/mean` | 优势值均值 |
| `critic/advantages/max` | 优势值最大值 |
| `critic/advantages/min` | 优势值最小值 |

#### 回报指标
| 指标名 | 描述 |
|--------|------|
| `critic/returns/mean` | 回报均值 |
| `critic/returns/max` | 回报最大值 |
| `critic/returns/min` | 回报最小值 |

#### 价值指标（仅当 use_critic=True）
| 指标名 | 描述 |
|--------|------|
| `critic/values/mean` | 价值估计均值 |
| `critic/values/max` | 价值估计最大值 |
| `critic/values/min` | 价值估计最小值 |
| `critic/vf_explained_var` | 价值函数解释方差 |

#### 响应长度指标
| 指标名 | 描述 |
|--------|------|
| `response_length/mean` | 响应长度均值 |
| `response_length/max` | 响应长度最大值 |
| `response_length/min` | 响应长度最小值 |
| `response_length/clip_ratio` | 达到最大长度的比例 |

#### 非中止响应长度指标
| 指标名 | 描述 |
|--------|------|
| `response_length_non_aborted/mean` | 非中止样本的响应长度均值 |
| `response_length_non_aborted/max` | 非中止样本的响应长度最大值 |
| `response_length_non_aborted/min` | 非中止样本的响应长度最小值 |
| `response_length_non_aborted/clip_ratio` | 非中止样本达到最大长度的比例 |

#### 中止比例指标
| 指标名 | 描述 |
|--------|------|
| `response/aborted_ratio` | 响应长度为零的样本比例 |

#### 提示长度指标
| 指标名 | 描述 |
|--------|------|
| `prompt_length/mean` | 提示长度均值 |
| `prompt_length/max` | 提示长度最大值 |
| `prompt_length/min` | 提示长度最小值 |
| `prompt_length/clip_ratio` | 达到最大长度的比例 |

#### 多轮对话指标（可选）
| 指标名 | 描述 |
|--------|------|
| `num_turns/min` | 对话轮次最小值 |
| `num_turns/max` | 对话轮次最大值 |
| `num_turns/mean` | 对话轮次均值 |

#### 工具调用指标（可选）
| 指标名 | 描述 |
|--------|------|
| `tool_call_counts/min` | 工具调用次数最小值 |
| `tool_call_counts/max` | 工具调用次数最大值 |
| `tool_call_counts/mean` | 工具调用次数均值 |

**价值函数解释方差计算**：
```python
# Explained variance: 1 - Var(returns - values) / Var(returns)
return_diff_var = torch.var(valid_returns - valid_values)
return_var = torch.var(valid_returns)
vf_explained_var = 1.0 - return_diff_var / (return_var + 1e-5)
```

---

## 四、时间指标计算

### 4.1 compute_timing_metrics

```python
def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
```

**功能**：计算各处理阶段的时间指标

**参数**：
- `batch`: DataProto 对象
- `timing_raw`: 字典，阶段名称映射到执行时间（秒）

**返回值**：包含以下指标的字典：
| 指标名 | 描述 |
|--------|------|
| `timing_s/{name}` | 各阶段的原始时间（秒） |
| `timing_per_token_ms/{name}` | 各阶段的每 token 时间（毫秒） |

**不同阶段的 token 计数**：
| 阶段 | token 计数方式 |
|------|---------------|
| `gen` | 仅响应 token |
| `ref`, `values`, `adv`, `update_critic`, `update_actor` | 所有 token（提示 + 响应） |

**实现细节**：
```python
num_tokens_of_section = {
    "gen": num_response_tokens,
    **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
}
```

---

## 五、吞吐量指标计算

### 5.1 compute_throughout_metrics

```python
def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
```

**功能**：计算 PPO 训练的吞吐量指标

**参数**：
- `batch`: DataProto 对象，包含 token 数量信息
- `timing_raw`: 时间字典，必须包含 `"step"` 键
- `n_gpus`: 训练使用的 GPU 数量

**返回值**：
| 指标名 | 描述 |
|--------|------|
| `perf/total_num_tokens` | 批次中的总 token 数 |
| `perf/time_per_step` | 单步时间（秒） |
| `perf/throughput` | 每 GPU 每秒处理的 token 数 |

**吞吐量计算公式**：
```python
throughput = total_num_tokens / (time * n_gpus)
```

---

## 六、Bootstrap 统计函数

### 6.1 bootstrap_metric

```python
def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
```

**功能**：使用 Bootstrap 重采样估计指标的统计量

**参数**：
| 参数 | 描述 |
|------|------|
| `data` | 数据点列表 |
| `subset_size` | 每次采样的大小 |
| `reduce_fns` | 归约函数列表 |
| `n_bootstrap` | Bootstrap 迭代次数，默认 1000 |
| `seed` | 随机种子，默认 42 |

**返回值**：元组列表，每个元组包含 (均值, 标准差)

**使用示例**：
```python
>>> data = [1, 2, 3, 4, 5]
>>> reduce_fns = [np.mean, np.max]
>>> bootstrap_metric(data, 3, reduce_fns)
[(3.0, 0.5), (4.5, 0.3)]  # 示例值
```

**实现流程**：
1. 设置随机种子
2. 进行 n_bootstrap 次迭代：
   - 有放回地随机采样 subset_size 个数据点
   - 对每个归约函数计算结果
3. 计算每个归约函数结果的均值和标准差

---

## 七、多数投票函数

### 7.1 calc_maj_val

```python
def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
```

**功能**：基于多数投票计算值

**参数**：
| 参数 | 描述 |
|------|------|
| `data` | 字典列表，每个字典包含投票键和值键 |
| `vote_key` | 用于投票的键 |
| `val_key` | 返回值对应的键 |

**返回值**：多数票对应的第一个值

**使用示例**：
```python
>>> data = [
...     {"pred": "A", "val": 0.9},
...     {"pred": "B", "val": 0.8},
...     {"pred": "A", "val": 0.7}
... ]
>>> calc_maj_val(data, vote_key="pred", val_key="val")
0.9  # 返回多数票 "A" 对应的第一个值
```

**实现逻辑**：
1. 按 vote_key 分组收集 val_key 的值
2. 找到出现次数最多的 vote
3. 返回该组的第一个 val

---

## 八、验证指标处理

### 8.1 process_validation_metrics

```python
def process_validation_metrics(
    data_sources: list[str],
    sample_uids: list[str],
    infos_dict: dict[str, list[Any]],
    seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
```

**功能**：处理验证指标，生成结构化的统计分析结果

**参数**：
| 参数 | 描述 |
|------|------|
| `data_sources` | 每个样本的数据源标识符列表 |
| `sample_uids` | 每个样本的 UID 列表 |
| `infos_dict` | 变量名到值列表的字典 |
| `seed` | Bootstrap 采样的随机种子，默认 42 |

**返回值**：嵌套字典结构：
```python
{
    data_source: {
        variable_name: {
            metric_name: value
        }
    }
}
```

**生成的指标类型**：
| 指标名格式 | 描述 |
|------------|------|
| `mean@N` | N 个样本的均值 |
| `std@N` | N 个样本的标准差 |
| `best@N/mean` | Bootstrap 采样中最佳值的均值 |
| `best@N/std` | Bootstrap 采样中最佳值的标准差 |
| `worst@N/mean` | Bootstrap 采样中最差值的均值 |
| `worst@N/std` | Bootstrap 采样中最差值的标准差 |
| `maj@N/mean` | 多数投票结果的均值（若存在 "pred" 键） |
| `maj@N/std` | 多数投票结果的标准差 |

**处理流程**：

1. **按数据源、UID 和变量分组**：
```python
data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for sample_idx, data_source in enumerate(data_sources):
    uid = sample_uids[sample_idx]
    var2vals = data_src2uid2var2vals[data_source][uid]
    for var_name, var_vals in infos_dict.items():
        var2vals[var_name].append(var_vals[sample_idx])
```

2. **计算每组的指标**：
```python
for var_name, var_vals in var2vals.items():
    n_resps = len(var_vals)
    metric[f"mean@{n_resps}"] = np.mean(var_vals)

    if n_resps > 1:
        metric[f"std@{n_resps}"] = np.std(var_vals)
        # Bootstrap 采样计算 best@N, worst@N, maj@N
```

3. **跨 UID 聚合**：
```python
# 对每个 data_source 的所有 UID 的指标取均值
for metric_name, uid_vals in metric2uid_vals.items():
    data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)
```

**N 值的选择**：
- 从 2 开始，每次翻倍，直到达到实际响应数
- 例如：若有 16 个响应，N 取值为 [2, 4, 8, 16]

---

## 使用示例

### 计算数据指标

```python
from verl.trainer.ppo.metric_utils import compute_data_metrics
from verl import DataProto

batch = DataProto.from_single_dict(batch_dict)
metrics = compute_data_metrics(batch, use_critic=True)

print(f"Score mean: {metrics['critic/score/mean']}")
print(f"Response length mean: {metrics['response_length/mean']}")
```

### 计算时间指标

```python
from verl.trainer.ppo.metric_utils import compute_timing_metrics

timing_raw = {
    "gen": 10.5,
    "ref": 3.2,
    "values": 2.1,
    "update_actor": 8.3,
    "step": 25.0
}

metrics = compute_timing_metrics(batch, timing_raw)
print(f"Generation time per token: {metrics['timing_per_token_ms/gen']} ms")
```

### 处理验证指标

```python
from verl.trainer.ppo.metric_utils import process_validation_metrics

data_sources = ["gsm8k", "gsm8k", "math"]
sample_uids = ["q1", "q1", "q2"]
infos_dict = {
    "acc": [1.0, 0.0, 1.0],
    "pred": ["42", "43", "100"]
}

results = process_validation_metrics(data_sources, sample_uids, infos_dict)
print(f"GSM8K acc mean@2: {results['gsm8k']['acc']['mean@2']}")
```

---

## 总结

`metric_utils.py` 提供了 PPO 训练过程中完整的指标计算功能：

1. **数据指标**：分数、奖励、优势、回报、价值等统计量
2. **时间指标**：各阶段的执行时间和每 token 时间
3. **吞吐量指标**：token 处理速度
4. **验证指标**：带 Bootstrap 采样的统计分析，支持 best@N、maj@N 等高级指标

这些指标对于监控训练过程、诊断问题和评估模型性能至关重要。
