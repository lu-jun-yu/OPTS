# `reward.py` 文件详解

## 文件概述

`reward.py` 是 verl PPO 训练器的奖励计算模块。该文件提供了加载自定义奖励函数、奖励管理器以及计算奖励的功能，支持同步和异步奖励计算。

## 文件路径

```
LLM/verl/verl/trainer/ppo/reward.py
```

## 模块导入

```python
import importlib.util
import inspect
import multiprocessing
import os
import sys
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, cast

import ray
import torch

from verl.utils.reward_score import default_compute_score
from verl.utils.transferqueue_utils import tqbridge
```

---

## 一、辅助函数

### 1.1 _call_with_kwargs

```python
def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
```

**功能**：调用函数时合并额外的关键字参数

**参数**：
| 参数 | 描述 |
|------|------|
| `raw_fn` | 原始函数 |
| `extra_kwargs` | 要合并的额外关键字参数 |
| `*args` | 位置参数 |
| `**kwargs` | 调用时的关键字参数 |

**返回值**：函数调用结果

**实现逻辑**：
```python
merged_kwargs = {**kwargs, **extra_kwargs}  # extra_kwargs 优先级更高
return raw_fn(*args, **merged_kwargs)
```

### 1.2 _call_with_kwargs_async

```python
async def _call_with_kwargs_async(raw_fn, extra_kwargs, *args, **kwargs):
```

**功能**：异步版本的 `_call_with_kwargs`

**特点**：
- 使用 `async/await` 语法
- 支持异步奖励函数

---

## 二、自定义奖励函数加载

### 2.1 get_custom_reward_fn

```python
def get_custom_reward_fn(config: DictConfig) -> Optional[RawRewardFn]:
```

**功能**：从外部文件动态加载自定义奖励函数

**参数**：
- `config`: 配置字典，包含 `custom_reward_function` 设置

**配置结构**：
```yaml
custom_reward_function:
  path: "/path/to/reward_function.py"
  name: "compute_reward"
  reward_kwargs:
    key1: value1
    key2: value2
```

**返回值**：包装后的奖励函数，或 `None`（如果未配置）

**异常**：
- `FileNotFoundError`: 指定的文件不存在
- `RuntimeError`: 加载模块时出错
- `AttributeError`: 指定的函数名不存在

**实现流程**：

1. **检查配置**：
```python
reward_fn_config = config.get("custom_reward_function") or {}
file_path = reward_fn_config.get("path")
if not file_path:
    return None
```

2. **动态导入模块**：
```python
spec = importlib.util.spec_from_file_location("custom_module", file_path)
module = importlib.util.module_from_spec(spec)
sys.modules["custom_module"] = module
spec.loader.exec_module(module)
```

3. **获取函数并包装**：
```python
raw_fn = getattr(module, function_name)
reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

# 根据函数类型选择包装方式
if not inspect.iscoroutinefunction(raw_fn):
    return partial(_call_with_kwargs, raw_fn, reward_kwargs)
else:
    return partial(_call_with_kwargs_async, raw_fn, reward_kwargs)
```

---

## 三、奖励管理器加载

### 3.1 load_reward_manager

```python
def load_reward_manager(
    config: DictConfig, tokenizer: Any, num_examine: int, **reward_kwargs: Any
) -> AbstractRewardManager:
```

**功能**：根据配置加载并初始化奖励管理器

**参数**：
| 参数 | 描述 |
|------|------|
| `config` | PPO 训练器配置，包含 `reward_model` 字段 |
| `tokenizer` | 分词器对象 |
| `num_examine` | 要检查的样本数量 |
| `**reward_kwargs` | 传递给奖励管理器的额外参数 |

**返回值**：奖励管理器实例

**支持的奖励管理器来源**：

| source 值 | 描述 |
|-----------|------|
| `register` | 从注册表获取 |
| `importlib` | 从外部模块导入 |

**加载流程**：

1. **尝试获取自定义奖励函数**：
```python
compute_score = get_custom_reward_fn(config)
final_compute_score = compute_score
```

2. **根据 source 加载奖励管理器类**：
```python
if reward_manager_cfg.source == "register":
    from verl.workers.reward_manager import get_reward_manager_cls
    reward_manager_cls = get_reward_manager_cls(reward_manager_cfg.name)

elif reward_manager_cfg.source == "importlib":
    from verl.utils.import_utils import load_extern_object
    reward_manager_cls = load_extern_object(module_path=module_cfg.path, object_name=reward_manager_cls_name)
```

3. **设置默认 compute_score（如果未提供自定义）**：
```python
if compute_score is None:
    sandbox_config = config.reward_model.get("sandbox_fusion")
    sandbox_url = sandbox_config.get("url") if sandbox_config else None

    if sandbox_url:
        # 使用沙箱执行
        _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
        final_compute_score = partial(
            default_compute_score,
            sandbox_fusion_url=sandbox_url,
            concurrent_semaphore=_concurrent_semaphore,
            memory_limit_mb=memory_limit_mb,
        )
    else:
        final_compute_score = default_compute_score
```

4. **实例化奖励管理器**：
```python
# RewardLoopManagerBase 子类
if issubclass(reward_manager_cls, RewardLoopManagerBase):
    return reward_manager_cls(
        config=config,
        tokenizer=tokenizer,
        compute_score=final_compute_score,
        **reward_kwargs,
    )

# AbstractRewardManager 子类
else:
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )
```

---

## 四、奖励计算函数

### 4.1 compute_reward

```python
@tqbridge(put_data=False)
def compute_reward(data: DataProto, reward_fn: AbstractRewardManager) -> tuple[torch.Tensor, dict[str, Any]]:
```

**功能**：计算批次数据的奖励

**装饰器**：
- `@tqbridge(put_data=False)`: TransferQueue 桥接装饰器，用于进程间数据传输

**参数**：
| 参数 | 描述 |
|------|------|
| `data` | DataProto 对象，包含输入数据 |
| `reward_fn` | 奖励函数/奖励管理器 |

**返回值**：
- `reward_tensor`: 奖励张量
- `reward_extra_infos_dict`: 额外信息字典

**实现逻辑**：
```python
try:
    # 尝试使用返回字典的方式调用
    reward_result = reward_fn(data, return_dict=True)
    reward_tensor = reward_result["reward_tensor"]
    reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
except Exception as e:
    # 回退到简单调用方式
    print(f"Error in reward_fn: {e}")
    reward_tensor = reward_fn(data)
    reward_extra_infos_dict = {}

return reward_tensor, reward_extra_infos_dict
```

### 4.2 compute_reward_async

```python
@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config=None, tokenizer=None, reward_fn=None):
```

**功能**：异步计算奖励（在独立的 Ray worker 中运行）

**装饰器**：
- `@ray.remote(num_cpus=1)`: 作为 Ray 远程任务运行，使用 1 个 CPU

**参数**：
| 参数 | 描述 |
|------|------|
| `data` | DataProto 对象 |
| `config` | 配置（已弃用） |
| `tokenizer` | 分词器（已弃用） |
| `reward_fn` | 奖励函数（推荐） |

**使用方式**：

```python
# 推荐方式：传入 reward_fn
future = compute_reward_async.remote(data=batch, reward_fn=reward_fn)
reward_tensor, extra_info = ray.get(future)

# 已弃用方式：传入 config 和 tokenizer
future = compute_reward_async.remote(data=batch, config=config, tokenizer=tokenizer)
```

**注意事项**：
- 使用 `config` 和 `tokenizer` 的方式已被弃用
- 推荐直接传入预加载的 `reward_fn`

---

## 五、类型定义（TYPE_CHECKING）

```python
if TYPE_CHECKING:
    from omegaconf import DictConfig
    from verl import DataProto
    from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
    from verl.trainer.config.config import ModuleConfig, RewardManagerConfig
    from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn
```

**说明**：这些类型仅用于类型检查，不会在运行时导入

---

## 六、使用示例

### 6.1 自定义奖励函数

创建自定义奖励函数文件 `my_reward.py`：

```python
def my_compute_score(data, custom_param=1.0):
    """自定义奖励计算函数"""
    # data 是 DataProto 对象
    responses = data.batch["responses"]
    # 计算奖励逻辑
    rewards = compute_rewards_logic(responses, custom_param)
    return {
        "reward_tensor": rewards,
        "reward_extra_info": {"custom_metric": some_value}
    }
```

配置文件：

```yaml
custom_reward_function:
  path: "/path/to/my_reward.py"
  name: "my_compute_score"
  reward_kwargs:
    custom_param: 2.0
```

### 6.2 使用奖励管理器

```python
from verl.trainer.ppo.reward import load_reward_manager, compute_reward

# 加载奖励管理器
reward_manager = load_reward_manager(
    config=config,
    tokenizer=tokenizer,
    num_examine=10,
)

# 计算奖励
reward_tensor, extra_info = compute_reward(batch, reward_manager)
```

### 6.3 异步奖励计算

```python
import ray
from verl.trainer.ppo.reward import compute_reward_async

# 启动异步计算
future = compute_reward_async.remote(data=batch, reward_fn=reward_fn)

# 继续其他工作...
do_other_work()

# 获取结果
reward_tensor, extra_info = ray.get(future)
```

---

## 七、沙箱执行配置

对于需要代码执行的奖励计算（如代码正确性评估），可以配置沙箱：

```yaml
reward_model:
  sandbox_fusion:
    url: "http://sandbox-server:8080"
    max_concurrent: 64
    memory_limit_mb: 1024
```

**沙箱特性**：
- 使用信号量控制并发数
- 限制内存使用
- 安全执行用户代码

---

## 八、奖励管理器类型

### 8.1 AbstractRewardManager

传统的奖励管理器，参数包括：
- `tokenizer`
- `num_examine`
- `compute_score`
- `reward_fn_key`

### 8.2 RewardLoopManagerBase

实验性的奖励循环管理器，参数包括：
- `config`
- `tokenizer`
- `compute_score`

**区别**：`RewardLoopManagerBase` 子类不接受 `num_examine` 参数

---

## 总结

`reward.py` 提供了灵活的奖励计算框架：

1. **自定义奖励函数**：支持从外部 Python 文件动态加载
2. **奖励管理器**：支持注册表和动态导入两种加载方式
3. **同步/异步计算**：支持同步计算和 Ray 异步计算
4. **沙箱执行**：支持安全的代码执行环境
5. **类型安全**：使用 TYPE_CHECKING 提供完整的类型提示

该模块是 PPO 训练中奖励信号计算的核心，支持多种奖励函数形式和计算模式。
