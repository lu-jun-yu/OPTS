#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${LLM_DIR}"

export NCCL_DEBUG="${NCCL_DEBUG:-ERROR}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARN}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_SERVICE_WAIT="${WANDB_SERVICE_WAIT:-60}"

# Normalize proxy env vars for libraries that only respect specific cases/protocols.
if [[ -z "${https_proxy:-}" ]]; then
    if [[ -n "${HTTPS_PROXY:-}" ]]; then
        export https_proxy="${HTTPS_PROXY}"
    elif [[ -n "${http_proxy:-}" ]]; then
        export https_proxy="${http_proxy}"
    elif [[ -n "${HTTP_PROXY:-}" ]]; then
        export https_proxy="${HTTP_PROXY}"
    elif [[ -n "${all_proxy:-}" ]]; then
        export https_proxy="${all_proxy}"
    elif [[ -n "${ALL_PROXY:-}" ]]; then
        export https_proxy="${ALL_PROXY}"
    fi
fi
if [[ -z "${http_proxy:-}" ]]; then
    if [[ -n "${HTTP_PROXY:-}" ]]; then
        export http_proxy="${HTTP_PROXY}"
    elif [[ -n "${all_proxy:-}" ]]; then
        export http_proxy="${all_proxy}"
    elif [[ -n "${ALL_PROXY:-}" ]]; then
        export http_proxy="${ALL_PROXY}"
    fi
fi
if [[ -z "${all_proxy:-}" ]]; then
    if [[ -n "${ALL_PROXY:-}" ]]; then
        export all_proxy="${ALL_PROXY}"
    elif [[ -n "${http_proxy:-}" ]]; then
        export all_proxy="${http_proxy}"
    elif [[ -n "${https_proxy:-}" ]]; then
        export all_proxy="${https_proxy}"
    fi
fi
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-}}"
export ALL_PROXY="${ALL_PROXY:-${all_proxy:-}}"
export NO_PROXY="${NO_PROXY:-${no_proxy:-}}"
export no_proxy="${no_proxy:-${NO_PROXY:-}}"
export WANDB_PROXY="${WANDB_PROXY:-${HTTPS_PROXY:-${https_proxy:-}}}"

MODEL_SIZE="${MODEL_SIZE:-1.7B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opts_ttpo_0419_${MODEL_SIZE}}"
MODE="${MODE:-local}"

NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"

RAY_HEAD_ADDR="${RAY_HEAD_ADDR:-127.0.0.1}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_NODE_IP_ADDRESS="${RAY_NODE_IP_ADDRESS:-}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-}"
RAY_OBJECT_MANAGER_PORT="${RAY_OBJECT_MANAGER_PORT:-}"
RAY_NODE_MANAGER_PORT="${RAY_NODE_MANAGER_PORT:-}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-}"
RAY_WORKER_PORT_LIST="${RAY_WORKER_PORT_LIST:-}"
RAY_DASHBOARD_AGENT_LISTEN_PORT="${RAY_DASHBOARD_AGENT_LISTEN_PORT:-}"
RAY_DASHBOARD_AGENT_GRPC_PORT="${RAY_DASHBOARD_AGENT_GRPC_PORT:-}"
RAY_RUNTIME_ENV_AGENT_PORT="${RAY_RUNTIME_ENV_AGENT_PORT:-}"
RAY_METRICS_EXPORT_PORT="${RAY_METRICS_EXPORT_PORT:-}"
RAY_START_EXTRA_ARGS="${RAY_START_EXTRA_ARGS:-}"
RAY_ADDRESS="${RAY_HEAD_ADDR}:${RAY_HEAD_PORT}"

mkdir -p logs

RAY_EXTRA_ARGS=()
if [[ -n "${RAY_START_EXTRA_ARGS}" ]]; then
    read -r -a RAY_EXTRA_ARGS <<< "${RAY_START_EXTRA_ARGS}"
fi

print_usage() {
    cat <<EOF
Usage:
  MODE=local  bash scripts/run_opts_ttpo.sh
  MODE=head   RAY_HEAD_ADDR=<head_ip> bash scripts/run_opts_ttpo.sh
  MODE=worker RAY_HEAD_ADDR=<head_ip> bash scripts/run_opts_ttpo.sh
  MODE=train  NNODES=2 GPUS_PER_NODE=1 RAY_HEAD_ADDR=<head_ip> bash scripts/run_opts_ttpo.sh

Notes:
  - local: 兼容原来的单机单卡训练
  - head:  启动 Ray head 节点
  - worker: 启动 Ray worker 节点并接入 head
  - train: 在已有 Ray 集群上发起训练
  - 同 IP 多实例请显式设置 RAY_TEMP_DIR 和各类 Ray 端口，避免端口冲突
EOF
}

append_ray_node_args() {
    if [[ -n "${RAY_NODE_IP_ADDRESS}" ]]; then
        RAY_NODE_ARGS+=(--node-ip-address "${RAY_NODE_IP_ADDRESS}")
    fi
    if [[ -n "${RAY_TEMP_DIR}" ]]; then
        RAY_NODE_ARGS+=(--temp-dir "${RAY_TEMP_DIR}")
    fi
    if [[ -n "${RAY_OBJECT_MANAGER_PORT}" ]]; then
        RAY_NODE_ARGS+=(--object-manager-port "${RAY_OBJECT_MANAGER_PORT}")
    fi
    if [[ -n "${RAY_NODE_MANAGER_PORT}" ]]; then
        RAY_NODE_ARGS+=(--node-manager-port "${RAY_NODE_MANAGER_PORT}")
    fi
    if [[ -n "${RAY_MIN_WORKER_PORT}" ]]; then
        RAY_NODE_ARGS+=(--min-worker-port "${RAY_MIN_WORKER_PORT}")
    fi
    if [[ -n "${RAY_MAX_WORKER_PORT}" ]]; then
        RAY_NODE_ARGS+=(--max-worker-port "${RAY_MAX_WORKER_PORT}")
    fi
    if [[ -n "${RAY_WORKER_PORT_LIST}" ]]; then
        RAY_NODE_ARGS+=(--worker-port-list "${RAY_WORKER_PORT_LIST}")
    fi
    if [[ -n "${RAY_DASHBOARD_AGENT_LISTEN_PORT}" ]]; then
        RAY_NODE_ARGS+=(--dashboard-agent-listen-port "${RAY_DASHBOARD_AGENT_LISTEN_PORT}")
    fi
    if [[ -n "${RAY_DASHBOARD_AGENT_GRPC_PORT}" ]]; then
        RAY_NODE_ARGS+=(--dashboard-agent-grpc-port "${RAY_DASHBOARD_AGENT_GRPC_PORT}")
    fi
    if [[ -n "${RAY_RUNTIME_ENV_AGENT_PORT}" ]]; then
        RAY_NODE_ARGS+=(--runtime-env-agent-port "${RAY_RUNTIME_ENV_AGENT_PORT}")
    fi
    if [[ -n "${RAY_METRICS_EXPORT_PORT}" ]]; then
        RAY_NODE_ARGS+=(--metrics-export-port "${RAY_METRICS_EXPORT_PORT}")
    fi
    RAY_NODE_ARGS+=("${RAY_EXTRA_ARGS[@]}")
}

start_ray_node_with_python() {
    local node_mode="$1"
    RAY_LAUNCH_MODE="${node_mode}" python3 - <<'PY'
import os

import ray
import ray._private.services as services
from ray._private.node import Node
from ray._private.parameter import RayParams


def env_str(name):
    value = os.environ.get(name, "")
    return value if value else None


def env_int(name):
    value = os.environ.get(name, "")
    return int(value) if value else None


launch_mode = os.environ["RAY_LAUNCH_MODE"]
is_head = launch_mode == "head"

head_addr = os.environ["RAY_HEAD_ADDR"]
head_port = int(os.environ["RAY_HEAD_PORT"])
gcs_address = f"{head_addr}:{head_port}"

ray_params = RayParams(
    node_ip_address=env_str("RAY_NODE_IP_ADDRESS"),
    node_name=env_str("RAY_NODE_IP_ADDRESS"),
    min_worker_port=env_int("RAY_MIN_WORKER_PORT"),
    max_worker_port=env_int("RAY_MAX_WORKER_PORT"),
    worker_port_list=env_str("RAY_WORKER_PORT_LIST"),
    object_manager_port=env_int("RAY_OBJECT_MANAGER_PORT"),
    node_manager_port=env_int("RAY_NODE_MANAGER_PORT") or 0,
    num_cpus=env_int("RAY_NUM_CPUS"),
    num_gpus=env_int("GPUS_PER_NODE"),
    temp_dir=env_str("RAY_TEMP_DIR"),
    include_dashboard=None,
    dashboard_host="0.0.0.0" if is_head else None,
    dashboard_port=env_int("RAY_DASHBOARD_PORT") if is_head else None,
    dashboard_agent_listen_port=env_int("RAY_DASHBOARD_AGENT_LISTEN_PORT"),
    metrics_agent_port=env_int("RAY_DASHBOARD_AGENT_GRPC_PORT"),
    runtime_env_agent_port=env_int("RAY_RUNTIME_ENV_AGENT_PORT"),
    metrics_export_port=env_int("RAY_METRICS_EXPORT_PORT"),
)

if is_head:
    ray_params.gcs_server_port = head_port
    ray_params.update_if_absent(node_ip_address=services.get_node_ip_address())
    node = Node(ray_params, head=True, shutdown_at_exit=False, spawn_reaper=False)
    print(f"Ray head ready at {node.address}")
else:
    bootstrap_address = services.canonicalize_bootstrap_address(
        gcs_address, temp_dir=ray_params.temp_dir
    )
    if bootstrap_address is None:
        raise RuntimeError(f"Cannot canonicalize Ray bootstrap address: {gcs_address}")

    ray_params.gcs_address = bootstrap_address
    ray_params.update_if_absent(node_ip_address=services.get_node_ip_address(bootstrap_address))
    node = Node(ray_params, head=False, shutdown_at_exit=False, spawn_reaper=False)
    node.check_version_info()
    print(f"Ray worker joined {bootstrap_address}")
PY
}

build_train_cmd() {
    TRAIN_CMD=(
        python3 -m trainer.main_opts_ttpo
        algorithm.adv_estimator=treegae
        data.train_files=data/train.parquet
        data.val_files=data/test.parquet
        data.train_batch_size=512
        data.max_prompt_length=1024
        data.max_response_length=2048
        data.filter_overlong_prompts=True
        actor_rollout_ref.model.path="models/Qwen3-${MODEL_SIZE}"
        actor_rollout_ref.actor.optim.lr=1e-6
        actor_rollout_ref.actor.optim.weight_decay=0.1
        actor_rollout_ref.actor.optim.lr_warmup_steps=10
        actor_rollout_ref.actor.ppo_mini_batch_size=512
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16
        actor_rollout_ref.actor.use_kl_loss=False
        actor_rollout_ref.rollout.name=vllm
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128
        actor_rollout_ref.rollout.tensor_model_parallel_size=1
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8
        actor_rollout_ref.rollout.search=opts
        actor_rollout_ref.rollout.n=4
        actor_rollout_ref.rollout.c=1.0
        actor_rollout_ref.rollout.max_search_per_tree=4
        critic.enable=True
        critic.optim.lr=1e-5
        critic.model.path="models/Qwen3-${MODEL_SIZE}"
        critic.ppo_micro_batch_size_per_gpu=32
        custom_reward_function.path=utils/reward_fn.py
        custom_reward_function.name=compute_score
        algorithm.use_kl_in_reward=False
        algorithm.kl_ctrl.kl_coef=0.0
        algorithm.lam=1.0
        trainer.logger='["console","wandb"]'
        trainer.val_before_train=False
        trainer.n_gpus_per_node="${GPUS_PER_NODE}"
        trainer.nnodes="${NNODES}"
        trainer.project_name="opts_ttpo_${MODEL_SIZE}"
        trainer.experiment_name="${EXPERIMENT_NAME}"
        trainer.save_freq=20
        trainer.test_freq=10
        trainer.total_epochs=15
        +trainer.wandb_init_timeout="${WANDB_INIT_TIMEOUT}"
        +trainer.wandb_service_wait="${WANDB_SERVICE_WAIT}"
    )

    if [[ -n "${RAY_NUM_CPUS}" ]]; then
        TRAIN_CMD+=("ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS}")
    fi

    if [[ "${MODE}" == "train" ]]; then
        TRAIN_CMD+=("+ray_kwargs.ray_init.address=${RAY_ADDRESS}")
    fi
}

start_ray_head() {
    start_ray_node_with_python head
}

start_ray_worker() {
    start_ray_node_with_python worker
}

run_training() {
    build_train_cmd
    "${TRAIN_CMD[@]}" 2>&1 | tee "logs/${EXPERIMENT_NAME}.log"
}

case "${MODE}" in
    local)
        run_training
        ;;
    head)
        start_ray_head
        ;;
    worker)
        start_ray_worker
        ;;
    train)
        run_training
        ;;
    help|-h|--help)
        print_usage
        ;;
    *)
        echo "Unknown MODE: ${MODE}" >&2
        print_usage >&2
        exit 1
        ;;
esac
