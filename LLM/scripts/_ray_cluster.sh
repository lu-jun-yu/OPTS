#!/usr/bin/env bash
# Shared Ray cluster helpers for scripts/run_opts_ttpo.sh-style multi-node launches.
#
# After sourcing, define a script-specific `build_train_cmd` (which populates
# the TRAIN_CMD array) and then dispatch on ${MODE} via `run_training` / the
# `start_ray_head` / `start_ray_worker` helpers.
#
# Variables expected before sourcing:
#   MODE, NNODES, GPUS_PER_NODE, RAY_HEAD_ADDR, RAY_HEAD_PORT,
#   RAY_DASHBOARD_PORT, RAY_* port overrides, RAY_START_EXTRA_ARGS, RAY_ADDRESS.

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

RAY_EXTRA_ARGS=()
if [[ -n "${RAY_START_EXTRA_ARGS:-}" ]]; then
    read -r -a RAY_EXTRA_ARGS <<< "${RAY_START_EXTRA_ARGS}"
fi

append_ray_node_args() {
    if [[ -n "${RAY_NODE_IP_ADDRESS:-}" ]]; then
        RAY_NODE_ARGS+=(--node-ip-address "${RAY_NODE_IP_ADDRESS}")
    fi
    if [[ -n "${RAY_TEMP_DIR:-}" ]]; then
        RAY_NODE_ARGS+=(--temp-dir "${RAY_TEMP_DIR}")
    fi
    if [[ -n "${RAY_OBJECT_MANAGER_PORT:-}" ]]; then
        RAY_NODE_ARGS+=(--object-manager-port "${RAY_OBJECT_MANAGER_PORT}")
    fi
    if [[ -n "${RAY_NODE_MANAGER_PORT:-}" ]]; then
        RAY_NODE_ARGS+=(--node-manager-port "${RAY_NODE_MANAGER_PORT}")
    fi
    if [[ -n "${RAY_MIN_WORKER_PORT:-}" ]]; then
        RAY_NODE_ARGS+=(--min-worker-port "${RAY_MIN_WORKER_PORT}")
    fi
    if [[ -n "${RAY_MAX_WORKER_PORT:-}" ]]; then
        RAY_NODE_ARGS+=(--max-worker-port "${RAY_MAX_WORKER_PORT}")
    fi
    if [[ -n "${RAY_WORKER_PORT_LIST:-}" ]]; then
        RAY_NODE_ARGS+=(--worker-port-list "${RAY_WORKER_PORT_LIST}")
    fi
    if [[ -n "${RAY_DASHBOARD_AGENT_LISTEN_PORT:-}" ]]; then
        RAY_NODE_ARGS+=(--dashboard-agent-listen-port "${RAY_DASHBOARD_AGENT_LISTEN_PORT}")
    fi
    if [[ -n "${RAY_DASHBOARD_AGENT_GRPC_PORT:-}" ]]; then
        RAY_NODE_ARGS+=(--dashboard-agent-grpc-port "${RAY_DASHBOARD_AGENT_GRPC_PORT}")
    fi
    if [[ -n "${RAY_RUNTIME_ENV_AGENT_PORT:-}" ]]; then
        RAY_NODE_ARGS+=(--runtime-env-agent-port "${RAY_RUNTIME_ENV_AGENT_PORT}")
    fi
    if [[ -n "${RAY_METRICS_EXPORT_PORT:-}" ]]; then
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

start_ray_head() {
    start_ray_node_with_python head
}

start_ray_worker() {
    start_ray_node_with_python worker
}

# Caller must populate TRAIN_CMD array first and set RUN_LOG (output log path).
run_training() {
    mkdir -p "$(dirname "${RUN_LOG:-logs/run.log}")"
    "${TRAIN_CMD[@]}" 2>&1 | tee "${RUN_LOG:-logs/run.log}"
}
