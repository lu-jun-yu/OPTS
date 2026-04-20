#!/usr/bin/env bash
# Generate N_SAMPLES i.i.d. responses per prompt for DAPO / GPG / PPO /
# REINFORCE++ / OPTS-TTPO step-300 checkpoints, on a Ray cluster that can be
# single-node or two-machine single-GPU (mirrors scripts/run_opts_ttpo.sh).
#
# Usage:
#   MODE=local  bash scripts/run_parallel_generation.sh      # single node
#   MODE=head   RAY_HEAD_ADDR=<head_ip> bash scripts/run_parallel_generation.sh
#   MODE=worker RAY_HEAD_ADDR=<head_ip> bash scripts/run_parallel_generation.sh
#   MODE=train  NNODES=2 GPUS_PER_NODE=1 RAY_HEAD_ADDR=<head_ip> \
#       bash scripts/run_parallel_generation.sh               # drives the cluster
#
# Each method's FSDP actor is merged to HF on the node that runs MODE=train/local
# (merge output path must be visible to all Ray workers — use shared storage for
# a two-node run). Idempotent: merge is skipped if the HF dir has weights and
# generation is skipped if the parquet already exists.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${LLM_DIR}"

export NCCL_DEBUG="${NCCL_DEBUG:-ERROR}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARN}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

# ---- generation / checkpoint knobs ----
MODEL_SIZE="${MODEL_SIZE:-1.7B}"
STEP="${STEP:-300}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints/opts_ttpo_${MODEL_SIZE}}"
DATA_PATH="${DATA_PATH:-data/test.parquet}"

N_SAMPLES="${N_SAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
PROMPT_LENGTH="${PROMPT_LENGTH:-1024}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-2048}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
ROLLOUT_MODE="${ROLLOUT_MODE:-sync}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-262144}"

# ---- Ray cluster knobs (same shape as run_opts_ttpo.sh) ----
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

# Export so the Python bootstrap spawned by start_ray_node_with_python inherits
# them (it reads RAY_HEAD_ADDR/PORT/... via os.environ).
export MODE NNODES GPUS_PER_NODE
export RAY_HEAD_ADDR RAY_HEAD_PORT RAY_DASHBOARD_PORT
export RAY_NODE_IP_ADDRESS RAY_NUM_CPUS RAY_TEMP_DIR
export RAY_OBJECT_MANAGER_PORT RAY_NODE_MANAGER_PORT
export RAY_MIN_WORKER_PORT RAY_MAX_WORKER_PORT RAY_WORKER_PORT_LIST
export RAY_DASHBOARD_AGENT_LISTEN_PORT RAY_DASHBOARD_AGENT_GRPC_PORT
export RAY_RUNTIME_ENV_AGENT_PORT RAY_METRICS_EXPORT_PORT
export RAY_START_EXTRA_ARGS RAY_ADDRESS

OUT_ROOT="${OUT_ROOT:-outputs/step${STEP}}"
MERGED_ROOT="${MERGED_ROOT:-${OUT_ROOT}/merged}"
GEN_ROOT="${GEN_ROOT:-${OUT_ROOT}/gen}"
LOG_ROOT="${LOG_ROOT:-logs/step${STEP}}"
mkdir -p "${MERGED_ROOT}" "${GEN_ROOT}" "${LOG_ROOT}"

# shellcheck source=scripts/_ray_cluster.sh
source "${SCRIPT_DIR}/_ray_cluster.sh"

print_usage() {
    cat <<EOF
Usage:
  MODE=local  bash scripts/run_parallel_generation.sh
  MODE=head   RAY_HEAD_ADDR=<head_ip> bash scripts/run_parallel_generation.sh
  MODE=worker RAY_HEAD_ADDR=<head_ip> bash scripts/run_parallel_generation.sh
  MODE=train  NNODES=2 GPUS_PER_NODE=1 RAY_HEAD_ADDR=<head_ip> bash scripts/run_parallel_generation.sh

Notes:
  - local  : 单机运行（merge + generation for all methods）
  - head   : 启动 Ray head 节点
  - worker : 启动 Ray worker 节点并接入 head
  - train  : 在已有 Ray 集群上驱动所有方法的 merge + generation
  - 同 IP 多实例请显式设置 RAY_TEMP_DIR 和各类 Ray 端口，避免端口冲突
  - 多机运行时 ${MERGED_ROOT} 与 ${GEN_ROOT} 需要落在共享存储
EOF
}

declare -A CKPT_NAMES=(
    [dapo]="dapo_0326_${MODEL_SIZE}"
    [gpg]="gpg_0326_${MODEL_SIZE}"
    [ppo]="ppo_0326_${MODEL_SIZE}"
    [reinforce_pp]="reinforce_pp_0326_${MODEL_SIZE}"
    [opts_ttpo]="opts_ttpo_0326_${MODEL_SIZE}"
)
DEFAULT_METHODS=("dapo" "gpg" "ppo" "reinforce_pp" "opts_ttpo")
read -r -a METHODS <<< "${METHODS:-${DEFAULT_METHODS[*]}}"

has_hf_weights() {
    local d="$1"
    [[ -f "${d}/config.json" ]] && ls "${d}"/*.safetensors >/dev/null 2>&1
}

# Build TRAIN_CMD for a given method's i.i.d. generation. In train mode we
# append +ray_kwargs.ray_init.address so main_generation connects to the
# existing Ray cluster instead of starting a local one.
build_train_cmd() {
    local method="$1"
    local model_path="$2"
    local output_path="$3"

    TRAIN_CMD=(
        python3 -m verl.trainer.main_generation
        trainer.nnodes=${NNODES}
        trainer.n_gpus_per_node=${GPUS_PER_NODE}
        data.path="${DATA_PATH}"
        data.prompt_key=prompt
        data.batch_size=${BATCH_SIZE}
        data.n_samples=${N_SAMPLES}
        data.output_path="${output_path}"
        model.path="${model_path}"
        rollout.temperature=${TEMPERATURE}
        rollout.top_p=${TOP_P}
        rollout.prompt_length=${PROMPT_LENGTH}
        rollout.response_length=${RESPONSE_LENGTH}
        rollout.tensor_model_parallel_size=${TP_SIZE}
        +rollout.pipeline_model_parallel_size=${PP_SIZE}
        rollout.mode=${ROLLOUT_MODE}
        rollout.gpu_memory_utilization=${GPU_MEM_UTIL}
        rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
    )
    if [[ -n "${RAY_NUM_CPUS}" ]]; then
        TRAIN_CMD+=("ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS}")
    fi
    if [[ "${MODE}" == "train" ]]; then
        TRAIN_CMD+=("+ray_kwargs.ray_init.address=${RAY_ADDRESS}")
    fi
}

run_all_methods() {
    for method in "${METHODS[@]}"; do
        local ckpt_name="${CKPT_NAMES[${method}]:-}"
        if [[ -z "${ckpt_name}" ]]; then
            echo "[skip] unknown method: ${method}" >&2
            continue
        fi
        local src_actor="${CKPT_ROOT}/${ckpt_name}/global_step_${STEP}/actor"
        local dst_actor="${MERGED_ROOT}/${method}_actor"
        local output_path="${GEN_ROOT}/${method}_iid_n${N_SAMPLES}.parquet"
        local time_path="${LOG_ROOT}/${method}_iid.time"

        echo "========== ${method}  (${ckpt_name}) =========="
        if has_hf_weights "${dst_actor}"; then
            echo "[skip merge] ${dst_actor}"
        else
            if [[ ! -f "${src_actor}/fsdp_config.json" ]]; then
                echo "[err] missing FSDP checkpoint: ${src_actor}" >&2
                continue
            fi
            echo "[merge] ${src_actor}  ->  ${dst_actor}"
            python3 -m verl.model_merger merge --backend fsdp \
                --local_dir "${src_actor}" --target_dir "${dst_actor}"
        fi

        if [[ -f "${output_path}" ]]; then
            echo "[skip gen] ${output_path}"
            continue
        fi

        build_train_cmd "${method}" "${dst_actor}" "${output_path}"
        RUN_LOG="${LOG_ROOT}/${method}_iid.log"
        echo "[gen iid] ${method}  ->  ${output_path}"
        local s
        s=$(date +%s.%N)
        run_training
        local e
        e=$(date +%s.%N)
        awk -v s="${s}" -v e="${e}" -v m="${method}" \
            'BEGIN{ printf "%s_iid elapsed_seconds=%.2f\n", m, e-s }' | tee "${time_path}"
    done

    echo "Done. Parquets:"
    ls -1 "${GEN_ROOT}"/*_iid_n${N_SAMPLES}.parquet 2>/dev/null || true
}

case "${MODE}" in
    local)
        run_all_methods
        ;;
    head)
        start_ray_head
        ;;
    worker)
        start_ray_worker
        ;;
    train)
        run_all_methods
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
