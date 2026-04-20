#!/usr/bin/env bash
# OPTS tree-search generation for the OPTS-TTPO step-300 checkpoint, on a Ray
# cluster that can be single-node or two-machine single-GPU (mirrors
# scripts/run_opts_ttpo.sh).
#
# Usage:
#   MODE=local  bash scripts/run_opts_generation.sh
#   MODE=head   RAY_HEAD_ADDR=<head_ip> bash scripts/run_opts_generation.sh
#   MODE=worker RAY_HEAD_ADDR=<head_ip> bash scripts/run_opts_generation.sh
#   MODE=train  NNODES=2 GPUS_PER_NODE=1 RAY_HEAD_ADDR=<head_ip> \
#       bash scripts/run_opts_generation.sh
#
# Merges actor + critic FSDP checkpoints to HF (on the driver node), then runs
# trainer.main_opts_generation with n_samples=128 across the Ray cluster. The
# output parquet carries responses + sample_indices + global_indices (the
# chronological counter that main_eval.py uses for opts@k). Different
# reward_mode values are written to different parquet/log/time files.
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
CKPT_NAME="${CKPT_NAME:-opts_ttpo_0326_${MODEL_SIZE}}"
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
CRITIC_FWD_BSZ_PER_GPU="${CRITIC_FWD_BSZ_PER_GPU:-8}"
TUCT_C="${TUCT_C:-1.0}"
MAX_SEARCH_PER_TREE="${MAX_SEARCH_PER_TREE:-4}"
LAM="${LAM:-0.999}"
REWARD_MODE="${REWARD_MODE:-reward}"
OPTS_GEN_TAG="${OPTS_GEN_TAG:-${REWARD_MODE}}"

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
  MODE=local  bash scripts/run_opts_generation.sh
  MODE=head   RAY_HEAD_ADDR=<head_ip> bash scripts/run_opts_generation.sh
  MODE=worker RAY_HEAD_ADDR=<head_ip> bash scripts/run_opts_generation.sh
  MODE=train  NNODES=2 GPUS_PER_NODE=1 RAY_HEAD_ADDR=<head_ip> bash scripts/run_opts_generation.sh

Notes:
  - local  : 单机运行
  - head   : 启动 Ray head 节点
  - worker : 启动 Ray worker 节点并接入 head
  - train  : 在已有 Ray 集群上驱动 merge + OPTS generation
  - 多机运行时 ${MERGED_ROOT} 与 ${GEN_ROOT} 需要落在共享存储
EOF
}

src_actor="${CKPT_ROOT}/${CKPT_NAME}/global_step_${STEP}/actor"
src_critic="${CKPT_ROOT}/${CKPT_NAME}/global_step_${STEP}/critic"
dst_actor="${MERGED_ROOT}/opts_ttpo_actor"
dst_critic="${MERGED_ROOT}/opts_ttpo_critic"
output_path="${GEN_ROOT}/opts_ttpo_opts_${OPTS_GEN_TAG}_n${N_SAMPLES}.parquet"
time_path="${LOG_ROOT}/opts_ttpo_opts_${OPTS_GEN_TAG}.time"

has_hf_weights() {
    local d="$1"
    [[ -f "${d}/config.json" ]] && ls "${d}"/*.safetensors >/dev/null 2>&1
}

build_train_cmd() {
    TRAIN_CMD=(
        python3 -m trainer.main_opts_generation
        trainer.nnodes=${NNODES}
        trainer.n_gpus_per_node=${GPUS_PER_NODE}
        data.val_files="${DATA_PATH}"
        data.prompt_key=prompt
        data.val_batch_size=${BATCH_SIZE}
        +data.n_samples=${N_SAMPLES}
        +data.reward_mode=${REWARD_MODE}
        +data.output_path="${output_path}"
        actor_rollout_ref.model.path="${dst_actor}"
        critic.model.path="${dst_critic}"
        critic.forward_micro_batch_size_per_gpu=${CRITIC_FWD_BSZ_PER_GPU}
        actor_rollout_ref.rollout.name=vllm
        actor_rollout_ref.rollout.load_format=auto
        actor_rollout_ref.rollout.enforce_eager=True
        actor_rollout_ref.rollout.c=${TUCT_C}
        actor_rollout_ref.rollout.max_search_per_tree=${MAX_SEARCH_PER_TREE}
        actor_rollout_ref.rollout.temperature=${TEMPERATURE}
        actor_rollout_ref.rollout.top_p=${TOP_P}
        actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH}
        actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH}
        actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}
        actor_rollout_ref.rollout.pipeline_model_parallel_size=${PP_SIZE}
        actor_rollout_ref.rollout.mode=${ROLLOUT_MODE}
        actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}
        actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
        algorithm.lam=${LAM}
    )
    if [[ -n "${RAY_NUM_CPUS}" ]]; then
        TRAIN_CMD+=("ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS}")
    fi
    if [[ "${MODE}" == "train" ]]; then
        TRAIN_CMD+=("+ray_kwargs.ray_init.address=${RAY_ADDRESS}")
    fi
}

run_opts_pipeline() {
    for pair in "${src_actor}:${dst_actor}" "${src_critic}:${dst_critic}"; do
        local src="${pair%:*}" dst="${pair#*:}"
        if has_hf_weights "${dst}"; then
            echo "[skip merge] ${dst}"
            continue
        fi
        if [[ ! -f "${src}/fsdp_config.json" ]]; then
            echo "[err] missing FSDP checkpoint: ${src}" >&2
            exit 1
        fi
        echo "[merge] ${src}  ->  ${dst}"
        python3 -m verl.model_merger merge --backend fsdp \
            --local_dir "${src}" --target_dir "${dst}"
    done

    if [[ -f "${output_path}" ]]; then
        echo "[skip gen] ${output_path}"
        return 0
    fi

    build_train_cmd
    RUN_LOG="${LOG_ROOT}/opts_ttpo_opts_${OPTS_GEN_TAG}.log"
    echo "[gen opts:${REWARD_MODE}] opts_ttpo  ->  ${output_path}"
    local s
    s=$(date +%s.%N)
    run_training
    local e
    e=$(date +%s.%N)
    awk -v s="${s}" -v e="${e}" -v tag="${OPTS_GEN_TAG}" \
        'BEGIN{ printf "opts_ttpo_opts_%s elapsed_seconds=%.2f\n", tag, e-s }' | tee "${time_path}"

    echo "Done. Output: ${output_path}"
}

case "${MODE}" in
    local)
        run_opts_pipeline
        ;;
    head)
        start_ray_head
        ;;
    worker)
        start_ray_worker
        ;;
    train)
        run_opts_pipeline
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
