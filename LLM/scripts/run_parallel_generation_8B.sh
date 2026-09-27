#!/usr/bin/env bash
# Generate N_SAMPLES i.i.d. responses per prompt for the 8B DAPO / PPO /
# REINFORCE++ / OPTS-TTPO checkpoints (single node).
# Merges each method's FSDP actor to HF, then runs verl.trainer.main_generation.
# Idempotent: merge is skipped if the HF dir has weights, generation is
# skipped if the parquet already exists.
#
# Run from anywhere:  bash scripts/run_parallel_generation_8B.sh
# To change settings (checkpoints, budget, methods), edit the variables below.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export NCCL_DEBUG=ERROR
export TRANSFORMERS_VERBOSITY=error
export VLLM_LOGGING_LEVEL=WARN
# Respect user-supplied GPU selection; default to 2 GPUs. GPU parallelism
# (n_gpus_per_node) is auto-derived from CUDA_VISIBLE_DEVICES below; TP stays
# 1, so extra GPUs simply add data-parallel replicas.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
IFS=',' read -ra _gpu_arr <<< "${CUDA_VISIBLE_DEVICES}"
N_GPUS=${#_gpu_arr[@]}
unset _gpu_arr

# Checkpoint selection is centralized in scripts/run_eval_8B.sh, which exports
# MODEL_SIZE / STEP / METHODS; the defaults below match it so this script
# also works standalone.
MODEL_SIZE=${MODEL_SIZE:-8B}
STEP=${STEP:-400}
CKPT_ROOT="${OPTS_CHECKPOINT_ROOT:-checkpoints}/opts_ttpo_${MODEL_SIZE}"
DATA_PATH=data/test.parquet
N_SAMPLES=128
TOP_K=${TOP_K:--1}
METHODS=${METHODS:-"dapo_0805_n8 ppo_0805_n8 reinforce_pp_baseline_0805_n8 opts_ttpo_exp8_3_0810_n8"}

OUT_ROOT="results/${MODEL_SIZE}/step${STEP}"
MERGED_ROOT="${OUT_ROOT}/merged"
GEN_ROOT="${OUT_ROOT}/gen"
LOG_ROOT="logs/${MODEL_SIZE}/step${STEP}"
mkdir -p "${MERGED_ROOT}" "${GEN_ROOT}" "${LOG_ROOT}"

for method in ${METHODS}; do
    src_actor="${CKPT_ROOT}/${method}_${MODEL_SIZE}/global_step_${STEP}/actor"
    dst_actor="${MERGED_ROOT}/${method}_actor"
    output_path="${GEN_ROOT}/${method}_iid_topk${TOP_K}_n${N_SAMPLES}.parquet"

    echo "========== ${method} =========="
    if [[ -f "${dst_actor}/config.json" ]] && ls "${dst_actor}"/*.safetensors >/dev/null 2>&1; then
        echo "[skip merge] ${dst_actor}"
    else
        echo "[merge] ${src_actor}  ->  ${dst_actor}"
        python3 -m verl.model_merger merge --backend fsdp --local_dir "${src_actor}" --target_dir "${dst_actor}"
    fi

    if [[ -f "${output_path}" ]]; then
        echo "[skip gen] ${output_path}"
        continue
    fi

    s=$(date +%s.%N)
    python3 -m verl.trainer.main_generation \
     trainer.nnodes=1 \
     trainer.n_gpus_per_node=${N_GPUS} \
     data.path="${DATA_PATH}" \
     data.prompt_key=prompt \
     data.batch_size=1024 \
     data.n_samples=${N_SAMPLES} \
     data.output_path="${output_path}" \
     model.path="${dst_actor}" \
     rollout.temperature=1.0 \
     rollout.top_k=${TOP_K} \
     rollout.top_p=0.95 \
     rollout.prompt_length=1024 \
     rollout.response_length=2048 \
     rollout.tensor_model_parallel_size=1 \
     +rollout.pipeline_model_parallel_size=1 \
     rollout.mode=sync \
     rollout.gpu_memory_utilization=0.7 \
     rollout.max_num_batched_tokens=262144 \
     2>&1 | tee "${LOG_ROOT}/${method}_iid.log"
    e=$(date +%s.%N)
    awk -v s="${s}" -v e="${e}" -v m="${method}" \
        'BEGIN{ printf "%s_iid elapsed_seconds=%.2f\n", m, e-s }' | tee "${LOG_ROOT}/${method}_iid.time" || true
done

echo "Done. Parquets:"
ls -1 "${GEN_ROOT}"/*_iid_topk${TOP_K}_n${N_SAMPLES}.parquet 2>/dev/null || true
