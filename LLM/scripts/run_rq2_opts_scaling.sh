#!/usr/bin/env bash
# RQ2: OPTS tree-count scaling generation (single node).
# Merges actor + critic FSDP checkpoints to HF, then runs
# experiments/rq2_opts_scaling.py (reward mode + zero OTRC baseline,
# terminates after trees_per_prompt * len(test) trees are opened).
# Idempotent: merge is skipped if the HF dir has weights, generation is
# skipped if the parquet already exists.
#
# Run from anywhere:  bash scripts/run_rq2_opts_scaling.sh
# Knobs (env-overridable): OPTS_METHOD STEP MODEL_SIZE BS TREES_PER_PROMPT
#   MAX_SEARCH_PER_TREE CUDA_VISIBLE_DEVICES
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

MODEL_SIZE=${MODEL_SIZE:-1.7B}
STEP=${STEP:-400}
OPTS_METHOD=${OPTS_METHOD:-opts_ttpo_exp8_3_0810_n8}
CKPT_NAME="${OPTS_METHOD}_${MODEL_SIZE}"
CKPT_ROOT="/share/lujunyu/ckpts/opts_ckpts/opts_ttpo_${MODEL_SIZE}"
DATA_PATH=data/test.parquet

BS=${BS:-902}
TREES_PER_PROMPT=${TREES_PER_PROMPT:-32}
MAX_SEARCH_PER_TREE=${MAX_SEARCH_PER_TREE:-7}

OUT_ROOT="results/step${STEP}"
MERGED_ROOT="${OUT_ROOT}/merged"
RQ2_ROOT="${OUT_ROOT}/rq2"
LOG_ROOT="logs/step${STEP}"
mkdir -p "${MERGED_ROOT}" "${RQ2_ROOT}" "${LOG_ROOT}"

src_actor="${CKPT_ROOT}/${CKPT_NAME}/global_step_${STEP}/actor"
src_critic="${CKPT_ROOT}/${CKPT_NAME}/global_step_${STEP}/critic"
dst_actor="${MERGED_ROOT}/opts_ttpo_actor"
dst_critic="${MERGED_ROOT}/opts_ttpo_critic"
# e.g. opts_ttpo_exp8_3_0810_n8_rq2_opts_s7_t32_bs902.parquet
GEN_BASENAME="${OPTS_METHOD}_rq2_opts_s${MAX_SEARCH_PER_TREE}_t${TREES_PER_PROMPT}_bs${BS}"
output_path="${RQ2_ROOT}/${GEN_BASENAME}.parquet"

for pair in "${src_actor}:${dst_actor}" "${src_critic}:${dst_critic}"; do
    src="${pair%:*}"
    dst="${pair#*:}"
    if [[ -f "${dst}/config.json" ]] && ls "${dst}"/*.safetensors >/dev/null 2>&1; then
        echo "[skip merge] ${dst}"
        continue
    fi
    echo "[merge] ${src}  ->  ${dst}"
    python3 -m verl.model_merger merge --backend fsdp --local_dir "${src}" --target_dir "${dst}"
done

if [[ -f "${output_path}" ]]; then
    echo "[skip gen] ${output_path}"
    exit 0
fi

s=$(date +%s.%N)
python3 -m experiments.rq2_opts_scaling \
 trainer.nnodes=1 \
 trainer.n_gpus_per_node=${N_GPUS} \
 data.val_files="${DATA_PATH}" \
 data.prompt_key=prompt \
 data.val_batch_size=${BS} \
 +data.trees_per_prompt=${TREES_PER_PROMPT} \
 +data.output_path="${output_path}" \
 actor_rollout_ref.model.path="${dst_actor}" \
 critic.model.path="${dst_critic}" \
 critic.model.use_remove_padding=True \
 critic.value_head_activation=sigmoid \
 critic.forward_micro_batch_size_per_gpu=64 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.search=opts \
 actor_rollout_ref.rollout.load_format=auto \
 actor_rollout_ref.rollout.max_search_per_tree=${MAX_SEARCH_PER_TREE} \
 actor_rollout_ref.rollout.temperature=1.0 \
 actor_rollout_ref.rollout.top_p=0.95 \
 actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
 actor_rollout_ref.rollout.prompt_length=1024 \
 actor_rollout_ref.rollout.response_length=2048 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
 actor_rollout_ref.rollout.mode=async \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
 actor_rollout_ref.rollout.max_num_batched_tokens=262144 \
 reward_model.use_reward_loop=False \
 algorithm.lam=0.999 \
 2>&1 | tee "${LOG_ROOT}/${GEN_BASENAME}.log"
e=$(date +%s.%N)
awk -v s="${s}" -v e="${e}" -v tag="${GEN_BASENAME}" \
    'BEGIN{ printf "%s elapsed_seconds=%.2f\n", tag, e-s }' | tee "${LOG_ROOT}/${GEN_BASENAME}.time" || true

echo "Done. Output: ${output_path}"
