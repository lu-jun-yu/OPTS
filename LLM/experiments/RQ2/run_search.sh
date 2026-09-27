#!/usr/bin/env bash
# RQ2: OPTS tree-count scaling generation + evaluation (single node).
# For each SEARCH_ROUNDS value: merge actor+critic, run
# experiments/RQ2/search.py, score with trainer.main_eval opts-avg@k.
# A run at the largest s covers all smaller depths: snapshots are taken after
# branch rounds SNAPSHOT_ROUNDS and sliced at eval time by OPTS_AVG_SLICES
# (s0 = roots only, sN>=1 = online snapshot opts_avg_responses_s{N}).
# Idempotent: existing merged weights / parquets are skipped.
#
# Run:  bash experiments/RQ2/run_search.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export NCCL_DEBUG=ERROR
export TRANSFORMERS_VERBOSITY=error
export VLLM_LOGGING_LEVEL=WARN
# n_gpus_per_node is derived from CUDA_VISIBLE_DEVICES; TP=1, extra GPUs add DP replicas.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
IFS=',' read -ra _gpu_arr <<< "${CUDA_VISIBLE_DEVICES}"
N_GPUS=${#_gpu_arr[@]}
unset _gpu_arr

MODEL_SIZE=${MODEL_SIZE:-1.7B}
STEP=${STEP:-400}
OPTS_METHOD=${OPTS_METHOD:-opts_ttpo_exp8_3_0810_n8}
CKPT_NAME="${OPTS_METHOD}_${MODEL_SIZE}"
CKPT_ROOT="${OPTS_CHECKPOINT_ROOT:-checkpoints}/opts_ttpo_${MODEL_SIZE}"
DATA_PATH=data/test.parquet

BS=${BS:-902}
TREES_PER_PROMPT=${TREES_PER_PROMPT:-32}
SEARCH_ROUNDS=${SEARCH_ROUNDS:-"15"}
SNAPSHOT_ROUNDS=${SNAPSHOT_ROUNDS:-"1 3 7 15"}
SNAPSHOT_ROUNDS_LIST="[$(echo ${SNAPSHOT_ROUNDS} | tr ' ' ',')]"
OPTS_AVG_KS=${OPTS_AVG_KS:-"32"}
OPTS_AVG_KS_TAG="${OPTS_AVG_KS// /-}"
OPTS_AVG_SLICES=${OPTS_AVG_SLICES:-"0 1 3 7 15"}
OPTS_AVG_SLICES_TAG="${OPTS_AVG_SLICES// /-}"

OUT_ROOT="results/step${STEP}"
MERGED_ROOT="${OUT_ROOT}/merged"
RQ2_ROOT="${OUT_ROOT}/rq2"
EVAL_ROOT="${RQ2_ROOT}/eval"
LOG_ROOT="logs/step${STEP}"
mkdir -p "${MERGED_ROOT}" "${RQ2_ROOT}" "${EVAL_ROOT}" "${LOG_ROOT}"

src_actor="${CKPT_ROOT}/${CKPT_NAME}/global_step_${STEP}/actor"
src_critic="${CKPT_ROOT}/${CKPT_NAME}/global_step_${STEP}/critic"
dst_actor="${MERGED_ROOT}/opts_ttpo_actor"
dst_critic="${MERGED_ROOT}/opts_ttpo_critic"

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

for s in ${SEARCH_ROUNDS}; do
    # e.g. opts_ttpo_exp8_3_0810_n8_rq2_opts_s7_t32_bs902.parquet
    GEN_BASENAME="${OPTS_METHOD}_rq2_opts_s${s}_t${TREES_PER_PROMPT}_bs${BS}"
    output_path="${RQ2_ROOT}/${GEN_BASENAME}.parquet"

    echo "========== RQ2 generation: branch_rounds=${s} =========="
    if [[ -f "${output_path}" ]]; then
        echo "[skip gen] ${output_path}"
    else
        s_time=$(date +%s.%N)
        python3 -m experiments.RQ2.search \
         trainer.nnodes=1 \
         trainer.n_gpus_per_node=${N_GPUS} \
         data.val_files="${DATA_PATH}" \
         data.prompt_key=prompt \
         data.val_batch_size=${BS} \
         +data.trees_per_prompt=${TREES_PER_PROMPT} \
         +data.n_branch_rounds=${s} \
         +data.opts_avg_snapshot_rounds="${SNAPSHOT_ROUNDS_LIST}" \
         +data.output_path="${output_path}" \
         actor_rollout_ref.model.path="${dst_actor}" \
         critic.model.path="${dst_critic}" \
         critic.model.use_remove_padding=True \
         +critic.value_head_activation=sigmoid \
         critic.forward_micro_batch_size_per_gpu=64 \
         actor_rollout_ref.rollout.name=vllm \
         +actor_rollout_ref.rollout.search=opts \
         actor_rollout_ref.rollout.load_format=auto \
         +actor_rollout_ref.rollout.max_search_per_tree=${s} \
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
        e_time=$(date +%s.%N)
        awk -v s="${s_time}" -v e="${e_time}" -v tag="${GEN_BASENAME}" \
            'BEGIN{ printf "%s elapsed_seconds=%.2f\n", tag, e-s }' | tee "${LOG_ROOT}/${GEN_BASENAME}.time" || true
    fi

    echo "========== RQ2 evaluation: branch_rounds=${s}, opts-avg slices ${OPTS_AVG_SLICES}, k=${OPTS_AVG_KS} =========="
    python3 -m trainer.main_eval \
        --pregenerated_parquet "${output_path}" \
        --metrics opts-avg --k ${OPTS_AVG_KS} \
        --opts_avg_slices ${OPTS_AVG_SLICES} \
        --output_tag "${GEN_BASENAME}_opts-avg_s${OPTS_AVG_SLICES_TAG}_k${OPTS_AVG_KS_TAG}" \
        --output_dir "${EVAL_ROOT}"
done

echo
echo "Done. Parquets: ${RQ2_ROOT}"
echo "Eval JSONs: ${EVAL_ROOT}"
