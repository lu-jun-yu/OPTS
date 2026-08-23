#!/usr/bin/env bash
# OPTS tree-search generation for the OPTS-TTPO checkpoint (single node).
# Merges actor + critic FSDP checkpoints to HF, then runs
# trainer.main_opts_generation with n_samples responses per prompt.
# Idempotent: merge is skipped if the HF dir has weights, generation is
# skipped if the parquet already exists.
#
# Run from anywhere:  bash scripts/run_opts_generation.sh
# To change settings (checkpoint, budget, reward mode), edit the variables below.
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

# Checkpoint selection is centralized in scripts/run_eval.sh, which exports
# MODEL_SIZE / STEP / OPTS_METHOD; the defaults below match it so this script
# also works standalone.
MODEL_SIZE=${MODEL_SIZE:-1.7B}
STEP=${STEP:-400}
OPTS_METHOD=${OPTS_METHOD:-opts_ttpo_exp8_3_0810_n8}
CKPT_NAME="${OPTS_METHOD}_${MODEL_SIZE}"
CKPT_ROOT="/share/lujunyu/ckpts/opts_ckpts/opts_ttpo_${MODEL_SIZE}"
DATA_PATH=data/test.parquet
# Method tag used in output filenames, e.g. opts_ttpo_exp8_3_0810_n8.
METHOD_TAG="${OPTS_METHOD}"

N_SAMPLES=128
REWARD_MODE=${REWARD_MODE:-reward}
if [ "${REWARD_MODE}" = "reward" ]; then
    OTRC_BASELINE=zero
else
    OTRC_BASELINE=mean
fi
MAX_SEARCH_PER_TREE=${MAX_SEARCH_PER_TREE:-7}
OPTS_GEN_TAG="${REWARD_MODE}_s${MAX_SEARCH_PER_TREE}"
OPTS_KS="8 16 32 64 128"
OPTS_KS_LIST="[$(echo ${OPTS_KS} | tr ' ' ',')]"

OUT_ROOT="results/step${STEP}"
MERGED_ROOT="${OUT_ROOT}/merged"
GEN_ROOT="${OUT_ROOT}/gen"
LOG_ROOT="logs/step${STEP}"
mkdir -p "${MERGED_ROOT}" "${GEN_ROOT}" "${LOG_ROOT}"

src_actor="${CKPT_ROOT}/${CKPT_NAME}/global_step_${STEP}/actor"
src_critic="${CKPT_ROOT}/${CKPT_NAME}/global_step_${STEP}/critic"
dst_actor="${MERGED_ROOT}/opts_ttpo_actor"
dst_critic="${MERGED_ROOT}/opts_ttpo_critic"
# e.g. opts_ttpo_exp8_3_0810_n8_opts_reward_s3_n128.parquet
GEN_BASENAME="${METHOD_TAG}_opts_${OPTS_GEN_TAG}"
output_path="${GEN_ROOT}/${GEN_BASENAME}_n${N_SAMPLES}.parquet"

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
python3 -m trainer.main_opts_generation \
 trainer.nnodes=1 \
 trainer.n_gpus_per_node=${N_GPUS} \
 data.val_files="${DATA_PATH}" \
 data.prompt_key=prompt \
 data.val_batch_size=902 \
 +data.n_samples=${N_SAMPLES} \
 +data.reward_mode=${REWARD_MODE} \
 +data.opts_snapshot_ks="${OPTS_KS_LIST}" \
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
 +algorithm.otrc_baseline=${OTRC_BASELINE} \
 2>&1 | tee "${LOG_ROOT}/${GEN_BASENAME}.log"
e=$(date +%s.%N)
awk -v s="${s}" -v e="${e}" -v tag="${GEN_BASENAME}" \
    'BEGIN{ printf "%s elapsed_seconds=%.2f\n", tag, e-s }' | tee "${LOG_ROOT}/${GEN_BASENAME}.time" || true

echo "Done. Output: ${output_path}"
