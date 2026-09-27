#!/usr/bin/env bash
# Learned-critic E5: value-guided search from the completed reward run's roots.
# No reward-guided search or root rollout is regenerated.  Defaults to H200
# devices 0,1,2,3.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export NCCL_DEBUG=ERROR
export TRANSFORMERS_VERBOSITY=error
export VLLM_LOGGING_LEVEL=WARN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
IFS=',' read -ra gpu_array <<< "${CUDA_VISIBLE_DEVICES}"
N_GPUS=${#gpu_array[@]}
unset gpu_array

PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/opts_verl/bin/python}
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
MODEL_SIZE=${MODEL_SIZE:-1.7B}
STEP=${STEP:-400}
OPTS_METHOD=${OPTS_METHOD:-opts_ttpo_exp8_3_0810_n8}
CKPT_NAME="${OPTS_METHOD}_${MODEL_SIZE}"
CKPT_ROOT=${CKPT_ROOT:-${OPTS_CHECKPOINT_ROOT:-checkpoints}/opts_ttpo_${MODEL_SIZE}}
DATA_PATH=${DATA_PATH:-data/test.parquet}

TREES_PER_PROMPT=${TREES_PER_PROMPT:-32}
SEARCH_ROUNDS=${SEARCH_ROUNDS:-15}
BS=${BS:-902}
SNAPSHOT_ROUNDS=${SNAPSHOT_ROUNDS:-"0 1 3 7 15"}
SNAPSHOT_LIST="[$(echo "${SNAPSHOT_ROUNDS}" | tr ' ' ',')]"

OUT_ROOT=${OUT_ROOT:-results/step${STEP}/e5_learned_critic}
MERGED_ROOT=${MERGED_ROOT:-results/step${STEP}/merged}
LOG_ROOT=${LOG_ROOT:-logs/step${STEP}/e5_learned_critic}
EVAL_ROOT=${EVAL_ROOT:-${OUT_ROOT}/eval}
RAY_TMP=${RAY_TMP:-/tmp/ray_e5_learned_critic}
mkdir -p "${OUT_ROOT}" "${MERGED_ROOT}" "${LOG_ROOT}" "${EVAL_ROOT}" "${RAY_TMP}"

src_actor="${CKPT_ROOT}/${CKPT_NAME}/global_step_${STEP}/actor"
src_critic="${CKPT_ROOT}/${CKPT_NAME}/global_step_${STEP}/critic"
dst_actor="${MERGED_ROOT}/opts_ttpo_actor"
dst_critic="${MERGED_ROOT}/opts_ttpo_critic"

for pair in "${src_actor}:${dst_actor}" "${src_critic}:${dst_critic}"; do
    src="${pair%:*}"
    dst="${pair#*:}"
    if [[ -f "${dst}/config.json" ]] && compgen -G "${dst}/*.safetensors" >/dev/null; then
        echo "[skip merge] ${dst}"
    else
        echo "[merge] ${src} -> ${dst}"
        "${PYTHON_BIN}" -m verl.model_merger merge \
            --backend fsdp --local_dir "${src}" --target_dir "${dst}"
    fi
done

tag="${OPTS_METHOD}_reward-text-roots_s${SEARCH_ROUNDS}_t${TREES_PER_PROMPT}_bs${BS}"
root_cache="${OUT_ROOT}/${tag}_roots.pkl"
value_output="${OUT_ROOT}/${tag}_value.parquet"
reward_source=${REWARD_SOURCE:-results/step${STEP}/rq2/${OPTS_METHOD}_rq2_opts_s${SEARCH_ROUNDS}_t${TREES_PER_PROMPT}_bs${BS}.parquet}
log_path="${LOG_ROOT}/${tag}.log"
time_path="${LOG_ROOT}/${tag}.time"

if [[ ! -f "${reward_source}" ]]; then
    echo "Missing completed reward-guided source: ${reward_source}" >&2
    exit 1
fi

if [[ -f "${value_output}" ]]; then
    echo "[skip generation] value-guided output already exists"
else
    start_time=$(date +%s.%N)
    "${PYTHON_BIN}" -m experiments.RQ2.guidance \
        +ray_kwargs.ray_init._temp_dir="${RAY_TMP}" \
        +ray_kwargs.ray_init.dashboard_port=8715 \
        +ray_kwargs.ray_init._metrics_export_port=18715 \
        ray_kwargs.ray_init.num_cpus=64 \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node="${N_GPUS}" \
        data.val_files="${DATA_PATH}" \
        data.prompt_key=prompt \
        data.val_batch_size="${BS}" \
        +data.trees_per_prompt="${TREES_PER_PROMPT}" \
        +data.n_branch_rounds="${SEARCH_ROUNDS}" \
        +data.opts_avg_snapshot_rounds="${SNAPSHOT_LIST}" \
        +data.adv_snapshot_rounds="${SNAPSHOT_LIST}" \
        +data.shared_guidance_phase=value \
        +data.root_cache_path="${root_cache}" \
        +data.root_source_parquet="${reward_source}" \
        +data.output_path_value="${value_output}" \
        +data.value_perf_diff_baseline=mean \
        +data.rollout_seed_base=20260915 \
        actor_rollout_ref.model.path="${dst_actor}" \
        critic.model.path="${dst_critic}" \
        critic.model.use_remove_padding=True \
        +critic.value_head_activation=sigmoid \
        critic.forward_micro_batch_size_per_gpu=64 \
        actor_rollout_ref.rollout.name=vllm \
        +actor_rollout_ref.rollout.search=opts \
        actor_rollout_ref.rollout.load_format=auto \
        +actor_rollout_ref.rollout.max_search_per_tree="${SEARCH_ROUNDS}" \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.top_p=0.95 \
        actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
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
        2>&1 | tee "${log_path}"
    end_time=$(date +%s.%N)
    awk -v s="${start_time}" -v e="${end_time}" -v tag="${tag}" \
        'BEGIN{ printf "%s elapsed_seconds=%.2f\n", tag, e-s }' | tee "${time_path}"
fi

eval_tag="${tag}_value_opts-avg_s$(echo "${SNAPSHOT_ROUNDS}" | tr ' ' '-')_k32"
"${PYTHON_BIN}" -m trainer.main_eval \
    --pregenerated_parquet "${value_output}" \
    --metrics opts-avg \
    --k 32 \
    --opts_avg_slices ${SNAPSHOT_ROUNDS} \
    --output_tag "${eval_tag}" \
    --output_dir "${EVAL_ROOT}"

echo "Existing reward-guided source: ${reward_source}"
echo "Value-guided output: ${value_output}"
echo "Reconstructed shared-root cache: ${root_cache}"
