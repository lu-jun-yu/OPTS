#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <dapo|reinforce> <cuda-visible-devices>" >&2
    exit 2
fi

ALGORITHM=$1
GPU_IDS=$2
MODEL_SIZE=8B
PROJECT_NAME="opts_ttpo_${MODEL_SIZE}"
CHECKPOINT_BASE="/share/lujunyu/ckpts/opts_ckpts/${PROJECT_NAME}"
OUTPUT_BASE="$(pwd)/logs/backfill_aime24_aime26"
PYTHON_ENV="/root/miniconda3/envs/opts_verl"

case "${ALGORITHM}" in
    dapo)
        EXPERIMENT_NAME="dapo_0805_n8_${MODEL_SIZE}"
        ADV_ESTIMATOR=grpo
        EXTRA_OVERRIDES=(
            reward_model.reward_manager=dapo
            +reward_model.overlong_buffer_cfg.enable=True
            +reward_model.overlong_buffer_cfg.len=1024
            +reward_model.overlong_buffer_cfg.penalty_factor=1.0
            +reward_model.overlong_buffer_cfg.log=False
            +reward_model.max_resp_len=2048
        )
        ;;
    reinforce)
        EXPERIMENT_NAME="reinforce_pp_baseline_0805_n8_${MODEL_SIZE}"
        ADV_ESTIMATOR=reinforce_plus_plus_baseline
        EXTRA_OVERRIDES=()
        ;;
    *)
        echo "unknown algorithm: ${ALGORITHM}" >&2
        exit 2
        ;;
esac

CHECKPOINT_ROOT="${CHECKPOINT_BASE}/${EXPERIMENT_NAME}"
RESULTS_DIR="${OUTPUT_BASE}/${EXPERIMENT_NAME}/metrics"
RAY_TEMP_DIR="/tmp/ray/ab_8b_${ALGORITHM}"
LOG_PATH="${OUTPUT_BASE}/${EXPERIMENT_NAME}/runner.log"
mkdir -p "${RESULTS_DIR}" "${RAY_TEMP_DIR}"

export PATH="${PYTHON_ENV}/bin:${PATH}"
export PYTHONPATH="$(pwd):$(pwd)/verl${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export CHECKPOINT_EVAL_ROOT="${CHECKPOINT_ROOT}"
export CHECKPOINT_EVAL_RESULTS="${RESULTS_DIR}"
export NCCL_DEBUG=ERROR
export TRANSFORMERS_VERBOSITY=error
export VLLM_LOGGING_LEVEL=WARN
export TOKENIZERS_PARALLELISM=false

cleanup() {
    status=$?
    trap - EXIT INT TERM
    find "${RAY_TEMP_DIR}" -depth -delete 2>/dev/null || true
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python -m trainer.main_checkpoint_eval \
    "algorithm.adv_estimator=${ADV_ESTIMATOR}" \
    data.train_files=data/train.parquet \
    'data.val_files=[data/aime24/test.parquet,data/aime26/test.parquet]' \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.nccl_timeout=3600 \
    "actor_rollout_ref.model.path=models/Qwen3-${MODEL_SIZE}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    'actor_rollout_ref.actor.checkpoint.load_contents=[model]' \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    critic.enable=False \
    custom_reward_function.path=utils/reward_fn.py \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.logger='["console"]' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    "trainer.project_name=${PROJECT_NAME}" \
    "trainer.experiment_name=${EXPERIMENT_NAME}_aime24_aime26_backfill" \
    "trainer.default_local_dir=${CHECKPOINT_ROOT}" \
    ray_kwargs.ray_init.num_cpus=24 \
    +ray_kwargs.ray_init.include_dashboard=False \
    +ray_kwargs.ray_init._temp_dir="${RAY_TEMP_DIR}" \
    "${EXTRA_OVERRIDES[@]}" 2>&1 | tee -a "${LOG_PATH}"
