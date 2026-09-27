#!/usr/bin/env bash
# Multi-node OPTS-TTPO paper defaults: n=8, max_search_per_tree=3,
# on 2 nodes x 2 GPUs.
#
#   head   (HEAD_IP, GPUs 0,1):  MODE=head   bash scripts/run_opts_ttpo_s3_multi-node.sh
#   worker (WORKER_IP, GPUs 3,4):  MODE=worker bash scripts/run_opts_ttpo_s3_multi-node.sh
#   train  (HEAD_IP):            MODE=train  bash scripts/run_opts_ttpo_s3_multi-node.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${LLM_DIR}"

export PATH=/root/miniconda3/envs/opts_verl/bin:$PATH
export NCCL_DEBUG=ERROR
export TRANSFORMERS_VERBOSITY=error
export VLLM_LOGGING_LEVEL=WARN
export WANDB_INIT_TIMEOUT=300
export WANDB_SERVICE_WAIT=60

MODEL_SIZE=1.7B-Base
Max_Search=3
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opts_ttpo_paper_s3_n8_${MODEL_SIZE}}"
MODE="${MODE:?set MODE=head|worker|train}"

NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
HEAD_IP="${HEAD_IP:-${RAY_HEAD_ADDR:-}}"
WORKER_IP="${WORKER_IP:-}"

RAY_HEAD_ADDR="${RAY_HEAD_ADDR:-${HEAD_IP}}"
RAY_HEAD_PORT=6379
RAY_DASHBOARD_PORT=8279
RAY_NUM_CPUS=32
RAY_TEMP_DIR="/tmp/ray/${EXPERIMENT_NAME}"
RAY_ADDRESS="${RAY_HEAD_ADDR}:${RAY_HEAD_PORT}"

cleanup_local_ray_temp() {
    local status=$?
    trap - EXIT INT TERM
    timeout 20s ray stop --force >/dev/null 2>&1 || true
    find "${RAY_TEMP_DIR}" -depth -delete 2>/dev/null || true
    exit "${status}"
}

case "${MODE}" in
    head)
        export CUDA_VISIBLE_DEVICES=0,1
        export RAY_NODE_IP_ADDRESS="${HEAD_IP:?Set HEAD_IP to the Ray head address}"
        ;;
    worker)
        : "${RAY_HEAD_ADDR:?Set HEAD_IP or RAY_HEAD_ADDR to the Ray head address}"
        export CUDA_VISIBLE_DEVICES=3,4
        export RAY_NODE_IP_ADDRESS="${WORKER_IP:?Set WORKER_IP to the Ray worker address}"
        ;;
    train)
        : "${RAY_HEAD_ADDR:?Set HEAD_IP or RAY_HEAD_ADDR to the Ray head address}"
        ;;
    local)
        # Fallback: single-node run on this machine (same math); multi-node can
        # later resume from its checkpoints.
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,4}"
        NNODES=1
        IFS=',' read -ra _gpu_arr <<< "${CUDA_VISIBLE_DEVICES}"
        GPUS_PER_NODE=${#_gpu_arr[@]}
        unset _gpu_arr
        find "${RAY_TEMP_DIR}" -depth -delete 2>/dev/null || true
        mkdir -p logs "${RAY_TEMP_DIR}"
        trap cleanup_local_ray_temp EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM
        ;;
    *)
        echo "Unknown MODE: ${MODE} (expected head|worker|train|local)" >&2
        exit 1
        ;;
esac

source "${SCRIPT_DIR}/_ray_cluster.sh"

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
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32
        actor_rollout_ref.actor.use_kl_loss=False
        actor_rollout_ref.rollout.name=vllm
        actor_rollout_ref.rollout.temperature=1.0
        actor_rollout_ref.rollout.top_p=1.0
        actor_rollout_ref.rollout.top_k=-1
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128
        actor_rollout_ref.rollout.tensor_model_parallel_size=1
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7
        +actor_rollout_ref.rollout.search=opts
        actor_rollout_ref.rollout.n=8
        actor_rollout_ref.rollout.val_kwargs.n=32
        actor_rollout_ref.rollout.val_kwargs.do_sample=True
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95
        actor_rollout_ref.rollout.val_kwargs.top_k=-1
        +actor_rollout_ref.rollout.max_search_per_tree=${Max_Search}
        critic.enable=True
        critic.optim.lr=1e-5
        critic.optim.weight_decay=0.1
        critic.optim.lr_warmup_steps=10
        critic.model.path="models/Qwen3-${MODEL_SIZE}"
        critic.model.use_remove_padding=True
        critic.ppo_micro_batch_size_per_gpu=128
        +critic.value_head_activation=sigmoid
        custom_reward_function.path=utils/reward_fn.py
        custom_reward_function.name=compute_score
        algorithm.use_kl_in_reward=False
        algorithm.kl_ctrl.kl_coef=0.0
        algorithm.lam=0.999
        +algorithm.reward_min=0.0
        +algorithm.reward_max=1.0
        +algorithm.max_searched_tree_ratio=0.3
        +algorithm.perf_diff_baseline=zero
        trainer.logger='["console","wandb"]'
        trainer.val_before_train=False
        trainer.n_gpus_per_node="${GPUS_PER_NODE}"
        trainer.nnodes="${NNODES}"
        trainer.project_name="opts_ttpo_${MODEL_SIZE}"
        trainer.experiment_name="${EXPERIMENT_NAME}"
        trainer.default_local_dir="${OPTS_CHECKPOINT_ROOT:-checkpoints}/opts_ttpo_${MODEL_SIZE}/${EXPERIMENT_NAME}"
        trainer.save_freq=20
        trainer.test_freq=20
        trainer.total_epochs=400
        trainer.total_training_steps=400
        +trainer.wandb_init_timeout="${WANDB_INIT_TIMEOUT}"
        +trainer.wandb_service_wait="${WANDB_SERVICE_WAIT}"
    )
    if [[ "${MODE}" == "train" ]]; then
        TRAIN_CMD+=("+ray_kwargs.ray_init.address=${RAY_ADDRESS}")
    else
        TRAIN_CMD+=("ray_kwargs.ray_init.num_cpus=64" "+ray_kwargs.ray_init._temp_dir=${RAY_TEMP_DIR}")
    fi
}

RUN_LOG="logs/${EXPERIMENT_NAME}.log"

case "${MODE}" in
    head)   start_ray_head ;;
    worker) start_ray_worker ;;
    train|local)  build_train_cmd; run_training ;;
esac
