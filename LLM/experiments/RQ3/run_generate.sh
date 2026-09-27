#!/usr/bin/env bash
# Sample OPTS trees, independent fixed-128 controls, and reference chains.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=${PYTHON_BIN:-python}
export RQ3_ROOT=${RQ3_ROOT:-results/rq3}
ACTOR=${ACTOR:-results/step400/merged/opts_ttpo_actor}
CRITIC=${CRITIC:-results/step400/merged/opts_ttpo_critic}
PROMPTS=${PROMPTS:-data/train.parquet}
read -r -a GPU_IDS <<< "${GPUS:-0 1 2 3}"
((${#GPU_IDS[@]} > 0)) || exit 1
command -v ninja >/dev/null || { echo 'Activate the existing VeRL environment (ninja must be on PATH).' >&2; exit 1; }
mkdir -p "${RQ3_ROOT}"/{opts,controls,logs}
"${PY}" - "${PROMPTS}" "${RQ3_ROOT}" <<'PY'
import sys
from pathlib import Path
import pandas as pd
frame = pd.read_parquet(sys.argv[1])
root = Path(sys.argv[2])
if len(frame) != 16384:
    raise ValueError(f'Expected 16,384 prompts; got {len(frame)}')
if any(root.glob('opts/opts_bias*.parquet')) or any(root.glob('controls/*_full_shard*.parquet')):
    raise FileExistsError('RQ3_ROOT already contains generated samples; select an empty output directory')
for shard in range(8):
    frame.iloc[shard*2048:(shard+1)*2048].reset_index(drop=True).to_parquet(root/'opts'/f'train_shard_{shard}.parquet')
PY
RAY_ROOT=$(mktemp -d /tmp/ray_rq3.XXXXXX)
worker() {
    local slot=$1 gpu=${GPU_IDS[$1]} shard mode seed output
    for ((shard=slot; shard<8; shard+=${#GPU_IDS[@]})); do
        common=(
            ray_kwargs.ray_init.num_cpus=20 trainer.nnodes=1 trainer.n_gpus_per_node=1
            actor_rollout_ref.model.path="${ACTOR}"
            actor_rollout_ref.rollout.name=vllm +actor_rollout_ref.rollout.search=opts
            actor_rollout_ref.rollout.load_format=auto
            actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0
            actor_rollout_ref.rollout.top_k=-1 actor_rollout_ref.rollout.val_kwargs.top_p=1.0
            actor_rollout_ref.rollout.prompt_length=1152 actor_rollout_ref.rollout.response_length=2048
            actor_rollout_ref.rollout.tensor_model_parallel_size=1
            actor_rollout_ref.rollout.pipeline_model_parallel_size=1
            actor_rollout_ref.rollout.mode=async actor_rollout_ref.rollout.gpu_memory_utilization=0.85
            reward_model.use_reward_loop=False
        )
        CUDA_VISIBLE_DEVICES=${gpu} "${PY}" -m experiments.RQ2.search "${common[@]}" \
            +ray_kwargs.ray_init._temp_dir="${RAY_ROOT}/opts_${shard}" \
            +ray_kwargs.ray_init.dashboard_port=$((8301+shard)) \
            +ray_kwargs.ray_init._metrics_export_port=$((18001+shard)) \
            data.val_files="${RQ3_ROOT}/opts/train_shard_${shard}.parquet" \
            data.prompt_key=prompt data.val_batch_size=2048 data.max_prompt_length=1152 \
            +data.rollout_seed_base=20260915 +data.prompt_index_offset=$((shard*2048)) \
            +data.trees_per_prompt=8 +data.n_branch_rounds=7 \
            +data.opts_avg_snapshot_rounds='[]' +data.adv_snapshot_rounds='[0,1,3,7]' \
            +data.adv_backups='[max]' +data.select_rule=perf_diff \
            +data.output_path="${RQ3_ROOT}/opts/opts_bias_s7_t8_full_shard${shard}.parquet" \
            critic.model.path="${CRITIC}" critic.model.use_remove_padding=True \
            +critic.value_head_activation=sigmoid critic.forward_micro_batch_size_per_gpu=64 \
            +actor_rollout_ref.rollout.max_search_per_tree=7 algorithm.lam=0.999 \
            actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.max_model_len=4096 \
            actor_rollout_ref.rollout.max_num_seqs=256 actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
            > "${RQ3_ROOT}/logs/opts_${shard}.log" 2>&1 || return 1
        for mode in fixed gstar; do
            seed=20260915; [[ ${mode} != gstar ]] || seed=120260915
            output="${RQ3_ROOT}/controls/${mode}_full_shard${shard}.parquet"
            CUDA_VISIBLE_DEVICES=${gpu} "${PY}" -m experiments.RQ3.controls "${common[@]}" \
                +ray_kwargs.ray_init._temp_dir="${RAY_ROOT}/${mode}_${shard}" \
                +ray_kwargs.ray_init.dashboard_port=$((8701+shard)) \
                +ray_kwargs.ray_init._metrics_export_port=$((18701+shard)) \
                +data.control_mode="${mode}" +data.output_path="${output}" \
                +data.prompt_source="${PROMPTS}" +data.shard=${shard} +data.shard_size=2048 \
                +data.rollout_seed_base=${seed} +data.reward_procs=16 \
                actor_rollout_ref.rollout.enforce_eager=False \
                actor_rollout_ref.rollout.max_num_seqs=1024 \
                actor_rollout_ref.rollout.max_num_batched_tokens=262144 \
                > "${RQ3_ROOT}/logs/${mode}_${shard}.log" 2>&1 || return 1
        done
        "${PY}" -m experiments.RQ3.validate_controls \
            --fixed "${RQ3_ROOT}/controls/fixed_full_shard${shard}.parquet" \
            --gstar "${RQ3_ROOT}/controls/gstar_full_shard${shard}.parquet" \
            --c "${RQ3_ROOT}/opts/opts_bias_s7_t8_full_shard${shard}.parquet" \
            --prompts "${PROMPTS}" --model "${ACTOR}" --prompt-offset=$((shard*2048)) \
            --output "${RQ3_ROOT}/controls/validation_shard${shard}.json" || return 1
    done
}
pids=()
for slot in "${!GPU_IDS[@]}"; do worker "${slot}" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
((failed == 0)) || exit 1
echo "RQ3 generation complete: ${RQ3_ROOT}"
