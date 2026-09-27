#!/usr/bin/env bash
# Appendix 3 (performance-difference vs. random vs. midpoint rebranch position) + RQ2 search-budget rerun.
#   perf_diff / random / midpoint arms, each s=7 with snapshots 1/3/7, concurrently on disjoint GPU sets
#   (the perf_diff arm doubles as the RQ2 search-budget scaling run, now up to s=7).
#   Both non-performance-difference rules keep the performance-difference rule's selected trees and budget; only the node position changes.
# Eval: opts-avg@32 / opts-pass@32 per slice (pooled + per benchmark). Idempotent.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH=/root/miniconda3/envs/opts_verl/bin:$PATH
export NCCL_DEBUG=ERROR TRANSFORMERS_VERBOSITY=error VLLM_LOGGING_LEVEL=WARN

GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
ACTOR=${ACTOR:-results/step400/merged/opts_ttpo_actor}     # opts_ttpo_exp8_3_0810_n8_1.7B step 400
CRITIC=${CRITIC:-results/step400/merged/opts_ttpo_critic}
DATA=data/test.parquet
BS=902; TREES=32
OUT=${OUT:-results/step400/appendix3_20260915}
LOG=logs/step400/appendix3_20260915
RAY_TMP=/tmp/ray_appendix3
mkdir -p "${OUT}/eval" "${LOG}" "${RAY_TMP}"
log() { echo "[$(date '+%F %T')] $*"; }

run_arm() {  # rule s snapshots gpus port_offset
    local rule=$1 s=$2 snaps=$3 gpus=$4 po=$5
    local n_gpus=$(echo "${gpus}" | tr ',' '\n' | wc -l)
    local name="opts_ttpo_exp8_3_0810_n8_${rule}_s${s}_t${TREES}_bs${BS}"
    local parquet="${OUT}/${name}.parquet"
    if [[ -f "${parquet}" ]]; then log "[skip gen] ${parquet}"; else
        log "[gen] ${rule} s=${s} on GPUs ${gpus}"
        local t0=$(date +%s)
        CUDA_VISIBLE_DEVICES=${gpus} python -m experiments.RQ2.search \
         +ray_kwargs.ray_init._temp_dir="${RAY_TMP}/${rule}" \
         +ray_kwargs.ray_init.dashboard_port=$((8265 + po)) \
         +ray_kwargs.ray_init._metrics_export_port=$((18100 + po)) \
         ray_kwargs.ray_init.num_cpus=$((16 * n_gpus)) \
         trainer.nnodes=1 trainer.n_gpus_per_node=${n_gpus} \
         data.val_files="${DATA}" data.prompt_key=prompt data.val_batch_size=${BS} \
         +data.trees_per_prompt=${TREES} +data.n_branch_rounds=${s} \
         +data.opts_avg_snapshot_rounds="[${snaps}]" +data.adv_snapshot_rounds="[]" \
         +data.select_rule=${rule} +data.select_seed=0 \
         +data.output_path="${parquet}" \
         actor_rollout_ref.model.path="${ACTOR}" critic.model.path="${CRITIC}" \
         critic.model.use_remove_padding=True +critic.value_head_activation=sigmoid \
         critic.forward_micro_batch_size_per_gpu=64 \
         actor_rollout_ref.rollout.name=vllm +actor_rollout_ref.rollout.search=opts \
         actor_rollout_ref.rollout.load_format=auto \
         +actor_rollout_ref.rollout.max_search_per_tree=${s} \
         actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=0.95 \
         actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
         actor_rollout_ref.rollout.prompt_length=1024 actor_rollout_ref.rollout.response_length=2048 \
         actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
         actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
         actor_rollout_ref.rollout.mode=async actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
         actor_rollout_ref.rollout.max_num_batched_tokens=262144 \
         reward_model.use_reward_loop=False algorithm.lam=0.999 \
         > "${LOG}/${name}.log" 2>&1
        local rc=$?
        echo "${name} elapsed_seconds=$(( $(date +%s) - t0 )) rc=${rc}" | tee "${LOG}/${name}.time"
        pkill -f "temp-dir=${RAY_TMP}/${rule} " >/dev/null 2>&1 || true
        [[ ${rc} -ne 0 || ! -f "${parquet}" ]] && { log "[gen] ${rule} FAILED rc=${rc}"; return 1; }
    fi
    local slices="0 ${snaps//,/ }"
    python -m trainer.main_eval --pregenerated_parquet "${parquet}" \
        --metrics opts-avg opts-pass --k 32 --opts_avg_slices ${slices} \
        --output_tag "opts-avg-pass_s${slices// /-}_k32" --output_dir "${OUT}/eval" \
        > "${LOG}/${name}_eval.log" 2>&1 && log "[eval done] ${name}" || { log "[eval] ${name} FAILED"; return 1; }
}

# Two phases on the given GPU set: perf_diff on all GPUs first (Figure 3 / RQ2 search budget),
# then random and midpoint concurrently on the two halves. All arms s=7 (user decision 2026-09-15).
IFS=',' read -ra G <<< "${GPUS}"; half=$(( ${#G[@]} / 2 ))
G1=$(IFS=,; echo "${G[*]:0:${half}}"); G2=$(IFS=,; echo "${G[*]:${half}}")
if [[ "${SKIP_PERF_DIFF:-0}" == "1" ]]; then log "phase 1 skipped (perf_diff handled elsewhere)"; else
log "phase 1: perf_diff s=7 on ${GPUS}"
run_arm perf_diff 7 "1,3,7" "${GPUS}" 0 || { touch "${OUT}/FAILED"; log "perf_diff FAILED"; exit 1; }
fi
log "phase 2: random on ${G1}, midpoint on ${G2}"
run_arm random 7 "1,3,7" "${G1}" 1 & p1=$!
sleep 30
run_arm midpoint 7 "1,3,7" "${G2}" 2 & p2=$!
rc=0; wait ${p1} || rc=1; wait ${p2} || rc=1
[[ ${rc} -eq 0 ]] && { touch "${OUT}/DONE"; log "ALL DONE"; } || { touch "${OUT}/FAILED"; log "FAILED"; }
exit ${rc}
