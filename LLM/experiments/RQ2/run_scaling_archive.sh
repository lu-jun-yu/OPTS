#!/usr/bin/env bash
# RQ2 matched-rollout (compute) scaling rerun for opts_ttpo_exp8_3_0810_n8_1.7B step 400:
# regenerate reward- and value-mode OPTS parquets (s=3, top-k -1, 128 per prompt) with the
# current search/reward code, then rescore them and the existing i.i.d. parquet with the
# current evaluator. Old parquets/evals are archived, not deleted.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH=/root/miniconda3/envs/opts_verl/bin:$PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export STEP=400 MODEL_SIZE=1.7B OPTS_METHOD=opts_ttpo_exp8_3_0810_n8 TOP_K=-1
M=${OPTS_METHOD}; GEN=results/step400/gen; EVAL=results/step400/eval; ARCH=results/step400/archive_20260815
S=3; KS="8 16 32 64 128"; KTAG="${KS// /-}"
IID_PARQUET="${GEN}/${M}_iid_topk-1_n128.parquet"
[[ -f "${IID_PARQUET}" ]] || { echo "Missing ${IID_PARQUET}; generate the matching IID baseline with TOP_K=-1 first." >&2; exit 1; }
mkdir -p "${ARCH}/gen" "${ARCH}/eval" logs/step400
log() { echo "[$(date '+%F %T')] $*"; }
for mode in reward value; do
    f="${GEN}/${M}_opts_${mode}_s${S}_topk50_n128.parquet"
    [[ -f "${f}" && ! -f "${ARCH}/gen/$(basename ${f})" ]] && { log "archive ${f}"; mv "${f}" "${ARCH}/gen/"; }
done
for j in "${EVAL}"/${M}_*; do [[ -f "$j" ]] && mv "$j" "${ARCH}/eval/" 2>/dev/null; done
for mode in reward value; do
    log "OPTS generation: ${mode} s=${S}"
    REWARD_MODE=${mode} MAX_SEARCH_PER_TREE=${S} bash experiments/RQ2/run_generate.sh > "logs/step400/rq2_compute_rerun_${mode}.log" 2>&1 \
        || { log "generation ${mode} FAILED"; exit 1; }
done
log "rescoring"
python -m trainer.main_eval --pregenerated_parquet "${IID_PARQUET}" --metrics avg pass --k 32 \
    --output_tag "task1_avg-pass_k32" --output_dir "${EVAL}" > logs/step400/rq2_compute_rerun_eval_task1.log 2>&1 || exit 1
python -m trainer.main_eval --pregenerated_parquet "${IID_PARQUET}" --metrics pass --k ${KS} \
    --output_tag "task2_iid_pass_k${KTAG}" --output_dir "${EVAL}" > logs/step400/rq2_compute_rerun_eval_iid.log 2>&1 || exit 1
python -m trainer.main_eval --pregenerated_parquet "${IID_PARQUET}" --metrics cons --k ${KS} \
    --output_tag "task3_iid_cons_k${KTAG}" --output_dir "${EVAL}" > logs/step400/rq2_compute_rerun_eval_iid_cons.log 2>&1 || exit 1
python -m trainer.main_eval --pregenerated_parquet "${GEN}/${M}_opts_reward_s${S}_topk-1_n128.parquet" --metrics opts --k ${KS} \
    --output_tag "task2_reward_opts_k${KTAG}" --output_dir "${EVAL}" > logs/step400/rq2_compute_rerun_eval_reward.log 2>&1 || exit 1
python -m trainer.main_eval --pregenerated_parquet "${GEN}/${M}_opts_value_s${S}_topk-1_n128.parquet" --metrics opts --k ${KS} \
    --output_tag "task3_value_opts_k${KTAG}" --output_dir "${EVAL}" > logs/step400/rq2_compute_rerun_eval_value.log 2>&1 || exit 1
log "DONE"; touch results/step400/RQ2_COMPUTE_RERUN_DONE
