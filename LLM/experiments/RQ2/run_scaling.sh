#!/usr/bin/env bash
# RQ2 compute scaling, unified settings (lam=0.999, top_k=-1 like the i.i.d. baseline): regenerate
# reward/value OPTS s=3 x 128 on the given GPUs, rescore, replot Figure 4 + appendix A1.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH=/root/miniconda3/envs/opts_verl/bin:$PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export STEP=400 MODEL_SIZE=1.7B OPTS_METHOD=opts_ttpo_exp8_3_0810_n8 TOP_K=-1
M=${OPTS_METHOD}; GEN=results/step400/gen; EVAL=results/step400/eval; S=3; KS="8 16 32 64 128"; KTAG="${KS// /-}"
IID_PARQUET="${GEN}/${M}_iid_topk-1_n128.parquet"
[[ -f "${IID_PARQUET}" ]] || { echo "Missing ${IID_PARQUET}; generate the matching IID baseline with TOP_K=-1 first." >&2; exit 1; }
log() { echo "[$(date '+%F %T')] $*"; }
for mode in reward value; do
    log "OPTS generation: ${mode} s=${S} top_k=-1 lam=0.999"
    REWARD_MODE=${mode} MAX_SEARCH_PER_TREE=${S} bash experiments/RQ2/run_generate.sh > "logs/step400/rq2_compute_rerun_v2_${mode}.log" 2>&1 \
        || { log "generation ${mode} FAILED"; exit 1; }
done
log "rescoring"
python -m trainer.main_eval --pregenerated_parquet "${IID_PARQUET}" --metrics pass --k ${KS} \
    --output_tag "task2_iid_pass_k${KTAG}" --output_dir "${EVAL}" > logs/step400/rq2_compute_rerun_v2_eval_iid.log 2>&1 || exit 1
python -m trainer.main_eval --pregenerated_parquet "${IID_PARQUET}" --metrics cons --k ${KS} \
    --output_tag "task3_iid_cons_k${KTAG}" --output_dir "${EVAL}" > logs/step400/rq2_compute_rerun_v2_eval_iid_cons.log 2>&1 || exit 1
python -m trainer.main_eval --pregenerated_parquet "${GEN}/${M}_opts_reward_s${S}_topk-1_n128.parquet" --metrics opts --k ${KS} \
    --output_tag "task2_reward_opts_k${KTAG}" --output_dir "${EVAL}" > logs/step400/rq2_compute_rerun_v2_eval_reward.log 2>&1 || exit 1
python -m trainer.main_eval --pregenerated_parquet "${GEN}/${M}_opts_value_s${S}_topk-1_n128.parquet" --metrics opts --k ${KS} \
    --output_tag "task3_value_opts_k${KTAG}" --output_dir "${EVAL}" > logs/step400/rq2_compute_rerun_v2_eval_value.log 2>&1 || exit 1
log "plotting"
python visual/plot_rq2_compute_scaling.py \
  --reward-iid "${EVAL}/${M}_iid_topk-1_n128__task2_iid_pass_k${KTAG}.json" \
  --value-iid "${EVAL}/${M}_iid_topk-1_n128__task3_iid_cons_k${KTAG}.json" \
  --iid-parquet "${IID_PARQUET}" \
  --reward-opts "${EVAL}/${M}_opts_reward_s${S}_topk-1_n128__task2_reward_opts_k${KTAG}.json" \
  --value-opts "${EVAL}/${M}_opts_value_s${S}_topk-1_n128__task3_value_opts_k${KTAG}.json" \
  --reward-parquet "${GEN}/${M}_opts_reward_s${S}_topk-1_n128.parquet" --value-parquet "${GEN}/${M}_opts_value_s${S}_topk-1_n128.parquet" \
    > logs/step400/rq2_compute_rerun_v2_plot.log 2>&1 || exit 1
log "DONE"; touch results/step400/RQ2_COMPUTE_RERUN_V2_DONE
