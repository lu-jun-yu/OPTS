#!/usr/bin/env bash
# Full step-400 evaluation pipeline for the 8B checkpoints.
#
#   1) Call scripts/run_parallel_generation_8B.sh — merge FSDP actors + generate
#      N_SAMPLES (128) i.i.d. responses per prompt for DAPO, PPO,
#      REINFORCE++, OPTS-TTPO.
#   2) OPTS tree-search generation remains disabled below; this script evaluates
#      the i.i.d. generations needed for the 8B training-method comparison.
#   3) Score every parquet with trainer.main_eval --pregenerated_parquet:
#        - Task 1: avg@PASSCONS_K, pass@PASSCONS_K, cons@PASSCONS_K over the
#          first PASSCONS_K of N_SAMPLES responses (default K=32).
#        - Task 2: pass@k for OPTS-TTPO's i.i.d. parquet for each k in OPTS_KS
#          (default 8 16 32 64 128); reward-guided OPTS is evaluated only when
#          a matching parquet already exists in the isolated 8B directory.
#        - Task 3: cons@k for OPTS-TTPO's i.i.d. parquet for each k in OPTS_KS;
#          value-guided OPTS is evaluated only when a matching parquet exists.
#   4) Summarize generation wall-clock times.
#
# CUDA_VISIBLE_DEVICES set on this script (e.g. `CUDA_VISIBLE_DEVICES=6 bash
# scripts/run_eval_8B.sh`) propagates to the generation script, which derives
# n_gpus_per_node from it automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${LLM_DIR}"

# ---- checkpoint selection (the single place to edit) ----
STEP=400
MODEL_SIZE=8B
METHODS=${METHODS:-"dapo_0805_n8 ppo_0805_n8 reinforce_pp_baseline_0805_n8 opts_ttpo_exp8_3_0810_n8"}
OPTS_METHOD="opts_ttpo_exp8_3_0810_n8"
export STEP MODEL_SIZE METHODS OPTS_METHOD
# ---------------------------------------------------------

N_SAMPLES=128
export TOP_K=${TOP_K:--1}
PASSCONS_K=32
OPTS_KS="8 16 32 64 128"
OPTS_KS_TAG="${OPTS_KS// /-}"
SKIP_GEN=0
OPTS_MAX_SEARCHES=${OPTS_MAX_SEARCHES:-"3"}

OUT_ROOT="results/${MODEL_SIZE}/step${STEP}"
GEN_ROOT="${OUT_ROOT}/gen"
EVAL_ROOT="${OUT_ROOT}/eval"
LOG_ROOT="logs/${MODEL_SIZE}/step${STEP}"
mkdir -p "${EVAL_ROOT}"

if [[ "${SKIP_GEN}" != "1" ]]; then
    echo "========== Stage 1: i.i.d. generation for ${METHODS} =========="
    bash "${SCRIPT_DIR}/run_parallel_generation_8B.sh"

    # OPTS tree-search generation is intentionally disabled for this 8B
    # training-method evaluation. Do not point the 1.7B generation script at
    # these checkpoints because its merged/output directories are different.
fi

echo "========== Stage 2: Task 1 — avg@${PASSCONS_K}, pass@${PASSCONS_K}, cons@${PASSCONS_K} =========="
for method in ${METHODS}; do
    parquet="${GEN_ROOT}/${method}_iid_topk${TOP_K}_n${N_SAMPLES}.parquet"
    if [[ ! -f "${parquet}" ]]; then
        echo "[missing] ${parquet}" >&2
        continue
    fi
    echo "--- ${method} ---"
    python3 -m trainer.main_eval \
        --pregenerated_parquet "${parquet}" \
        --metrics avg pass cons --k ${PASSCONS_K} \
        --output_tag "task1_avg-pass-cons_k${PASSCONS_K}" \
        --output_dir "${EVAL_ROOT}"
done

echo "========== Stage 2: Task 2 — opts@k (reward) + pass@k (i.i.d.) for opts_ttpo =========="
iid_parquet="${GEN_ROOT}/${OPTS_METHOD}_iid_topk${TOP_K}_n${N_SAMPLES}.parquet"
for s in ${OPTS_MAX_SEARCHES}; do
    reward_opts_parquet="${GEN_ROOT}/${OPTS_METHOD}_opts_reward_s${s}_topk${TOP_K}_n${N_SAMPLES}.parquet"
    if [[ -f "${reward_opts_parquet}" ]]; then
        echo "--- opts_ttpo OPTS parquet (reward, s=${s}) ---"
        python3 -m trainer.main_eval \
            --pregenerated_parquet "${reward_opts_parquet}" \
            --metrics opts --k ${OPTS_KS} \
            --output_tag "task2_reward_opts_k${OPTS_KS_TAG}" \
            --output_dir "${EVAL_ROOT}"
    else
        echo "[missing] ${reward_opts_parquet}" >&2
    fi
done
if [[ -f "${iid_parquet}" ]]; then
    echo "--- opts_ttpo i.i.d. parquet (pass@k reference) ---"
    python3 -m trainer.main_eval \
        --pregenerated_parquet "${iid_parquet}" \
        --metrics pass --k ${OPTS_KS} \
        --output_tag "task2_iid_pass_k${OPTS_KS_TAG}" \
        --output_dir "${EVAL_ROOT}"
else
    echo "[missing] ${iid_parquet}" >&2
fi

echo "========== Stage 2: Task 3 — opts@k (value) + cons@k (i.i.d.) for opts_ttpo =========="
for s in ${OPTS_MAX_SEARCHES}; do
    value_opts_parquet="${GEN_ROOT}/${OPTS_METHOD}_opts_value_s${s}_topk${TOP_K}_n${N_SAMPLES}.parquet"
    if [[ -f "${value_opts_parquet}" ]]; then
        echo "--- opts_ttpo OPTS parquet (value, s=${s}) ---"
        python3 -m trainer.main_eval \
            --pregenerated_parquet "${value_opts_parquet}" \
            --metrics opts --k ${OPTS_KS} \
            --output_tag "task3_value_opts_k${OPTS_KS_TAG}" \
            --output_dir "${EVAL_ROOT}"
    else
        echo "[missing] ${value_opts_parquet}" >&2
    fi
done
if [[ -f "${iid_parquet}" ]]; then
    echo "--- opts_ttpo i.i.d. parquet (cons@k reference) ---"
    python3 -m trainer.main_eval \
        --pregenerated_parquet "${iid_parquet}" \
        --metrics cons --k ${OPTS_KS} \
        --output_tag "task3_iid_cons_k${OPTS_KS_TAG}" \
        --output_dir "${EVAL_ROOT}"
else
    echo "[missing] ${iid_parquet}" >&2
fi

echo "========== Timings =========="
shopt -s nullglob
for f in "${LOG_ROOT}"/*.time; do
    cat "$f"
done
shopt -u nullglob

echo
echo "Eval JSONs: ${EVAL_ROOT}"
