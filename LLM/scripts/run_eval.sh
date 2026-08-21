#!/usr/bin/env bash
# Full step-300 evaluation pipeline.
#
#   1) Call scripts/run_parallel_generation.sh — merge FSDP actors + generate
#      N_SAMPLES (128) i.i.d. responses per prompt for DAPO, GPG, PPO,
#      REINFORCE++, OPTS-TTPO.
#   2) Call scripts/run_opts_generation.sh — merge OPTS-TTPO actor+critic and
#      run trainer.main_opts_generation (OPTS tree search). The reward mode is
#      set inside that script; Task 3 below needs a separate REWARD_MODE=value
#      run of it.
#   3) Score every parquet with trainer.main_eval --pregenerated_parquet:
#        - Task 1: avg@PASSCONS_K, pass@PASSCONS_K, cons@PASSCONS_K over the
#          first PASSCONS_K of N_SAMPLES responses (default K=32).
#        - Task 2: opts@k (from reward-guided OPTS parquet) and pass@k (from
#          OPTS-TTPO's i.i.d. parquet) for each k in OPTS_KS
#          (default 8 16 32 64 128).
#        - Task 3: value-guided opts@k from online greedy-path snapshots and
#          cons@k (from OPTS-TTPO's i.i.d. parquet) for each k in OPTS_KS.
#   4) Summarize generation wall-clock times so pass@k-style i.i.d. sampling
#      and OPTS tree-search can be compared directly.
#
# Generation knobs (checkpoint, budget, reward mode) are set inside the two
# generation scripts; eval knobs are the variables directly below.
# Set SKIP_GEN=1 to bypass steps 1 and 2 (evaluate-only on existing parquets).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${LLM_DIR}"

STEP=400
N_SAMPLES=128
PASSCONS_K=32
OPTS_KS="8 16 32 64 128"
OPTS_KS_TAG="${OPTS_KS// /-}"
SKIP_GEN=0
# max_search_per_tree values to generate/evaluate; results are saved with a
# "_s${s}" filename tag so different search depths can be compared.
OPTS_MAX_SEARCHES=${OPTS_MAX_SEARCHES:-"3 7"}

OUT_ROOT="results/step${STEP}"
GEN_ROOT="${OUT_ROOT}/gen"
EVAL_ROOT="${OUT_ROOT}/eval"
LOG_ROOT="logs/step${STEP}"
mkdir -p "${EVAL_ROOT}"

# Match the actual filenames produced by scripts/run_parallel_generation.sh.
METHODS=("dapo_0703_n8" "ppo_0704_n8" "reinforce_pp_baseline_0703_n8" "opts_ttpo_exp8_3_0810_n8")

if [[ "${SKIP_GEN}" != "1" ]]; then
    echo "========== Stage 1: i.i.d. generation for ${METHODS[*]} =========="
    bash "${SCRIPT_DIR}/run_parallel_generation.sh"

    echo "========== Stage 2: OPTS tree-search generation for opts_ttpo =========="
    for s in ${OPTS_MAX_SEARCHES}; do
        REWARD_MODE=reward MAX_SEARCH_PER_TREE=${s} bash "${SCRIPT_DIR}/run_opts_generation.sh"
        REWARD_MODE=value MAX_SEARCH_PER_TREE=${s} bash "${SCRIPT_DIR}/run_opts_generation.sh"
    done
fi

echo "========== Stage 3: Task 1 — avg@${PASSCONS_K}, pass@${PASSCONS_K}, cons@${PASSCONS_K} =========="
for method in "${METHODS[@]}"; do
    parquet="${GEN_ROOT}/${method}_iid_n${N_SAMPLES}.parquet"
    if [[ ! -f "${parquet}" ]]; then
        echo "[missing] ${parquet} — re-run without SKIP_GEN=1" >&2
        continue
    fi
    echo "--- ${method} ---"
    python3 -m trainer.main_eval \
        --pregenerated_parquet "${parquet}" \
        --metrics avg pass cons --k ${PASSCONS_K} \
        --output_tag "task1_avg-pass-cons_k${PASSCONS_K}" \
        --output_dir "${EVAL_ROOT}"
done

echo "========== Stage 3: Task 2 — opts@k (reward) + pass@k (i.i.d.) for opts_ttpo =========="
iid_parquet="${GEN_ROOT}/opts_ttpo_exp8_3_0810_n8_iid_n${N_SAMPLES}.parquet"
for s in ${OPTS_MAX_SEARCHES}; do
    reward_opts_parquet="${GEN_ROOT}/opts_ttpo_opts_reward_s${s}_n${N_SAMPLES}.parquet"
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

echo "========== Stage 3: Task 3 — opts@k (value) + cons@k (i.i.d.) for opts_ttpo =========="
for s in ${OPTS_MAX_SEARCHES}; do
    value_opts_parquet="${GEN_ROOT}/opts_ttpo_opts_value_s${s}_n${N_SAMPLES}.parquet"
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
