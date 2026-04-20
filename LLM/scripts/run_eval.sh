#!/usr/bin/env bash
# Full step-300 evaluation pipeline.
#
#   1) Call scripts/run_parallel_generation.sh — merge FSDP actors + generate
#      N_SAMPLES (128) i.i.d. responses per prompt for DAPO, GPG, PPO,
#      REINFORCE++, OPTS-TTPO.
#   2) Call scripts/run_opts_generation.sh for each reward mode in
#      OPTS_REWARD_MODES — merge OPTS-TTPO actor+critic + run
#      trainer.main_opts_generation with n_samples=128 (OPTS tree search).
#   3) Score every parquet with trainer.main_eval --pregenerated_parquet:
#        - Task 1: avg@PASSCONS_K, pass@PASSCONS_K, cons@PASSCONS_K over the
#          first PASSCONS_K of N_SAMPLES responses (default K=32).
#        - Task 2: opts@k (from reward-guided OPTS parquet) and pass@k (from
#          OPTS-TTPO's i.i.d. parquet) for each k in OPTS_KS
#          (default 8 16 32 64 128).
#        - Task 3: opts@k (from value-guided OPTS parquet) and cons@k (from
#          OPTS-TTPO's i.i.d. parquet) for each k in OPTS_KS.
#   4) Summarize generation wall-clock times so pass@k-style i.i.d. sampling
#      and OPTS tree-search can be compared directly.
#
# Ray modes (forwarded to the generation scripts via env vars):
#   MODE=local (default)         — single-node single-GPU end-to-end
#   MODE=train NNODES=2 \        — drive merge+gen on an already-started
#       GPUS_PER_NODE=1 \          two-machine single-GPU Ray cluster.
#       RAY_HEAD_ADDR=<head_ip>    Start the cluster first with
#                                  `MODE=head` / `MODE=worker bash scripts/run_parallel_generation.sh`
#                                  (or `scripts/run_opts_generation.sh`) on the respective machines.
#
# Set SKIP_GEN=1 to bypass steps 1 and 2 (evaluate-only on existing parquets).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${LLM_DIR}"

MODEL_SIZE="${MODEL_SIZE:-1.7B}"
STEP="${STEP:-300}"
N_SAMPLES="${N_SAMPLES:-128}"
PASSCONS_K="${PASSCONS_K:-32}"
OPTS_KS="${OPTS_KS:-8 16 32 64 128}"
OPTS_REWARD_MODES="${OPTS_REWARD_MODES:-reward value}"
OPTS_KS_TAG="${OPTS_KS// /-}"

OUT_ROOT="${OUT_ROOT:-outputs/step${STEP}}"
GEN_ROOT="${GEN_ROOT:-${OUT_ROOT}/gen}"
EVAL_ROOT="${EVAL_ROOT:-${OUT_ROOT}/eval}"
LOG_ROOT="${LOG_ROOT:-logs/step${STEP}}"
mkdir -p "${EVAL_ROOT}"

# Propagate common vars (generation + Ray) into the two sub-scripts. The Ray
# knobs below are no-ops when MODE=local (default) but let the orchestrator
# drive a two-machine Ray cluster without re-setting them on each call.
export MODEL_SIZE STEP N_SAMPLES OUT_ROOT GEN_ROOT LOG_ROOT
export MODE="${MODE:-local}"
export NNODES="${NNODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
export RAY_HEAD_ADDR="${RAY_HEAD_ADDR:-127.0.0.1}"
export RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"

METHODS=("dapo" "gpg" "ppo" "reinforce_pp" "opts_ttpo")
read -r -a OPTS_REWARD_MODE_ARR <<< "${OPTS_REWARD_MODES}"

if [[ "${SKIP_GEN:-0}" != "1" ]]; then
    echo "========== Stage 1: i.i.d. generation for ${METHODS[*]}  (MODE=${MODE}) =========="
    bash "${SCRIPT_DIR}/run_parallel_generation.sh"

    echo "========== Stage 2: OPTS tree-search generation for opts_ttpo  (MODE=${MODE}) =========="
    for reward_mode in "${OPTS_REWARD_MODE_ARR[@]}"; do
        echo "--- opts_ttpo reward_mode=${reward_mode} ---"
        REWARD_MODE="${reward_mode}" OPTS_GEN_TAG="${reward_mode}" \
            bash "${SCRIPT_DIR}/run_opts_generation.sh"
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
iid_parquet="${GEN_ROOT}/opts_ttpo_iid_n${N_SAMPLES}.parquet"
reward_opts_parquet="${GEN_ROOT}/opts_ttpo_opts_reward_n${N_SAMPLES}.parquet"
if [[ -f "${reward_opts_parquet}" ]]; then
    echo "--- opts_ttpo OPTS parquet (reward) ---"
    python3 -m trainer.main_eval \
        --pregenerated_parquet "${reward_opts_parquet}" \
        --metrics opts --k ${OPTS_KS} \
        --output_tag "task2_reward_opts_k${OPTS_KS_TAG}" \
        --output_dir "${EVAL_ROOT}"
else
    echo "[missing] ${reward_opts_parquet}" >&2
fi
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
value_opts_parquet="${GEN_ROOT}/opts_ttpo_opts_value_n${N_SAMPLES}.parquet"
if [[ -f "${value_opts_parquet}" ]]; then
    echo "--- opts_ttpo OPTS parquet (value) ---"
    python3 -m trainer.main_eval \
        --pregenerated_parquet "${value_opts_parquet}" \
        --metrics opts --k ${OPTS_KS} \
        --output_tag "task3_value_opts_k${OPTS_KS_TAG}" \
        --output_dir "${EVAL_ROOT}"
else
    echo "[missing] ${value_opts_parquet}" >&2
fi
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
