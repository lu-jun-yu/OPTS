#!/usr/bin/env bash
# Direct V=0 gradients, M=1 bias/coverage, and same-tree prefix credit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=${PYTHON_BIN:-python}
ROOT=${RQ3_ROOT:-results/rq3}
MODEL=${ACTOR:-results/step400/merged/opts_ttpo_actor}
PROMPTS=${PROMPTS:-data/train.parquet}
ACCS_ON=${ACCS_ON:-gpu}
read -r -a GPU_IDS <<< "${GPUS:-0 1 2 3}"
GRAD=${ROOT}/gradients
MASK=${ROOT}/allocation.npz
mkdir -p "${GRAD}"/{est,opts,gstar} "${ROOT}/logs"
# Check all controls before creating coefficients or starting gradient work.
for shard in {0..7}; do
    "${PY}" -m experiments.RQ3.validate_controls \
        --fixed "${ROOT}/controls/fixed_full_shard${shard}.parquet" \
        --gstar "${ROOT}/controls/gstar_full_shard${shard}.parquet" \
        --c "${ROOT}/opts/opts_bias_s7_t8_full_shard${shard}.parquet" \
        --prompts "${PROMPTS}" --model "${MODEL}" --prompt-offset=$((shard*2048)) \
        --output "${ROOT}/controls/validation_shard${shard}.json"
done
"${PY}" -m experiments.RQ3.allocation --opts-dir "${ROOT}/opts" --prompts "${PROMPTS}" \
    --tree-data "${ROOT}"/controls/fixed_full_shard*.parquet --out "${MASK}"
"${PY}" -m experiments.RQ3.coefficients --in-dir "${ROOT}/opts" --out-dir "${ROOT}/coefficients"
worker() {
    local slot=$1 gpu=${GPU_IDS[$1]} group kind count mode
    for kind in est opts gstar; do
        count=8; [[ ${kind} != gstar ]] || count=32
        for ((group=slot; group<count; group+=${#GPU_IDS[@]})); do
            mode=treegrad
            extra=()
            case ${kind} in
                est) data=("${ROOT}"/controls/fixed_full_shard*.parquet); extra=(--grad-ckpt) ;;
                gstar) data=("${ROOT}"/controls/gstar_full_shard*.parquet); extra=(--backbone-only) ;;
                opts) data=("${ROOT}"/coefficients/*_max.pt); mode=optsgrad ;;
            esac
            CUDA_VISIBLE_DEVICES=${gpu} "${PY}" -m experiments.RQ3.gradients \
                --mode "${mode}" --model "${MODEL}" --prompts "${PROMPTS}" \
                --group "${group}" --accs-on "${ACCS_ON}" --out-dir "${GRAD}/${kind}" \
                --tree-data "${data[@]}" --mask-file "${MASK}" "${extra[@]}" \
                > "${ROOT}/logs/${kind}_g${group}.log" 2>&1 || return 1
        done
    done
}
pids=()
for slot in "${!GPU_IDS[@]}"; do worker "${slot}" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
((failed == 0)) || exit 1
"${PY}" -m experiments.RQ3.metrics --grad "${GRAD}"
"${PY}" -m experiments.RQ3.coverage --mask "${MASK}" --bias-root "${ROOT}/opts" \
    --fixed-data "${ROOT}"/controls/fixed_full_shard*.parquet --out "${ROOT}/coverage.json"
"${PY}" -m experiments.RQ3.credit --in-dir "${ROOT}/opts" --out-dir "${ROOT}/credit"
"${PY}" -m experiments.RQ3.plot --metrics "${GRAD}/metrics.json" \
    --coverage "${ROOT}/coverage.json" --credit "${ROOT}/credit/credit_stats.json" \
    --output "${ROOT}/figures/mechanism.pdf"
