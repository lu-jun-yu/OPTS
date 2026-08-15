#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/backfill_aime24_aime26

declare -a JOBS=(
    "dapo 0"
    "gpg 1"
    "ppo 2"
    "reinforce 3"
    "opts 5,6"
)

declare -a PIDS=()

cleanup() {
    status=$?
    trap - EXIT INT TERM
    for pid in "${PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for job in "${JOBS[@]}"; do
    read -r algorithm gpu_ids <<<"${job}"
    log_path="logs/backfill_aime24_aime26/launcher_${algorithm}.log"
    bash scripts/run_aime_checkpoint_backfill.sh "${algorithm}" "${gpu_ids}" \
        >"${log_path}" 2>&1 &
    PIDS+=("$!")
    echo "${algorithm} pid=$! gpu=${gpu_ids} log=${log_path}"
done

status=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"
