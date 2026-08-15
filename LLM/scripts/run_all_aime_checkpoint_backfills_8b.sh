#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/backfill_aime24_aime26

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

bash scripts/run_aime_checkpoint_backfill_8b.sh dapo 0,1 \
    >logs/backfill_aime24_aime26/launcher_dapo_8b.log 2>&1 &
PIDS+=("$!")
echo "dapo_8b pid=$! gpu=0,1"

bash scripts/run_aime_checkpoint_backfill_8b.sh reinforce 2,3 \
    >logs/backfill_aime24_aime26/launcher_reinforce_8b.log 2>&1 &
PIDS+=("$!")
echo "reinforce_8b pid=$! gpu=2,3"

status=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"
