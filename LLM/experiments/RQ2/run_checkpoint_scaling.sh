#!/usr/bin/env bash
# RQ2 learning curve: for each checkpoint step in STEPS, run
# experiments/RQ2/checkpoint_scaling.py (merge + search generation +
# opts-avg@k eval), serially.
#
# Run:  bash experiments/RQ2/run_checkpoint_scaling.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export NCCL_DEBUG=ERROR
export TRANSFORMERS_VERBOSITY=error
export VLLM_LOGGING_LEVEL=WARN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2}

export OPTS_METHOD=${OPTS_METHOD:-opts_ttpo_exp8_3_0810_n8}
export MODEL_SIZE=${MODEL_SIZE:-1.7B}
export TREES_PER_PROMPT=${TREES_PER_PROMPT:-32}
export SEARCH_ROUNDS=${SEARCH_ROUNDS:-3}
export SNAPSHOT_ROUNDS=${SNAPSHOT_ROUNDS:-"1 3"}
export OPTS_AVG_SLICES=${OPTS_AVG_SLICES:-"3"}
export OPTS_AVG_KS=${OPTS_AVG_KS:-"32"}
export BS=${BS:-902}
STEPS=${STEPS:-"20 40 80 160 320"}

for step in ${STEPS}; do
    echo "========== RQ2 learning curve: step=${step} =========="
    python3 experiments/RQ2/checkpoint_scaling.py --step "${step}"
done

echo
echo "Done. Parquets: results/rq2_learn_scaling"
echo "Eval JSONs: results/rq2_learn_scaling/eval"
