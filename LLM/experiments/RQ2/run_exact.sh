#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON_BIN=${PYTHON_BIN:-python3}
OUTPUT_DIR=${OUTPUT_DIR:-results/e5_exact_deterministic}
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" -m experiments.RQ2.exact_mdp \
    --output-dir "${OUTPUT_DIR}" \
    "$@" 2>&1 | tee "${OUTPUT_DIR}/run.log"
