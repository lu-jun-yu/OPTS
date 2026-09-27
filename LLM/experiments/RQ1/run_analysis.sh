#!/usr/bin/env bash
# RQ1 全局 token-level 归一化版（实验侧 8 组 + 独立 g* 32 组，M=1/2/4）：
#   A. 实验侧 treegrad --pg-norm global（旧 8 组中点树，tree_gen_train）
#   B. g* 侧 treegrad --pg-norm global --backbone-only（32 组，tree_gen_train_gstar）
#   C. groupmetrics --pg-norm global（per-M bias/var/mse/cos + 恒等式校验）
# 幂等：已存在的输出自动跳过。采样数据全部复用，不重采。
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
PY=${PYTHON_BIN:-python}
EXP="${SCRIPT_DIR}/estimation.py"
MODEL=results/step400/merged/ppo_0704_n8_actor
PROMPTS=data/train.parquet
EST_GEN=results/step400/rq1/tree_gen_train
GSTAR_GEN=results/step400/rq1/tree_gen_train_gstar
# 组梯度文件很大（est 单组 ~60G x 8 + gstar 单组 ~7G x 32），/dev/shm 放不下时
# 用 NFS。可用 OUT_ROOT 环境变量覆盖（例如指回 /dev/shm/rq1_global）。
OUT_ROOT=${OUT_ROOT:-results/step400/rq1/global}
EST_DIR=${OUT_ROOT}/est
GSTAR_DIR=${OUT_ROOT}/gstar
LOG=logs/step400/rq1
mkdir -p "${EST_DIR}" "${GSTAR_DIR}" "${LOG}"
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "OUT_ROOT=${OUT_ROOT}"

# ---- A. 实验侧 treegrad（8 组，global 口径，4 卡两波）----
NGPU=${NGPU:-4}
# 物理卡映射：用 GPU_LIST 环境变量指定（空格分隔），默认 0..NGPU-1
GPU_LIST=(${GPU_LIST:-$(seq -s' ' 0 $((NGPU - 1)))})
log "A: estimator treegrad (8 groups, global, ${NGPU} GPUs)"
for wave in 0 1; do
    pids=()
    for slot in $(seq 0 $((NGPU - 1))); do
        g=$((wave * NGPU + slot))
        (
            f="${EST_DIR}/treegrad_g$(printf %02d "${g}")_rank0.pt"
            [[ -f "${f}" ]] && exit 0
            CUDA_VISIBLE_DEVICES=${GPU_LIST[${slot}]} "${PY}" "${EXP}" --mode treegrad --model "${MODEL}" \
                --group "${g}" --accs-on gpu --pg-norm global --prompts "${PROMPTS}" \
                --tree-data "${EST_GEN}"/tree_data_train_s*.parquet --out-dir "${EST_DIR}" \
                > "${LOG}/tg_global_est_g0${g}.log" 2>&1 || echo "[treegrad] est group ${g} FAILED"
        ) &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "${p}"; done
    log "A wave ${wave} done"
done
n=$(ls "${EST_DIR}"/treegrad_g*_rank0.pt 2>/dev/null | wc -l)
log "A done, ${n} estimator group files"
[[ "${n}" -eq 8 ]] || { echo "estimator treegrad incomplete" >&2; exit 1; }

# ---- B. g* 侧 treegrad（32 组 backbone-only，global 口径，4 卡 8 波）----
log "B: gstar treegrad (32 groups, backbone-only, global, ${NGPU} GPUs)"
n_waves=$(( 32 / NGPU ))
for wave in $(seq 0 $((n_waves - 1))); do
    pids=()
    for slot in $(seq 0 $((NGPU - 1))); do
        g=$((wave * NGPU + slot))
        (
            f="${GSTAR_DIR}/treegrad_g$(printf %02d "${g}")_rank0.pt"
            [[ -f "${f}" ]] && exit 0
            if [[ "${g}" -lt 8 ]]; then
                DATA_GLOB="${GSTAR_GEN}"/tree_data_gstar_s*.parquet
            else
                DATA_GLOB="${GSTAR_GEN}"/tree_data_gstar_e_s*.parquet
            fi
            CUDA_VISIBLE_DEVICES=${GPU_LIST[${slot}]} "${PY}" "${EXP}" --mode treegrad --model "${MODEL}" \
                --group "${g}" --accs-on gpu --pg-norm global --prompts "${PROMPTS}" \
                --backbone-only \
                --tree-data ${DATA_GLOB} --out-dir "${GSTAR_DIR}" \
                > "${LOG}/tg_global_gstar_g$(printf %02d "${g}").log" 2>&1 || echo "[treegrad] gstar group ${g} FAILED"
        ) &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "${p}"; done
    log "B wave ${wave} done"
done
n=$(ls "${GSTAR_DIR}"/treegrad_g*_rank0.pt 2>/dev/null | wc -l)
log "B done, ${n} gstar group files"
[[ "${n}" -eq 32 ]] || { echo "gstar treegrad incomplete" >&2; exit 1; }

# ---- C. groupmetrics（global 口径，独立 g* 32 组）----
log "C: groupmetrics (global, independent g* 32 groups)"
"${PY}" "${EXP}" --mode groupmetrics --pg-norm global \
    --in-dir "${EST_DIR}" --gstar-dir "${GSTAR_DIR}" \
    --ms 1,2,4 --out-dir results/step400/rq1 --tag rq1_train16k_global32_metrics.json \
    2>&1 | tee "${LOG}/groupmetrics_global32.log"
log "ALL DONE"
