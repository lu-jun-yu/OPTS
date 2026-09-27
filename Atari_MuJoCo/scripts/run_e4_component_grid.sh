#!/usr/bin/env bash
set -u

# E4 component ablations at fixed xi=0.6, s=1; seeds 1--10 for both variants.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PARALLEL=${PARALLEL:-80}
PYTHON=${PYTHON:-python}
RESULTS_ROOT=${RESULTS_ROOT:-/data/results}

TASKS=(Humanoid-v4 Ant-v4 Walker2d-v4 Hopper-v4 HalfCheetah-v4)
XIS=(0.6)
SEARCHES=(1)
SEEDS=(1 2 3 4 5 6 7 8 9 10)

run_one() {
    variant=$1
    task=$2
    xi=$3
    search=$4
    seed=$5

    if [[ "$variant" == "random" ]]; then
        algo="random_ttpo_continuous_action_wEqual-bMean-nLen_s${search}_20260920"
        result_file="$RESULTS_ROOT/1_2048/$algo/${task}_${seed}.json"
        [[ -f "$result_file" ]] && { echo "Skip (exists): $algo/${task}_${seed}"; return 0; }
        "$PYTHON" cleanrl/cleanrl/opts_ttpo_continuous_action_wEqual-bMean-nLen.py \
            --branch-selection random \
            --results-root "$RESULTS_ROOT" \
            --env-id "$task" \
            --total-timesteps 1000000 \
            --num-steps 2048 \
            --num-minibatches 32 \
            --max-search-per-tree "$search" \
            --no-cuda \
            --seed "$seed" > /dev/null 2>&1 \
            && echo "done random task=$task s=$search seed=$seed" \
            || echo "FAILED random task=$task s=$search seed=$seed"
        return
    fi

    algo="opts_ttpo_continuous_action_wNone-bMax-nLen_xi${xi}_s${search}_20260817"
    result_file="$RESULTS_ROOT/1_2048/$algo/${task}_${seed}.json"
    [[ -f "$result_file" ]] && { echo "Skip (exists): $algo/${task}_${seed}"; return 0; }
    "$PYTHON" cleanrl/cleanrl/opts_ttpo_continuous_action_wEqual-bMax-nLen.py \
        --aggregation none \
        --results-root "$RESULTS_ROOT" \
        --env-id "$task" \
        --total-timesteps 1000000 \
        --num-steps 2048 \
        --num-minibatches 32 \
        --max-search-per-tree "$search" \
        --baseline mean \
        --xi "$xi" \
        --no-cuda \
        --seed "$seed" > /dev/null 2>&1 \
        && echo "done naive task=$task xi=$xi s=$search seed=$seed" \
        || echo "FAILED naive task=$task xi=$xi s=$search seed=$seed"
}
export -f run_one
export PYTHON RESULTS_ROOT

jobs_file=$(mktemp "/tmp/e4_component_grid.XXXXXX")
trap 'rm -f "$jobs_file"' EXIT

for task in "${TASKS[@]}"; do
    for search in "${SEARCHES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            printf 'random\t%s\t_\t%s\t%s\n' "$task" "$search" "$seed" >> "$jobs_file"
        done
    done

    for xi in "${XIS[@]}"; do
        for search in "${SEARCHES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                printf 'naive\t%s\t%s\t%s\t%s\n' "$task" "$xi" "$search" "$seed" >> "$jobs_file"
            done
        done
    done
done

job_count=$(wc -l < "$jobs_file")
echo "E4 starting: jobs=$job_count parallel=$PARALLEL results=$RESULTS_ROOT"
xargs -P "$PARALLEL" -n 5 bash -c 'run_one "$@"' _ < "$jobs_file"
echo "E4 finished"
