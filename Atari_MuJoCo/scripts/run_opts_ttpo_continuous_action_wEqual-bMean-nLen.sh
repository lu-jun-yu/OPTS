# OPTS-TTPO wEqual-bMean-nLen MuJoCo: fixed xi=0.6, s=1; 10 seeds x 5 tasks, one job queue (40 workers), Humanoid first.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEEDS=(1 2 3 4 5 6 7 8 9 10)
SEARCHES=(1)
TASKS=(Humanoid-v4 Ant-v4 HalfCheetah-v4 Walker2d-v4 Hopper-v4)
XIS=(0.6)
PARALLEL=${PARALLEL:-40}
PYTHON=${PYTHON:-python}

total_timesteps=1000000
num_steps=2048
num_minibatches=32
RESULTS_ROOT=/data/results/1_${num_steps}

run_one() {
    task=$1; xi=$2; search=$3; seed=$4
    algo="opts_ttpo_continuous_action_wEqual-bMean-nLen_xi${xi}_s${search}_20260817"
    [ -f "$RESULTS_ROOT/$algo/${task}_${seed}.json" ] && { echo "Skip (exists): $algo/${task}_${seed}"; return 0; }
    "$PYTHON" cleanrl/cleanrl/opts_ttpo_continuous_action_wEqual-bMean-nLen.py \
        --env-id "$task" --total-timesteps "$total_timesteps" --num-steps "$num_steps" \
        --num-minibatches "$num_minibatches" --max-search-per-tree "$search" \
        --baseline mean --xi "$xi" --no-cuda --seed "$seed" > /dev/null 2>&1 \
        && echo "done $task xi=$xi s=$search seed=$seed" || echo "FAILED $task xi=$xi s=$search seed=$seed"
}
export -f run_one
export RESULTS_ROOT total_timesteps num_steps num_minibatches PYTHON

for task in "${TASKS[@]}"; do for xi in "${XIS[@]}"; do for search in "${SEARCHES[@]}"; do for seed in "${SEEDS[@]}"; do
    echo "$task $xi $search $seed"
done; done; done; done | xargs -P "$PARALLEL" -n 4 bash -c 'run_one "$@"' _

echo "OPTS-TTPO wEqual-bMean-nLen runs done"
