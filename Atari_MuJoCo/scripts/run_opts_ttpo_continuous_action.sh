# Run tasks sequentially; use 40 parallel workers per task (5 seeds x 8 searches).

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEEDS=(1 2 3 4 5)
SEARCHES=(1 2 3 4 5 6 7 8)
TASKS=(Walker2d-v4 Hopper-v4 HalfCheetah-v4 Ant-v4 Humanoid-v4)

total_timesteps=1000000
num_steps=2048
num_minibatches=32
tau=0.7

for task in "${TASKS[@]}"; do
    echo "Starting $task with ${#SEEDS[@]} seeds x ${#SEARCHES[@]} searches..."
    for search in "${SEARCHES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            python cleanrl/cleanrl/opts_ttpo_continuous_action.py \
                --env-id "$task" \
                --total-timesteps "$total_timesteps" \
                --num-steps "$num_steps" \
                --num-minibatches "$num_minibatches" \
                --max-search-per-tree "$search" \
                --tau "$tau" \
                --no-cuda \
                --seed "$seed" &
        done
    done
    wait
    echo "$task done"
done

echo "OPTS-TTPO continuous-action runs done"
