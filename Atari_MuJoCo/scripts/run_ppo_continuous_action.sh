# Run MuJoCo continuous-action tasks with PPO default rollout settings.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEEDS=(1 2 3 4 5 6 7 8 9 10)
TASKS=(Walker2d-v4 Hopper-v4 HalfCheetah-v4 Ant-v4 Humanoid-v4)

total_timesteps=1000000
num_steps=2048
num_minibatches=32

RESULTS_ROOT=/data/results/1_${num_steps}

skip_if_done() {
    # $1=algo dir, $2=task, $3=seed
    if [ -f "$RESULTS_ROOT/$1/$2_$3.json" ]; then
        echo "Skip (exists): $RESULTS_ROOT/$1/$2_$3.json"
        return 0
    fi
    return 1
}

for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ALGO=ppo_continuous_action_20260413
        skip_if_done "$ALGO" "$task" "$seed" && continue
        python cleanrl/cleanrl/ppo_continuous_action.py \
            --env-id "$task" \
            --total-timesteps "$total_timesteps" \
            --num-steps "$num_steps" \
            --num-minibatches "$num_minibatches" \
            --no-cuda \
            --seed "$seed" &
    done
done

wait
echo "OPTS-TTPO continuous-action runs done"
