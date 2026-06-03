# Run MuJoCo continuous-action tasks with PPO default rollout settings.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEEDS=(1 2 3 4 5)
TASKS=(Walker2d-v4 Hopper-v4 HalfCheetah-v4 Ant-v4 Humanoid-v4)

total_timesteps=1000000
num_steps=2048
num_minibatches=32

for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
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
