# Run all MuJoCo continuous action tasks with multiple seeds in parallel

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEEDS=(1 2 3 4 5)

# PPO
for task in HalfCheetah-v4 Walker2d-v4 Hopper-v4 Ant-v4 Humanoid-v4; do
    for seed in "${SEEDS[@]}"; do
        python cleanrl/ppo_continuous_action.py \
            --env-id $task \
            --total-timesteps 3000000 \
            --num-steps 4096 \
            --no-cuda \
            --seed $seed &
    done
done

wait
echo "PPO done"


# RPO
for task in HalfCheetah-v4 Walker2d-v4 Hopper-v4 Ant-v4 Humanoid-v4; do
    for seed in "${SEEDS[@]}"; do
        python cleanrl/rpo_continuous_action.py \
            --env-id $task \
            --total-timesteps 3000000 \
            --num-steps 4096 \
            --no-cuda \
            --seed $seed &
    done
done

wait
echo "RPO done"
