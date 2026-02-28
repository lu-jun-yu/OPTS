# Run all MuJoCo continuous action tasks with multiple seeds in parallel

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEEDS=(1 2 3 4 5)

# PPO
for task in HalfCheetah-v4 Walker2d-v4 Hopper-v4; do
    for seed in "${SEEDS[@]}"; do
        python cleanrl/ppo_continuous_action.py \
            --env-id $task \
            --total-timesteps 1000000 \
            --num-envs 2 \
            --no-cuda \
            --seed $seed &
    done
done

for task in Ant-v4 Humanoid-v4; do
    for seed in "${SEEDS[@]}"; do
        python cleanrl/ppo_continuous_action.py \
            --env-id $task \
            --total-timesteps 3000000 \
            --num-envs 2 \
            --no-cuda \
            --seed $seed &
    done
done

wait
echo "PPO done"


# RPO
for task in HalfCheetah-v4 Walker2d-v4 Hopper-v4; do
    for seed in "${SEEDS[@]}"; do
        python cleanrl/rpo_continuous_action.py \
            --env-id $task \
            --total-timesteps 1000000 \
            --num-envs 2 \
            --no-cuda \
            --seed $seed &
    done
done

for task in Ant-v4 Humanoid-v4; do
    for seed in "${SEEDS[@]}"; do
        python cleanrl/rpo_continuous_action.py \
            --env-id $task \
            --total-timesteps 3000000 \
            --num-envs 2 \
            --no-cuda \
            --seed $seed &
    done
done

wait
echo "RPO done"


# A2C
for task in HalfCheetah-v4 Walker2d-v4 Hopper-v4; do
    for seed in "${SEEDS[@]}"; do
        python cleanrl/a2c_continuous_action.py \
            --env-id $task \
            --total-timesteps 1000000 \
            --num-envs 2 \
            --no-cuda \
            --seed $seed &
    done
done

for task in Ant-v4 Humanoid-v4; do
    for seed in "${SEEDS[@]}"; do
        python cleanrl/a2c_continuous_action.py \
            --env-id $task \
            --total-timesteps 3000000 \
            --num-envs 2 \
            --no-cuda \
            --seed $seed &
    done
done

wait
echo "A2C done"
