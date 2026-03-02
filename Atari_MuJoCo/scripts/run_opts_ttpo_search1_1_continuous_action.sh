# Run all MuJoCo continuous action tasks with multiple seeds in parallel

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEEDS=(1 2 3 4 5)

for task in HalfCheetah-v4 Walker2d-v4 Hopper-v4; do
    for seed in "${SEEDS[@]}"; do
        python cleanrl/opts_ttpo_continuous_action.py \
            --env-id $task \
            --total-timesteps 1000000 \
            --max-search-per-tree 1 \
            --num-envs 1 \
            --no-cuda \
            --seed $seed &
    done
done

for task in Ant-v4 Humanoid-v4; do
    for seed in "${SEEDS[@]}"; do
        python cleanrl/opts_ttpo_continuous_action.py \
            --env-id $task \
            --total-timesteps 3000000 \
            --max-search-per-tree 1 \
            --num-envs 1 \
            --no-cuda \
            --seed $seed &
    done
done