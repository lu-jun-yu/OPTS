# Run all MuJoCo continuous action tasks with multiple seeds in parallel

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEEDS=(1 2 3 4 5)

for task in Ant-v4 Humanoid-v4; do
    for seed in "${SEEDS[@]}"; do
        python cleanrl/cleanrl/opts_ttpo_continuous_action.py \
            --env-id $task \
            --total-timesteps 3000000 \
            --num-steps 4096 \
            --num-envs 1 \
            --max-search-per-tree 4 \
            --no-cuda \
            --seed $seed &
    done
done

for task in HalfCheetah-v4 Walker2d-v4 Hopper-v4; do
    for seed in "${SEEDS[@]}"; do
        python cleanrl/cleanrl/opts_ttpo_continuous_action.py \
            --env-id $task \
            --total-timesteps 3000000 \
            --num-steps 4096 \
            --num-envs 1 \
            --max-search-per-tree 4 \
            --no-cuda \
            --seed $seed &
    done
done