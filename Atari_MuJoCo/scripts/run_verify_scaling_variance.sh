# Run Verification 2: PPO vs OPTS scaling variance

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEEDS=(1 2 3 4 5)

for task in HalfCheetah-v4 Walker2d-v4 Hopper-v4 Ant-v4 Humanoid-v4; do
    for seed in "${SEEDS[@]}"; do
        python experiments/verify_scaling_variance.py \
            --env-id $task \
            --seed $seed \
            --total-steps 1000000 \
            --num-steps 10000 \
            --batch-sizes 64,128,256,512,1024,2048,4096 \
            --num-bootstrap 100 \
            --max-search-per-tree 2 \
            --no-cuda &
        # wait
    done
    # wait
done
