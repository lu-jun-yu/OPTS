# Run Verification 1: positive vs negative trajectory PG variance

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEEDS=(1 2 3 4 5)

for task in HalfCheetah-v4 Walker2d-v4 Hopper-v4 Ant-v4 Humanoid-v4; do
    for seed in "${SEEDS[@]}"; do
        python experiments/verify_pg_variance.py \
            --env-id $task \
            --seed $seed \
            --num-steps 1000000 \
            --alpha 0.2 \
            --num-bootstrap 200 \
            --no-cuda &
    done
    wait
done
wait
