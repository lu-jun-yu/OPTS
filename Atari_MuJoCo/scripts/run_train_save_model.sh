# Train OPTS_TTPO on 5 MuJoCo tasks and save checkpoints

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for task in HalfCheetah-v4 Walker2d-v4 Hopper-v4 Ant-v4 Humanoid-v4; do
    python cleanrl/cleanrl/opts_ttpo_continuous_action.py \
        --env-id $task \
        --total-timesteps 1000000 \
        --num-steps 4096 \
        --save-model \
        --no-cuda \
        --seed 1 &
done
wait
