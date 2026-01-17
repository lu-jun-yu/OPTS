for seed in {1..10}; do
    python cleanrl/opts_ttpo_continuous_action.py \
        --env-id HalfCheetah-v4 \
        --total-timesteps 1000000 \
        --learning-rate 3e-4 \
        --root_tuct 0.5 \
        --num_steps 4096 \
        --seed $seed
done

for seed in {1..10}; do
    python cleanrl/ppo_continuous_action.py \
        --env-id HalfCheetah-v4 \
        --total-timesteps 1000000 \
        --learning-rate 3e-4 \
        --root_tuct 0.5 \
        --num_steps 4096 \
        --seed $seed
done