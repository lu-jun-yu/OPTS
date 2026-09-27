# RQ1: Tree-gradient estimation

Paper protocol: frozen step-400 Qwen3-1.7B PPO actor, 16,384 training prompts,
eight midpoint-branch tree groups, K={0,1,3,7,15}, and 32 independent chain
groups for the reference gradient. Use global token-level aggregation,
M={1,2,4}, binary returns, gamma=lambda=1, and no learned baseline.

- `estimation.py`: sampling, gradient accumulation, and metrics.
- `run_analysis.sh`: accumulate gradients and compute metrics from sampled trees.
- `plot.py`: plot the metrics using the existing paper style.

Use the existing LLM environment. From `LLM/`, the default inputs are
`data/train.parquet` and the merged PPO actor at
`results/step400/merged/ppo_0704_n8_actor`.

Generate eight estimator groups and 32 independent reference groups in eight
prompt shards. The reference groups use a separate sampling seed:

```bash
for shard in {0..7}; do
    python experiments/RQ1/estimation.py --mode gen \
        --shard "$shard" --num-shards 8 --n-groups 8 --seed-base 20260824 \
        --out "results/step400/rq1/tree_gen_train/tree_data_train_s${shard}.parquet"
    python experiments/RQ1/estimation.py --mode gen --backbone-only \
        --shard "$shard" --num-shards 8 --n-groups 8 --seed-base 20261115 \
        --out "results/step400/rq1/tree_gen_train_gstar/tree_data_gstar_s${shard}.parquet"
    python experiments/RQ1/estimation.py --mode gen --backbone-only \
        --shard "$shard" --num-shards 8 --n-groups 24 --group-offset 8 --seed-base 20261115 \
        --out "results/step400/rq1/tree_gen_train_gstar/tree_data_gstar_e_s${shard}.parquet"
done

bash experiments/RQ1/run_analysis.sh
python experiments/RQ1/plot.py
```

The gradient launcher defaults to four GPUs (0–3); `GPU_LIST` selects the four
devices, `PYTHON_BIN` selects Python, and `OUT_ROOT` selects gradient storage.
Metrics are written to `results/step400/rq1/rq1_train16k_global32_metrics.json`;
figures are written to `paper/figures/`.

The fixed-token-128 E1/E2 mechanism experiment is in `../RQ3/`.
