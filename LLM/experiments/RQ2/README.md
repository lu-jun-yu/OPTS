# RQ2: OPTS search quality and test-time scaling

| Experiment | Implementation | Launch scripts |
| --- | --- | --- |
| Exact-value deterministic MDP | `exact_mdp.py` | `run_exact.sh` |
| Learned-critic search-budget scaling | `search.py`, `guidance.py` | `run_search.sh`, `run_guidance.sh` |
| Matched rollout-budget scaling | shared `trainer/main_opts_generation.py` | `run_generate.sh`, `run_scaling.sh` |

Use the existing LLM environment. The LLM experiments require `data/test.parquet`
(902 problems) and the frozen step-400 Qwen3-1.7B OPTS-TTPO actor and critic.

From `LLM/`, run the exact-value experiment, then the learned-critic experiment:

```bash
bash experiments/RQ2/run_exact.sh
bash experiments/RQ2/run_search.sh
bash experiments/RQ2/run_guidance.sh
python visual/plot_e5_exact_learned.py
```

The exact experiment uses 32 depth-4 configurations, lambda={0,.3,.6,.95}, and
budgets 0–15. Learned-critic search uses 32 trees per prompt, lambda=.999, and
S={0,1,3,7,15}. The value-guided run requires the reward-guided output and
reuses its root-response texts.

For matched budgets, first generate the IID baseline, then run reward-guided
and value-guided OPTS at S=3 and k={8,16,32,64,128}; `run_scaling.sh` also plots:

```bash
MODEL_SIZE=1.7B STEP=400 METHODS=opts_ttpo_exp8_3_0810_n8 TOP_K=-1 \
    bash scripts/run_parallel_generation.sh
bash experiments/RQ2/run_scaling.sh
```

Shared generation/evaluation utilities are in `scripts/` and `trainer/`;
plotting scripts are in `visual/`.

Additional experiments: `checkpoint_scaling.py` / `run_checkpoint_scaling.sh`
for checkpoint scaling and `run_position_ablation.sh` for the S=7 position
ablation. `run_scaling_archive.sh` archives existing outputs before rerunning.
