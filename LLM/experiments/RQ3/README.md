# RQ3: Coverage–bias and prefix credit

This directory contains the paper's E1/E2 computations: `p=1`, `M=1` theory
aggregation, search snapshots `{0,1,3,7}`, and direct returns with `V=0`.
E1 compares Fixed + TTPG, Fixed + NaivePG, and OPTS + max-backup TTPG.
E2 compares max/mean return backup on the same OPTS trees at λ=1 and 0.999.

Use the existing VeRL environment and the **step-400 Qwen3-1.7B OPTS-TTPO**
actor/critic, not the PPO checkpoint used by RQ1. Defaults are
`results/step400/merged/opts_ttpo_actor`, `opts_ttpo_critic`, and
`data/train.parquet` (16,384 prompts). The merged checkpoints and dataset must
be prepared using the repository's existing setup. OPTS generation reuses
`experiments.RQ2.search`.

From `LLM/`:

```bash
GPUS="0 1 2 3" bash experiments/RQ3/run_generate.sh
GPUS="0 1 2 3" bash experiments/RQ3/run_analysis.sh
```

`PYTHON_BIN`, `ACTOR`, `CRITIC`, `PROMPTS`, and `RQ3_ROOT` override paths;
`GPUS` sets available devices. `ACCS_ON=cpu` stores gradient accumulators in
host memory. Raw full-model gradient files require roughly 0.9 TB of storage;
GPU accumulation for the fixed controls alone takes about 48 GB plus model
and activation memory. Defaults write samples, gradients, JSON summaries and
the two-panel figure under `results/rq3/`. Use a fresh `RQ3_ROOT` for each run.

The input uses a 1,152-token prompt region,
including the 128-token fixed branch prefix, and keeps all 16,384 prompts.
