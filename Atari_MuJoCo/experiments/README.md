# MuJoCo Reproduction Guide

This directory provides reproducible MuJoCo training, checkpointing, policy evaluation, and statistical analysis. Run the commands below from the repository root. The PPO and OPTS-TTPO implementations are in `../cleanrl/cleanrl/`; experiment scripts here call those implementations directly.

For a first run, check the Python environment, run the tests, preview the job queue, and complete a small smoke test before launching a full 50-seed sweep. The full default sweep contains 2,500 independent training/evaluation jobs, so choose the worker count based on available CPU and memory.

## Files

| File | Purpose |
| --- | --- |
| `run_queue.py` | Queue training and evaluation jobs (up to 80 at once); preview, pause, and resume jobs |
| `worker.py` | Run one configuration, validate its outputs, and evaluate the final policy |
| `mujoco_config.py` | Configure rollout sizes, environments, methods, and result paths |
| `axap_checkpoint.py` | Atomic intermediate/final checkpoints and normalization statistics |
| `evaluate.py` | Deterministic policy evaluation using raw environment rewards, without search |
| `summarize.py` | Matched-seed per-task and per-10-seed summaries, with paired bootstrap intervals |
| `test_mujoco.py` | Check timeout handling, TreeGAE, and experiment configuration |
| `test_statistics.py` | Check statistical summaries, input validation, and command-line use |

## Protocol

| Setting | Rollout size | Requested steps | Actual steps | Minibatch size | Rollout iterations | Optimizer steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `2048_1m` | 2048 | 1,000,000 | 999,424 | 64 | 488 | 156,160 |
| `4096_2m` | 4096 | 2,000,000 | 1,998,848 | 128 | 488 | 156,160 |
| `8192_4m` (select with `--settings`) | 8192 | 4,000,000 | 3,997,696 | 256 | 488 | 156,160 |

Each run uses one environment, 32 minibatches per rollout, and 10 optimization epochs. The learning rate is 3e-4, discount factor 0.99, and GAE lambda 0.95; KL early stopping is disabled. Larger settings double both rollout samples and environment interactions while keeping the optimizer-step count fixed. Compare them as different data budgets, not as fixed-interaction-budget experiments.

Tasks: Humanoid-v4, Ant-v4, HalfCheetah-v4, Walker2d-v4, and Hopper-v4. Methods: `ppo` (baseline), `full` (performance-difference selection, max backup, equal branch weights), `unweighted` (same search and backup with unweighted aggregation), `random_mean` (random selection, mean backup), and `guided_mean` (performance-difference selection, mean backup). Search uses xi=0.6, a mean baseline, and at most one additional search per tree.

By default, the queue runs `2048_1m` and `4096_2m` for every task and method with training seeds 1–50 (2,500 jobs). Mode `zero` assigns a value of zero at time-limit truncations. Mode `bootstrap` instead estimates the value from the final observation when a time limit is reached without true termination. Both modes use ordinary value bootstrap at nonterminal rollout boundaries. Selecting both modes doubles the default queue to 5,000 jobs. The summary script reads existing results; it does not launch missing jobs.

## Environment

The reference environment uses Python 3.10, PyTorch 2.4.1, Gymnasium 0.29.1, MuJoCo 2.3.3, NumPy 2.2.6, Tyro 1.0.15, and TensorBoard 2.21.0. Training expects the Gymnasium 0.29 final-observation API and the v4 MuJoCo environments. Install the dependencies before running the scripts; they do not modify your environment.

In a fresh Python 3.10 environment, install the packages used by these MuJoCo scripts:

```bash
python -m pip install "torch==2.4.1" "gymnasium==0.29.1" "mujoco==2.3.3" "numpy==2.2.6" "tyro==1.0.15" "tensorboard==2.21.0"
```

`summarize.py` needs only Python and NumPy. `test_mujoco.py` also needs PyTorch, and training/evaluation need the full environment. Save results outside the source directories; every job creates its own training logs, checkpoints, and evaluation files.

Verify the Python interpreter and run the lightweight tests before training:

```bash
python --version
python Atari_MuJoCo/experiments/test_mujoco.py
python Atari_MuJoCo/experiments/test_statistics.py
```

## Inspect and launch

Preview the job count and selected configurations without starting training or creating a results directory:

```bash
python Atari_MuJoCo/experiments/run_queue.py --results-root ./results/mujoco --dry-run
```

Run a small end-to-end check first: this runs two rollout iterations, saves checkpoints, and performs two evaluation episodes. The check writes to a separate `smoke/` directory:

```bash
python Atari_MuJoCo/experiments/worker.py 2048_1m zero full HalfCheetah-v4 1 --results-root ./results/mujoco --smoke --eval-episodes 2
```

Run the default sweep (both settings, five methods, five tasks, 50 seeds). The example uses 80 concurrent workers; reduce `--max-workers` on smaller machines:

```bash
python Atari_MuJoCo/experiments/run_queue.py --results-root ./results/mujoco --max-workers 80
```

Run the larger 8192/4M setting separately (1,250 jobs):

```bash
python Atari_MuJoCo/experiments/run_queue.py --results-root ./results/mujoco_8192_4m --settings 8192_4m --modes zero --seed-start 1 --seed-end 50 --max-workers 80
```

For a smaller run, select settings, timeout modes, methods, tasks, and seeds explicitly:

```bash
python Atari_MuJoCo/experiments/run_queue.py --results-root ./results/mujoco --settings 2048_1m --modes zero bootstrap --variants ppo full --envs HalfCheetah-v4 --seed-start 1 --seed-end 10 --max-workers 8
```

Each worker hides GPUs and constrains numeric-library threads to one. The queue limit includes evaluation because evaluation runs inside the same worker slot. A results-root lock prevents overlapping queues on the same root, and each job has its own OS lock. Running multiple queues with different roots requires managing their combined resource use separately.

The queue verifies a completed job's configuration, source hashes, training curve, checkpoints, evaluation metadata, and output hashes before skipping it. To rerun a failed or incomplete job, use `--retry-failed`. Its earlier output is placed in `previous_attempts/`, and training restarts from the specified seed (the simulator state is not restored exactly).

For each run, outputs are stored under `RESULTS_ROOT/jobs/<setting>/<mode>/<variant>/<environment>/seed_<NN>/`. Look for `config.json`, `training.log`, `checkpoints/`, `independent_eval/results.json`, and `complete.json`. A job writes `failure.json` on failure; check the logs before retrying. Smoke-test jobs use `smoke/` instead of `jobs/`.

Creating an empty `PAUSE_NEW_JOBS` file under the results root stops new launches. Active jobs finish and the queue exits with code 2. Remove the marker and rerun the same command to continue. Any failed worker makes the queue return nonzero, with details in `queue_logs`, the job logs and `failure.json`.

## Checkpoints and metrics

Each full run saves a checkpoint every 50 rollout iterations and a final checkpoint at iteration 488 (10 files per job). Checkpoints contain the model, optimizer, observation/reward normalization statistics, run arguments, training seed, timeout mode, and PyTorch/NumPy RNG states. They omit simulator and tree-search state, so loading a checkpoint does not exactly resume an interrupted training run.

You can run an algorithm entrypoint directly with `--checkpoint-every 0` to disable periodic checkpoints. `worker.py` enables checkpointing and sets the per-job output directory through `AXAP_JOB_DIR`. The algorithm's `--save-model` option remains available.

Policy evaluation loads the last checkpoint, freezes observation normalization, and acts with the actor's deterministic mean action without tree search. It reports raw, undiscounted environment return; each episode ends at termination or the environment's 1,000-step limit. For training seed `s`, the ten evaluation seeds are `100000 + 100 * s + i` for `i = 0, ..., 9`, shared across methods. Evaluation returns are never used to select a checkpoint.

Training-tail and independent-evaluation results are distinct. Training-tail averages the last 100 of the 488 logged training points. PPO logs ordinary episode returns. Every tree variant, including `unweighted`, logs branch-weighted returns within each tree and then averages trees equally; `unweighted` changes optimization weights after that logging computation. Training-tail is therefore not an independent no-search policy evaluation.

## Summaries

The summary command currently supports `2048_1m` and `4096_2m`. The `8192_4m` run creates the same per-job output files under its own results root, but the summary command must be extended before analyzing that setting.

After the full default sweep has finished (both supported settings, five methods, five tasks, and matching seeds 1–50):

```bash
python Atari_MuJoCo/experiments/summarize.py --results-root ./results/mujoco --output-dir ./results/mujoco_summary
```

To summarize a previously exported results snapshot without loading the checkpoints:

```bash
python Atari_MuJoCo/experiments/summarize.py --input /path/to/seed50_snapshot.json --output-dir ./results/snapshot_summary
```

The summary requires complete matched seeds for all five tasks/methods and both settings, rejects missing/duplicate/nonfinite records, and refuses to overwrite an existing output directory. It writes `summary.json` and `summary.md`, with per-task mean/SD, each nonoverlapping 10-seed block and the full requested cohort. By default, each metric uses 5000 paired seed-bootstrap resamples independently within each task (base seed 20271109). Relative gains are computed from each task's two method means and then averaged equally over the five tasks; episodes are not treated as separate training seeds. Nonpositive ratio denominators cause an explicit error rather than dropping draws. These intervals are not adjusted for multiple comparisons.

