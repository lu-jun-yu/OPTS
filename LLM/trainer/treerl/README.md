# TreeRL baseline

This directory ports the public TreeRL baseline to this repository's VeRL
training stack.  It follows the TreeRL paper and the released implementation at
commit `63ea99e0253cc320e53b2b47eb1bcfa8adafd666` for the algorithm-specific
parts:

- EPTree uses `(M, N, L, T) = (6, 2, 1, 2)`.
- Forking points are ranked by sampled-token surprisal, `-log p(y_t)`.
- The selected token is regenerated: a fork at token `t` continues from
  `y_{<t}`.
- Every complete root-to-leaf response is used for training.
- Each tree segment receives `(GA + LA) / sqrt(number of descendant leaves)`.

The common optimizer, rollout, data, verifier, evaluation, and 400-step settings
match the paper's LLM RLVR protocol rather than TreeRL's original Qwen-2.5/OpenRLHF
launcher.  In particular, this matched baseline uses Qwen3, 2,048 response
tokens, temperature 1.0, top-p 1.0, actor learning rate `1e-6`, symmetric PPO
clip 0.2, token-level loss aggregation, and no KL penalty.

## Exact 4,096-rollout budget

A full TreeRL prompt produces `6 * (1 + 2 * 1 * 2) = 30` completed responses.
The trainer therefore uses 136 full TreeRL prompt groups (4,080 responses) and
one deterministic budget-filler prompt with 16 independent roots.  This gives
exactly 4,096 completed responses without changing the TreeRL hyperparameters
for any full tree group.  The filler roots use the same global/local process
advantage; they simply have no branch nodes.

Run a one-update integration check on physical GPU 6:

```bash
bash scripts/run_treerl_smoke.sh
```

Run the matched 400-step baseline (also defaults to physical GPU 6):

```bash
bash scripts/run_treerl.sh
```

