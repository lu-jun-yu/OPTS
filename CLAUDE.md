# OPTS

**OPTS** = **O**n-policy **P**arallel **T**ree **S**earch. Core algorithm: **OPTS-TTPO** (On-policy Parallel Tree Search + **T**ree **T**rajectory **P**olicy **O**ptimization). A general RL algorithm, applied in two domains: classic RL (Atari/MuJoCo) and LLM.

## Core algorithm locations

| Domain | Path |
| --- | --- |
| Atari/MuJoCo shared core (TreeGAE + OTRC selection) | [Atari_MuJoCo/cleanrl/cleanrl/opts_ttpo_core.py](Atari_MuJoCo/cleanrl/cleanrl/opts_ttpo_core.py) |
| MuJoCo (continuous control) main loop | [Atari_MuJoCo/cleanrl/cleanrl/opts_ttpo_continuous_action.py](Atari_MuJoCo/cleanrl/cleanrl/opts_ttpo_continuous_action.py) |
| Atari (discrete control) main loop | [Atari_MuJoCo/cleanrl/cleanrl/opts_ttpo_atari.py](Atari_MuJoCo/cleanrl/cleanrl/opts_ttpo_atari.py) |
| LLM RLVR (training) | [LLM/trainer/opts_ttpo/](LLM/trainer/opts_ttpo/) — package; entry [LLM/trainer/main_opts_ttpo.py](LLM/trainer/main_opts_ttpo.py) |
| LLM test-time scaling | [LLM/trainer/main_opts_generation.py](LLM/trainer/main_opts_generation.py) |

`*_v1.py` files alongside the above are older variants; the non-`v1` files are current.

## Repo layout

- `Atari_MuJoCo/` — classic RL, based on CleanRL. Shared algo (TreeGAE + OTRC selection) in `cleanrl/cleanrl/opts_ttpo_core.py`; per-domain main loops in `opts_ttpo_continuous_action.py` / `opts_ttpo_atari.py`; run scripts in `scripts/`.
- `LLM/` — LLM RL, based on verl.
  - `LLM/verl/` — upstream verl library (do not modify copyright headers).
  - `LLM/trainer/` — custom trainers (OPTS-TTPO). Key pieces: `opts_ttpo/core_algos.py` (TreeGAE advantage), `opts_ttpo/ray_trainer.py` (tree search / training loop).
  - `LLM/utils/reward_fn.py` — reward (`compute_score`, called by verl's `NaiveRewardManager`).
  - `LLM/data_preprocess/` — dataset prep (math12k, aime25, amc23, ...).
  - `LLM/scripts/` — training/eval shell scripts.

## Notes

- All `LLM/` code except `LLM/verl/` is authored by Junyu Lu (Julian Lou).
- Comments: keep sparse — none, or one short line.
- Baselines: compare OPTS-TTPO against DAPO (not DAPO-n8).
