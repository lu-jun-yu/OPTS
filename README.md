# OPTS: On-policy Parallel Tree Search for Enhanced Policy Optimization

<p align="center">
  <a href="#"><b>Paper</b></a> &nbsp;|&nbsp;
  <a href="#installation"><b>Installation</b></a> &nbsp;|&nbsp;
  <a href="#usage"><b>Usage</b></a> &nbsp;|&nbsp;
  <a href="#citation"><b>Citation</b></a>
</p>

<!-- TODO: Add framework diagram -->
<!-- <p align="center"><img src="assets/framework.png" width="90%"></p> -->

## News

- Coming soon.

## Introduction

This repository provides the official implementation of **OPTS** (On-policy Parallel Tree Search) and **TTPO** (Tree Trajectory Policy Optimization), covering both classical RL (Atari & MuJoCo) and LLM post-training scenarios.

### OPTS (On-policy Parallel Tree Search)

OPTS is a novel tree search method designed for on-policy reinforcement learning. Unlike traditional Monte Carlo Tree Search (MCTS), which exhaustively expands all possible actions and relies on separate tree and rollout policies (off-policy), OPTS builds search trees by iteratively sampling trajectory batches under the current policy itself (on-policy).

Key features of OPTS:
- **Sampling-based expansion**: Instead of enumerating all actions, each round generates a batch of trajectories, progressively building the tree.
- **TUCT (Trajectory-level Upper Confidence bound for Trees)**: A selection criterion that balances exploitation (expected improvement from branching) and exploration (penalizing over-searched nodes) to choose optimal states for the next round of expansion.
- **Backtracking to earlier states**: OPTS can branch from any state along the trajectory, not just leaf nodes, enabling re-exploration from earlier decision points—particularly beneficial in language generation settings.

| Feature | MCTS | OPTS |
|---------|------|------|
| Expansion | Exhaustive over all actions | Sampling-based, one batch per round |
| Policy type | Off-policy (tree policy vs. rollout policy) | On-policy (sampling policy = optimization policy) |
| State selection | Children only | Any state in the tree |
| Backtracking | Post-simulation backup | Supports re-expansion from earlier states |

### TTPO (Tree Trajectory Policy Optimization)

TTPO extends standard policy gradient methods (e.g., PPO) to tree-structured trajectories collected by OPTS, ensuring unbiased gradient estimation:

- **TreeGAE**: Generalizes Generalized Advantage Estimation to tree structures. At branching nodes, the advantage is computed by averaging the first-token advantages across all child branches, enabling advantage information to flow from child trajectories back through the tree.
- **Branch Weight Correction**: Introduces a weight factor $W_t$ that equals the cumulative product of branch counts from the root to each node. The policy gradient is divided by this weight, correcting for the non-uniform sampling frequency induced by branching, thereby guaranteeing an unbiased policy gradient estimate.

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_t \frac{1}{W_t} \nabla_\theta \log \pi_\theta(a_t|s_t) \hat{A}_t \right]
$$

## Installation

### Requirements

- OS: Ubuntu 22.04
- CUDA: >= 12.6
- Python: 3.10

### Clone the Repository

```bash
git clone https://github.com/lu-jun-yu/OPTS.git
cd OPTS
git submodule init
git submodule update
```

### Atari & MuJoCo

**Hardware**: Multi-core CPU is sufficient.

```bash
cd Atari_MuJoCo/cleanrl
conda create -n cleanrl python==3.10
conda activate cleanrl
pip install uv
uv pip install .
uv pip install ".[atari]"
uv pip install ".[mujoco]"
```

### LLM

**Hardware**: >= 8x A100 (80GB) recommended. You may adjust script parameters for lower configurations.

> For more details, refer to the [VERL documentation](https://woniu9524.github.io/verl-doc/start/install.html).

```bash
conda create -n opts_verl python==3.10
conda activate opts_verl
```

**Install CUDA:**

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/

apt-get update
apt-get -y install cuda-toolkit-12-8
apt-get install -y nvidia-open
apt-get install -y cuda-drivers

update-alternatives --set cuda /usr/local/cuda-12.8
```

**Install cuDNN:**

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.20.0/local_installers/cudnn-local-repo-ubuntu2204-9.20.0_1.0-1_amd64.deb
dpkg -i cudnn-local-repo-ubuntu2204-9.20.0_1.0-1_amd64.deb
cp /var/cudnn-local-repo-ubuntu2204-9.20.0/cudnn-*-keyring.gpg /usr/share/keyrings/

apt-get update
apt-get -y install cudnn
apt-get -y install cudnn9-cuda-12
```

**Install dependencies:**

```bash
cd LLM/verl
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math_verify
```

**Install NVIDIA Apex:**

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
# Set MAX_JOB according to the number of CPU cores on your machine (avoid setting it too high)
MAX_JOB=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

**Install VERL:**

```bash
cd LLM/verl
pip install --no-deps -e .
```

## Usage

### Atari & MuJoCo

**Run baselines:**

```bash
cd Atari_MuJoCo/cleanrl
../scripts/run_ppo_atari.sh                       # Requirement: CPU Cores > 57
../scripts/run_all_baselines_continuous_action.sh # Requirement: CPU Cores > 25
```

**Run OPTS-TTPO:**

```bash
cd Atari_MuJoCo/cleanrl
../scripts/run_opts_ttpo_atari.sh                 # Requirement: CPU Cores > 57
../scripts/run_opts_ttpo_continuous_action.sh     # Requirement: CPU Cores > 25
```

**Visualization:**

```bash
cd Atari_MuJoCo/visual

# Plot Atari results
python plot_atari.py ../cleanrl/results ppo_atari opts_ttpo_atari

# Plot MuJoCo results
python plot_mujoco.py ../cleanrl/results ppo_continuous_action rpo_continuous_action opts_ttpo_continuous_action
```

### LLM

**Download models:**

```bash
cd LLM
hf download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B
hf download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
```

**Run baselines:**

```bash
cd LLM
./scripts/run_ppo.sh
./scripts/run_dapo.sh
./scripts/run_gpg.sh
./scripts/run_reinforce_pp.sh
```

**Run OPTS-TTPO (train-time searching):**

```bash
cd LLM
./scripts/run_opts_ttpo.sh
```

> Entry point: `LLM/trainer/main_opts_ttpo.py`

**Run OPTS (test-time searching):**

```bash
cd LLM
./scripts/run_opts_generation.sh
```

> Entry point: `LLM/trainer/main_opts_generation.py`

## Data

The training and test data are already included in the `LLM/data/` directory—no additional download is needed.

| Split | Dataset | Source |
|-------|---------|--------|
| Train | math12k | [hiyouga/math12k](https://huggingface.co/datasets/hiyouga/math12k) (split: train) |
| Train | NuminaMath-1.5-RL-Verifiable | [nlile/NuminaMath-1.5-RL-Verifiable](https://huggingface.co/datasets/nlile/NuminaMath-1.5-RL-Verifiable) (split: train) |
| Test | math12k | [hiyouga/math12k](https://huggingface.co/datasets/hiyouga/math12k) (split: test) |
| Test | MinervaMath | [math-ai/minervamath](https://huggingface.co/datasets/math-ai/minervamath) (split: test) |
| Test | AMC23 | [math-ai/amc23](https://huggingface.co/datasets/math-ai/amc23) (split: test) |
| Test | AIME25 | [math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25) (split: test) |

### Input/Output Format

The LLM experiments use the following chat format:

```json
[
    {
        "role": "system",
        "content": "You are a math problem solver. For each problem, think through it step by step within <think> </think> tags, then provide your final numerical answer using \\boxed{}.\n\nRequirements:\n- Show your complete reasoning process inside <think> tags.\n- Your final answer must be a single number inside \\boxed{}.\n\nExample:\nUser: If 3x + 7 = 22, what is x?\nAssistant: <think>\n3x + 7 = 22\n3x = 22 - 7 = 15\nx = 15 / 3 = 5\n</think>\nThe answer is \\boxed{5}."
    },
    {
        "role": "user",
        "content": "<question>"
    },
    {
        "role": "assistant",
        "content": "<think>\n...\n</think>\n...\\boxed{<answer>}..."
    }
]
```

## Main Results

<!-- TODO: Add experimental results -->

Coming soon.

## Project Structure

```
OPTS/
├── Atari_MuJoCo/
│   ├── cleanrl/                         # CleanRL framework (git submodule)
│   │   └── cleanrl/
│   │       ├── ppo_atari.py             # PPO baseline for Atari
│   │       ├── ppo_continuous_action.py # PPO baseline for MuJoCo
│   │       ├── rpo_continuous_action.py # RPO baseline for MuJoCo
│   │       ├── opts_ttpo_atari.py       # OPTS-TTPO for Atari
│   │       └── opts_ttpo_continuous_action.py  # OPTS-TTPO for MuJoCo
│   ├── scripts/                         # Training launch scripts
│   └── visual/                          # Visualization tools
│       ├── plot_atari.py
│       └── plot_mujoco.py
├── LLM/
│   ├── verl/                            # VERL framework (git submodule)
│   ├── trainer/
│   │   ├── opts_ttpo/                   # OPTS-TTPO trainer for LLM
│   │   │   ├── ray_trainer.py           # RayOPTSTTPOTrainer (tree search + training loop)
│   │   │   └── core_algos.py            # Core algorithms (TreeGAE, branch weight, loss)
│   │   ├── main_opts_ttpo.py            # Train-time searching entry point
│   │   ├── main_opts_generation.py      # Test-time searching entry point
│   │   └── main_eval.py                 # Evaluation entry point
│   ├── utils/
│   │   └── reward_fn.py                 # Custom reward function
│   ├── data_preprocess/                 # Data preprocessing scripts
│   ├── data/                            # Training and test datasets (parquet)
│   ├── scripts/                         # Training launch scripts
│   └── models/                          # Model checkpoints (downloaded by user)
└── README.md
```

## Citation

<!-- TODO: Add citation -->

Coming soon.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgements

This project builds upon the following open-source frameworks:

- **Atari & MuJoCo**: [CleanRL](https://github.com/vwxyzjn/cleanrl) — A clean and simple implementation of RL algorithms.
- **LLM**: [VERL](https://github.com/verl-project/verl) — A flexible and efficient RL training framework for LLMs.
