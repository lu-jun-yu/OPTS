# Quick Environment Migration and Start

This note is for quickly restoring the packed `opts_verl` Conda environment and preparing the local `verl` package for development.

Assumptions:

- The machine is Linux x86_64.
- CUDA and cuDNN are already installed and available on the machine.
- The OPTS repository is available at `~/OPTS`.

## 1. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init
```

## 2. Configure the Conda Environment Path

```bash
echo 'export CONDA_ENVS_PATH=~/miniconda3/envs' >> ~/.bashrc
source ~/.bashrc
```

## 3. Restore the Packed Environment

```bash
pip install -U huggingface_hub
hf download Julian2002/OPTS-TTPO-ENV --repo-type dataset --local-dir ./OPTS-TTPO-ENV
mkdir -p ~/miniconda3/envs/opts_verl
tar -xzf ./OPTS-TTPO-ENV/miniconda_opts_verl.tar.gz -C ~/miniconda3/envs/opts_verl
cd ~/miniconda3/envs/opts_verl
./bin/conda-unpack
```

## 4. Activate the Environment and Install VERL

```bash
conda activate opts_verl
cd ~/OPTS/LLM/verl
pip install --no-deps -e .
```

The Conda environment setup is complete.

## 5. Download the Base Model

```bash
cd ~/OPTS/LLM
hf download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
```

## 6. Start the Baseline Run

```bash
cd ~/OPTS/LLM
bash scripts/run_ppo_8B.sh
```
