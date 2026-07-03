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

## 5. Install CUDA:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/

apt-get update
apt-get -y install cuda-toolkit-12-8

update-alternatives --set cuda /usr/local/cuda-12.8

# Add CUDA to PATH permanently
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

## 6. Install cuDNN:

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.20.0/local_installers/cudnn-local-repo-ubuntu2204-9.20.0_1.0-1_amd64.deb
dpkg -i cudnn-local-repo-ubuntu2204-9.20.0_1.0-1_amd64.deb
cp /var/cudnn-local-repo-ubuntu2204-9.20.0/cudnn-*-keyring.gpg /usr/share/keyrings/

apt-get update
apt-get -y install cudnn
apt-get -y install cudnn9-cuda-12
```

## 7. Download the Base Model

```bash
cd ~/OPTS/LLM
hf download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
```

## 8. Start the Baseline Run

```bash
cd ~/OPTS/LLM
bash scripts/run_ppo_8B.sh
```
