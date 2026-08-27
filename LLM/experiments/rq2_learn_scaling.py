# Copyright 2025 Junyu Lu (Julian Lou). All rights reserved.

"""
RQ2 learning curve: run the rq2_opts_scaling pipeline (reward mode, zero
baseline, s=SEARCH_ROUNDS) for ONE checkpoint step, then score with
trainer.main_eval opts-avg@k over OPTS_AVG_SLICES. Driven by
scripts/run_rq2_learn_scaling.sh.
"""

import argparse
import os
import subprocess
import sys
import time

LLM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(LLM_DIR)

CKPT_ROOT = os.environ.get("CKPT_ROOT", "/share/lujunyu/ckpts/opts_ckpts/opts_ttpo_1.7B")
METHOD = os.environ.get("OPTS_METHOD", "opts_ttpo_exp8_3_0810_n8")
MODEL_SIZE = os.environ.get("MODEL_SIZE", "1.7B")
CKPT_DIR = f"{CKPT_ROOT}/{METHOD}_{MODEL_SIZE}"

TREES_PER_PROMPT = int(os.environ.get("TREES_PER_PROMPT", "32"))
SEARCH_ROUNDS = int(os.environ.get("SEARCH_ROUNDS", "3"))
SNAPSHOT_ROUNDS = os.environ.get("SNAPSHOT_ROUNDS", "1 3")
OPTS_AVG_SLICES = os.environ.get("OPTS_AVG_SLICES", "3")
OPTS_AVG_KS = os.environ.get("OPTS_AVG_KS", "32")
BS = int(os.environ.get("BS", "902"))
DATA_PATH = os.environ.get("DATA_PATH", "data/test.parquet")
N_GPUS = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(","))

OUT_ROOT = os.environ.get("OUT_ROOT", "results/rq2_learn_scaling")
MERGED_ROOT = os.path.join(OUT_ROOT, "merged")
EVAL_ROOT = os.path.join(OUT_ROOT, "eval")
LOG_ROOT = os.environ.get("LOG_ROOT", "logs/rq2_learn_scaling")
for d in (MERGED_ROOT, EVAL_ROOT, LOG_ROOT):
    os.makedirs(d, exist_ok=True)


def run(cmd, log_path):
    with open(log_path, "w") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def merge(src, dst):
    if os.path.exists(os.path.join(dst, "config.json")) and any(
        f.endswith(".safetensors") for f in os.listdir(dst)
    ):
        print(f"[skip merge] {dst}")
        return
    print(f"[merge] {src} -> {dst}")
    run([sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
         "--local_dir", src, "--target_dir", dst],
        os.path.join(LOG_ROOT, f"merge_{os.path.basename(dst)}.log"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    step = args.step

    name = f"{METHOD}_step{step}_s{SEARCH_ROUNDS}_t{TREES_PER_PROMPT}_bs{BS}"
    output_path = os.path.join(OUT_ROOT, f"{name}.parquet")

    actor_path = os.path.join(MERGED_ROOT, f"{METHOD}_step{step}_actor")
    merge(f"{CKPT_DIR}/global_step_{step}/actor", actor_path)
    critic_path = os.path.join(MERGED_ROOT, f"{METHOD}_step{step}_critic")
    merge(f"{CKPT_DIR}/global_step_{step}/critic", critic_path)

    if os.path.exists(output_path):
        print(f"[skip gen] {output_path}")
    else:
        t0 = time.time()
        run([
            sys.executable, "-m", "experiments.rq2_opts_scaling",
            "trainer.nnodes=1", f"trainer.n_gpus_per_node={N_GPUS}",
            f"data.val_files={DATA_PATH}", "data.prompt_key=prompt", f"data.val_batch_size={BS}",
            f"+data.trees_per_prompt={TREES_PER_PROMPT}", f"+data.n_branch_rounds={SEARCH_ROUNDS}",
            f"+data.opts_avg_snapshot_rounds=[{SNAPSHOT_ROUNDS.replace(' ', ',')}]",
            f"+data.output_path={output_path}",
            f"actor_rollout_ref.model.path={actor_path}",
            f"critic.model.path={critic_path}",
            "critic.model.use_remove_padding=True",
            "critic.value_head_activation=sigmoid",
            "critic.forward_micro_batch_size_per_gpu=64",
            "actor_rollout_ref.rollout.name=vllm",
            "actor_rollout_ref.rollout.search=opts",
            "actor_rollout_ref.rollout.load_format=auto",
            f"actor_rollout_ref.rollout.max_search_per_tree={SEARCH_ROUNDS}",
            "actor_rollout_ref.rollout.temperature=1.0",
            "actor_rollout_ref.rollout.top_p=0.95",
            "actor_rollout_ref.rollout.val_kwargs.top_p=0.95",
            "actor_rollout_ref.rollout.prompt_length=1024",
            "actor_rollout_ref.rollout.response_length=2048",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "actor_rollout_ref.rollout.pipeline_model_parallel_size=1",
            "actor_rollout_ref.rollout.mode=async",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.85",
            "actor_rollout_ref.rollout.max_num_batched_tokens=262144",
            "reward_model.use_reward_loop=False",
            "algorithm.lam=0.999",
        ], os.path.join(LOG_ROOT, f"{name}.log"))
        with open(os.path.join(LOG_ROOT, f"{name}.time"), "w") as f:
            f.write(f"{name} elapsed_seconds={time.time() - t0:.2f}\n")

    print(f"========== eval: {name} ==========")
    run([
        sys.executable, "-m", "trainer.main_eval",
        "--pregenerated_parquet", output_path,
        "--metrics", "opts-avg", "--k", *OPTS_AVG_KS.split(),
        "--opts_avg_slices", *OPTS_AVG_SLICES.split(),
        "--output_tag", f"{name}_opts-avg_s{OPTS_AVG_SLICES.replace(' ', '-')}_k{OPTS_AVG_KS.replace(' ', '-')}",
        "--output_dir", EVAL_ROOT,
    ], os.path.join(LOG_ROOT, f"{name}.eval.log"))


if __name__ == "__main__":
    main()
