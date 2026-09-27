# Copyright 2025 Anonymous authors. All rights reserved.

"""
RQ2: OPTS tree-count scaling. Reward mode + zero performance-difference baseline, two phases:
  - call 0: open all trees (trees_per_prompt * dataset_size prompts, one drawn
    prompt = one tree, uid = tree ID) and generate all roots in one batch;
  - the next n_branch_rounds calls: branch all performance-difference-qualifying positions of the
    existing trees in one batch per round; no new trees.
After branch rounds in opts_avg_snapshot_rounds, snapshot each tree's greedy
terminal response into column opts_avg_responses_s{round} for opts-avg@k eval.
"""

import os
import sys
from collections import defaultdict

LLM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if LLM_DIR not in sys.path:
    sys.path.insert(0, LLM_DIR)

import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from pprint import pprint

import torch
import pandas as pd
from omegaconf import OmegaConf

from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

from trainer.opts_ttpo.core_algos import (
    compute_treegae_advantage_return,
)
from trainer.opts_ttpo.ray_trainer import (
    PromptBuffer,
    compute_episodic_returns,
    compute_response_mask,
    merge_batches,
    prepare_next_round_input,
    refresh_tree_search_states,
    set_opts_ttpo_info,
    select_next_states,
    selected_to_branch_points,
)
from verl.trainer.ppo.reward import compute_reward
from utils.response_boundary import decode_response_strs

REWARD_MODE = "reward"
PERF_DIFF_BASELINE_MODE = "zero"


def request_sampling_seed(base, global_prompt_idx, group, call_idx):
    """Stable, collision-free request seed within one E1 generation profile."""
    if call_idx == 0:
        return int(base + global_prompt_idx * 64 + group)
    return int(base + 10_000_000 + global_prompt_idx * 512 + group * 64 + call_idx)


def _select_first(config, *paths, default=None):
    for path in paths:
        value = OmegaConf.select(config, path)
        if value is not None:
            return value
    return default


def _require_config_value(config, *paths):
    value = _select_first(config, *paths)
    if value is None:
        joined_paths = ", ".join(paths)
        raise ValueError(f"Missing required config value. Tried: {joined_paths}")
    return value


def _get_actor_worker_config(config):
    return _select_first(config, "actor_rollout_ref", default=config)


def _get_rollout_config(config):
    return _require_config_value(config, "actor_rollout_ref.rollout", "rollout")


def reposition_selected_states(selected_states, rule, batch, tree_search_state_by_uid, max_prompt_length, rng):
    """Keep the performance-difference rule's selected trees; move each node to a random / midpoint position of its greedy path.

    The greedy path is the ancestor chain of the cached terminal trajectory. Positions are
    restricted exactly like the performance-difference clamp: within the prompt-length budget and at or before
    the terminal's first boxed token. Returns {uid: (traj_idx, token_pos)} selected nodes.
    """
    rid = batch.non_tensor_batch["rid"]
    pid = batch.non_tensor_batch["pid"]
    branch_pos = batch.non_tensor_batch["branch_pos"]
    rid2idx = {r: i for i, r in enumerate(rid)}
    prompt_len = batch.batch["input_ids"].shape[1] - batch.batch["responses"].shape[1]
    prompt_lengths = batch.batch["attention_mask"][:, :prompt_len].sum(dim=1).cpu().numpy()
    boxed = batch.non_tensor_batch.get("first_boxed_token_pos")
    resp_lens = batch.batch["response_mask"].sum(dim=1).cpu().numpy()
    out = {}
    for u in selected_states:
        state = tree_search_state_by_uid[u]
        term = rid2idx[state.terminal_rid]
        # segments root -> terminal: (traj_idx, last_local_t); the end-of-response
        # state (local_t == length) is never a regeneration target.
        chain = []
        cur, last_t = term, min(int(state.terminal_pos), int(resp_lens[term]) - 1)
        while True:
            chain.append((cur, last_t))
            if pid[cur] is None:
                break
            parent = rid2idx[pid[cur]]
            last_t = int(branch_pos[cur])
            cur = parent
        chain.reverse()
        path = []  # absolute path position -> (traj_idx, local_t)
        for traj, last_t in chain:
            for t in range(0, last_t + 1):
                path.append((traj, t))
        # validity as in refresh_tree_search_states: prompt budget and boxed limit
        # (first_boxed_token_pos is already in full-answer coordinates = path index)
        boxed_limit = len(path) - 1
        if boxed is not None and int(boxed[term]) >= 0:
            boxed_limit = min(boxed_limit, int(boxed[term]))
        last_valid = -1
        for k, (traj, t) in enumerate(path):
            if k > boxed_limit or prompt_lengths[traj] + t >= max_prompt_length:
                break
            last_valid = k
        if last_valid < 0:
            continue
        if rule == "random":
            k = int(rng.integers(0, last_valid + 1))
        else:
            k = last_valid // 2
        out[u] = path[k]
    return out


def select_fixed_root_states(batch, search_count, max_search_per_tree, selected_token_pos=128):
    """Select the same fixed response position on every eligible root trajectory.

    ``selected_token_pos`` is the token to replace.  The downstream
    ``selected_to_branch_points`` conversion therefore uses token 127 as the
    branch point and preserves exactly the first 128 response tokens.  This
    mode is used only to generate the prompt-matched unbiased control pool; it
    deliberately does not apply the performance-difference gate.
    """
    uids = batch.non_tensor_batch["uid"]
    pids = batch.non_tensor_batch["pid"]
    response_lens = batch.batch["response_mask"].sum(dim=1).cpu().numpy()
    selected = {}
    seen_roots = set()
    for traj_idx, (uid, pid, response_len) in enumerate(zip(uids, pids, response_lens)):
        if pid is not None or uid in seen_roots:
            continue
        seen_roots.add(uid)
        if search_count.get(uid, 0) >= max_search_per_tree:
            continue
        if int(response_len) <= selected_token_pos:
            continue
        selected[uid] = (traj_idx, selected_token_pos)
        search_count[uid] = search_count.get(uid, 0) + 1
    return selected


@hydra.main(config_path="pkg://verl.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    actor_worker_config = _get_actor_worker_config(config)
    rollout_config = _get_rollout_config(config)

    model_path = _require_config_value(config, "actor_rollout_ref.model.path", "model.path")
    data_path = _require_config_value(config, "data.path", "data.val_files", "data.train_files")
    batch_size = _require_config_value(config, "data.batch_size", "data.val_batch_size", "data.train_batch_size")
    trees_per_prompt = int(_select_first(config, "data.trees_per_prompt", default=32))
    output_path = _require_config_value(config, "data.output_path")
    trust_remote_code = _select_first(
        config,
        "actor_rollout_ref.model.trust_remote_code",
        "model.trust_remote_code",
        "data.trust_remote_code",
        default=False,
    )

    local_path = copy_to_local(model_path)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    assert batch_size >= 1, f"batch_size must be >= 1, got {batch_size}"
    assert trees_per_prompt >= 1, f"trees_per_prompt must be >= 1, got {trees_per_prompt}"

    prompt_length = rollout_config.prompt_length
    response_length = rollout_config.response_length
    max_search_per_tree = rollout_config.get("max_search_per_tree", 15)
    n_branch_rounds = int(_select_first(config, "data.n_branch_rounds", default=max_search_per_tree))
    assert n_branch_rounds >= 0, f"n_branch_rounds must be >= 0, got {n_branch_rounds}"
    raw_snapshot_rounds = _select_first(config, "data.opts_avg_snapshot_rounds", default=[1, 3, 7, 15])
    if isinstance(raw_snapshot_rounds, str):
        raw_snapshot_rounds = raw_snapshot_rounds.strip("[]").replace(",", " ").split()
    elif isinstance(raw_snapshot_rounds, int):
        raw_snapshot_rounds = [raw_snapshot_rounds]
    opts_avg_snapshot_rounds = sorted({int(r) for r in raw_snapshot_rounds if int(r) >= 1})
    # RQ1-OPTS bias analysis: snapshot per-token TreeGAE advantages as of these
    # rounds (round 0 = roots only, i.e. s=0 plain GAE baseline).
    raw_adv_rounds = _select_first(config, "data.adv_snapshot_rounds", default=[0, 1, 3, 7, 15])
    if isinstance(raw_adv_rounds, str):
        raw_adv_rounds = raw_adv_rounds.strip("[]").replace(",", " ").split()
    elif isinstance(raw_adv_rounds, int):
        raw_adv_rounds = [raw_adv_rounds]
    adv_snapshot_rounds = sorted({int(r) for r in raw_adv_rounds if int(r) >= 0})
    gamma = config.algorithm.gamma
    lam = config.algorithm.lam
    # Appendix: rebranch-position rule. Performance-difference gates and picks; random/midpoint keep
    # the performance-difference tree set and budget but re-place the node on the same greedy path.
    select_rule = str(_select_first(config, "data.select_rule", default="perf_diff"))
    assert select_rule in ("perf_diff", "random", "midpoint", "fixed128_all"), select_rule
    select_seed = int(_select_first(config, "data.select_seed", default=0))
    select_rng = np.random.default_rng(select_seed)
    rollout_seed_base = int(_select_first(config, "data.rollout_seed_base", default=20260915))
    prompt_index_offset = int(_select_first(config, "data.prompt_index_offset", default=0))
    # Main-text bias analysis: also track mean-backup TreeGAE on the same trees.
    raw_backups = _select_first(config, "data.adv_backups", default=["max"])
    if isinstance(raw_backups, str):
        raw_backups = raw_backups.strip("[]").replace(",", " ").split()
    adv_backups = [str(b) for b in raw_backups]
    assert adv_backups and adv_backups[0] == "max" and set(adv_backups) <= {"max", "mean"}, adv_backups
    extra_backups = [b for b in adv_backups if b != "max"]

    dataset = pd.read_parquet(data_path)
    total_samples = len(dataset)
    trees_target = trees_per_prompt * total_samples

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from utils.reward_fn import compute_score_sync
    from verl.workers.reward_manager.naive import NaiveRewardManager
    reward_fn = NaiveRewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        compute_score=compute_score_sync,
    )

    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        use_gpu=True,
        max_colocate_count=2,
    )

    async_mode = rollout_config.get("mode", "sync") == "async"
    actor_worker_impl = AsyncActorRolloutRefWorker if async_mode else ActorRolloutRefWorker
    actor_rollout_cls = RayClassWithInitArgs(
        cls=ray.remote(actor_worker_impl),
        config=actor_worker_config,
        role="rollout",
    )
    critic_cls = RayClassWithInitArgs(cls=ray.remote(CriticWorker), config=config.critic)

    class_dict = {"actor_rollout": actor_rollout_cls, "critic": critic_cls}
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)

    wg_dict = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_dict_cls,
        device_name=config.trainer.device,
    )
    spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
    wg = spawn_wg["actor_rollout"]
    critic_wg = spawn_wg["critic"]

    wg.init_model()
    critic_wg.init_model()

    async_rollout_manager = None
    if async_mode:
        from verl.experimental.agent_loop import AgentLoopManager

        async_rollout_manager = AgentLoopManager(config=config, worker_group=wg, rm_resource_pool=None)

    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl.trainer.main_ppo import create_rl_dataset
    from verl.utils.dataset.rl_dataset import collate_fn as rl_collate_fn

    dataset_config = OmegaConf.create(OmegaConf.to_container(config.data, resolve=False))
    dataset_config.val_files = data_path
    dataset_config.max_prompt_length = prompt_length
    dataset_config.shuffle = False
    dataset_config.validation_shuffle = False

    rl_dataset = create_rl_dataset(
        data_path,
        dataset_config,
        tokenizer,
        processor=None,
        max_samples=dataset_config.get("val_max_samples", -1),
    )
    dataloader = StatefulDataLoader(
        dataset=rl_dataset,
        batch_size=batch_size,
        num_workers=dataset_config.get("dataloader_num_workers", 0),
        shuffle=False,
        drop_last=False,
        collate_fn=rl_collate_fn,
    )
    prompt_buffer = PromptBuffer(dataloader)
    if len(rl_dataset) != total_samples:
        raise ValueError(
            "PromptBuffer dataset size does not match the raw parquet size. "
            "This usually means the reused training data pipeline filtered or reordered samples. "
            f"prompt_buffer.total_samples={len(rl_dataset)}, total_samples={total_samples}"
        )

    idx_responses = defaultdict(list)
    idx_sample_counter = defaultdict(int)
    global_sample_counter = 0

    print(f"Starting RQ2 OPTS tree-count scaling: "
          f"trees_per_prompt={trees_per_prompt}, trees_target={trees_target}, "
          f"batch_size={batch_size}, n_branch_rounds={n_branch_rounds}, "
          f"reward_mode={REWARD_MODE}, perf_diff_baseline={PERF_DIFF_BASELINE_MODE}, "
          f"max_search_per_tree={max_search_per_tree}")

    global_batch = None
    next_states = {}
    search_count = {}
    max_perf_diffs = {}
    tree_search_state_by_uid = {}
    uid_to_dataset_idx = {}
    prompt_cursor = 0
    trees_opened = 0
    response_by_rid = {}
    response_ids_by_rid = {}
    adv_snapshots_by_round = {}  # round -> {rid: [per-token advantage floats]}
    extra_adv_snapshots_by_round = {b: {} for b in extra_backups}
    # lam=1 advantage snapshots (gradient-side: subtree return aggregation - V(s_t))
    lam1_adv_snapshots = {b: {} for b in ["max"] + extra_backups}
    values_by_rid = {}   # rid -> per-token critic values (E2 credit analysis)
    reward_by_rid = {}   # rid -> scalar 0/1 reward (E1 frontier coverage)
    selection_events = []  # (tree_seq, dataset_idx, round, candidate_selection_score)
    tree_seq = {}
    opts_avg_snapshots = {r: [[] for _ in range(total_samples)] for r in opts_avg_snapshot_rounds}

    total_calls = 1 + n_branch_rounds
    for call_idx in range(total_calls):
        root_phase = call_idx == 0
        print(f"[call {call_idx + 1}/{total_calls}] Start. "
              f"phase={'root' if root_phase else 'branch'}, "
              f"trees_opened={trees_opened}/{trees_target}, "
              f"continuations={len(next_states)}.")

        if root_phase:
            batch = prompt_buffer.draw(trees_target)
            for uid in batch.non_tensor_batch["uid"]:
                uid_to_dataset_idx[uid] = prompt_cursor
                prompt_cursor = (prompt_cursor + 1) % total_samples
                tree_seq[str(uid)] = len(tree_seq)
            trees_opened += len(batch.non_tensor_batch["uid"])
        else:
            if not next_states:
                print(f"[call {call_idx + 1}/{total_calls}] No performance-difference candidates left; stopping early.")
                break
            batch = prepare_next_round_input(
                global_batch=global_batch,
                next_states=next_states,
                pad_token_id=tokenizer.pad_token_id,
            )
        request_seeds = []
        for uid in batch.non_tensor_batch["uid"]:
            uid_str = str(uid)
            local_prompt_idx = int(uid_to_dataset_idx[uid])
            seq = int(tree_seq[uid_str])
            group = seq // total_samples
            if seq % total_samples != local_prompt_idx:
                raise ValueError(
                    f"tree/prompt order mismatch: tree_seq={seq}, prompt={local_prompt_idx}"
                )
            global_prompt_idx = prompt_index_offset + local_prompt_idx
            request_seeds.append(
                request_sampling_seed(rollout_seed_base, global_prompt_idx, group, call_idx)
            )
        batch.non_tensor_batch["sampling_seed"] = np.asarray(request_seeds, dtype=np.int64)
        batch.meta_info["temperature"] = rollout_config.temperature

        if async_mode:
            size_divisor = rollout_config.agent.num_workers
            batch_padded, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
            output = unpad_dataproto(
                async_rollout_manager.generate_sequences(
                    batch_padded, sleep_after=(call_idx == total_calls - 1)
                ),
                pad_size=pad_size,
            )
        else:
            batch_padded, pad_size = pad_dataproto_to_divisor(batch, wg.world_size)
            output_padded = wg.generate_sequences(batch_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)

        # Both rollout frontends are allowed to rebuild non_tensor_batch.  The
        # generation result order is the unpadded input order, so restore all
        # request metadata from the source batch for both sync and async modes.
        for key in (
            "uid", "raw_prompt_len", "data_source", "reward_model", "extra_info",
            "raw_prompt", "sampling_seed",
        ):
            if key in batch.non_tensor_batch:
                output.non_tensor_batch[key] = batch.non_tensor_batch[key]

        if "response_mask" not in output.batch.keys():
            output.batch["response_mask"] = compute_response_mask(output)

        values_input_padded, values_pad_size = pad_dataproto_to_divisor(output, critic_wg.world_size)
        values_output_padded = critic_wg.compute_values(values_input_padded)
        values_output = unpad_dataproto(values_output_padded, pad_size=values_pad_size)
        output = output.union(values_output)

        full_response_strs = decode_response_strs(output, tokenizer, prompt_length, response_length)
        if "extra_info" not in output.non_tensor_batch:
            output.non_tensor_batch["extra_info"] = np.array(
                [{} for _ in range(len(full_response_strs))], dtype=object)
        for i, s in enumerate(full_response_strs):
            output.non_tensor_batch["extra_info"][i]["full_response_str"] = s
        reward_tensor, _ = compute_reward(output, reward_fn)
        output.batch["token_level_rewards"] = reward_tensor

        new_sample_indices = set_opts_ttpo_info(output, global_batch, next_states, call_idx)
        output.non_tensor_batch["episodic_returns"] = compute_episodic_returns(output, global_batch)
        cur_batch_size, cur_response_len = output.batch["responses"].shape
        output.batch["state_branches"] = torch.ones(cur_batch_size, cur_response_len)
        output.batch["advantages"] = torch.zeros(cur_batch_size, cur_response_len)
        output.batch["returns"] = torch.zeros(cur_batch_size, cur_response_len)
        for b in extra_backups:
            output.batch[f"advantages_{b}"] = torch.zeros(cur_batch_size, cur_response_len)

        if global_batch is None:
            global_batch = output
        else:
            global_batch = merge_batches(global_batch, output)

        advantages, returns = compute_treegae_advantage_return(
            token_level_rewards=global_batch.batch["token_level_rewards"],
            values=global_batch.batch["values"],
            response_mask=global_batch.batch["response_mask"],
            attention_mask=global_batch.batch["attention_mask"],
            gamma=gamma,
            lam=lam,
            rid=list(global_batch.non_tensor_batch["rid"]),
            pid=list(global_batch.non_tensor_batch["pid"]),
            branch_pos=list(global_batch.non_tensor_batch["branch_pos"]),
            cid=list(global_batch.non_tensor_batch["cid"]),
            new_sample_indices=new_sample_indices,
            raw_prompt_len=global_batch.non_tensor_batch["raw_prompt_len"],
            max_prompt_len=global_batch.batch["attention_mask"].shape[1] - global_batch.batch["response_mask"].shape[1],
            advantages=global_batch.batch["advantages"],
        )
        global_batch.batch["advantages"] = advantages
        global_batch.batch["returns"] = returns
        for b in extra_backups:
            adv_b, _ = compute_treegae_advantage_return(
                token_level_rewards=global_batch.batch["token_level_rewards"],
                values=global_batch.batch["values"],
                response_mask=global_batch.batch["response_mask"],
                attention_mask=global_batch.batch["attention_mask"],
                gamma=gamma,
                lam=lam,
                rid=list(global_batch.non_tensor_batch["rid"]),
                pid=list(global_batch.non_tensor_batch["pid"]),
                branch_pos=list(global_batch.non_tensor_batch["branch_pos"]),
                cid=list(global_batch.non_tensor_batch["cid"]),
                new_sample_indices=new_sample_indices,
                raw_prompt_len=global_batch.non_tensor_batch["raw_prompt_len"],
                max_prompt_len=global_batch.batch["attention_mask"].shape[1] - global_batch.batch["response_mask"].shape[1],
                advantages=global_batch.batch[f"advantages_{b}"],
                backup=b,
            )
            global_batch.batch[f"advantages_{b}"] = adv_b

        # RQ1-OPTS: snapshot per-token advantages over the FULL tree as of this
        # round (ancestor advantages can change when new branches appear, so the
        # snapshot must cover every rid, not just this round's new samples).
        if call_idx in adv_snapshot_rounds:
            adv_cpu = global_batch.batch["advantages"].detach().cpu()
            mask_cpu = global_batch.batch["response_mask"].detach().cpu().bool()
            snap = {}
            for rid_, adv_row, m_row in zip(
                global_batch.non_tensor_batch["rid"], adv_cpu, mask_cpu
            ):
                snap[str(rid_)] = [float(x) for x in adv_row[m_row].tolist()]
            adv_snapshots_by_round[call_idx] = snap
            for b in extra_backups:
                adv_b_cpu = global_batch.batch[f"advantages_{b}"].detach().cpu()
                snap_b = {}
                for rid_, adv_row, m_row in zip(global_batch.non_tensor_batch["rid"], adv_b_cpu, mask_cpu):
                    snap_b[str(rid_)] = [float(x) for x in adv_row[m_row].tolist()]
                extra_adv_snapshots_by_round[b][call_idx] = snap_b
            print(f"[call {call_idx + 1}/{total_calls}] advantage snapshot: "
                  f"{len(snap)} rids", flush=True)

            # Full lam=1 snapshots for E2. A fresh zero buffer requires ALL
            # trajectories, not just this round's new suffixes; otherwise old
            # unaffected suffixes silently become zero. E1 separately rebuilds
            # V=0 returns from topology/rewards in opts_bias_prep.
            for b in ["max"] + extra_backups:
                adv1, _ = compute_treegae_advantage_return(
                    token_level_rewards=global_batch.batch["token_level_rewards"],
                    values=global_batch.batch["values"],
                    response_mask=global_batch.batch["response_mask"],
                    attention_mask=global_batch.batch["attention_mask"],
                    gamma=gamma,
                    lam=1.0,
                    rid=list(global_batch.non_tensor_batch["rid"]),
                    pid=list(global_batch.non_tensor_batch["pid"]),
                    branch_pos=list(global_batch.non_tensor_batch["branch_pos"]),
                    cid=list(global_batch.non_tensor_batch["cid"]),
                    new_sample_indices=list(range(len(global_batch.non_tensor_batch["rid"]))),
                    raw_prompt_len=global_batch.non_tensor_batch["raw_prompt_len"],
                    max_prompt_len=global_batch.batch["attention_mask"].shape[1] - global_batch.batch["response_mask"].shape[1],
                    advantages=torch.zeros_like(global_batch.batch["advantages"]),
                    backup=b,
                )
                adv1_cpu = adv1.detach().cpu()
                snap1 = {}
                for rid_, adv_row, m_row in zip(
                    global_batch.non_tensor_batch["rid"], adv1_cpu, mask_cpu
                ):
                    snap1[str(rid_)] = [float(x) for x in adv_row[m_row].tolist()]
                lam1_adv_snapshots[b][call_idx] = snap1
                del adv1, adv1_cpu

        response_strs = decode_response_strs(output, tokenizer, prompt_length, response_length)
        resp_ids_tensor = output.batch["responses"].detach().cpu()
        resp_mask_tensor = output.batch["response_mask"].detach().cpu().bool()
        values_tensor = output.batch["values"].detach().cpu()
        rewards_cpu = reward_tensor.detach().cpu()
        output_uids = output.non_tensor_batch["uid"]
        current_rids = output.non_tensor_batch["rid"]
        current_pids = output.non_tensor_batch["pid"]
        current_branch_pos = output.non_tensor_batch["branch_pos"]
        current_sampling_seeds = output.non_tensor_batch["sampling_seed"]

        for local_idx, (resp, uid) in enumerate(zip(response_strs, output_uids)):
            dataset_idx = uid_to_dataset_idx[uid]
            idx_sample_counter[dataset_idx] += 1
            global_sample_counter += 1
            pid = current_pids[local_idx]
            rid_str = str(current_rids[local_idx])
            response_by_rid[rid_str] = resp
            ids_row = resp_ids_tensor[local_idx][resp_mask_tensor[local_idx]]
            response_ids_by_rid[rid_str] = [int(t) for t in ids_row.tolist()]
            values_by_rid[rid_str] = [
                float(x) for x in values_tensor[local_idx][resp_mask_tensor[local_idx]].tolist()
            ]
            reward_by_rid[rid_str] = float(rewards_cpu[local_idx].sum())
            idx_responses[dataset_idx].append({
                "response": resp,
                "uid": str(uid),
                "tree_seq": int(tree_seq[str(uid)]),
                "sample_index": idx_sample_counter[dataset_idx],
                "global_index": global_sample_counter,
                "rid": rid_str,
                "pid": None if pid is None else str(pid),
                "branch_pos": int(current_branch_pos[local_idx]),
                "round": call_idx,
                "sampling_seed": int(current_sampling_seeds[local_idx]),
            })

        refresh_tree_search_states(
            batch=global_batch,
            affected_uids=set(output_uids),
            tree_search_state_by_uid=tree_search_state_by_uid,
            gamma=gamma,
            max_prompt_length=prompt_length,
            tokenizer=tokenizer,
            round_idx=call_idx,
        )

        # States persist across calls, so unbranched trees keep their latest
        # terminal — the snapshot is exactly the "as of this round" view.
        if call_idx in opts_avg_snapshots:
            snapshot_entries = defaultdict(list)
            for uid, state in tree_search_state_by_uid.items():
                dataset_idx = uid_to_dataset_idx[uid]
                snapshot_entries[dataset_idx].append((tree_seq[uid], response_by_rid[state.terminal_rid]))
            for dataset_idx, entries in snapshot_entries.items():
                assert len(entries) == trees_per_prompt, (
                    f"snapshot s{call_idx}: prompt {dataset_idx} has {len(entries)} trees, "
                    f"expected {trees_per_prompt}."
                )
                entries.sort(key=lambda item: item[0])
                opts_avg_snapshots[call_idx][dataset_idx] = [response for _, response in entries]

        print(f"[call {call_idx + 1}/{total_calls}] Done. "
              f"Collected {sum(len(v) for v in idx_responses.values())} total responses, "
              f"{trees_opened}/{trees_target} trees opened.")

        # search_batch_size=trees_target: every qualifying tree may branch once per round.
        if call_idx < total_calls - 1:
            if select_rule == "fixed128_all":
                selected_states = select_fixed_root_states(
                    batch=global_batch,
                    search_count=search_count,
                    max_search_per_tree=max_search_per_tree,
                )
            else:
                selected_states = select_next_states(
                    batch=global_batch,
                    search_count=search_count,
                    max_perf_diffs=max_perf_diffs,
                    max_search_per_tree=max_search_per_tree,
                    tree_search_state_by_uid=tree_search_state_by_uid,
                    max_searched_tree_ratio=1.0,
                    search_batch_size=trees_target,
                    perf_diff_baseline_mode=PERF_DIFF_BASELINE_MODE,
                )
            # E1: per-round selection log. A tree's entry score is the
            # length-normalized candidate score at its FIRST selection (used for the
            # training-faithful top-p scan at metrics time).
            for u in selected_states:
                selection_events.append({
                    "tree_seq": int(tree_seq[str(u)]),
                    "dataset_idx": int(uid_to_dataset_idx[u]),
                    "round": call_idx + 1,  # the new suffix is generated next call
                    "score": float(tree_search_state_by_uid[u].candidate_selection_score),
                })
            if select_rule in ("random", "midpoint"):
                selected_states = reposition_selected_states(
                    selected_states, select_rule, global_batch, tree_search_state_by_uid,
                    prompt_length, select_rng)
            next_states = selected_to_branch_points(selected_states, global_batch)

    assert trees_opened == trees_target, (
        f"Tree-count invariant violated: opened {trees_opened}, target {trees_target}."
    )

    final_advantages_by_rid = {}
    final_advantages = global_batch.batch["advantages"].detach().cpu()
    final_response_mask = global_batch.batch["response_mask"].detach().cpu().bool()
    for rid, advantages, valid_mask in zip(
        global_batch.non_tensor_batch["rid"],
        final_advantages,
        final_response_mask,
    ):
        final_advantages_by_rid[str(rid)] = [float(x) for x in advantages[valid_mask].tolist()]

    output_responses = [[] for _ in range(total_samples)]
    output_sample_indices = [[] for _ in range(total_samples)]
    output_global_indices = [[] for _ in range(total_samples)]
    output_tree_uids = [[] for _ in range(total_samples)]
    output_tree_seqs = [[] for _ in range(total_samples)]
    output_tree_rids = [[] for _ in range(total_samples)]
    output_tree_pids = [[] for _ in range(total_samples)]
    output_tree_branch_pos = [[] for _ in range(total_samples)]
    output_tree_advantages = [[] for _ in range(total_samples)]
    output_response_ids = [[] for _ in range(total_samples)]
    output_tree_values = [[] for _ in range(total_samples)]
    output_tree_rewards = [[] for _ in range(total_samples)]
    output_tree_rounds = [[] for _ in range(total_samples)]
    output_tree_sampling_seeds = [[] for _ in range(total_samples)]
    output_lam1_snaps = {b: {r: [[] for _ in range(total_samples)] for r in adv_snapshot_rounds}
                         for b in ["max"] + extra_backups}
    output_adv_snaps = {r: [[] for _ in range(total_samples)] for r in adv_snapshot_rounds}
    output_extra_adv = {b: [[] for _ in range(total_samples)] for b in extra_backups}
    output_extra_adv_snaps = {b: {r: [[] for _ in range(total_samples)] for r in adv_snapshot_rounds}
                              for b in extra_backups}
    extra_final_adv_by_rid = {}
    for b in extra_backups:
        adv_b = global_batch.batch[f"advantages_{b}"].detach().cpu()
        extra_final_adv_by_rid[b] = {
            str(rid): [float(x) for x in a[m].tolist()]
            for rid, a, m in zip(global_batch.non_tensor_batch["rid"], adv_b, final_response_mask)}

    for dataset_idx, resp_list in idx_responses.items():
        for record in resp_list:
            rid_str = record["rid"]
            output_responses[dataset_idx].append(record["response"])
            output_sample_indices[dataset_idx].append(record["sample_index"])
            output_global_indices[dataset_idx].append(record["global_index"])
            output_tree_uids[dataset_idx].append(record["uid"])
            output_tree_seqs[dataset_idx].append(record["tree_seq"])
            output_tree_rids[dataset_idx].append(rid_str)
            output_tree_pids[dataset_idx].append(record["pid"])
            output_tree_branch_pos[dataset_idx].append(record["branch_pos"])
            output_tree_advantages[dataset_idx].append(final_advantages_by_rid[rid_str])
            output_response_ids[dataset_idx].append(response_ids_by_rid[rid_str])
            output_tree_values[dataset_idx].append(values_by_rid[rid_str])
            output_tree_rewards[dataset_idx].append(reward_by_rid[rid_str])
            output_tree_rounds[dataset_idx].append(record["round"])
            output_tree_sampling_seeds[dataset_idx].append(record["sampling_seed"])
            for b in ["max"] + extra_backups:
                for r in adv_snapshot_rounds:
                    output_lam1_snaps[b][r][dataset_idx].append(
                        lam1_adv_snapshots[b].get(r, {}).get(rid_str, [])
                    )
            for r in adv_snapshot_rounds:
                # a rid created after round r has no entry; store [] as sentinel
                output_adv_snaps[r][dataset_idx].append(
                    adv_snapshots_by_round.get(r, {}).get(rid_str, [])
                )
            for b in extra_backups:
                output_extra_adv[b][dataset_idx].append(extra_final_adv_by_rid[b][rid_str])
                for r in adv_snapshot_rounds:
                    output_extra_adv_snaps[b][r][dataset_idx].append(
                        extra_adv_snapshots_by_round[b].get(r, {}).get(rid_str, []))

    dataset["responses"] = output_responses
    dataset["sample_indices"] = output_sample_indices
    dataset["global_indices"] = output_global_indices
    dataset["tree_uids"] = output_tree_uids
    dataset["tree_seqs"] = output_tree_seqs
    dataset["tree_rids"] = output_tree_rids
    dataset["tree_pids"] = output_tree_pids
    dataset["tree_branch_pos"] = output_tree_branch_pos
    dataset["tree_advantages"] = output_tree_advantages
    dataset["response_ids"] = output_response_ids
    dataset["tree_values"] = output_tree_values
    dataset["tree_rewards"] = output_tree_rewards
    dataset["tree_rounds"] = output_tree_rounds
    dataset["tree_sampling_seeds"] = output_tree_sampling_seeds
    for b in ["max"] + extra_backups:
        lam1_col = "tree_advantages_lam1" if b == "max" else f"tree_advantages_{b}_lam1"
        for r in adv_snapshot_rounds:
            dataset[f"{lam1_col}_s{r}"] = output_lam1_snaps[b][r]
    for r in adv_snapshot_rounds:
        dataset[f"tree_advantages_s{r}"] = output_adv_snaps[r]
    for b in extra_backups:
        dataset[f"tree_advantages_{b}"] = output_extra_adv[b]
        for r in adv_snapshot_rounds:
            dataset[f"tree_advantages_{b}_s{r}"] = output_extra_adv_snaps[b][r]
    dataset["select_rule"] = select_rule
    for r, snapshots in opts_avg_snapshots.items():
        if any(len(s) > 0 for s in snapshots):
            dataset[f"opts_avg_responses_s{r}"] = snapshots
    dataset["opts_reward_mode"] = REWARD_MODE

    output_dir = os.path.dirname(output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(output_path)
    print(f"Results saved to {output_path}")

    sel_path = output_path.replace(".parquet", "_selections.parquet")
    pd.DataFrame(
        selection_events, columns=["tree_seq", "dataset_idx", "round", "score"]
    ).to_parquet(sel_path, index=False)
    print(f"Selection events saved to {sel_path} ({len(selection_events)} rows)")


if __name__ == "__main__":
    main()
