#!/usr/bin/env python3
"""Value-guided OPTS search from an existing reward-guided run's roots.

The root response texts and tree identities are extracted from the completed
reward-guided parquet, canonically retokenized, passed through the frozen
critic, and stored as a restartable DataProto cache.  Value-guided OPTS then
performs only the additional branch-search rounds.  Consequently both
experiments have the same 902 prompts, 32 decoded root responses per prompt,
and tree ordering at S_max=0; the existing reward-guided search is never rerun.

The output schema is compatible with ``trainer.main_eval --metrics opts-avg``
and retains per-round advantages, true verifier rewards, guidance rewards, and
online greedy-terminal snapshots for later E5 analysis.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

LLM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if LLM_DIR not in sys.path:
    sys.path.insert(0, LLM_DIR)

import hydra
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import ray
import torch
from omegaconf import OmegaConf

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from experiments.RQ2.search import request_sampling_seed
from trainer.opts_ttpo.core_algos import compute_treegae_advantage_return
from trainer.opts_ttpo.ray_trainer import (
    PromptBuffer,
    compute_episodic_returns,
    compute_response_mask,
    merge_batches,
    prepare_next_round_input,
    refresh_tree_search_states,
    select_next_states,
    selected_to_branch_points,
    set_opts_ttpo_info,
)
from utils.response_boundary import decode_response_strs
from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.reward import compute_reward
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker


CACHE_VERSION = 6
GUIDANCE_MODES = ("reward", "value")


def _select_first(config, *paths, default=None):
    for path in paths:
        value = OmegaConf.select(config, path)
        if value is not None:
            return value
    return default


def _require_config_value(config, *paths):
    value = _select_first(config, *paths)
    if value is None:
        raise ValueError(f"Missing required config value. Tried: {', '.join(paths)}")
    return value


def _parse_int_list(value, *, minimum=0):
    if isinstance(value, str):
        value = value.strip("[]").replace(",", " ").split()
    elif isinstance(value, int):
        value = [value]
    return sorted({int(item) for item in value if int(item) >= minimum})


def _get_actor_worker_config(config):
    return _select_first(config, "actor_rollout_ref", default=config)


def _get_rollout_config(config):
    return _require_config_value(config, "actor_rollout_ref.rollout", "rollout")


def make_guidance_rewards(batch: DataProto, mode: str) -> torch.Tensor:
    """Return token rewards for one mode without mutating the shared cache."""

    if mode not in GUIDANCE_MODES:
        raise ValueError(f"Unknown guidance mode: {mode}")
    true_rewards = batch.batch["true_token_level_rewards"]
    if mode == "reward":
        return true_rewards.clone()

    response_mask = batch.batch["response_mask"]
    values = batch.batch["values"]
    last_pos = (response_mask.sum(dim=1) - 1).clamp(min=0).long()
    rewards = torch.zeros_like(values)
    row = torch.arange(values.size(0), device=values.device)
    rewards[row, last_pos] = values[row, last_pos]
    return rewards


def root_fingerprint(batch: DataProto) -> str:
    """Identify the shared stochastic roots independently of branch outputs."""

    digest = hashlib.sha256()
    for key in ("responses", "response_mask"):
        tensor = batch.batch[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    seeds = np.asarray(batch.non_tensor_batch["sampling_seed"], dtype=np.int64)
    digest.update(seeds.tobytes())
    return digest.hexdigest()


def _atomic_save_dataproto(batch: DataProto, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    batch.save_to_disk(temporary)
    os.replace(temporary, path)


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _cache_manifest_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".json")


def _validate_cache_manifest(manifest: dict, expected: dict, cache_path: Path) -> None:
    for key, expected_value in expected.items():
        actual = manifest.get(key)
        if actual != expected_value:
            raise ValueError(
                f"Root cache mismatch for {key}: expected {expected_value!r}, "
                f"found {actual!r} in {_cache_manifest_path(cache_path)}"
            )


def _snapshot_advantages(batch: DataProto) -> dict[str, list[float]]:
    advantages = batch.batch["advantages"].detach().cpu()
    mask = batch.batch["response_mask"].detach().cpu().bool()
    return {
        str(rid): [float(x) for x in row[row_mask].tolist()]
        for rid, row, row_mask in zip(batch.non_tensor_batch["rid"], advantages, mask)
    }


def _snapshot_terminals(
    tree_search_state_by_uid,
    uid_to_dataset_idx,
    tree_seq,
    response_by_rid,
    total_samples,
    trees_per_prompt,
):
    entries_by_dataset = defaultdict(list)
    for uid, state in tree_search_state_by_uid.items():
        dataset_idx = uid_to_dataset_idx[uid]
        entries_by_dataset[dataset_idx].append(
            (tree_seq[uid], response_by_rid[state.terminal_rid])
        )
    snapshot = [[] for _ in range(total_samples)]
    for dataset_idx, entries in entries_by_dataset.items():
        entries.sort(key=lambda item: item[0])
        if len(entries) != trees_per_prompt:
            raise RuntimeError(
                f"Snapshot for prompt {dataset_idx} contains {len(entries)} trees; "
                f"expected {trees_per_prompt}."
            )
        snapshot[dataset_idx] = [response for _, response in entries]
    return snapshot


@hydra.main(config_path="pkg://verl.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = {
            "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}
        }
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))
    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    OmegaConf.resolve(config)
    rollout_config = _get_rollout_config(config)
    actor_worker_config = _get_actor_worker_config(config)

    model_path = _require_config_value(config, "actor_rollout_ref.model.path", "model.path")
    critic_path = _require_config_value(config, "critic.model.path")
    data_path = str(_require_config_value(config, "data.path", "data.val_files", "data.train_files"))
    batch_size = int(
        _require_config_value(config, "data.batch_size", "data.val_batch_size", "data.train_batch_size")
    )
    trees_per_prompt = int(_select_first(config, "data.trees_per_prompt", default=32))
    n_branch_rounds = int(
        _select_first(config, "data.n_branch_rounds", default=rollout_config.max_search_per_tree)
    )
    cache_path = Path(str(_require_config_value(config, "data.root_cache_path")))
    phase = str(_select_first(config, "data.shared_guidance_phase", default="value"))
    if phase not in ("root", "value"):
        raise ValueError(f"Unknown data.shared_guidance_phase={phase!r}")
    root_source_path = Path(str(_require_config_value(config, "data.root_source_parquet")))
    if not root_source_path.exists():
        raise FileNotFoundError(f"Reward-guided source parquet not found: {root_source_path}")

    snapshot_rounds = _parse_int_list(
        _select_first(config, "data.opts_avg_snapshot_rounds", default=[0, 1, 3, 7, 15])
    )
    advantage_rounds = _parse_int_list(
        _select_first(config, "data.adv_snapshot_rounds", default=snapshot_rounds)
    )
    requested_modes = ["value"] if phase == "value" else []
    output_paths = {
        "value": Path(str(_require_config_value(config, "data.output_path_value")))
    } if requested_modes else {}
    modes_to_run = [mode for mode in requested_modes if not output_paths[mode].exists()]
    if requested_modes and not modes_to_run:
        print("All requested guidance outputs already exist; nothing to do.")
        return

    dataset = pd.read_parquet(data_path)
    total_samples = len(dataset)
    trees_target = total_samples * trees_per_prompt
    prompt_length = int(rollout_config.prompt_length)
    response_length = int(rollout_config.response_length)
    max_search_per_tree = int(rollout_config.max_search_per_tree)
    gamma = float(config.algorithm.gamma)
    lam = float(config.algorithm.lam)
    rollout_seed_base = int(_select_first(config, "data.rollout_seed_base", default=20260915))
    prompt_index_offset = int(_select_first(config, "data.prompt_index_offset", default=0))
    baseline_by_mode = {
        "value": str(_select_first(config, "data.value_perf_diff_baseline", default="mean")),
    }
    if any(value not in ("zero", "mean") for value in baseline_by_mode.values()):
        raise ValueError(f"Unsupported performance-difference baseline: {baseline_by_mode}")

    expected_manifest = {
        "cache_version": CACHE_VERSION,
        "data_path": str(Path(data_path).resolve()),
        "model_path": str(Path(str(model_path)).resolve()),
        "critic_path": str(Path(str(critic_path)).resolve()),
        "total_prompts": total_samples,
        "trees_per_prompt": trees_per_prompt,
        "root_count": trees_target,
        "prompt_length": prompt_length,
        "response_length": response_length,
        "rollout_seed_base": rollout_seed_base,
        "prompt_index_offset": prompt_index_offset,
        "root_source_parquet": str(root_source_path.resolve()),
        "root_source_size": root_source_path.stat().st_size,
        "root_source_mtime_ns": root_source_path.stat().st_mtime_ns,
    }

    local_path = copy_to_local(model_path)
    tokenizer = hf_tokenizer(
        local_path,
        trust_remote_code=bool(
            _select_first(config, "actor_rollout_ref.model.trust_remote_code", default=False)
        ),
    )
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
        cls=ray.remote(actor_worker_impl), config=actor_worker_config, role="rollout"
    )
    critic_cls = RayClassWithInitArgs(cls=ray.remote(CriticWorker), config=config.critic)
    class_dict = {"actor_rollout": actor_rollout_cls, "critic": critic_cls}
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    wg_dict = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_dict_cls,
        device_name=config.trainer.device,
    )
    workers = wg_dict.spawn(prefix_set=class_dict.keys())
    actor_wg = workers["actor_rollout"]
    critic_wg = workers["critic"]
    actor_wg.init_model()
    critic_wg.init_model()

    async_rollout_manager = None
    if async_mode:
        from verl.experimental.agent_loop import AgentLoopManager

        async_rollout_manager = AgentLoopManager(
            config=config, worker_group=actor_wg, rm_resource_pool=None
        )

    def generate_and_score(batch: DataProto, *, sleep_after: bool) -> DataProto:
        batch.meta_info["temperature"] = rollout_config.temperature
        if async_mode:
            divisor = rollout_config.agent.num_workers
            padded, pad_size = pad_dataproto_to_divisor(batch, divisor)
            output = unpad_dataproto(
                async_rollout_manager.generate_sequences(padded, sleep_after=sleep_after),
                pad_size=pad_size,
            )
        else:
            padded, pad_size = pad_dataproto_to_divisor(batch, actor_wg.world_size)
            output = unpad_dataproto(actor_wg.generate_sequences(padded), pad_size=pad_size)

        for key in (
            "uid",
            "raw_prompt_len",
            "data_source",
            "reward_model",
            "extra_info",
            "raw_prompt",
            "sampling_seed",
        ):
            if key in batch.non_tensor_batch:
                output.non_tensor_batch[key] = batch.non_tensor_batch[key]
        if "response_mask" not in output.batch:
            output.batch["response_mask"] = compute_response_mask(output)

        values_input, values_pad = pad_dataproto_to_divisor(output, critic_wg.world_size)
        values_output = unpad_dataproto(
            critic_wg.compute_values(values_input), pad_size=values_pad
        )
        output = output.union(values_output)

        decoded = decode_response_strs(output, tokenizer, prompt_length, response_length)
        if "extra_info" not in output.non_tensor_batch:
            output.non_tensor_batch["extra_info"] = np.array(
                [{} for _ in decoded], dtype=object
            )
        for index, response in enumerate(decoded):
            output.non_tensor_batch["extra_info"][index]["full_response_str"] = response
        true_rewards, _ = compute_reward(output, reward_fn)
        output.batch["true_token_level_rewards"] = true_rewards
        return output

    manifest_path = _cache_manifest_path(cache_path)
    if cache_path.exists() != manifest_path.exists():
        raise FileNotFoundError(
            f"Incomplete shared-root cache: expected both {cache_path} and {manifest_path}"
        )

    if not cache_path.exists():
        from torchdata.stateful_dataloader import StatefulDataLoader
        from verl.trainer.main_ppo import create_rl_dataset
        from verl.utils.dataset.rl_dataset import collate_fn as rl_collate_fn
        root_records = []
        source_columns = ["responses", "tree_uids", "tree_rids", "tree_advantages"]
        source_file = pq.ParquetFile(root_source_path)
        for dataset_idx, record_batch in enumerate(
            source_file.iter_batches(batch_size=1, columns=source_columns)
        ):
            row = record_batch.to_pylist()[0]
            fields = (
                list(row["responses"]),
                list(row["tree_uids"]),
                list(row["tree_rids"]),
                list(row["tree_advantages"]),
            )
            lengths = {len(field) for field in fields}
            if len(lengths) != 1:
                raise ValueError(
                    f"Mismatched nested fields in reward-guided row {dataset_idx}: {lengths}"
                )
            for response, uid, rid, advantages in zip(*fields):
                rid = str(rid)
                if not rid.startswith("r0_"):
                    continue
                seq = int(rid.split("_", 1)[1])
                if seq % total_samples != dataset_idx:
                    raise ValueError(
                        f"Root {rid} is stored under prompt {dataset_idx}; "
                        f"expected {seq % total_samples}."
                    )
                root_records.append(
                    (seq, str(uid), str(response), len(advantages))
                )
        root_records.sort(key=lambda record: record[0])
        actual_sequences = [record[0] for record in root_records]
        if actual_sequences != list(range(trees_target)):
            raise ValueError(
                f"Reward-guided source does not contain exactly {trees_target} ordered roots."
            )
        if len({record[1] for record in root_records}) != trees_target:
            raise ValueError("Reward-guided source root UIDs are not unique.")
        del source_file

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
        if len(rl_dataset) != total_samples:
            raise ValueError(
                f"RL dataset has {len(rl_dataset)} rows; expected {total_samples}."
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
        root_input = prompt_buffer.draw(trees_target)
        source_uids = np.asarray([record[1] for record in root_records], dtype=object)
        source_texts = [record[2] for record in root_records]
        expected_lengths = [record[3] for record in root_records]
        root_input.non_tensor_batch["uid"] = source_uids
        root_input.non_tensor_batch["sampling_seed"] = np.full(
            trees_target, -1, dtype=np.int64
        )
        root_input.non_tensor_batch["full_response_str"] = np.asarray(
            source_texts, dtype=object
        )
        if "extra_info" not in root_input.non_tensor_batch:
            root_input.non_tensor_batch["extra_info"] = np.array(
                [{} for _ in source_texts], dtype=object
            )
        else:
            root_input.non_tensor_batch["extra_info"] = np.array(
                [copy.copy(item) for item in root_input.non_tensor_batch["extra_info"]],
                dtype=object,
            )
        for index, response in enumerate(source_texts):
            root_input.non_tensor_batch["extra_info"][index][
                "full_response_str"
            ] = response

        response_ids = []
        retokenization_deltas = defaultdict(int)
        text_truncated_root_count = 0
        eos_dropped_at_limit_count = 0
        eos_appended_count = 0
        roundtrip_mismatch_sequences = []
        encode_batch_size = 256
        for start in range(0, trees_target, encode_batch_size):
            texts = source_texts[start : start + encode_batch_size]
            encoded_batch = tokenizer(texts, add_special_tokens=False)["input_ids"]
            for offset, encoded in enumerate(encoded_batch):
                expected = expected_lengths[start + offset]
                encoded = list(encoded)
                original_encoded_length = len(encoded)
                retokenization_deltas[expected - original_encoded_length] += 1
                if tokenizer.eos_token_id is None:
                    raise ValueError("Tokenizer has no EOS token for reconstructed roots.")
                roundtrip = tokenizer.decode(
                    encoded,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if roundtrip != texts[offset]:
                    # Tokenizer normalization can canonicalize equivalent
                    # Unicode spellings (observed for one combining-dot form).
                    # Keep the source text for reward/evaluation and record the
                    # affected root for audit.
                    roundtrip_mismatch_sequences.append(start + offset)
                # The legacy response mask includes EOS, while the saved text
                # was decoded with skip_special_tokens=True.  An old trajectory
                # shorter than the 2048-token cap therefore ended with EOS.  At
                # the cap, append EOS only when it exactly explains the one-token
                # difference; otherwise treat the trajectory as length-limited.
                should_append_eos = (
                    expected < response_length
                    or expected == original_encoded_length + 1
                )
                if should_append_eos and (
                    not encoded or encoded[-1] != int(tokenizer.eos_token_id)
                ):
                    encoded.append(int(tokenizer.eos_token_id))
                    eos_appended_count += 1
                if len(encoded) > response_length:
                    # Decoding is not injective: a canonical re-encoding can be
                    # slightly longer than the sampled tokenization. Append EOS
                    # first, then retain the original 2048-token context limit.
                    encoded = encoded[:response_length]
                    if original_encoded_length > response_length:
                        text_truncated_root_count += 1
                    else:
                        eos_dropped_at_limit_count += 1
                response_ids.append(encoded)

        prompt_ids = root_input.batch["input_ids"]
        prompt_mask = root_input.batch["attention_mask"]
        prompt_positions = root_input.batch["position_ids"]
        responses = torch.full(
            (trees_target, response_length),
            tokenizer.pad_token_id,
            dtype=prompt_ids.dtype,
        )
        response_mask = torch.zeros(
            (trees_target, response_length), dtype=prompt_mask.dtype
        )
        for index, ids in enumerate(response_ids):
            length = len(ids)
            responses[index, :length] = torch.as_tensor(ids, dtype=responses.dtype)
            response_mask[index, :length] = 1
        input_ids = torch.cat([prompt_ids, responses], dim=-1)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=-1)
        delta = torch.arange(1, response_length + 1, dtype=prompt_positions.dtype)
        delta = delta.unsqueeze(0).expand(trees_target, -1)
        if prompt_positions.dim() == 3:
            delta = delta.view(trees_target, 1, -1).expand(
                trees_target, prompt_positions.size(1), -1
            )
        position_ids = torch.cat(
            [prompt_positions, prompt_positions[..., -1:] + delta], dim=-1
        )
        root_batch = DataProto.from_dict(
            tensors={
                "prompts": prompt_ids,
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": response_mask,
            },
            non_tensors=root_input.non_tensor_batch,
            meta_info=root_input.meta_info,
        )
        # Recompute only critic values and verifier rewards; no root generation.
        values_input, values_pad = pad_dataproto_to_divisor(root_batch, critic_wg.world_size)
        values_output = unpad_dataproto(
            critic_wg.compute_values(values_input), pad_size=values_pad
        )
        root_batch = root_batch.union(values_output)
        decode_response_strs(root_batch, tokenizer, prompt_length, response_length)
        true_rewards, _ = compute_reward(root_batch, reward_fn)
        root_batch.batch["true_token_level_rewards"] = true_rewards
        root_batch.to("cpu")
        fingerprint = root_fingerprint(root_batch)
        manifest = dict(expected_manifest)
        manifest["fingerprint"] = fingerprint
        manifest["root_text_source"] = "reward-guided parquet r0_* responses"
        manifest["retokenization_length_delta"] = {
            str(delta): count for delta, count in sorted(retokenization_deltas.items())
        }
        manifest["text_truncated_root_count"] = text_truncated_root_count
        manifest["eos_dropped_at_limit_count"] = eos_dropped_at_limit_count
        manifest["eos_appended_count"] = eos_appended_count
        manifest["roundtrip_mismatch_sequences"] = roundtrip_mismatch_sequences
        _atomic_save_dataproto(root_batch, cache_path)
        _atomic_write_json(manifest, manifest_path)
        print(
            f"Reconstructed {trees_target} existing reward-guided roots and saved "
            f"{cache_path} (sha256={fingerprint})."
        )
        print(
            "Retokenization audit: "
            f"stored_minus_canonical={dict(sorted(retokenization_deltas.items()))}, "
            f"eos_appended={eos_appended_count}, "
            f"text_truncated={text_truncated_root_count}, "
            f"eos_dropped_at_limit={eos_dropped_at_limit_count}, "
            f"roundtrip_mismatch={roundtrip_mismatch_sequences}."
        )
        del root_batch, root_input, prompt_buffer, dataloader, rl_dataset, root_records
        gc.collect()
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_cache_manifest(manifest, expected_manifest, cache_path)
        print(f"Reusing shared roots from {cache_path} (sha256={manifest['fingerprint']}).")

    if phase == "root":
        return

    root_fingerprint_expected = manifest["fingerprint"]
    for mode_index, mode in enumerate(modes_to_run):
        output_path = output_paths[mode]
        root_batch = DataProto.load_from_disk(cache_path)
        actual_fingerprint = root_fingerprint(root_batch)
        if actual_fingerprint != root_fingerprint_expected:
            raise RuntimeError(
                f"Shared-root cache fingerprint changed: expected {root_fingerprint_expected}, "
                f"found {actual_fingerprint}."
            )
        print(
            f"Starting {mode}-guided search from shared roots: "
            f"S_max={n_branch_rounds}, trees_per_prompt={trees_per_prompt}, "
            f"baseline={baseline_by_mode[mode]}."
        )

        global_batch = None
        next_states = {}
        search_count = {}
        max_perf_diffs = {}
        tree_search_state_by_uid = {}
        uid_to_dataset_idx = {}
        tree_seq = {}
        response_by_rid = {}
        response_ids_by_rid = {}
        values_by_rid = {}
        true_reward_by_rid = {}
        guidance_reward_by_rid = {}
        idx_responses = defaultdict(list)
        idx_sample_counter = defaultdict(int)
        global_sample_counter = 0
        selection_events = []
        terminal_snapshots = {}
        advantage_snapshots = {}

        def ingest_output(output: DataProto, round_index: int):
            nonlocal global_batch, global_sample_counter
            output.batch["token_level_rewards"] = make_guidance_rewards(output, mode)
            new_indices = set_opts_ttpo_info(output, global_batch, next_states, round_index)
            output.non_tensor_batch["episodic_returns"] = compute_episodic_returns(
                output, global_batch
            )
            current_size, current_response_len = output.batch["responses"].shape
            output.batch["state_branches"] = torch.ones(current_size, current_response_len)
            output.batch["advantages"] = torch.zeros(current_size, current_response_len)
            output.batch["returns"] = torch.zeros(current_size, current_response_len)
            global_batch = output if global_batch is None else merge_batches(global_batch, output)
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
                new_sample_indices=new_indices,
                raw_prompt_len=global_batch.non_tensor_batch["raw_prompt_len"],
                max_prompt_len=prompt_length,
                advantages=global_batch.batch["advantages"],
            )
            global_batch.batch["advantages"] = advantages
            global_batch.batch["returns"] = returns

            responses = decode_response_strs(output, tokenizer, prompt_length, response_length)
            response_ids = output.batch["responses"].detach().cpu()
            response_mask = output.batch["response_mask"].detach().cpu().bool()
            values = output.batch["values"].detach().cpu()
            true_rewards = output.batch["true_token_level_rewards"].detach().cpu()
            guidance_rewards = output.batch["token_level_rewards"].detach().cpu()
            for local_index, (response, uid) in enumerate(
                zip(responses, output.non_tensor_batch["uid"])
            ):
                dataset_idx = uid_to_dataset_idx[uid]
                idx_sample_counter[dataset_idx] += 1
                global_sample_counter += 1
                rid = str(output.non_tensor_batch["rid"][local_index])
                pid = output.non_tensor_batch["pid"][local_index]
                valid = response_mask[local_index]
                response_by_rid[rid] = response
                response_ids_by_rid[rid] = [
                    int(x) for x in response_ids[local_index][valid].tolist()
                ]
                values_by_rid[rid] = [float(x) for x in values[local_index][valid].tolist()]
                true_reward_by_rid[rid] = float(true_rewards[local_index].sum())
                guidance_reward_by_rid[rid] = float(guidance_rewards[local_index].sum())
                idx_responses[dataset_idx].append(
                    {
                        "response": response,
                        "uid": str(uid),
                        "tree_seq": int(tree_seq[uid]),
                        "sample_index": idx_sample_counter[dataset_idx],
                        "global_index": global_sample_counter,
                        "rid": rid,
                        "pid": None if pid is None else str(pid),
                        "branch_pos": int(output.non_tensor_batch["branch_pos"][local_index]),
                        "round": round_index,
                        "sampling_seed": int(output.non_tensor_batch["sampling_seed"][local_index]),
                    }
                )

            affected_uids = set(output.non_tensor_batch["uid"])
            refresh_tree_search_states(
                batch=global_batch,
                affected_uids=affected_uids,
                tree_search_state_by_uid=tree_search_state_by_uid,
                gamma=gamma,
                max_prompt_length=prompt_length,
                tokenizer=tokenizer,
                round_idx=round_index,
            )
            if round_index in advantage_rounds:
                advantage_snapshots[round_index] = _snapshot_advantages(global_batch)
            if round_index in snapshot_rounds:
                terminal_snapshots[round_index] = _snapshot_terminals(
                    tree_search_state_by_uid,
                    uid_to_dataset_idx,
                    tree_seq,
                    response_by_rid,
                    total_samples,
                    trees_per_prompt,
                )
            return affected_uids

        root_uids = root_batch.non_tensor_batch["uid"]
        for seq, uid in enumerate(root_uids):
            uid_to_dataset_idx[uid] = seq % total_samples
            tree_seq[uid] = seq
        ingest_output(root_batch, 0)

        for round_index in range(1, n_branch_rounds + 1):
            selected_states = select_next_states(
                batch=global_batch,
                search_count=search_count,
                max_perf_diffs=max_perf_diffs,
                max_search_per_tree=max_search_per_tree,
                tree_search_state_by_uid=tree_search_state_by_uid,
                max_searched_tree_ratio=1.0,
                search_batch_size=trees_target,
                perf_diff_baseline_mode=baseline_by_mode[mode],
            )
            for uid in selected_states:
                selection_events.append(
                    {
                        "tree_seq": int(tree_seq[uid]),
                        "dataset_idx": int(uid_to_dataset_idx[uid]),
                        "round": round_index,
                        "score": float(tree_search_state_by_uid[uid].candidate_selection_score),
                    }
                )
            next_states = selected_to_branch_points(selected_states, global_batch)
            if not next_states:
                print(f"[{mode}] no candidates at round {round_index}; stopping early.")
                break

            branch_input = prepare_next_round_input(
                global_batch=global_batch,
                next_states=next_states,
                pad_token_id=tokenizer.pad_token_id,
            )
            branch_input.non_tensor_batch["sampling_seed"] = np.asarray(
                [
                    request_sampling_seed(
                        rollout_seed_base,
                        prompt_index_offset + uid_to_dataset_idx[uid],
                        tree_seq[uid] // total_samples,
                        round_index,
                    )
                    for uid in branch_input.non_tensor_batch["uid"]
                ],
                dtype=np.int64,
            )
            is_last_generation = (
                mode_index == len(modes_to_run) - 1 and round_index == n_branch_rounds
            )
            branch_output = generate_and_score(
                branch_input, sleep_after=is_last_generation
            )
            ingest_output(branch_output, round_index)
            print(
                f"[{mode}] round {round_index}/{n_branch_rounds}: "
                f"{len(branch_output)} suffixes, {global_sample_counter} total responses."
            )

        completed_rounds = sorted(terminal_snapshots)
        if not completed_rounds:
            raise RuntimeError(f"No terminal snapshots were created for {mode} mode.")
        last_terminal_round = completed_rounds[-1]
        completed_advantage_rounds = sorted(advantage_snapshots)
        last_advantage_round = completed_advantage_rounds[-1]
        for round_index in snapshot_rounds:
            if round_index <= n_branch_rounds and round_index not in terminal_snapshots:
                terminal_snapshots[round_index] = terminal_snapshots[last_terminal_round]
        for round_index in advantage_rounds:
            if round_index <= n_branch_rounds and round_index not in advantage_snapshots:
                advantage_snapshots[round_index] = advantage_snapshots[last_advantage_round]

        final_advantages = _snapshot_advantages(global_batch)
        columns = {
            name: [[] for _ in range(total_samples)]
            for name in (
                "responses",
                "sample_indices",
                "global_indices",
                "tree_uids",
                "tree_seqs",
                "tree_rids",
                "tree_pids",
                "tree_branch_pos",
                "tree_advantages",
                "response_ids",
                "tree_values",
                "tree_rewards",
                "tree_guidance_rewards",
                "tree_rounds",
                "tree_sampling_seeds",
            )
        }
        advantage_columns = {
            round_index: [[] for _ in range(total_samples)] for round_index in advantage_rounds
        }
        for dataset_idx, records in idx_responses.items():
            for record in records:
                rid = record["rid"]
                columns["responses"][dataset_idx].append(record["response"])
                columns["sample_indices"][dataset_idx].append(record["sample_index"])
                columns["global_indices"][dataset_idx].append(record["global_index"])
                columns["tree_uids"][dataset_idx].append(record["uid"])
                columns["tree_seqs"][dataset_idx].append(record["tree_seq"])
                columns["tree_rids"][dataset_idx].append(rid)
                columns["tree_pids"][dataset_idx].append(record["pid"])
                columns["tree_branch_pos"][dataset_idx].append(record["branch_pos"])
                columns["tree_advantages"][dataset_idx].append(final_advantages[rid])
                columns["response_ids"][dataset_idx].append(response_ids_by_rid[rid])
                columns["tree_values"][dataset_idx].append(values_by_rid[rid])
                columns["tree_rewards"][dataset_idx].append(true_reward_by_rid[rid])
                columns["tree_guidance_rewards"][dataset_idx].append(
                    guidance_reward_by_rid[rid]
                )
                columns["tree_rounds"][dataset_idx].append(record["round"])
                columns["tree_sampling_seeds"][dataset_idx].append(record["sampling_seed"])
                for round_index in advantage_rounds:
                    advantage_columns[round_index][dataset_idx].append(
                        advantage_snapshots[round_index].get(rid, [])
                    )

        mode_dataset = dataset.copy(deep=True)
        for name, values in columns.items():
            mode_dataset[name] = values
        for round_index, values in advantage_columns.items():
            mode_dataset[f"tree_advantages_s{round_index}"] = values
        for round_index, values in terminal_snapshots.items():
            mode_dataset[f"opts_avg_responses_s{round_index}"] = values
        mode_dataset["opts_reward_mode"] = mode
        mode_dataset["perf_diff_baseline"] = baseline_by_mode[mode]
        mode_dataset["shared_root_fingerprint"] = root_fingerprint_expected

        makedirs(str(output_path.parent), exist_ok=True)
        mode_dataset.to_parquet(output_path)
        selection_path = output_path.with_name(output_path.stem + "_selections.parquet")
        pd.DataFrame(
            selection_events,
            columns=["tree_seq", "dataset_idx", "round", "score"],
        ).to_parquet(selection_path, index=False)
        print(f"Saved {mode}-guided results to {output_path}.")
        print(f"Saved {mode}-guided selections to {selection_path}.")

        del root_batch, global_batch, mode_dataset
        gc.collect()


if __name__ == "__main__":
    main()
