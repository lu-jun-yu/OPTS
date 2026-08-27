# Copyright 2025 Junyu Lu (Julian Lou). All rights reserved.

"""RQ1: does the unbiased trajectory return suffice for unbiased policy
gradients on tree trajectories?

Reference gradient (per-trajectory token-mean REINFORCE, gamma=1, lam=1, no
baseline, binary 0/1 rewards):

    g(tau) = (1/|tau|) sum_t R(tau) grad log pi(a_t|s_t)
    g*     = (1/(902*256)) sum_{x,j} g(tau_{x,j})

estimated from the two stored 128 x 902 i.i.d. rollout parquets
(ppo_0704_n8_iid_n128 + ppo_new_0704_n8_iid_n128).

Tree data: 16 groups x 902 prompts of backbone trajectories; each backbone is
split at its response-token midpoint (branch point) and 15 extra on-policy
suffixes are sampled from the branch state (budget 2048 - prefix_len, same
sampling config as the stored rollouts). Only the K=15 tree is generated;
K in {0,1,3,7} reuses the first K extra suffixes.

Per-tree aggregations (z_e = R_e grad log pi(a_e|s_e); prefix tokens carry the
branch-mean reward Rbar_K = (R_0 + ... + R_K)/(K+1) -- the same return
assignment for both methods -- suffix tokens carry their own MC return R_j;
TTPG weights prefix tokens 1 and each suffix 1/(K+1)):

    g_Naive(T) = sum_e z_e / |T|
    g_TTPG(T)  = sum_e w_e z_e / sum_e w_e

Modes:
  gen      : vLLM generation of backbones (n=16/prompt) + 15 extra suffixes
             per branch point, plus rule-based rewards. Token ids are stored
             exactly as sampled (no detokenize/retokenize round trip).
  gstar    : per-trajectory-normalized gradient sum over a 128 x 902 parquet
             (one fp32 accumulator; --rank/--world-size prompt sharding).
  treegrad : per-group tree gradient accumulation. All 9 accumulators
             (Naive K in {1,3,7,15}, TTPG K in {1,3,7,15}, shared K=0) live on
             the GPU (~62 GiB); one group's 902 trees per job.
  metrics  : combine g* and the 16 group files; report Bias/Var/Cos for
             K in {0,1,3,7,15} and render the 1x3 figure.
"""

import argparse
import json
import os
import re
import sys
import time

LLM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if LLM_DIR not in sys.path:
    sys.path.insert(0, LLM_DIR)

import numpy as np
import pandas as pd

RESPONSE_LENGTH = 2048
GROUPS = 16
N_EXTRA = 15
KS = (0, 1, 3, 7, 15)
# Accumulator slots: K=0 is method-independent (Naive == TTPG there).
ACC_KEYS = ["k0"] + [f"k{k}_{m}" for k in (1, 3, 7, 15) for m in ("naive", "ttpg")]
TEST_DATA = os.path.join(LLM_DIR, "data/test.parquet")
MODEL_DEFAULT = os.path.join(LLM_DIR, "results/step400/merged/ppo_0704_n8_actor")
OUT_DEFAULT = os.path.join(LLM_DIR, "results/step400/rq1")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _prompt_ids(row, tokenizer):
    messages = [dict(m) for m in row["prompt"]]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)


def _flat_grad(params):
    import torch

    return torch.cat([p.grad.detach().view(-1) for p in params]).to(torch.float32)


def _load_actor(args, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cuda" and dtype == torch.float32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    if getattr(args, "grad_ckpt", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # HF GradientCheckpointingLayer only checkpoints when self.training is
        # set; Qwen3 has no dropout/BN, so train() is numerically identical to
        # eval() here.
        model.train()
    else:
        model.gradient_checkpointing_disable()
        model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    return model, tokenizer, params


def _chunked_token_logp(model, input_ids, pad_id, device, log_chunk):
    """Forward pass; yields per-chunk (tok_logp [1, e-s], slice) so callers can
    assemble arbitrary masked losses without materializing a fp32 logits copy.

    Returns (loss_terms, n_vocab_positions) where loss_terms is the list of
    per-chunk token logprobs for positions 1..L-1 (graph-retaining).
    """
    import torch

    attention_mask = (input_ids != pad_id).long()
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    shift_labels = input_ids[:, 1:]
    terms = []
    for s in range(0, logits.shape[1] - 1, log_chunk):
        e = min(s + log_chunk, logits.shape[1] - 1)
        logp_c = torch.log_softmax(logits[:, s:e].float(), dim=-1)
        tok_logp_c = torch.gather(logp_c, 2, shift_labels[:, s:e].unsqueeze(-1)).squeeze(-1)
        terms.append((tok_logp_c, s, e))
    return terms


def _masked_sum_loss(terms, mask):
    """loss = -sum(mask * tok_logp); mask covers positions 1..L-1."""
    loss = None
    for tok_logp_c, s, e in terms:
        part = -(tok_logp_c * mask[:, s:e]).sum()
        loss = part if loss is None else loss + part
    return loss


# ---------------------------------------------------------------------------
# gen mode
# ---------------------------------------------------------------------------

def _reward_text_task(task):
    key, data_source, text, ground_truth = task
    from utils.reward_fn import compute_score

    return key, float(compute_score(data_source, text, ground_truth)["score"])


def run_gen(args):
    from multiprocessing import Pool
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    df = pd.read_parquet(args.prompts)
    n_prompts = len(df)
    prompt_indices = list(range(args.shard, n_prompts, args.num_shards))
    if args.max_prompts > 0:
        prompt_indices = prompt_indices[: args.max_prompts]
    print(f"[gen shard {args.shard}/{args.num_shards}] {len(prompt_indices)} prompts", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eos_id = tokenizer.eos_token_id
    prompt_ids = {i: _prompt_ids(df.iloc[i], tokenizer) for i in prompt_indices}

    llm = LLM(
        model=args.model,
        dtype="auto",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=4096,
        max_num_batched_tokens=32768,
        seed=args.seed_base,
    )

    # ---- backbones: n_groups per prompt ----
    # per-request seeds make the draws independent of engine scheduling/sharding
    n_groups, g_off = args.n_groups, args.group_offset
    bb_params_list = [
        SamplingParams(
            n=n_groups, temperature=1.0, top_p=0.95, top_k=50,
            max_tokens=RESPONSE_LENGTH, seed=args.seed_base + i * 64 + g_off,
        )
        for i in prompt_indices
    ]
    t0 = time.time()
    bb_outs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids[i]) for i in prompt_indices], bb_params_list
    )
    print(f"[gen shard {args.shard}] backbones done in {time.time() - t0:.1f}s", flush=True)

    # ---- branch states + extra suffixes ----
    records = {}  # (prompt_idx, group) -> row dict
    suf_requests, suf_params_list, suf_keys = [], [], []
    for req_i, i in enumerate(prompt_indices):
        out = bb_outs[req_i]
        assert len(out.outputs) == n_groups
        for g in range(n_groups):
            comp = out.outputs[g]
            bb_ids = list(comp.token_ids)
            if len(bb_ids) == 0:
                print(f"[gen] WARNING empty backbone at prompt {i} group {g}; skipped", flush=True)
                continue
            prefix_len = min(max(1, len(bb_ids) // 2), len(bb_ids) - 1)
            if len(bb_ids) == 1:
                prefix_len = 1
            rec = {
                "prompt_idx": i,
                "group": g_off + g,
                "prefix_len": prefix_len,
                "backbone_ids": np.asarray(bb_ids, dtype=np.int32),
                "backbone_finish": comp.finish_reason,
            }
            records[(i, g_off + g)] = rec
            if args.backbone_only:
                continue
            branch_ids = prompt_ids[i] + bb_ids[:prefix_len]
            suf_requests.append(TokensPrompt(prompt_token_ids=branch_ids))
            suf_params_list.append(
                SamplingParams(
                    n=N_EXTRA,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=50,
                    max_tokens=RESPONSE_LENGTH - prefix_len,
                    seed=args.seed_base + 10_000_000 + i * 64 + g_off + g,
                )
            )
            suf_keys.append((i, g_off + g))

    if args.backbone_only:
        empty = np.asarray([], dtype=np.int32)
        for rec in records.values():
            rec["suffix_ids"] = [empty] * N_EXTRA
            rec["suffix_finish"] = ["length"] * N_EXTRA
        suf_outs = []
    else:
        t0 = time.time()
        suf_outs = llm.generate(suf_requests, suf_params_list)
        print(
            f"[gen shard {args.shard}] {len(suf_requests)} suffix requests done in {time.time() - t0:.1f}s",
            flush=True,
        )
    for key, out in zip(suf_keys, suf_outs):
        rec = records[key]
        rec["suffix_ids"] = [np.asarray(list(c.token_ids), dtype=np.int32) for c in out.outputs]
        rec["suffix_finish"] = [c.finish_reason for c in out.outputs]
    del llm, bb_outs, suf_outs

    # ---- rewards (CPU parallel) ----
    tasks = []
    for (i, g), rec in records.items():
        row = df.iloc[i]
        gt = row["reward_model"]["ground_truth"]
        ds = row["data_source"]
        bb_text = tokenizer.decode(rec["backbone_ids"], skip_special_tokens=True)
        tasks.append(((i, g, -1), ds, bb_text, gt))
        if args.backbone_only:
            continue
        prefix = rec["backbone_ids"][: rec["prefix_len"]]
        for j, suf in enumerate(rec["suffix_ids"]):
            text = tokenizer.decode(np.concatenate([prefix, suf]), skip_special_tokens=True)
            tasks.append(((i, g, j), ds, text, gt))
    t0 = time.time()
    with Pool(processes=args.reward_procs) as pool:
        for done, (key, score) in enumerate(
            pool.imap_unordered(_reward_text_task, tasks, chunksize=16)
        ):
            i, g, j = key
            rec = records[(i, g)]
            if j < 0:
                rec["backbone_reward"] = score
            else:
                rec.setdefault("suffix_rewards", {})[j] = score
            if (done + 1) % 20000 == 0 or done + 1 == len(tasks):
                print(f"[gen shard {args.shard}] rewards {done + 1}/{len(tasks)}", flush=True)
    print(f"[gen shard {args.shard}] rewards done in {time.time() - t0:.1f}s", flush=True)

    rows = []
    for (i, g) in sorted(records):
        rec = records[(i, g)]
        rows.append(
            {
                "prompt_idx": i,
                "group": g,
                "prefix_len": rec["prefix_len"],
                "backbone_ids": rec["backbone_ids"],
                "backbone_finish": rec["backbone_finish"],
                "backbone_reward": np.float32(rec.get("backbone_reward", 0.0)),
                "suffix_ids": rec["suffix_ids"],
                "suffix_finish": rec["suffix_finish"],
                "suffix_rewards": np.asarray(
                    [rec.get("suffix_rewards", {}).get(j, 0.0) for j in range(N_EXTRA)],
                    dtype=np.float32,
                ),
            }
        )
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_parquet(args.out)
    n_bb = len(out_df)
    print(
        f"[gen shard {args.shard}] saved {n_bb} trees -> {args.out}; "
        f"backbone reward mean={out_df['backbone_reward'].mean():.4f}, "
        f"suffix reward mean={np.concatenate(out_df['suffix_rewards'].to_list()).mean():.4f}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# gstar mode
# ---------------------------------------------------------------------------

def run_gstar(args):
    import torch
    from experiments.pg_sample_complexity import _build_samples, _make_micro_batches

    device = torch.device(args.device)
    df = pd.read_parquet(args.data)
    rewards = np.load(args.rewards)["rewards"]
    assert rewards.shape[0] == len(df)

    n_prompts = len(df)
    prompt_indices = list(range(args.rank, n_prompts, args.world_size))
    if args.max_prompts > 0:
        prompt_indices = prompt_indices[: args.max_prompts]

    model, tokenizer, params = _load_actor(args, device)
    pad_id = tokenizer.pad_token_id
    numel = sum(p.numel() for p in params)

    t0 = time.time()
    samples, stats = _build_samples(df, rewards, prompt_indices, tokenizer)
    if args.max_responses > 0:
        samples = samples[: args.max_responses]
    print(f"[gstar {args.tag} rank {args.rank}] tokenize done in {time.time() - t0:.1f}s: {stats}",
          flush=True)

    acc = torch.zeros(numel, dtype=torch.float32, device=device)
    n_passes = len(samples)
    for i in range(n_passes):
        pass_samples = samples[i]
        if not pass_samples:
            continue
        micro_batches = _make_micro_batches(pass_samples, args.token_budget)
        for mb in micro_batches:
            maxlen = max(len(item[0]) for item in mb)
            input_ids = torch.full((len(mb), maxlen), pad_id, dtype=torch.long)
            resp_mask = torch.zeros((len(mb), maxlen), dtype=torch.float32)
            for b, (seq, resp_len, _, _) in enumerate(mb):
                input_ids[b, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                resp_mask[b, len(seq) - resp_len : len(seq)] = 1.0
            input_ids = input_ids.to(device, non_blocking=True)
            resp_mask = resp_mask.to(device, non_blocking=True)
            # per-sample loss weights: 1/|tau| for per-trajectory token-mean
            # ("mean"), 1 for the standard episodic trajectory-sum PG ("sum")
            if args.pg_norm == "mean":
                inv_len = torch.tensor(
                    [1.0 / item[1] for item in mb], dtype=torch.float32
                ).to(device, non_blocking=True)
            else:
                inv_len = torch.ones(len(mb), dtype=torch.float32, device=device)
            terms = _chunked_token_logp(model, input_ids, pad_id, device, args.log_chunk)
            loss = None
            for tok_logp_c, s, e in terms:
                per_sample = (tok_logp_c * resp_mask[:, 1:][:, s:e]).sum(1)
                part = -(per_sample * inv_len).sum()
                loss = part if loss is None else loss + part
            model.zero_grad(set_to_none=True)
            loss.backward()
            acc += _flat_grad(params)
            del terms, loss
        if args.rank == 0 and ((i + 1) % 16 == 0 or i + 1 == n_passes):
            print(f"[gstar {args.tag} rank 0] pass {i + 1}/{n_passes}, "
                  f"elapsed {time.time() - t0:.1f}s", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"gstar_sum_{args.tag}_rank{args.rank}.pt")
    torch.save(
        {"acc": acc.cpu(), "n_prompts_shard": len(prompt_indices), "tag": args.tag,
         "data": args.data, "stats": stats},
        out_path,
    )
    print(f"[gstar {args.tag} rank {args.rank}] saved {out_path}", flush=True)


# ---------------------------------------------------------------------------
# treegrad mode
# ---------------------------------------------------------------------------

def _response_terms_loss(model, seq_ids, keep_ranges, pad_id, device, log_chunk):
    """Forward one sequence; return per-range masked losses.

    keep_ranges: list of (lo, hi) response-token index ranges (0-based within
    seq_ids; token at index t is scored by the logit at t-1).
    """
    import torch

    input_ids = torch.tensor(seq_ids, dtype=torch.long, device=device).unsqueeze(0)
    terms = _chunked_token_logp(model, input_ids, pad_id, device, log_chunk)
    losses = []
    for lo, hi in keep_ranges:
        if hi <= lo:
            losses.append(None)
            continue
        mask = torch.zeros(1, input_ids.shape[1] - 1, dtype=torch.float32, device=device)
        mask[:, lo - 1 : hi - 1] = 1.0  # response token t <- shift position t-1
        losses.append(_masked_sum_loss(terms, mask))
    return losses


def run_treegrad(args):
    import torch

    device = torch.device(args.device)
    dfs = []
    for path in sorted(args.tree_data):
        dfs.append(pd.read_parquet(path))
    tdf = pd.concat(dfs, ignore_index=True)
    tdf = tdf[tdf["group"] == args.group].sort_values("prompt_idx").reset_index(drop=True)
    n_trees_total = len(tdf)
    tree_indices = list(range(args.rank, n_trees_total, args.world_size))
    if args.max_trees > 0:
        tree_indices = tree_indices[: args.max_trees]
    print(f"[treegrad g{args.group} rank {args.rank}] {len(tree_indices)}/{n_trees_total} trees",
          flush=True)

    df_test = pd.read_parquet(args.prompts)
    model, tokenizer, params = _load_actor(args, device)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    numel = sum(p.numel() for p in params)

    # Pre-tokenize prompts once, then order trees longest-first: peak
    # activation memory is reached within the first few trees, so an OOM would
    # surface immediately rather than hours into the run (the gradient sum is
    # order-invariant).
    t0 = time.time()
    prompt_ids_map = {}
    for ti in tree_indices:
        pi = int(tdf.iloc[ti]["prompt_idx"])
        if pi not in prompt_ids_map:
            prompt_ids_map[pi] = _prompt_ids(df_test.iloc[pi], tokenizer)
    tree_indices.sort(
        key=lambda ti: -(
            len(prompt_ids_map[int(tdf.iloc[ti]["prompt_idx"])])
            + len(tdf.iloc[ti]["backbone_ids"])
        )
    )
    print(f"[treegrad g{args.group}] prompts tokenized + sorted in {time.time() - t0:.1f}s",
          flush=True)

    # Accumulators: on GPU (fast) or CPU (for cards sharing memory with other
    # tenants). C/S live on the GPU; grads are folded into them with
    # foreach_add over per-param views, so no 6.4 GiB flat-grad temp is ever
    # materialized.
    acc_device = device if args.accs_on == "gpu" else torch.device("cpu")
    acc_keys = ("k0",) if args.backbone_only else ACC_KEYS
    accs = {k: torch.zeros(numel, dtype=torch.float32, device=acc_device) for k in acc_keys}
    C = torch.zeros(numel, dtype=torch.float32, device=device)
    S = torch.zeros(numel, dtype=torch.float32, device=device)

    def _views(flat):
        outs, o = [], 0
        for p in params:
            n = p.numel()
            outs.append(flat[o : o + n])
            o += n
        return outs

    Cv, Sv = _views(C), _views(S)

    def _fold_grad(dst_views, scale):
        torch._foreach_add_(dst_views, [p.grad.view(-1) for p in params], alpha=scale)

    def _upd(acc, alpha_c, alpha_s):
        if acc.is_cuda:
            acc.add_(C, alpha=alpha_c)
            acc.add_(S, alpha=alpha_s)
        else:
            tmp = C.mul(alpha_c)
            tmp.add_(S, alpha=alpha_s)
            acc.add_(tmp.cpu())
            del tmp

    diag = {"r0_pos": 0, "rs_pos": 0, "rs_tot": 0, "lens": [], "prefix_lens": []}

    t0 = time.time()
    for done, ti in enumerate(tree_indices):
        row = tdf.iloc[ti]
        prompt_ids = prompt_ids_map[int(row["prompt_idx"])]
        bb_ids = [int(t) for t in row["backbone_ids"]]
        if row["backbone_finish"] == "stop" and (not bb_ids or bb_ids[-1] != eos_id):
            bb_ids = bb_ids + [eos_id]
        plen = int(row["prefix_len"])
        r0 = float(row["backbone_reward"])
        suf_ids = []
        for j in range(N_EXTRA):
            ids = [int(t) for t in row["suffix_ids"][j]]
            if row["suffix_finish"][j] == "stop" and (not ids or ids[-1] != eos_id):
                ids = ids + [eos_id]
            suf_ids.append(ids)
        rs = [float(v) for v in row["suffix_rewards"]]
        diag["r0_pos"] += int(r0 > 0)
        diag["rs_pos"] += sum(int(v > 0) for v in rs)
        diag["rs_tot"] += N_EXTRA
        diag["lens"].append(len(bb_ids))
        diag["prefix_lens"].append(plen)

        l_pre, l0 = plen, len(bb_ids) - plen
        n_prompt = len(prompt_ids)
        C.zero_()  # C = G_pre (unscaled prefix-token gradient sum)
        S.zero_()  # S = sum over processed suffixes of R_j * G_j
        # the prefix gradient is needed whenever any branch reward is positive
        need_pre = (r0 > 0) or any(v > 0 for v in rs)
        dirty = False

        # backbone: one forward, separate prefix / suffix0 backwards
        if need_pre:
            seq = prompt_ids + bb_ids
            loss_pre, loss_s0 = _response_terms_loss(
                model, seq, [(n_prompt, n_prompt + l_pre), (n_prompt + l_pre, len(seq))],
                pad_id, device, args.log_chunk,
            )
            model.zero_grad(set_to_none=True)
            loss_pre.backward(retain_graph=True)
            _fold_grad(Cv, 1.0)
            if r0 > 0 and loss_s0 is not None:
                model.zero_grad(set_to_none=True)
                loss_s0.backward()
                _fold_grad(Sv, r0)
            del loss_pre, loss_s0
            dirty = True
        # K = 0 checkpoint (Rbar_0 = R_0; shared by Naive and TTPG)
        if dirty and r0 > 0:
            if args.pg_norm == "mean":
                d0 = l_pre + l0
                _upd(accs["k0"], r0 / d0, 1.0 / d0)
            else:
                _upd(accs["k0"], r0, 1.0)

        # extra suffixes; checkpoints at K = 1, 3, 7, 15
        if args.backbone_only:
            if (done + 1) % 25 == 0 or done + 1 == len(tree_indices):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                print(f"[treegrad g{args.group} rank {args.rank}] {done + 1}/{len(tree_indices)} trees, "
                      f"elapsed {time.time() - t0:.1f}s", flush=True)
            continue
        l_suf_cum = l0  # token-count running sum over suffixes 0..j
        r_cum = r0      # reward running sum over suffixes 0..j
        for j in range(1, N_EXTRA + 1):
            ids, rj = suf_ids[j - 1], rs[j - 1]
            l_suf_cum += len(ids)
            r_cum += rj
            if rj > 0 and ids:
                seq = prompt_ids + bb_ids[:plen] + ids
                (loss_j,) = _response_terms_loss(
                    model, seq, [(len(seq) - len(ids), len(seq))], pad_id, device, args.log_chunk
                )
                model.zero_grad(set_to_none=True)
                loss_j.backward()
                _fold_grad(Sv, rj)
                del loss_j
            if j in (1, 3, 7, 15) and dirty:
                rbar = r_cum / (j + 1)  # prefix-token return: mean over the K+1 branches
                if args.pg_norm == "mean":
                    dN = l_pre + l_suf_cum
                    dT = l_pre + l_suf_cum / (j + 1)
                    _upd(accs[f"k{j}_naive"], rbar / dN, 1.0 / dN)
                    _upd(accs[f"k{j}_ttpg"], rbar / dT, 1.0 / (dT * (j + 1)))
                else:
                    # standard episodic PG: within-tree sum, across-tree mean
                    _upd(accs[f"k{j}_naive"], rbar, 1.0)
                    _upd(accs[f"k{j}_ttpg"], rbar, 1.0 / (j + 1))

        if (done + 1) % 25 == 0 or done + 1 == len(tree_indices):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            print(f"[treegrad g{args.group} rank {args.rank}] {done + 1}/{len(tree_indices)} trees, "
                  f"elapsed {time.time() - t0:.1f}s", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"treegrad_g{args.group:02d}_rank{args.rank}.pt")
    # atomic: tmp + rename so a concurrent reducer never reads a partial file;
    # retry on transient ENOSPC (tmpfs full while the reducer is mid-fold)
    tmp_path = out_path + ".tmp"
    payload = {
        "accs": {k: v.cpu() for k, v in accs.items()},
        "group": args.group,
        "n_trees_total": n_trees_total,
        "n_trees_shard": len(tree_indices),
        "model": args.model,
        "pg_norm": args.pg_norm,
        "diag": {k: v for k, v in diag.items() if k != "lens" and k != "prefix_lens"},
    }
    for attempt in range(30):
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, out_path)
            break
        except (OSError, RuntimeError) as exc:
            print(f"[treegrad g{args.group}] save attempt {attempt + 1} failed: {exc}; "
                  f"retrying in 60s", flush=True)
            time.sleep(60)
    else:
        raise RuntimeError(f"could not save {out_path} after 30 attempts")
    np.savez(
        os.path.join(args.out_dir, f"treegrad_g{args.group:02d}_rank{args.rank}_stats.npz"),
        backbone_len=np.asarray(diag["lens"], dtype=np.int32),
        prefix_len=np.asarray(diag["prefix_lens"], dtype=np.int32),
    )
    print(f"[treegrad g{args.group} rank {args.rank}] saved {out_path}", flush=True)


# ---------------------------------------------------------------------------
# reduce mode: fold per-group accumulator files into block-reduction state
# (64 x 902 design with事后 regrouping over M in {2,4,8})
# ---------------------------------------------------------------------------

MS = (2, 4, 8)
N_GROUPS = 64


def _gstar_vector(out_dir):
    """g* = (S_old + S_new) / (256 * 902), fp32."""
    import torch

    g = None
    for tag in ("old", "new"):
        files = sorted(
            f for f in os.listdir(out_dir)
            if f.startswith(f"gstar_sum_{tag}_rank") and f.endswith(".pt")
        )
        assert files, f"no gstar files for tag {tag} in {out_dir}"
        for f in files:
            ck = torch.load(os.path.join(out_dir, f), map_location="cpu")
            v = ck["acc"]
            g = v.clone() if g is None else g + v
            del ck
    return g / ((2 * 128) * 902)


def run_reduce(args):
    """Fold complete treegrad_g*_rank0.pt files from --in-dir into a reduction
    state under the --state PREFIX (kept on /dev/shm):

      PREFIX.gstar.npy : fp32 reference gradient
      PREFIX.ckpt.pt   : {s1, Bopen, blocks, next, ...} written at 8-group closes
      PREFIX.next      : text marker of the last closed fold position
      PREFIX.done.pt   : final export for metrics2

    s1 and block accumulators live in RAM (~270GB for MS=(4,8,16)); a
    checkpoint is written after every 8 folds, so a crash costs at most the
    current 8-group block. Folded group files are deleted at each close.
    --init-from migrates an existing torch state once.
    """
    import torch

    global MS
    MS = tuple(int(x) for x in args.ms.split(","))
    torch.set_num_threads(max(1, min(64, os.cpu_count() or 1)))
    prefix = args.state
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    ckpt_path = prefix + ".ckpt.pt"
    gstar_path = prefix + ".gstar.npy"

    s1, Bopen = None, None
    if os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location="cpu")
        s1, Bopen, blocks = st["s1"], st["Bopen"], st["blocks"]
        nxt, numel, n_trees = st["next"], st["numel"], st["n_trees_per_group"]
        print(f"[reduce] resuming at group {nxt}", flush=True)
    else:
        if args.init_from:
            st = torch.load(args.init_from, map_location="cpu")
            nxt, n_trees = st["next"], st["n_trees_per_group"]
            numel = st["s1"][ACC_KEYS[0]].numel()
            np.save(gstar_path, st["g_star"].numpy())
            s1 = dict(st["s1"])
            blocks = {}
            for s in ACC_KEYS:
                for m in MS:
                    for q, (sq, dt) in st["blocks"].get(s, {}).get(m, {}).items():
                        blocks[f"{s}|{m}|{q}"] = [sq, dt]
            del st
            print(f"[reduce] migrated torch state (next={nxt})", flush=True)
        else:
            g = _gstar_vector(args.in_dir)
            np.save(gstar_path, g.numpy())
            nxt, blocks, n_trees = 0, {}, None
            numel = g.numel()
            del g

    g64 = torch.from_numpy(np.load(gstar_path)).to(torch.float64)
    # B per M; entries with M<=8 are zero right after each close. Bopen
    # persists B[M>8] across closes (checkpointed); B[M<=8] are folded per
    # 8-block and never need persistence.
    B = None

    t0 = time.time()
    while nxt < args.expect:
        path = os.path.join(args.in_dir, f"treegrad_g{nxt + args.reduce_offset:02d}_rank0.pt")
        if not os.path.exists(path):
            if args.watch:
                time.sleep(30)
                continue
            break
        ck = torch.load(path, map_location="cpu", mmap=True)
        n_trees = ck["n_trees_total"]
        if s1 is None:
            numel = ck["accs"][ACC_KEYS[0]].numel()
            s1 = {s: torch.zeros(numel, dtype=torch.float32) for s in ACC_KEYS}
        if B is None:
            B = {m: {s: torch.zeros(numel, dtype=torch.float32) for s in ACC_KEYS} for m in MS}
            if Bopen is not None:
                for m in MS:
                    if m > 8:
                        for s in ACC_KEYS:
                            B[m][s] += Bopen[s]
                Bopen = None
        for s in ACC_KEYS:
            v = ck["accs"][s] / n_trees
            s1[s] += v
            for m in MS:
                B[m][s] += v
                if (nxt + 1) % m == 0:
                    Bd = B[m][s].to(torch.float64)
                    blocks[f"{s}|{m}|{nxt // m}"] = [Bd.dot(Bd).item(), Bd.dot(g64).item()]
                    B[m][s].zero_()
            del v
        del ck
        nxt += 1
        print(f"[reduce] folded group {nxt - 1} ({time.time() - t0:.1f}s)", flush=True)
        if nxt % 8 == 0 or nxt >= args.expect:
            Bopen = {s: B[max(MS)][s].clone() for s in ACC_KEYS} if max(MS) > 8 else None
            state = {"s1": s1, "Bopen": Bopen, "blocks": blocks, "next": nxt,
                     "numel": numel, "n_trees_per_group": n_trees, "pg_norm": args.pg_norm}
            torch.save(state, ckpt_path)
            with open(prefix + ".next", "w") as f:
                f.write(str(nxt))
            if not args.no_delete:
                for g_old in range(nxt - 8, nxt):
                    p_old = os.path.join(
                        args.in_dir, f"treegrad_g{g_old + args.reduce_offset:02d}_rank0.pt")
                    if os.path.exists(p_old):
                        os.remove(p_old)
            print(f"[reduce] checkpoint at next={nxt}", flush=True)
    print(f"[reduce] done at next={nxt}", flush=True)

    if nxt >= args.expect:
        state = {
            "g_star": torch.from_numpy(np.load(gstar_path)),
            "s1": s1,
            "blocks": {s: {m: {int(q): tuple(v) for q, v in
                               ((kk.split("|")[2], vv) for kk, vv in blocks.items()
                                if kk.startswith(f"{s}|{m}|"))}
                           for m in MS} for s in ACC_KEYS},
            "next": nxt, "pg_norm": args.pg_norm, "n_trees_per_group": n_trees,
        }
        torch.save(state, prefix + ".done.pt")
        print(f"[reduce] exported {prefix}.done.pt", flush=True)


# ---------------------------------------------------------------------------
# groupmetrics mode: direct metrics from a handful of group files (<=16),
# reference gradient taken from the groups' own K=0 (backbone) slot
# ---------------------------------------------------------------------------

def run_groupmetrics(args):
    """Direct Bias/Var/MSE/Cos for a small number of treegrad group files.

    g*: with --gstar-dir, the mean k0 (backbone-only) gradient of INDEPENDENT
    backbone-only group files; otherwise the mean k0 of the estimator groups
    themselves (shared design). Blocks for M in --ms are consecutive groups.
    Includes the MSE = Bias^2 + (R-1)/R Var identity check on every point.
    """
    import torch

    global MS
    MS = tuple(int(x) for x in args.ms.split(","))
    torch.set_num_threads(max(1, min(64, os.cpu_count() or 1)))

    import glob as _glob
    files = sorted(_glob.glob(os.path.join(args.in_dir, "treegrad_g*_rank0.pt")))
    assert files, f"no group files in {args.in_dir}"
    n_groups = len(files)
    print(f"[groupmetrics] {n_groups} estimator groups from {args.in_dir}", flush=True)

    def _load_k0_mean(dir_path):
        """Mean of per-group k0 vectors (fp64), one file at a time."""
        fs = sorted(_glob.glob(os.path.join(dir_path, "treegrad_g*_rank0.pt")))
        assert fs, f"no group files in {dir_path}"
        acc, n_tr = None, None
        for f in fs:
            ck = torch.load(f, map_location="cpu")
            n_tr = ck["n_trees_total"]
            v = ck["accs"]["k0"].to(torch.float64) / n_tr
            acc = v if acc is None else acc.add_(v)
            del ck, v
        return acc / len(fs), len(fs), n_tr

    # ---- reference gradient ----
    indep = bool(args.gstar_dir) and os.path.abspath(args.gstar_dir) != os.path.abspath(args.in_dir)
    if indep:
        g, n_gstar_groups, _ = _load_k0_mean(args.gstar_dir)
        print(f"[groupmetrics] independent g* from {n_gstar_groups} groups in {args.gstar_dir}",
              flush=True)
    else:
        g, n_gstar_groups = None, n_groups  # shared: built from ghat["k0"] below
        print("[groupmetrics] shared g* (estimator groups' own k0)", flush=True)

    # ---- estimator side: single pass over files, keep all slots fp32 ----
    slots = ["k0"] + [f"k{k}_{m}" for k in KS if k > 0 for m in ("naive", "ttpg")]
    ghat = {s: [] for s in slots}
    n_prompts = None
    for f in files:
        ck = torch.load(f, map_location="cpu")
        n_prompts = ck["n_trees_total"]
        for s in slots:
            ghat[s].append(ck["accs"][s] / n_prompts)  # fp32, ~6.9GB per vector
        del ck
        print(f"[groupmetrics] loaded {os.path.basename(f)}", flush=True)

    if g is None:  # shared design: g* = mean of the estimator groups' k0
        g = ghat["k0"][0].to(torch.float64)
        for v in ghat["k0"][1:]:
            g += v
        g /= n_groups
    gn = g.norm().item()
    gn2 = gn * gn

    gout = {"per_k": {}}
    for k in KS:
        for meth in ("naive", "ttpg"):
            s = "k0" if k == 0 else f"k{k}_{meth}"
            vs = ghat[s]
            gbar = torch.zeros_like(g, dtype=torch.float64)
            for v in vs:
                gbar += v
            gbar /= n_groups
            bias = (gbar - g).norm().item() / gn
            e = {"bias": bias, "norm_gbar": gbar.norm().item(), "per_m": {}}
            gbar2 = gbar.dot(gbar).item()
            for m in MS:
                r = n_groups // m
                assert n_groups % m == 0
                vals = []
                for q in range(r):
                    Bv = torch.zeros_like(g, dtype=torch.float64)
                    for v in vs[q * m:(q + 1) * m]:
                        Bv += v
                    Bv /= m
                    vals.append((Bv.norm().item() ** 2, Bv.dot(g).item()))
                    del Bv
                sum_sq = sum(vv[0] for vv in vals)
                var = max((sum_sq - r * gbar2) / (r - 1), 0.0) / gn2 if r > 1 else float("nan")
                mse = sum(vv[0] - 2.0 * vv[1] + gn2 for vv in vals) / r / gn2
                cos = sum(vv[1] / max(vv[0], 1e-300) ** 0.5 for vv in vals) / r / gn
                resid = mse - bias * bias - (r - 1) / r * var
                e["per_m"][str(m)] = {"var": var, "mse": mse, "cos": cos, "identity_resid": resid}
            gout["per_k"].setdefault(str(k), {})[meth] = e
            del gbar
        print(f"[groupmetrics] K={k} done", flush=True)
    gout["n_groups"] = n_groups
    gout["n_trees_per_group"] = n_prompts
    gout["gstar_independent"] = indep
    gout["n_gstar_groups"] = n_gstar_groups
    gout["norm_gstar"] = gn
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, args.tag or "groupmetrics.json")
    with open(out, "w") as f:
        json.dump(gout, f, indent=2)
    print(f"[groupmetrics] saved {out}", flush=True)
    for k in KS:
        for meth in ("naive", "ttpg"):
            e = gout["per_k"][str(k)][meth]
            cells = " | ".join(
                f"M{mm}: var={v['var']:.4f} mse={v['mse']:.4f} cos={v['cos']:.4f} resid={v['identity_resid']:+.4f}"
                for mm, v in e["per_m"].items())
            print(f"K={k:>2} {meth:>5}: bias={e['bias']:.4f} | {cells}", flush=True)


# ---------------------------------------------------------------------------
# metrics2 mode: 64-group regrouped Bias/Var/MSE/Cos + figures
# ---------------------------------------------------------------------------

def _metrics_from_state(state):
    """Compute all RQ1 quantities from a reduction state."""
    import torch

    g = state["g_star"].to(torch.float64)
    gn2 = g.dot(g).item()
    gn = gn2 ** 0.5
    s1, blocks = state["s1"], state["blocks"]
    n_groups = int(state["next"])  # actual folded group count (64 or 128)
    out = {"norm_gstar": gn, "per_k": {}}
    for k in KS:
        entry = {}
        for meth in ("naive", "ttpg"):
            s = "k0" if k == 0 else f"k{k}_{meth}"
            gbar = s1[s].to(torch.float64) / n_groups
            bias = (gbar - g).norm().item() / gn
            e = {"bias": bias, "norm_gbar": gbar.norm().item(), "per_m": {}}
            for m in MS:
                r = n_groups // m
                bl = blocks[s][m]
                assert len(bl) == r, f"slot {s} M={m}: {len(bl)} blocks != {r}"
                sum_tilde_sq = sum(sq for sq, _ in bl.values()) / (m * m)
                var = max((sum_tilde_sq - r * gbar.dot(gbar).item()) / (r - 1), 0.0) / gn2
                mse = sum(sq / (m * m) - 2.0 * d / m + gn2 for sq, d in bl.values()) / r / gn2
                cos = sum(d / max(sq, 1e-300) ** 0.5 for sq, d in bl.values()) / r / gn
                e["per_m"][str(m)] = {"var": var, "mse": mse, "cos": cos}
            entry[meth] = e
            del gbar
        out["per_k"][str(k)] = entry
    return out


def run_metrics2(args):
    import torch

    global MS
    MS = tuple(int(x) for x in args.ms.split(","))
    torch.set_num_threads(max(1, min(64, os.cpu_count() or 1)))
    res = {}
    for tag, path in (("mean", args.state), ("sum", args.state_sum)):
        state = torch.load(path, map_location="cpu")
        assert state["next"] == args.expect, f"{tag}: only {state['next']} groups folded"
        res[tag] = _metrics_from_state(state)
        print(f"[metrics2] {tag} metrics computed", flush=True)

    out_json = os.path.join(args.out_dir, "rq1_64_metrics.json")
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[metrics2] saved {out_json}", flush=True)
    _plot2(res, args.out_dir)


def _plot2(res, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = list(KS)
    xs = np.arange(len(ks))
    labels = {"naive": "Naive Aggregation", "ttpg": "TTPG Aggregation"}
    styles = {"naive": dict(marker="o", color="#d62728"), "ttpg": dict(marker="s", color="#1f77b4")}

    def get(tag, k, meth, key, m=None):
        e = res[tag]["per_k"][str(k)][meth]
        return e[key] if m is None else e["per_m"][str(m)][key]

    # ---- main figure (token-mean口径), 1x3: Bias / Var@M=2 / dMSE heatmap ----
    fig = plt.figure(figsize=(14.5, 4.0))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    for meth in ("naive", "ttpg"):
        ax1.plot(xs, [get("mean", k, meth, "bias") for k in ks], label=labels[meth],
                 linewidth=1.8, markersize=5, **styles[meth])
        ax2.plot(xs, [get("mean", k, meth, "var", m=MS[0]) for k in ks], label=labels[meth],
                 linewidth=1.8, markersize=5, **styles[meth])
    ax1.set_title("(a) Relative Gradient Bias $\\downarrow$")
    ax2.set_title("(b) Relative Gradient Variance $\\downarrow$")
    for ax in (ax1, ax2):
        ax.set_xticks(xs)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("$K$ (extra suffixes per tree)")
        ax.grid(alpha=0.3)
    ax1.legend(frameon=False)
    ax2.set_title(f"(b) Relative Gradient Variance $\\downarrow$ ($M={MS[0]}$)")
    # heatmap: dMSE = MSE_naive - MSE_ttpg over K x M
    ks_h = [1, 3, 7, 15]
    mat = np.array([
        [get("mean", k, "naive", "mse", m=m) - get("mean", k, "ttpg", "mse", m=m) for k in ks_h]
        for m in MS
    ])
    im = ax3.imshow(mat, cmap="RdBu_r", aspect="auto",
                    vmin=-np.abs(mat).max(), vmax=np.abs(mat).max())
    ax3.set_xticks(range(len(ks_h)))
    ax3.set_xticklabels([str(k) for k in ks_h])
    ax3.set_yticks(range(len(MS)))
    ax3.set_yticklabels([str(m) for m in MS])
    ax3.set_xlabel("$K$")
    ax3.set_ylabel("$M$")
    ax3.set_title("(c) $\\Delta$MSE = MSE$_{\\rm Naive}$ $-$ MSE$_{\\rm TTPG}$")
    for ii in range(len(MS)):
        for jj in range(len(ks_h)):
            ax3.text(jj, ii, f"{mat[ii, jj]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax3, fraction=0.046)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"rq1_64_main.{ext}"), dpi=200)
    plt.close(fig)

    # ---- appendix: per-M curves for both口径 ----
    for tag, tag_zh in (("mean", "token-mean"), ("sum", "trajectory-sum")):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
        for ax, key, title in (
            (axes[0], "var", "Relative Variance $\\downarrow$"),
            (axes[1], "mse", "Relative MSE $\\downarrow$"),
            (axes[2], "cos", "Cosine to $g^*$ $\\uparrow$"),
        ):
            for meth in ("naive", "ttpg"):
                for m in MS:
                    ax.plot(xs, [get(tag, k, meth, key, m=m) for k in ks],
                            label=f"{labels[meth]}, M={m}", linewidth=1.4, markersize=4,
                            linestyle={2: "-", 4: "--", 8: ":"}[m], **styles[meth])
            ax.set_yscale("log" if key in ("var", "mse") else "linear")
            ax.set_xticks(xs)
            ax.set_xticklabels([str(k) for k in ks])
            ax.set_xlabel("$K$")
            ax.set_title(f"{title} ({tag_zh})")
            ax.grid(alpha=0.3)
        axes[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"rq1_64_appendix_{tag}.{ext}"), dpi=200)
        plt.close(fig)
    print(f"[metrics2] figures saved under {out_dir}", flush=True)

def run_metrics(args):
    import torch

    torch.set_num_threads(max(1, min(64, os.cpu_count() or 1)))
    n_prompts = 902

    # ---- g* = (S_old + S_new) / (256 * 902), fp64 ----
    g_star = None
    for tag in ("old", "new"):
        files = sorted(
            f for f in os.listdir(args.out_dir)
            if f.startswith(f"gstar_sum_{tag}_rank") and f.endswith(".pt")
        )
        assert files, f"no gstar files for tag {tag} under {args.out_dir}"
        for f in files:
            ck = torch.load(os.path.join(args.out_dir, f), map_location="cpu")
            v = ck["acc"].to(torch.float64)
            g_star = v if g_star is None else g_star + v
    g_star /= (2 * 128) * n_prompts
    gstar_norm = g_star.norm().item()
    print(f"[metrics] ||g*|| = {gstar_norm:.6e}", flush=True)

    # ---- per-group vectors: slots -> fp64 running reductions ----
    slots = ["k0"] + [f"k{k}_{m}" for k in (1, 3, 7, 15) for m in ("naive", "ttpg")]
    sum_v = {s: torch.zeros_like(g_star) for s in slots}
    sum_sq = {s: 0.0 for s in slots}    # sum_i ||g_i||^2
    cos_sum = {s: 0.0 for s in slots}   # sum_i cos(g_i, g*)
    n_groups = None
    for g in range(GROUPS):
        files = sorted(
            f for f in os.listdir(args.out_dir)
            if f.startswith(f"treegrad_g{g:02d}_rank") and f.endswith(".pt")
        )
        assert files, f"missing treegrad files for group {g}"
        acc, n_trees = None, 0
        for f in files:
            ck = torch.load(os.path.join(args.out_dir, f), map_location="cpu")
            n_trees += ck["n_trees_shard"]
            if acc is None:
                acc = {s: ck["accs"][s].to(torch.float64) for s in slots}
            else:
                for s in slots:
                    acc[s] += ck["accs"][s].to(torch.float64)
            del ck
        assert n_trees == n_prompts, f"group {g}: {n_trees} trees != {n_prompts}"
        for s in slots:
            v = acc[s] / n_trees  # \hat g_{g,K}^M
            sum_v[s] += v
            vn = v.norm().item()
            sum_sq[s] += vn * vn
            cos_sum[s] += torch.dot(v, g_star).item() / max(vn * gstar_norm, 1e-300)
        del acc
        print(f"[metrics] group {g} merged", flush=True)
    n_groups = GROUPS

    results = {"norm_gstar": gstar_norm, "per_k": {}}
    for k in KS:
        entry = {}
        for m in ("naive", "ttpg"):
            s = "k0" if k == 0 else f"k{k}_{m}"
            g_bar = sum_v[s] / n_groups
            bias = (g_bar - g_star).norm().item() / gstar_norm
            var = (sum_sq[s] - n_groups * g_bar.norm().item() ** 2) / (n_groups - 1)
            var = max(var, 0.0) / gstar_norm ** 2
            cos = cos_sum[s] / n_groups
            entry[m] = {"bias": bias, "var": var, "cos": cos, "norm_gbar": g_bar.norm().item()}
            del g_bar
        results["per_k"][str(k)] = entry
        print(f"[metrics] K={k}: naive {entry['naive']}\n            ttpg {entry['ttpg']}", flush=True)

    out_json = os.path.join(args.out_dir, "rq1_metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[metrics] saved {out_json}", flush=True)
    _plot(results, args.out_dir)


def _plot(results, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = KS
    xs = np.arange(len(ks))
    series = {
        "naive": {m: [results["per_k"][str(k)]["naive"][m] for k in ks] for m in ("bias", "var", "cos")},
        "ttpg": {m: [results["per_k"][str(k)]["ttpg"][m] for k in ks] for m in ("bias", "var", "cos")},
    }
    titles = [
        ("bias", "(a) Relative Gradient Bias $\\downarrow$"),
        ("var", "(b) Relative Gradient Variance $\\downarrow$"),
        ("cos", "(c) Gradient Cosine Similarity $\\uparrow$"),
    ]
    labels = {"naive": "Naive Aggregation", "ttpg": "TTPG Aggregation"}
    styles = {"naive": dict(marker="o", color="#d62728"), "ttpg": dict(marker="s", color="#1f77b4")}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for ax, (m, title) in zip(axes, titles):
        for meth in ("naive", "ttpg"):
            ax.plot(xs, series[meth][m], label=labels[meth], linewidth=1.8, markersize=5,
                    **styles[meth])
        ax.set_xticks(xs)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("$K$ (extra suffixes per tree)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("relative bias")
    axes[1].set_ylabel("relative variance")
    axes[2].set_ylabel("cosine similarity")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"rq1_tree_pg.{ext}")
        fig.savefig(path, dpi=200)
        print(f"[metrics] saved {path}", flush=True)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["gen", "gstar", "treegrad", "metrics", "reduce", "metrics2", "groupmetrics"], required=True)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--out-dir", default=OUT_DEFAULT)
    # gen
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--out", default="", help="gen mode output parquet")
    parser.add_argument("--seed-base", type=int, default=20_260_824)
    parser.add_argument("--n-groups", type=int, default=GROUPS, help="gen: backbone samples per prompt")
    parser.add_argument("--group-offset", type=int, default=0, help="gen: first group id of this run")
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--reward-procs", type=int, default=32)
    parser.add_argument("--max-prompts", type=int, default=-1)
    # gstar
    parser.add_argument("--data", default=os.path.join(LLM_DIR, "results/step400/gen/ppo_0704_n8_iid_n128.parquet"))
    parser.add_argument("--prompts", default=TEST_DATA, help="gen/treegrad: prompt parquet (chat messages + reward_model)")
    parser.add_argument("--rewards", default=os.path.join(LLM_DIR, "results/step400/pg_sample_complexity/rewards_ppo_0704_n8.npz"))
    parser.add_argument("--tag", default="old")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-responses", type=int, default=-1)
    # grad shared
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--log-chunk", type=int, default=256)
    # treegrad
    parser.add_argument("--tree-data", nargs="+", default=[])
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--max-trees", type=int, default=-1)
    parser.add_argument("--accs-on", choices=["gpu", "cpu"], default="gpu",
                        help="treegrad: where the 9 fp32 accumulators (~62 GiB) live")
    parser.add_argument("--grad-ckpt", action="store_true",
                        help="enable gradient checkpointing (cuts activation memory, ~30%% slower)")
    parser.add_argument("--pg-norm", choices=["mean", "sum"], default="mean",
                        help="mean: per-trajectory/per-tree token-mean; sum: standard episodic "
                             "PG (within trajectory/tree sum, across trajectories/trees mean)")
    parser.add_argument("--backbone-only", action="store_true",
                        help="gen: skip suffix sampling/rewards; treegrad: accumulate only the "
                             "k0 (backbone-only) slot. For independent-g* reference passes.")
    # reduce / metrics2
    parser.add_argument("--in-dir", default="", help="reduce: dir with treegrad_g*.pt + gstar files")
    parser.add_argument("--gstar-dir", default="",
                        help="groupmetrics: dir with backbone-only (k0) group files used as an "
                             "INDEPENDENT g* reference; empty = reuse in-dir k0 (shared design)")
    parser.add_argument("--state", default="", help="reduce: reduction state path; metrics2: mean state")
    parser.add_argument("--state-sum", default="", help="metrics2: trajectory-sum state")
    parser.add_argument("--expect", type=int, default=N_GROUPS)
    parser.add_argument("--ms", default="2,4,8", help="reduce/metrics2: block sizes for regrouping")
    parser.add_argument("--reduce-offset", type=int, default=0,
                        help="reduce: group id of the first file (files are g<offset+r>)")
    parser.add_argument("--init-from", default="", help="reduce: migrate an existing torch state file once")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--no-delete", action="store_true")
    args = parser.parse_args()

    if args.mode == "gen":
        run_gen(args)
    elif args.mode == "gstar":
        run_gstar(args)
    elif args.mode == "treegrad":
        run_treegrad(args)
    elif args.mode == "reduce":
        run_reduce(args)
    elif args.mode == "metrics2":
        run_metrics2(args)
    elif args.mode == "groupmetrics":
        run_groupmetrics(args)
    else:
        run_metrics(args)


if __name__ == "__main__":
    main()
