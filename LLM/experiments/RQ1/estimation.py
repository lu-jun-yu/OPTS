# Copyright 2025 Anonymous authors. All rights reserved.

"""RQ1: does the unbiased trajectory return suffice for unbiased policy
gradients on tree trajectories?

Current design (train set, 16,384 prompts): 8 estimator groups of midpoint
branch trees (one backbone + up to 15 extra on-policy suffixes per tree; only
the K=15 tree is sampled, K in {0,1,3,7} reuses the first K extra suffixes),
plus an INDEPENDENT reference-gradient side of 32 backbone-only groups.
Fixed PPO step-400 actor; binary 0/1 rewards; gamma=1, lam=1, no baseline.

Aggregation norms (--pg-norm):
  mean   : per-tree token-mean, then mean over trees (local ratio per tree);
  global : per-group ratio sum(N_i)/sum(D_i) with unnormalized branch-weighted
           numerators and unconditional denominators (zero-reward trees count
           toward D). Estimator-side groups: 8; g* = sum(N*)/sum(D*) over the
           32 independent backbone groups.

Token return assignment (same for both methods): prefix tokens carry the
branch-mean reward Rbar_K = (R_0 + ... + R_K)/(K+1); suffix tokens carry their
own MC return R_j. TTPG weights prefix tokens 1 and each suffix 1/(K+1);
NaivePG weights all sampled tokens 1.

Modes:
  gen          : vLLM sampling of backbones + extra suffixes (+ rule-based
                 rewards; --backbone-only for g*-side sampling). Token ids are
                 stored exactly as sampled (no detokenize/retokenize).
  treegrad     : per-group tree gradient accumulation (9 accumulators on GPU,
                 ~62 GiB; backbone-only mode keeps just k0).
  optsgrad     : OPTS training-estimator gradients on performance-difference search trees (raw
                 TreeGAE advantage x TTPO branch weight, global aggregation),
                 from coefficient files produced by experiments/E1E2/opts_bias_prep.py.
  groupmetrics : Bias/Var/MSE/Cos vs the independent g*, per M in --ms, with
                 the MSE = Bias^2 + (R-1)/R Var identity check on every point
                 (mean and global payload layouts auto-detected via --pg-norm).
"""

import argparse
import json
import os
import re
import sys
import time

LLM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if LLM_DIR not in sys.path:
    sys.path.insert(0, LLM_DIR)

import numpy as np
import pandas as pd

RESPONSE_LENGTH = 2048
GROUPS = 8
N_EXTRA = 15
KS = (0, 1, 3, 7, 15)
# Accumulator slots: K=0 is method-independent (Naive == TTPG there).
ACC_KEYS = ["k0"] + [f"k{k}_{m}" for k in (1, 3, 7, 15) for m in ("naive", "ttpg")]
TEST_DATA = os.path.join(LLM_DIR, "data/train.parquet")
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
    from utils.reward_fn import compute_score_sync

    return key, float(compute_score_sync(data_source, text, ground_truth)["score"])


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
            n=n_groups, temperature=1.0, top_p=1.0, top_k=-1,
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
                    top_p=1.0,
                    top_k=-1,
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
    # global token-level aggregation (pg_norm="global"): numerators accumulate
    # UNNORMALIZED branch-weighted gradient sums; denominators accumulate token
    # weight sums UNCONDITIONALLY (zero-reward trees still count toward D).
    dens = {k: 0.0 for k in acc_keys}
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
        if args.pg_norm == "global":
            dens["k0"] += float(l_pre + l0)  # unconditional: zero-reward trees count too
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
            if j in KS:
                rbar = r_cum / (j + 1)  # prefix-token return: mean over the K+1 branches
                if args.pg_norm == "global":
                    # denominators unconditional: zero-reward trees still count
                    dens[f"k{j}_naive"] += float(l_pre + l_suf_cum)
                    dens[f"k{j}_ttpg"] += float(l_pre + l_suf_cum / (j + 1))
                if dirty:
                    if args.pg_norm == "mean":
                        dN = l_pre + l_suf_cum
                        dT = l_pre + l_suf_cum / (j + 1)
                        _upd(accs[f"k{j}_naive"], rbar / dN, 1.0 / dN)
                        _upd(accs[f"k{j}_ttpg"], rbar / dT, 1.0 / (dT * (j + 1)))
                    else:
                        # sum / global numerator: unnormalized branch-weighted sum
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
        "dens": dens if args.pg_norm == "global" else None,
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
# (raw TreeGAE advantage x TTPO branch weight, no whitening), from coef files
# produced by experiments/E1E2/opts_bias_prep.py
# ---------------------------------------------------------------------------

OPTS_SLOTS = (0, 1, 3, 7, 15)


def _weighted_terms_loss(model, seq_ids, lo, hi, coef, pad_id, device, log_chunk):
    """Forward one sequence; loss = -sum_t coef_t * logp_t over tokens [lo, hi).

    Indices are 0-based within seq_ids; token t is scored by the logit at t-1.
    """
    import torch

    input_ids = torch.tensor(seq_ids, dtype=torch.long, device=device).unsqueeze(0)
    terms = _chunked_token_logp(model, input_ids, pad_id, device, log_chunk)
    wvec = torch.zeros(1, input_ids.shape[1] - 1, dtype=torch.float32, device=device)
    wvec[:, lo - 1 : hi - 1] = torch.tensor(
        np.asarray(coef, dtype=np.float32), device=device
    )
    return _masked_sum_loss(terms, wvec)


def _weighted_terms_losses(model, seq_ids, lo, hi, coefs, pad_id, device, log_chunk):
    """One forward; one loss per coefficient vector (shared graph, backward with retain_graph)."""
    import torch

    input_ids = torch.tensor(seq_ids, dtype=torch.long, device=device).unsqueeze(0)
    terms = _chunked_token_logp(model, input_ids, pad_id, device, log_chunk)
    losses = []
    for coef in coefs:
        wvec = torch.zeros(1, input_ids.shape[1] - 1, dtype=torch.float32, device=device)
        wvec[:, lo - 1 : hi - 1] = torch.tensor(np.asarray(coef, dtype=np.float32), device=device)
        losses.append(_masked_sum_loss(terms, wvec))
    return losses


def run_optsgrad(args):
    """OPTS training-estimator gradients. --tree-data: max-backup coef files (slots
    max_s{r}); --tree-data-mean: mean-backup coef files on the SAME trees (slots
    mean_s{r}), sharing every forward pass with the max-backup slots."""
    import torch

    device = torch.device(args.device)
    trees = []
    for path in sorted(args.tree_data):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        for t in ck["trees"]:
            if t["group"] == args.group:
                t["shard"] = ck["shard"]
                trees.append(t)
    backups = ["max"]
    if args.tree_data_mean:
        backups.append("mean")
        mean_trees = []
        for path in sorted(args.tree_data_mean):
            ck = torch.load(path, map_location="cpu", weights_only=False)
            for t in ck["trees"]:
                if t["group"] == args.group:
                    mean_trees.append(t)
        assert len(mean_trees) == len(trees), (len(mean_trees), len(trees))
        for t, tm in zip(trees, mean_trees):
            assert t["prompt_row"] == tm["prompt_row"] and len(t["rids"]) == len(tm["rids"])
            t["slots_mean"] = tm["slots"]
        del mean_trees
    n_trees_total = len(trees)
    tree_indices = list(range(args.rank, n_trees_total, args.world_size))
    if args.max_trees > 0:
        tree_indices = tree_indices[: args.max_trees]
    print(f"[optsgrad g{args.group} rank {args.rank}] {len(tree_indices)}/{n_trees_total} trees",
          flush=True)

    # gen parquets (for prompts): coef_<name>.pt -> <name>.parquet in same dir
    gen_dfs = {}
    for path in sorted(args.tree_data):
        base = os.path.basename(path)
        sh = int(base.rsplit("shard", 1)[1].split(".")[0].split("_")[0])
        parquet = path.replace("coef_", "").replace(f"{args.coef_suffix}.pt", ".parquet")
        gen_dfs[sh] = pd.read_parquet(parquet)

    model, tokenizer, params = _load_actor(args, device)
    pad_id = tokenizer.pad_token_id
    numel = sum(p.numel() for p in params)

    accs = {f"{b}_s{s}": torch.zeros(numel, dtype=torch.float32, device=device)
            for b in backups for s in OPTS_SLOTS}
    # global-aggregation denominators: per-slot sum over trees of the tree's
    # branch-weight sum D_i (group estimate = acc / den, NOT a per-tree ratio)
    dens = {k: 0.0 for k in accs}
    flat_offsets = []
    off = 0
    for p in params:
        flat_offsets.append(off)
        off += p.numel()
    prompt_cache = {}

    def _get_prompt(shard, row):
        key = (shard, row)
        if key not in prompt_cache:
            prompt_cache[key] = _prompt_ids(gen_dfs[shard].iloc[row], tokenizer)
        return prompt_cache[key]

    def _full_resp(t, j, memo):
        if j in memo:
            return memo[j]
        pid, bp, ids = t["rids"][j]
        if pid < 0:
            fr = [int(x) for x in ids]
        else:
            fr = _full_resp(t, pid, memo)[: bp + 1] + [int(x) for x in ids]
        memo[j] = fr
        return fr

    t0 = time.time()
    for done, ti in enumerate(tree_indices):
        t = trees[ti]
        prompt_ids = _get_prompt(t["shard"], t["prompt_row"])
        memo = {}
        for s in OPTS_SLOTS:
            if s not in t["slots"]:
                continue
            slot_by_backup = {"max": t["slots"][s]}
            if "mean" in backups:
                slot_by_backup["mean"] = t["slots_mean"][s]
            for b, slot in slot_by_backup.items():
                dens[f"{b}_s{s}"] += float(slot["weight_sum"])
            views = {b: [accs[f"{b}_s{s}"][o : o + p.numel()] for o, p in zip(flat_offsets, params)]
                     for b in slot_by_backup}
            coef_by_rid = {b: dict(slot["segs"]) for b, slot in slot_by_backup.items()}
            for rid_idx in coef_by_rid["max"]:
                coefs = [coef_by_rid[b][rid_idx] for b in slot_by_backup]
                if all(len(c) == 0 for c in coefs):
                    continue
                pid, bp, ids = t["rids"][rid_idx]
                ctx = prompt_ids if pid < 0 else prompt_ids + _full_resp(t, pid, memo)[: bp + 1]
                seq = ctx + [int(x) for x in ids]
                losses = _weighted_terms_losses(
                    model, seq, len(ctx), len(seq), coefs, pad_id, device, args.log_chunk
                )
                for k, b in enumerate(slot_by_backup):
                    model.zero_grad(set_to_none=True)
                    losses[k].backward(retain_graph=k < len(losses) - 1)
                    torch._foreach_add_(views[b], [p.grad.view(-1) for p in params])
                del losses
        if (done + 1) % 25 == 0 or done + 1 == len(tree_indices):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            print(f"[optsgrad g{args.group} rank {args.rank}] {done + 1}/{len(tree_indices)} "
                  f"trees, elapsed {time.time() - t0:.1f}s", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"optsgrad_g{args.group:02d}_rank{args.rank}.pt")
    tmp_path = out_path + ".tmp"
    payload = {
        "accs": {k: v.cpu() for k, v in accs.items()},
        "dens": dens,
        "group": args.group,
        "n_trees_total": n_trees_total,
        "n_trees_shard": len(tree_indices),
        "model": args.model,
        "pg_norm": "opts_raw_treegae_global",
        "backups": backups,
    }
    for attempt in range(30):
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, out_path)
            break
        except (OSError, RuntimeError) as exc:
            print(f"[optsgrad g{args.group}] save attempt {attempt + 1} failed: {exc}; "
                  f"retrying in 60s", flush=True)
            time.sleep(60)
    else:
        raise RuntimeError(f"could not save {out_path} after 30 attempts")
    print(f"[optsgrad g{args.group} rank {args.rank}] saved {out_path}", flush=True)


# ---------------------------------------------------------------------------
# groupmetrics mode: direct metrics from a handful of group files (<=16),
# reference gradient taken from the groups' own K=0 (backbone) slot
# ---------------------------------------------------------------------------

def run_groupmetrics_global(args, files):
    """Global token-level aggregation: group estimate = sum(N_i)/sum(D_i).

    Every payload must carry pg_norm="global" and per-slot denominators.
    g* = (sum_g N*_g)/(sum_g D*_g) over the --gstar-dir backbone-only groups.
    Block estimates merge numerators/denominators BEFORE dividing; since ratios
    do not commute with averaging, Bias/Var/MSE/Cos are all computed per M
    (the MSE = Bias^2 + (R-1)/R Var identity holds per M).
    """
    import torch

    torch.set_num_threads(max(1, min(64, os.cpu_count() or 1)))
    import glob as _glob
    ms = tuple(int(x) for x in args.ms.split(","))
    slots = ["k0"] + [f"k{k}_{m}" for k in KS if k > 0 for m in ("naive", "ttpg")]

    def _load_global(dir_path, want_slots, workers=16):
        """Per-group (numerator, denominator) for each slot; parallel file reads
        (NFS single-stream throughput is low; parallel streams scale it)."""
        fs = sorted(_glob.glob(os.path.join(dir_path, "treegrad_g*_rank0.pt")))
        assert fs, f"no group files in {dir_path}"

        def _one(f):
            ck = torch.load(f, map_location="cpu")
            assert ck.get("pg_norm") == "global" and ck.get("dens"), (
                f"{f} is not a global-aggregation payload "
                f"(pg_norm={ck.get('pg_norm')!r}, has_dens={'dens' in ck})"
            )
            out = ({s: ck["accs"][s] for s in want_slots},
                   {s: float(ck["dens"][s]) for s in want_slots},
                   ck["n_trees_total"], os.path.basename(f))
            del ck
            return out

        from concurrent.futures import ThreadPoolExecutor
        nums = {s: [None] * len(fs) for s in want_slots}
        dens = {s: [None] * len(fs) for s in want_slots}
        n_tr = None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx, payload in zip(range(len(fs)), pool.map(_one, fs)):
                n, d, n_tr_, name = payload
                n_tr = n_tr_
                for s in want_slots:
                    nums[s][idx] = n[s]
                    dens[s][idx] = d[s]
                print(f"[groupmetrics/global] loaded {name}", flush=True)
        return nums, dens, len(fs), n_tr

    indep = bool(args.gstar_dir) and os.path.abspath(args.gstar_dir) != os.path.abspath(args.in_dir)
    assert indep, "global口径要求独立 g* (--gstar-dir)"
    sn, sd, n_gs, _ = _load_global(args.gstar_dir, ["k0"])
    N_star = sum(v.to(torch.float64) for v in sn["k0"])
    D_star = sum(sd["k0"])
    g = N_star / D_star
    gn = g.norm().item()
    gn2 = gn * gn
    del sn, sd, N_star
    print(f"[groupmetrics/global] g* from {n_gs} groups, ||g*||={gn:.6e}, "
          f"D*/(n_groups)={D_star / n_gs:.2f}", flush=True)

    n_groups = len(files)
    nums, dens, _, n_tr = _load_global(args.in_dir, slots)

    gout = {"per_k": {}, "pg_norm": "global", "n_groups": n_groups,
            "n_gstar_groups": n_gs, "norm_gstar": gn,
            "n_trees_per_group": n_tr}
    for k in KS:
        for meth in ("naive", "ttpg"):
            s = "k0" if k == 0 else f"k{k}_{meth}"
            per_m = {}
            for m in ms:
                r = n_groups // m
                assert n_groups % m == 0
                blocks = []
                for q in range(r):
                    NB = torch.zeros_like(g)
                    DB = 0.0
                    for gi in range(q * m, (q + 1) * m):
                        NB += nums[s][gi]
                        DB += dens[s][gi]
                    Bv = NB / DB
                    blocks.append([Bv.norm().item() ** 2, Bv.dot(g).item(), Bv])
                gbar = sum(b[2] for b in blocks) / r
                bias = (gbar - g).norm().item() / gn
                sum_sq = sum(b[0] for b in blocks)
                gbar2 = gbar.dot(gbar).item()
                var = max((sum_sq - r * gbar2) / (r - 1), 0.0) / gn2 if r > 1 else float("nan")
                mse = sum(b[0] - 2.0 * b[1] + gn2 for b in blocks) / r / gn2
                cos = sum(b[1] / max(b[0], 1e-300) ** 0.5 for b in blocks) / r / gn
                resid = mse - bias * bias - (r - 1) / r * var
                per_m[str(m)] = {"bias": bias, "var": var, "mse": mse, "cos": cos,
                                 "identity_resid": resid}
                for b in blocks:
                    b[2] = None  # release the fp64 block vector
                del blocks, gbar
            gout["per_k"].setdefault(str(k), {})[meth] = {"per_m": per_m}
        print(f"[groupmetrics/global] K={k} done", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, args.tag or "groupmetrics_global.json")
    with open(out, "w") as f:
        json.dump(gout, f, indent=2)
    print(f"[groupmetrics/global] saved {out}", flush=True)
    for k in KS:
        for meth in ("naive", "ttpg"):
            cells = " | ".join(
                f"M{mm}: bias={v['bias']:.4f} var={v['var']:.4f} mse={v['mse']:.4f} "
                f"cos={v['cos']:.4f} resid={v['identity_resid']:+.4f}"
                for mm, v in gout["per_k"][str(k)][meth]["per_m"].items())
            print(f"K={k:>2} {meth:>5}: {cells}", flush=True)


def run_groupmetrics_opts(args):
    """Bias/Var/MSE/Cos of the OPTS training estimator (A: max backup or A': mean
    backup, one --in-dir each) against the independent backbone-only g*.

    Payloads are optsgrad_g*_rank0.pt with slots s{r} (global aggregation:
    numerator = sum of w*A gradients, denominator = sum of branch weights).
    Same block-regrouping statistics as run_groupmetrics_global.
    """
    import glob as _glob
    import torch

    torch.set_num_threads(max(1, min(64, os.cpu_count() or 1)))
    ms = tuple(int(x) for x in args.ms.split(","))
    gfs = sorted(_glob.glob(os.path.join(args.gstar_dir, "treegrad_g*_rank0.pt")))
    assert gfs, f"no g* group files in {args.gstar_dir}"
    N_star, D_star = None, 0.0
    for f in gfs:
        ck = torch.load(f, map_location="cpu")
        assert ck.get("pg_norm") == "global" and ck.get("dens"), f"{f} is not a global payload"
        v = ck["accs"]["k0"].to(torch.float64)
        N_star = v if N_star is None else N_star.add_(v)
        D_star += float(ck["dens"]["k0"])
        del ck
    g = N_star / D_star
    gn = g.norm().item()
    gn2 = gn * gn
    del N_star
    print(f"[groupmetrics/opts] g* from {len(gfs)} groups, ||g*||={gn:.6e}", flush=True)

    files = sorted(_glob.glob(os.path.join(args.in_dir, "optsgrad_g*_rank0.pt")))
    assert files, f"no optsgrad group files in {args.in_dir}"
    n_groups = len(files)
    slots = [f"{args.backup}_s{r}" for r in OPTS_SLOTS]
    nums = {sl: [] for sl in slots}
    dens = {sl: [] for sl in slots}
    n_tr = None
    for f in files:
        ck = torch.load(f, map_location="cpu")
        n_tr = ck["n_trees_total"]
        for sl in slots:
            nums[sl].append(ck["accs"][sl])
            dens[sl].append(float(ck["dens"][sl]))
        del ck
        print(f"[groupmetrics/opts] loaded {os.path.basename(f)}", flush=True)

    gout = {"per_s": {}, "pg_norm": "opts_raw_treegae_global", "backup": args.backup,
            "n_groups": n_groups, "n_gstar_groups": len(gfs), "norm_gstar": gn,
            "n_trees_per_group": n_tr, "in_dir": args.in_dir}
    for sl in slots:
        per_m = {}
        for m in ms:
            r = n_groups // m
            assert n_groups % m == 0
            blocks = []
            for q in range(r):
                NB = torch.zeros_like(g)
                DB = 0.0
                for gi in range(q * m, (q + 1) * m):
                    NB += nums[sl][gi].to(torch.float64)
                    DB += dens[sl][gi]
                Bv = NB / DB
                blocks.append([Bv.norm().item() ** 2, Bv.dot(g).item(), Bv])
            gbar = sum(b[2] for b in blocks) / r
            bias = (gbar - g).norm().item() / gn
            sum_sq = sum(b[0] for b in blocks)
            gbar2 = gbar.dot(gbar).item()
            var = max((sum_sq - r * gbar2) / (r - 1), 0.0) / gn2 if r > 1 else float("nan")
            mse = sum(b[0] - 2.0 * b[1] + gn2 for b in blocks) / r / gn2
            cos = sum(b[1] / max(b[0], 1e-300) ** 0.5 for b in blocks) / r / gn
            per_m[str(m)] = {"bias": bias, "var": var, "mse": mse, "cos": cos,
                             "identity_resid": mse - bias * bias - (r - 1) / r * var,
                             "norm_gbar": gbar.norm().item() / gn}
            for b in blocks:
                b[2] = None
            del blocks, gbar
        gout["per_s"][sl.split("_s")[1]] = {"per_m": per_m}
        print(f"[groupmetrics/opts] {sl}: " + " | ".join(
            f"M{mm}: bias={v['bias']:.4f} var={v['var']:.4f} mse={v['mse']:.4f} cos={v['cos']:.4f}"
            for mm, v in per_m.items()), flush=True)
    # Full-group estimate per slot, saved for A - A' distances across in-dirs.
    full = {sl: (sum(v.to(torch.float64) for v in nums[sl]) / sum(dens[sl])) for sl in slots}
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, args.tag or "groupmetrics_opts.json")
    with open(out, "w") as f:
        json.dump(gout, f, indent=2)
    torch.save({sl: v.to(torch.float32) for sl, v in full.items()},
               out.replace(".json", "_full_estimates.pt"))
    print(f"[groupmetrics/opts] saved {out}", flush=True)


def run_groupmetrics(args):
    """Direct Bias/Var/MSE/Cos for a small number of treegrad group files.

    g*: with --gstar-dir, the mean k0 (backbone-only) gradient of INDEPENDENT
    backbone-only group files; otherwise the mean k0 of the estimator groups
    themselves (shared design). Blocks for M in --ms are consecutive groups.
    Includes the MSE = Bias^2 + (R-1)/R Var identity check on every point.
    """
    import torch

    if args.payload == "opts":
        run_groupmetrics_opts(args)
        return
    if args.pg_norm == "global":
        import glob as _glob2
        est_files = sorted(_glob2.glob(os.path.join(args.in_dir, "treegrad_g*_rank0.pt")))
        assert est_files, f"no group files in {args.in_dir}"
        run_groupmetrics_global(args, est_files)
        return

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
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gen", "treegrad", "optsgrad", "groupmetrics"], required=True)
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
    parser.add_argument("--prompts", default=TEST_DATA, help="gen/treegrad: prompt parquet (chat messages + reward_model)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
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
    parser.add_argument("--pg-norm", choices=["mean", "global"], default="global",
                        help="mean: per-tree token-mean (local ratio), then mean over trees; "
                             "global: unnormalized numerator + unconditional denominator, "
                             "group estimate = sum(N_i)/sum(D_i)")
    parser.add_argument("--backbone-only", action="store_true",
                        help="gen: skip suffix sampling/rewards; treegrad: accumulate only the "
                             "k0 (backbone-only) slot. For independent-g* reference passes.")
    # groupmetrics
    parser.add_argument("--in-dir", default="", help="groupmetrics: dir with treegrad_g*.pt group files")
    parser.add_argument("--gstar-dir", default="",
                        help="groupmetrics: dir with backbone-only (k0) group files used as an "
                             "INDEPENDENT g* reference; empty = reuse in-dir k0 (shared design)")
    parser.add_argument("--ms", default="1,2,4", help="groupmetrics: block sizes for regrouping")
    parser.add_argument("--ks", default="1,3,7,15",
                        help="treegrad/groupmetrics: suffix-count checkpoints K (<= 15); "
                             "the bias analysis matches them to OPTS mean suffix counts")
    parser.add_argument("--coef-suffix", default="",
                        help="optsgrad: suffix of coef files (e.g. _mean) to strip when locating gen parquets")
    parser.add_argument("--payload", choices=["tree", "opts"], default="tree",
                        help="groupmetrics: 'opts' reads optsgrad_g*.pt slots {backup}_s{r} (A / A')")
    parser.add_argument("--backup", choices=["max", "mean"], default="max",
                        help="groupmetrics --payload opts: which TreeGAE backup's slots to score")
    parser.add_argument("--tree-data-mean", nargs="+", default=[],
                        help="optsgrad: mean-backup coef files on the same trees (adds mean_s{r} slots)")
    args = parser.parse_args()

    global KS, ACC_KEYS
    ks = tuple(sorted({int(k) for k in args.ks.split(",") if int(k) > 0}))
    assert ks and ks[-1] <= N_EXTRA, ks
    KS = (0,) + ks
    ACC_KEYS = ["k0"] + [f"k{k}_{m}" for k in ks for m in ("naive", "ttpg")]

    if args.mode == "gen":
        run_gen(args)
    elif args.mode == "treegrad":
        run_treegrad(args)
    elif args.mode == "optsgrad":
        run_optsgrad(args)
    elif args.mode == "groupmetrics":
        run_groupmetrics(args)


if __name__ == "__main__":
    main()
