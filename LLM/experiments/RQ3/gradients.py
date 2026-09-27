"""RQ3 gradients: Fixed + TTPG/NaivePG and OPTS + max-backup TTPG, p=1."""
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

LLM_DIR = str(Path(__file__).resolve().parents[2])
if LLM_DIR not in sys.path:
    sys.path.insert(0, LLM_DIR)
BRANCH_POS = 128
N_EXTRA = 7
S_SLOTS = (1, 3, 7)
OPTS_SLOTS = (0, 1, 3, 7)
EST_SLOTS = ['s0'] + [f's{s}_{m}' for s in S_SLOTS for m in ('naive', 'ttpg')]

def _prompt_ids(row, tokenizer):
    messages = [dict(m) for m in row["prompt"]]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)


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

    attention_mask = torch.ones_like(input_ids)  # unpadded sequences; EOS may equal PAD
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


def _reward_text_task(task):
    key, data_source, text, ground_truth = task
    from utils.reward_fn import compute_score_sync

    return key, float(compute_score_sync(data_source, text, ground_truth)["score"])


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


def response_context(rids, j, prompt_ids, memo):
    """Build context using parent-local branch positions, including nested suffixes."""
    if j not in memo:
        parent, bp, ids = rids[j]
        if parent < 0:
            ctx = list(prompt_ids)
        else:
            parent_ctx = response_context(rids, parent, prompt_ids, memo)
            parent_ids = [int(x) for x in rids[parent][2]]
            if not 0 <= bp < len(parent_ids):
                raise ValueError("Invalid parent-local branch position")
            ctx = parent_ctx + parent_ids[:bp+1]
        memo[j] = ctx
    return memo[j]


def run_treegrad(args):
    """Accumulate raw per-tree gradient sums at the matched p=1 budgets."""
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

    # selection mask for this group (indexed by prompt_idx)
    if not args.backbone_only:
        mk = np.load(args.mask_file)
        if int(mk["branch_pos"]) != BRANCH_POS or str(mk["allocation"]) != "prompt_round_v2":
            raise ValueError("Expected fixed-128 prompt/round allocation mask")
        lengths = tdf.backbone_ids.str.len().to_numpy()
        if not np.array_equal(tdf.prefix_len.to_numpy(), np.minimum(lengths, BRANCH_POS)):
            raise ValueError("Tree data does not match fixed-128 mask")
        assert list(mk["snap_rounds"]) == list(S_SLOTS), (
            f"mask snapshots {list(mk['snap_rounds'])} != {list(S_SLOTS)}")
        k_arr = mk["k"][args.group]            # (n_prompts, len(S_SLOTS)) int8
        print(f"[treegrad g{args.group}] prompt/round matched mask {args.mask_file}", flush=True)

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

    acc_keys = ("s0",) if args.backbone_only else tuple(EST_SLOTS)
    acc_device = device if args.accs_on == "gpu" else torch.device("cpu")
    accs = {k: torch.zeros(numel, dtype=torch.float32, device=acc_device) for k in acc_keys}
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
        for dst, param in zip(dst_views, params):
            if param.grad is not None:
                dst.add_(param.grad.reshape(-1), alpha=scale)

    def _upd(acc, alpha_c, alpha_s):
        acc.add_(C.to(acc.device), alpha=alpha_c)
        acc.add_(S.to(acc.device), alpha=alpha_s)

    diag = {"r0_pos": 0, "rs_pos": 0, "rs_tot": 0, "lens": [], "prefix_lens": []}
    smax = max(S_SLOTS)

    t0 = time.time()
    for done, ti in enumerate(tree_indices):
        row = tdf.iloc[ti]
        pi = int(row["prompt_idx"])
        prompt_ids = prompt_ids_map[pi]
        bb_ids = [int(t) for t in row["backbone_ids"]]
        plen = int(row["prefix_len"])
        r0 = float(row["backbone_reward"])
        rs = [float(v) for v in row["suffix_rewards"]]
        l_pre, l0 = plen, len(bb_ids) - plen

        if args.backbone_only:
            kmax = 0
        else:
            k1, k3, k7 = (int(x) for x in k_arr[pi])
            assert 0 <= k1 <= k3 <= k7 <= N_EXTRA
            kmax = k7
            cps = [(k1, "s1"), (k3, "s3"), (k7, "s7")]
            assert all(0 <= k <= kmax for k, _ in cps)
            cps.sort(key=lambda item: item[0])
        diag["r0_pos"] += int(r0 > 0)
        diag["rs_pos"] += sum(int(rs[j] > 0) for j in range(kmax))
        diag["rs_tot"] += kmax
        diag["lens"].append(len(bb_ids))
        diag["prefix_lens"].append(plen)

        C.zero_()  # C = G_pre (unscaled prefix-token gradient sum)
        S.zero_()  # S = running sum of R_j * G_j over processed continuations
        need_pre = (r0 > 0) or any(rs[j] > 0 for j in range(kmax))

        # backbone: one forward, separate prefix / suffix0 backwards
        if need_pre:
            seq = prompt_ids + bb_ids
            loss_pre, loss_s0 = _response_terms_loss(
                model, seq, [(n_prompt := len(prompt_ids), n_prompt + l_pre),
                             (n_prompt + l_pre, len(seq))],
                pad_id, device, args.log_chunk,
            )
            model.zero_grad(set_to_none=True)
            if loss_pre is not None:
                loss_pre.backward(retain_graph=True)
                _fold_grad(Cv, 1.0)
            if r0 > 0 and loss_s0 is not None:
                model.zero_grad(set_to_none=True)
                loss_s0.backward()
                _fold_grad(Sv, r0)
            del loss_pre, loss_s0

        # ---- s0 slot: chain gradient (k=0 state), shared by both methods ----
        dens["s0"] += float(l_pre + l0)
        if need_pre:
            _upd(accs["s0"], r0, 1.0)

        if args.backbone_only:
            if (done + 1) % 25 == 0 or done + 1 == len(tree_indices):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                print(f"[treegrad g{args.group} rank {args.rank}] {done + 1}/{len(tree_indices)} trees, "
                      f"elapsed {time.time() - t0:.1f}s", flush=True)
            continue

        def _fold_ckpt(slot, k, r_cum_, l_cum_):
            """Fold the k-suffix state (C, S, rewards, lengths) into slot s_cp."""
            dens[f"{slot}_naive"] += float(l_pre + l_cum_)
            dens[f"{slot}_ttpg"] += float(l_pre + l_cum_ / (k + 1))
            if need_pre:
                rbar = r_cum_ / (k + 1)  # prefix-token return: branch-mean reward
                _upd(accs[f"{slot}_naive"], rbar, 1.0)
                _upd(accs[f"{slot}_ttpg"], rbar, 1.0 / (k + 1))

        # ---- snapshot slots: single ascending pass over this tree's suffixes ----
        l_suf_cum, r_cum = float(l0), r0  # token/reward sums over continuations 0..j
        ci = 0
        while ci < len(cps) and cps[ci][0] == 0:
            _fold_ckpt(cps[ci][1], 0, r_cum, l_suf_cum)
            ci += 1
        for j in range(1, kmax + 1):
            ids = [int(t) for t in row["suffix_ids"][j - 1]]
            rj = rs[j - 1]
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
            while ci < len(cps) and cps[ci][0] == j:
                _fold_ckpt(cps[ci][1], j, r_cum, l_suf_cum)
                ci += 1
        assert ci == len(cps)

        if (done + 1) % 25 == 0 or done + 1 == len(tree_indices):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            print(f"[treegrad g{args.group} rank {args.rank}] {done + 1}/{len(tree_indices)} trees, "
                  f"elapsed {time.time() - t0:.1f}s", flush=True)

    if args.no_save:
        print(
            f"[treegrad g{args.group} rank {args.rank}] no-save smoke complete; "
            f"trees={len(tree_indices)}, dens={dens}",
            flush=True,
        )
        return

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"treegrad_g{args.group:02d}_rank{args.rank}.pt")
    # atomic: tmp + rename so a concurrent reducer never reads a partial file;
    # retry on transient ENOSPC
    tmp_path = out_path + ".tmp"
    payload = {
        "accs": {k: v.cpu() for k, v in accs.items()},
        "dens": dens,
        "group": args.group,
        "n_trees_total": n_trees_total,
        "n_trees_shard": len(tree_indices),
        "model": args.model,
        "pg_norm": "masked_global",
        "schema_version": "exp1_v2",
        "parameter_names": [name for name, p in model.named_parameters() if p.requires_grad],
        "slots_meta": {"kind": "backbone_only" if args.backbone_only else "est",
                       "s_slots": S_SLOTS},
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


def run_optsgrad(args):
    """Share each response forward across the four max-backup snapshots."""
    import torch
    from experiments.RQ3.coefficients import RETURN_SCHEMA
    def check_return_schema(ck):
        if ck.get("credit_mode") == "return" and ck.get("return_schema") != RETURN_SCHEMA:
            raise ValueError("Return coefficients must use the full-tree zero-value schema")
    trees = []
    credit_modes = set()
    for path in args.tree_data:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        check_return_schema(ck)
        if ck.get("schema_version") != "exp1_v2":
            raise ValueError("Coefficient files must use schema exp1_v2")
        credit_modes.add(ck.get("credit_mode", "advantage"))
        for t in ck["trees"]:
            if t["group"] != args.group:
                continue
            t["shard"] = ck["shard"]
            t["prompt_idx"] = ck["shard"] * ck["shard_size"] + t["prompt_row"]
            t["gen_path"] = ck["gen_path"]
            trees.append(t)
    if not trees:
        raise ValueError("Incomplete coefficient inputs")
    if len(credit_modes) != 1:
        raise ValueError(f"Mixed coefficient credit modes: {sorted(credit_modes)}")
    credit_mode = credit_modes.pop()
    if credit_mode != "return":
        raise ValueError("RQ3 requires direct V=0 return coefficients")
    trees.sort(key=lambda t: t["prompt_idx"])
    if len({t["prompt_idx"] for t in trees}) != len(trees):
        raise ValueError("Duplicate prompt in OPTS group")
    mk = np.load(args.mask_file)
    if str(mk["allocation"]) != "prompt_round_v2":
        raise ValueError("Expected v2 prompt mask")
    excluded = set(mk["excluded_prompt_indices"].tolist()) if "excluded_prompt_indices" in mk.files else set()
    if excluded:
        raise ValueError("The allocation mask must include all prompts")
    if not trees:
        raise ValueError("No eligible C trees")
    tree_indices = list(range(args.rank, len(trees), args.world_size))
    if args.max_trees > 0:
        tree_indices = tree_indices[:args.max_trees]
    device = torch.device(args.device)
    model, tokenizer, params = _load_actor(args, device)
    pad_id = tokenizer.pad_token_id
    numel = sum(p.numel() for p in params)
    keys = [f"max_s{s}" for s in OPTS_SLOTS]
    acc_device = device if args.accs_on == "gpu" else torch.device("cpu")
    accs = {key: torch.zeros(numel, dtype=torch.float32, device=acc_device)
            for key in keys}
    dens = {key: 0.0 for key in keys}
    offsets = np.cumsum([0] + [p.numel() for p in params]).tolist()
    gen_dfs = {path: pd.read_parquet(path, columns=["prompt"])
               for path in {t["gen_path"] for t in trees}}
    prompt_cache = {}
    start = time.time()
    for done, ti in enumerate(tree_indices):
        t = trees[ti]
        pi = t["prompt_idx"]
        if pi not in prompt_cache:
            prompt_cache[pi] = _prompt_ids(gen_dfs[t["gen_path"]].iloc[t["prompt_row"]], tokenizer)
        prompt = prompt_cache[pi]
        slots = {f"max_s{s}": t["slots"][s] for s in OPTS_SLOTS}
        for key, slot in slots.items():
            dens[key] += float(slot["weight_sum"])
        # Map each distinct coefficient to all accumulator destinations.
        by_rid = {}
        for key, slot in slots.items():
            for rid_idx, coef in slot["segs"]:
                coef = np.asarray(coef, dtype=np.float32)
                if not np.isfinite(coef).all():
                    raise ValueError("Nonfinite gradient coefficient")
                if not np.any(coef):
                    continue
                unique = by_rid.setdefault(rid_idx, {})
                packed = coef.tobytes()
                if packed not in unique:
                    unique[packed] = [coef, []]
                unique[packed][1].append(key)
        memo = {}
        for rid_idx, unique in by_rid.items():
            ctx = response_context(t["rids"], rid_idx, prompt, memo)
            ids = [int(x) for x in t["rids"][rid_idx][2]]
            work = list(unique.values())
            losses = _weighted_terms_losses(model, ctx+ids, len(ctx), len(ctx)+len(ids),
                                            [x[0] for x in work], pad_id, device, args.log_chunk)
            for j, (_, destinations) in enumerate(work):
                model.zero_grad(set_to_none=True)
                losses[j].backward(retain_graph=j+1 < len(work))
                for key in destinations:
                    for q, param in enumerate(params):
                        if param.grad is not None:
                            accs[key][offsets[q]:offsets[q+1]].add_(
                                param.grad.detach().reshape(-1).to(acc_device))
            del losses
        if (done+1) % 25 == 0 or done+1 == len(tree_indices):
            if device.type == "cuda":
                torch.cuda.synchronize()
            print(f"[optsgrad g{args.group}] {done+1}/{len(tree_indices)} trees, "
                  f"elapsed {time.time()-start:.1f}s", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"optsgrad_g{args.group:02d}_rank{args.rank}.pt")
    payload = dict(accs={k: v.cpu() for k, v in accs.items()}, dens=dens,
                   group=args.group, n_trees_total=len(trees), n_trees_shard=len(tree_indices),
                   model=args.model, pg_norm="opts_raw_treegae_global", schema_version="exp1_v2",
                   credit_mode=credit_mode,
                   return_schema=RETURN_SCHEMA,
                   parameter_names=[name for name, p in model.named_parameters() if p.requires_grad])
    torch.save(payload, out+".tmp")
    os.replace(out+".tmp", out)
    print("saved", out, flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['treegrad','optsgrad'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--prompts', default='data/train.parquet')
    p.add_argument('--tree-data', nargs='+', required=True)
    p.add_argument('--mask-file', default='')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--group', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=['float32','bfloat16'], default='float32')
    p.add_argument('--accs-on', choices=['gpu','cpu'], default='gpu')
    p.add_argument('--log-chunk', type=int, default=256)
    p.add_argument('--grad-ckpt', action='store_true')
    p.add_argument('--backbone-only', action='store_true')
    args = p.parse_args()
    args.rank, args.world_size, args.max_trees, args.no_save = 0, 1, -1, False
    prefix = 'treegrad' if args.mode == 'treegrad' else 'optsgrad'
    target = Path(args.out_dir)/f'{prefix}_g{args.group:02d}_rank0.pt'
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite gradients: {target}')
    if args.mode == 'treegrad':
        run_treegrad(args)
    else:
        run_optsgrad(args)

if __name__ == '__main__':
    main()
