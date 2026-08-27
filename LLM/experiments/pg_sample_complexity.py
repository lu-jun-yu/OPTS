# Copyright 2025 Junyu Lu (Julian Lou). All rights reserved.

"""Raw unbiased policy-gradient (REINFORCE, gamma=1, lam=1, no value baseline)
sample-complexity analysis on stored i.i.d. rollouts.

Given a parquet with N prompts x 128 responses (results/step400/gen/*.parquet)
and the actor that generated them, compute

    g*_n = (1/N) * sum_p (1/n) * sum_{i<=n} R_{p,i} * grad log pi_theta(y_{p,i} | x_p)

for n in {8, 16, 32, 64, 128}, then report, for each n < 128,

    cos(g*_n, g*_128)   and   ||g*_n - g*_128|| / ||g*_128||.

With gamma=1, lam=1 and no value baseline, the GAE advantage of every response
token equals the terminal trajectory return, so the policy gradient reduces to
the reward-weighted score function above. Rewards are binary (rule-based
math_verify scoring in utils.reward_fn), so responses with R=0 contribute
exactly zero gradient and are skipped.

Modes:
  rewards   : precompute per-response rewards to an .npz cache (CPU parallel).
  grad      : one process per GPU; accumulates gradient buffers for its prompt
              shard over 128 passes (pass i covers the i-th response of every
              prompt) and saves raw buffers to disk. With --pass-grads-mmap it
              also persists every per-pass gradient G_i (fp32 memmap on disk),
              so any rollout-subset gradient (split halves, 128 vs 256, ...) is
              recomputable without another GPU pass. Also writes per-sample
              logprob/reward/length stats. Chunked fp32 log-softmax bounds
              LM-head memory.
  combine   : sum the per-rank buffers, form g*_n, and print/save the metrics.
  splithalf : random disjoint 64/64 split-half metrics from a saved
              --pass-grads-mmap file: cos(g*_64,A, g*_64,B) and
              ||g_A - g_B|| / ||(g_A + g_B)/2||, mean +/- std over K splits.
  gcmp      : from an old and a new pass-grads mmap (128 fresh i.i.d. rollouts
              per prompt each), report cos(g*_128, g*_256) and
              ||g*_128 - g*_256|| / ||g*_256||.
"""

import argparse
import json
import os
import sys
import time

LLM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if LLM_DIR not in sys.path:
    sys.path.insert(0, LLM_DIR)

import numpy as np
import pandas as pd

NS = (8, 16, 32, 64, 128)
RESPONSE_LENGTH = 2048  # generation config: actor_rollout_ref.rollout.response_length


# ---------------------------------------------------------------------------
# rewards mode
# ---------------------------------------------------------------------------

def _reward_task(args):
    idx, data_source, responses, ground_truth = args
    from utils.reward_fn import compute_score

    scores = np.empty(len(responses), dtype=np.float32)
    for j, resp in enumerate(responses):
        scores[j] = float(compute_score(data_source, resp, ground_truth)["score"])
    return idx, scores


def run_rewards(args):
    from multiprocessing import Pool

    df = pd.read_parquet(args.data)
    n_prompts, n_resp = len(df), len(df.iloc[0]["responses"])
    tasks = []
    for idx in range(n_prompts):
        row = df.iloc[idx]
        tasks.append((idx, row["data_source"], list(row["responses"]), row["reward_model"]["ground_truth"]))

    rewards = np.zeros((n_prompts, n_resp), dtype=np.float32)
    t0 = time.time()
    with Pool(processes=args.reward_procs) as pool:
        for done, (idx, scores) in enumerate(pool.imap_unordered(_reward_task, tasks, chunksize=4)):
            rewards[idx] = scores
            if (done + 1) % 50 == 0 or done + 1 == n_prompts:
                print(f"[rewards] {done + 1}/{n_prompts} prompts, elapsed {time.time() - t0:.1f}s", flush=True)

    os.makedirs(os.path.dirname(args.rewards) or ".", exist_ok=True)
    np.savez(args.rewards, rewards=rewards)
    print(f"[rewards] saved to {args.rewards}; mean={rewards.mean():.4f}, "
          f"nonzero={int((rewards > 0).sum())}/{rewards.size}")


# ---------------------------------------------------------------------------
# grad mode
# ---------------------------------------------------------------------------

def _build_samples(df, rewards, prompt_indices, tokenizer):
    """Tokenize prompts/responses for the shard; keep only R>0 responses.

    Returns per-response-index lists: samples[i] = [(seq_ids, resp_len,
    prompt_idx, resp_idx), ...] where seq_ids = prompt_ids + response_ids
    (EOS appended unless the response already fills the 2048-token
    generation budget).
    """
    samples = [[] for _ in range(len(df.iloc[0]["responses"]))]
    stats = {"total": 0, "kept": 0, "eos": 0}
    for idx in prompt_indices:
        row = df.iloc[idx]
        messages = [dict(m) for m in row["prompt"]]
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        for j, resp in enumerate(row["responses"]):
            stats["total"] += 1
            if rewards[idx, j] <= 0:
                continue
            resp_ids = tokenizer(resp, add_special_tokens=False)["input_ids"]
            if len(resp_ids) >= RESPONSE_LENGTH:
                resp_ids = resp_ids[:RESPONSE_LENGTH]  # hit generation cap: no EOS sampled
            else:
                resp_ids = resp_ids + [tokenizer.eos_token_id]
                stats["eos"] += 1
            samples[j].append((prompt_ids + resp_ids, len(resp_ids), idx, j))
            stats["kept"] += 1
    return samples, stats


def _make_micro_batches(pass_samples, token_budget):
    """Group (seq_ids, resp_len) into micro-batches with padded-token budget."""
    order = sorted(pass_samples, key=lambda item: -len(item[0]))
    batches, cur, cur_max = [], [], 0
    for item in order:
        item_max = max(cur_max, len(item[0]))
        if cur and (len(cur) + 1) * item_max > token_budget:
            batches.append(cur)
            cur, cur_max = [], 0
            item_max = len(item[0])
        cur.append(item)
        cur_max = item_max
    if cur:
        batches.append(cur)
    return batches


def run_grad(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cuda" and dtype == torch.float32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    df = pd.read_parquet(args.data)
    rewards = np.load(args.rewards)["rewards"]
    assert rewards.shape[0] == len(df)

    n_prompts = len(df)
    prompt_indices = list(range(args.rank, n_prompts, args.world_size))
    if args.max_prompts > 0:
        prompt_indices = prompt_indices[: args.max_prompts]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    print(f"[rank {args.rank}] tokenizing shard of {len(prompt_indices)} prompts...", flush=True)
    t0 = time.time()
    samples, stats = _build_samples(df, rewards, prompt_indices, tokenizer)
    if args.max_responses > 0:
        samples = samples[: args.max_responses]
    print(f"[rank {args.rank}] tokenize done in {time.time() - t0:.1f}s: {stats}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.gradient_checkpointing_disable()
    model.config.use_cache = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    numel = sum(p.numel() for p in params)
    # fp32 accumulation buffers. On GPU they cost 5 x numel x 4B but the
    # per-pass updates are free; on CPU they save GPU memory at the price of
    # a slow per-pass D2H copy + CPU adds.
    buf_device = device if args.buffers_on == "gpu" else torch.device("cpu")
    bufs = {n: torch.zeros(numel, dtype=torch.float32, device=buf_device) for n in NS}
    print(f"[rank {args.rank}] model loaded, {numel / 1e9:.2f}B params, "
          f"fp32 buffers on {buf_device}", flush=True)

    n_passes = len(samples)
    # Optional: persist every pass gradient G_i (sum over the shard's prompts
    # of R * grad log pi for the i-th response) to an on-disk fp32 memmap
    # ([n_passes, numel] + a .json sidecar), so arbitrary rollout-subset
    # gradients are recomputable without another GPU pass.
    pass_mmap = None
    if args.pass_grads_mmap:
        os.makedirs(os.path.dirname(args.pass_grads_mmap) or ".", exist_ok=True)
        pass_mmap = np.memmap(args.pass_grads_mmap, dtype=np.float32, mode="w+",
                              shape=(n_passes, numel))
        print(f"[rank {args.rank}] pass-grad memmap: {args.pass_grads_mmap} "
              f"({n_passes} x {numel} fp32 = {n_passes * numel * 4 / 2**30:.0f} GiB)", flush=True)

    # per-sample stats: log pi(y|x) sums and response token counts
    sample_prompt_idx, sample_resp_idx = [], []
    sample_logprob, sample_ntok = [], []

    for i in range(n_passes):
        pass_samples = samples[i]
        n_target = i + 1  # response index i contributes to g*_n for all n >= i + 1
        if not pass_samples:
            continue
        micro_batches = _make_micro_batches(pass_samples, args.token_budget)
        model.zero_grad(set_to_none=True)
        n_tokens = 0
        for mb in micro_batches:
            maxlen = max(len(item[0]) for item in mb)
            input_ids = torch.full((len(mb), maxlen), pad_id, dtype=torch.long)
            resp_mask = torch.zeros((len(mb), maxlen), dtype=torch.float32)
            for b, (seq, resp_len, _, _) in enumerate(mb):
                input_ids[b, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                resp_mask[b, len(seq) - resp_len : len(seq)] = 1.0
            input_ids = input_ids.to(device, non_blocking=True)
            resp_mask = resp_mask.to(device, non_blocking=True)
            attention_mask = (input_ids != pad_id).long()
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            # log pi(y_t | x, y_<t) over response tokens; loss = -sum log pi.
            # Chunk the vocab log-softmax (fp32) so the LM head never
            # materializes a full fp32 [tokens, vocab] tensor.
            shift_labels = input_ids[:, 1:]
            shift_mask = resp_mask[:, 1:]
            mb_logprob = torch.zeros(len(mb), device=device)
            loss = None
            for s in range(0, logits.shape[1] - 1, args.log_chunk):
                e = min(s + args.log_chunk, logits.shape[1] - 1)
                logp_c = torch.log_softmax(logits[:, s:e].float(), dim=-1)
                tok_logp_c = torch.gather(logp_c, 2, shift_labels[:, s:e].unsqueeze(-1)).squeeze(-1)
                tok_logp_c = tok_logp_c * shift_mask[:, s:e]
                mb_logprob += tok_logp_c.sum(1).detach()
                loss_c = -tok_logp_c.sum()
                loss = loss_c if loss is None else loss + loss_c
            loss.backward()
            n_tokens += int(shift_mask.sum().item())
            for b, (_, _, p_idx, r_idx) in enumerate(mb):
                sample_prompt_idx.append(p_idx)
                sample_resp_idx.append(r_idx)
                sample_logprob.append(float(mb_logprob[b]))
                sample_ntok.append(int(shift_mask[b].sum()))
            del logits, logp_c, tok_logp_c, loss
        grad_flat = torch.cat([p.grad.detach().view(-1) for p in params]).to(buf_device, torch.float32)
        for n in NS:
            if n_target <= n:
                bufs[n] += grad_flat / n
        if pass_mmap is not None:
            pass_mmap[i] = grad_flat.cpu().numpy()
            if n_target % 16 == 0 or n_target == n_passes:
                pass_mmap.flush()
        if args.rank == 0 and (n_target % 8 == 0 or n_target == n_passes):
            elapsed = time.time() - t0
            print(f"[rank 0] pass {n_target}/{n_passes} done "
                  f"({len(pass_samples)} seqs, {n_tokens} resp tokens, "
                  f"{len(micro_batches)} micro-batches), elapsed {elapsed:.1f}s", flush=True)

    model.zero_grad(set_to_none=True)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"grad_bufs_rank{args.rank}.pt")
    torch.save(
        {
            "bufs": {n: bufs[n].cpu() for n in NS},
            "ns": NS,
            "n_prompts_total": n_prompts,
            "shard_prompts": len(prompt_indices),
            "stats": stats,
        },
        out_path,
    )
    print(f"[rank {args.rank}] saved {out_path}", flush=True)

    if pass_mmap is not None:
        pass_mmap.flush()
        sidecar = {
            "n_passes": n_passes,
            "numel": numel,
            "dtype": "float32",
            "layout": "row i = G_i = sum over shard prompts of R * grad log pi(response i)",
            "n_prompts_total": n_prompts,
            "model": args.model,
            "data": args.data,
            "stats": stats,
        }
        with open(args.pass_grads_mmap + ".json", "w") as f:
            json.dump(sidecar, f, indent=2)
        print(f"[rank {args.rank}] pass-grad memmap flushed: {args.pass_grads_mmap}", flush=True)

    stats_path = os.path.join(args.out_dir, f"sample_stats_rank{args.rank}.npz")
    np.savez(
        stats_path,
        prompt_idx=np.array(sample_prompt_idx, dtype=np.int32),
        resp_idx=np.array(sample_resp_idx, dtype=np.int32),
        logprob=np.array(sample_logprob, dtype=np.float64),
        n_resp_tokens=np.array(sample_ntok, dtype=np.int32),
    )
    print(f"[rank {args.rank}] saved per-sample stats: {stats_path} "
          f"({len(sample_logprob)} samples)", flush=True)


# ---------------------------------------------------------------------------
# splithalf / gcmp modes (work on saved pass-grads memmaps)
# ---------------------------------------------------------------------------

def _open_pass_grads(path):
    """Zero-copy [n_passes, numel] fp32 torch view of a pass-grads memmap."""
    import torch

    with open(path + ".json") as f:
        meta = json.load(f)
    mm = np.memmap(path, dtype=np.float32, mode="r", shape=(meta["n_passes"], meta["numel"]))
    return torch.from_numpy(mm), meta


def run_split_half(args):
    """Random disjoint 64/64 split-half analysis of g*_64 from saved G_i.

    pass_grads[i] = G_i = sum over all prompts of R_{p,i} * grad log pi(y_{p,i}|x_p).
    For split k with half A: g*_64,A = (1/N)(1/64) sum_{i in A} G_i, and B is the
    complement. Reports cos(g*_64,A, g*_64,B) and
    ||g_A - g_B|| / ||(g_A + g_B)/2|| = 2||2A - T|| / ||T||, T = sum_i G_i.
    """
    import torch

    torch.set_num_threads(max(1, min(64, os.cpu_count() or 1)))
    pass_grads, meta = _open_pass_grads(args.pass_grads_mmap)
    n_prompts = meta["n_prompts_total"]
    n_passes, numel = pass_grads.shape
    half = n_passes // 2
    print(f"[split-half] loaded {args.pass_grads_mmap}: {n_passes} x {numel}", flush=True)
    t0 = time.time()
    total = pass_grads.sum(0, dtype=torch.float64)  # T
    total_norm = total.norm().item()
    # consistency: ||T||/(n_passes*N) must equal ||g*_128|| from combine mode
    print(f"[split-half] ||g*_{n_passes}|| check: {total_norm / (n_passes * n_prompts):.6e} "
          f"({time.time() - t0:.1f}s)", flush=True)

    gen = torch.Generator().manual_seed(args.split_seed)
    rows = []
    for k in range(args.split_half):
        perm = torch.randperm(n_passes, generator=gen)[:half].tolist()
        a = torch.zeros(numel, dtype=torch.float64)
        for s in range(0, half, 8):  # chunked: bound the fancy-index copy
            idx = perm[s : s + 8]
            a += pass_grads[idx].sum(0, dtype=torch.float64)
        b = total - a
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        rel = 2.0 * (2 * a - total).norm().item() / total_norm
        rows.append({"split": k, "cos": cos, "rel_norm_diff": rel})
        print(f"[split-half] split {k}: cos={cos:.8f} rel={rel:.8f} "
              f"(elapsed {time.time() - t0:.1f}s)", flush=True)
        del a, b

    cos_vals = [r["cos"] for r in rows]
    rel_vals = [r["rel_norm_diff"] for r in rows]
    results = {
        "pass_grads": args.pass_grads_mmap,
        "n_prompts": n_prompts,
        "n_passes": n_passes,
        "half": half,
        "n_splits": args.split_half,
        "seed": args.split_seed,
        "splits": rows,
        "cos_mean": float(np.mean(cos_vals)),
        "cos_std": float(np.std(cos_vals, ddof=1)),
        "rel_norm_diff_mean": float(np.mean(rel_vals)),
        "rel_norm_diff_std": float(np.std(rel_vals, ddof=1)),
    }
    print(f"[split-half] cos(g*_64,A, g*_64,B) = {results['cos_mean']:.6f} ± {results['cos_std']:.6f}")
    print(f"[split-half] ||d||/||mean||       = {results['rel_norm_diff_mean']:.6f} ± {results['rel_norm_diff_std']:.6f}")
    out_path = os.path.join(args.out_dir, "pg_split_half_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[split-half] metrics saved to {out_path}", flush=True)


def run_gcmp(args):
    """cos(g*_128, g*_256) and ||g*_128 - g*_256|| / ||g*_256||.

    g*_128 comes from the OLD dataset's pass-grads (T_old = sum_i G_i^old);
    g*_256 = (T_old + T_new) / (256 N) adds the NEW dataset's 128 fresh i.i.d.
    rollouts per prompt. Both datasets cover the same N prompts.
    """
    import torch

    torch.set_num_threads(max(1, min(64, os.cpu_count() or 1)))
    t0 = time.time()
    old, meta_old = _open_pass_grads(args.pass_grads_mmap)
    new, meta_new = _open_pass_grads(args.pass_grads_mmap_new)
    n_prompts = meta_old["n_prompts_total"]
    assert meta_new["n_prompts_total"] == n_prompts
    n_old, n_new = old.shape[0], new.shape[0]
    print(f"[gcmp] summing old {n_old} passes + new {n_new} passes...", flush=True)
    t_old_sum = old.sum(0, dtype=torch.float64)
    print(f"[gcmp] old done ({time.time() - t0:.1f}s)", flush=True)
    t_new_sum = new.sum(0, dtype=torch.float64)
    print(f"[gcmp] new done ({time.time() - t0:.1f}s)", flush=True)

    g128 = t_old_sum / (n_old * n_prompts)
    g256 = (t_old_sum + t_new_sum) / ((n_old + n_new) * n_prompts)
    cos = torch.nn.functional.cosine_similarity(g128, g256, dim=0).item()
    rel = (g128 - g256).norm().item() / g256.norm().item()
    results = {
        "old": args.pass_grads_mmap,
        "new": args.pass_grads_mmap_new,
        "n_prompts": n_prompts,
        "n_old_per_prompt": n_old,
        "n_new_per_prompt": n_new,
        "norm_g128": g128.norm().item(),
        "norm_g256": g256.norm().item(),
        "cos_g128_g256": cos,
        "rel_norm_diff": rel,
    }
    print(f"[gcmp] ||g*_{n_old}|| = {results['norm_g128']:.6e}, "
          f"||g*_{n_old + n_new}|| = {results['norm_g256']:.6e}")
    print(f"[gcmp] cos(g*_{n_old}, g*_{n_old + n_new}) = {cos:.8f}")
    print(f"[gcmp] ||g*_{n_old} - g*_{n_old + n_new}|| / ||g*_{n_old + n_new}|| = {rel:.8f}")
    out_path = os.path.join(args.out_dir, "pg_g128_vs_g256_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[gcmp] metrics saved to {out_path}", flush=True)


# ---------------------------------------------------------------------------
# combine mode
# ---------------------------------------------------------------------------

def run_combine(args):
    import torch

    files = sorted(
        os.path.join(args.out_dir, f) for f in os.listdir(args.out_dir) if f.startswith("grad_bufs_rank")
    )
    assert files, f"no grad_bufs_rank*.pt under {args.out_dir}"
    acc = {n: None for n in NS}
    n_prompts_total, total_stats = None, {"total": 0, "kept": 0, "eos": 0}
    for f in files:
        ckpt = torch.load(f, map_location="cpu")
        n_prompts_total = ckpt["n_prompts_total"]
        for k in total_stats:
            total_stats[k] += ckpt["stats"][k]
        for n in NS:
            v = ckpt["bufs"][n].to(torch.float64)
            acc[n] = v if acc[n] is None else acc[n] + v

    grads = {n: acc[n] / n_prompts_total for n in NS}  # g*_n = (1/N) sum_p (1/n) sum_i ...
    ref = grads[max(NS)]
    ref_norm = ref.norm().item()

    results = {
        "data": args.data,
        "n_prompts": n_prompts_total,
        "sample_sizes": list(NS),
        "reward_mean": total_stats["kept"] / max(total_stats["total"], 1),
        "grad_norms": {str(n): grads[n].norm().item() for n in NS},
        "vs_128": {},
    }
    print(f"n_prompts={n_prompts_total}, reward_mean={results['reward_mean']:.4f}")
    print(f"{'n':>5} {'||g*_n||':>14} {'cos(g*_n,g*_128)':>18} {'||g_n-g_128||/||g_128||':>26}")
    for n in NS:
        cos = torch.nn.functional.cosine_similarity(grads[n], ref, dim=0).item()
        rel = (grads[n] - ref).norm().item() / ref_norm
        results["vs_128"][str(n)] = {"cos": cos, "rel_norm_diff": rel}
        print(f"{n:>5} {grads[n].norm().item():>14.6e} {cos:>18.8f} {rel:>26.8f}")

    out_path = os.path.join(args.out_dir, "pg_sample_complexity_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"metrics saved to {out_path}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["rewards", "grad", "combine", "splithalf", "gcmp"], required=True)
    parser.add_argument("--data", default=os.path.join(LLM_DIR, "results/step400/gen/ppo_0704_n8_iid_n128.parquet"))
    parser.add_argument("--model", default=os.path.join(LLM_DIR, "results/step400/merged/ppo_0704_n8_actor"))
    parser.add_argument("--rewards", default=os.path.join(LLM_DIR, "results/step400/pg_sample_complexity/rewards_ppo_0704_n8.npz"))
    parser.add_argument("--out-dir", default=os.path.join(LLM_DIR, "results/step400/pg_sample_complexity"))
    parser.add_argument("--device", default="cuda:0", help="grad mode device; use cpu for smoke tests")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--buffers-on", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--token-budget", type=int, default=8192, help="padded tokens per micro-batch")
    parser.add_argument("--log-chunk", type=int, default=256, help="sequence chunk size for fp32 log-softmax")
    parser.add_argument("--reward-procs", type=int, default=32)
    parser.add_argument("--max-prompts", type=int, default=-1, help="debug: cap prompts per rank")
    parser.add_argument("--max-responses", type=int, default=-1, help="debug: cap response-index passes")
    parser.add_argument("--pass-grads-mmap", default="", help="grad mode: path to persist per-pass G_i (fp32 memmap)")
    parser.add_argument("--pass-grads-mmap-new", default="", help="gcmp mode: new dataset's pass-grads memmap")
    parser.add_argument("--split-half", type=int, default=16, help="splithalf mode: number of random disjoint 64/64 splits")
    parser.add_argument("--split-seed", type=int, default=12345)
    args = parser.parse_args()

    if args.mode == "rewards":
        run_rewards(args)
    elif args.mode == "grad":
        run_grad(args)
    elif args.mode == "combine":
        run_combine(args)
    elif args.mode == "splithalf":
        run_split_half(args)
    else:
        run_gcmp(args)


if __name__ == "__main__":
    main()
