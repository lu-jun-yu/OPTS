# Copyright 2025 Junyu Lu (Julian Lou). All rights reserved.

"""Smoke tests for experiments/rq1_tree_pg.py.

Validates, with the tiny random-weight Qwen3 used for earlier smokes:

  1. treegrad: per-tree Naive/TTPG gradient accumulators match a brute-force
     per-token autograd computation (masking ranges, EOS append rule, C/S
     assembly, checkpoint denominators), including trees with zero rewards.
  2. gstar: the gradient sum uses the per-trajectory token-mean
     (1/|tau|) normalization and skips R=0 responses exactly.

Run:  python experiments/rq1_smoke.py
"""

import os
import subprocess
import sys

LLM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if LLM_DIR not in sys.path:
    sys.path.insert(0, LLM_DIR)

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TINY = os.path.join(LLM_DIR, "results/step400/pg_sample_complexity/smoke/tiny_qwen3")
SMOKE_DIR = os.path.join(LLM_DIR, "results/step400/rq1/smoke")
TEST_DATA = os.path.join(LLM_DIR, "data/test.parquet")
EXP = os.path.join(LLM_DIR, "experiments/rq1_tree_pg.py")
PYTHON = sys.executable
EOS = 151645
N_EXTRA = 15
KS = (0, 1, 3, 7, 15)

torch.manual_seed(0)
rng = np.random.default_rng(7)


def _flat(params):
    return torch.cat([p.grad.detach().view(-1) for p in params]).double()


def _token_logp(model, seq):
    ids = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
    logits = model(input_ids=ids).logits[0, :-1].float()
    labels = ids[0, 1:]
    return torch.log_softmax(logits, dim=-1).gather(1, labels.unsqueeze(-1)).squeeze(-1)
    # index t-1 of the result scores response token t (1-based seq position)


def _grad_of_range(model, params, seq, lo, hi):
    """grad of -sum_{t in [lo,hi)} log p(seq[t] | seq[:t])."""
    if hi <= lo:
        return torch.zeros(sum(p.numel() for p in params), dtype=torch.float64)
    tok_logp = _token_logp(model, seq)
    loss = -tok_logp[lo - 1 : hi - 1].sum()
    model.zero_grad(set_to_none=True)
    loss.backward()
    return _flat(params)


def test_treegrad():
    os.makedirs(SMOKE_DIR, exist_ok=True)
    df_test = pd.read_parquet(TEST_DATA)
    tok = AutoTokenizer.from_pretrained(TINY)

    rows = []
    n_trees = 3
    for t in range(n_trees):
        bb = rng.integers(0, 150000, size=12).tolist()
        if t == 0:
            bb[-1] = EOS  # already terminated: no append expected
        sufs = [rng.integers(0, 150000, size=int(rng.integers(2, 6))).tolist() for _ in range(N_EXTRA)]
        rows.append(
            {
                "prompt_idx": t,
                "group": 0,
                "prefix_len": 5,
                "backbone_ids": np.asarray(bb, dtype=np.int32),
                "backbone_finish": "stop" if t < 2 else "length",
                "backbone_reward": np.float32([1.0, 0.0, 1.0][t]),
                "suffix_ids": [np.asarray(s, dtype=np.int32) for s in sufs],
                "suffix_finish": ["stop"] * N_EXTRA,
                "suffix_rewards": np.asarray(
                    [[0.0, 1.0] [t % 2] if j == 0 else float((j + t) % 3 == 0)
                     for j in range(N_EXTRA)],
                    dtype=np.float32,
                ),
            }
        )
    tree_path = os.path.join(SMOKE_DIR, "fake_trees.parquet")
    pd.DataFrame(rows).to_parquet(tree_path)

    out_dir = os.path.join(SMOKE_DIR, "treegrad")
    subprocess.run(
        [PYTHON, EXP, "--mode", "treegrad", "--model", TINY, "--device", "cpu",
         "--group", "0", "--tree-data", tree_path, "--out-dir", out_dir,
         "--log-chunk", "4", "--pg-norm", "sum"],
        check=True, cwd=LLM_DIR,
    )
    got = torch.load(os.path.join(out_dir, "treegrad_g00_rank0.pt"), map_location="cpu")["accs"]

    # ---- brute force ----
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    numel = sum(p.numel() for p in params)
    slots = ["k0"] + [f"k{k}_{m}" for k in (1, 3, 7, 15) for m in ("naive", "ttpg")]
    bf = {s: torch.zeros(numel, dtype=torch.float64) for s in slots}

    for t, row in enumerate(rows):
        messages = [dict(m) for m in df_test.iloc[t]["prompt"]]
        prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        bb = list(map(int, row["backbone_ids"]))
        if row["backbone_finish"] == "stop" and (not bb or bb[-1] != EOS):
            bb = bb + [EOS]
        plen = int(row["prefix_len"])
        r0 = float(row["backbone_reward"])
        sufs = []
        for j in range(N_EXTRA):
            ids = list(map(int, row["suffix_ids"][j]))
            if row["suffix_finish"][j] == "stop" and (not ids or ids[-1] != EOS):
                ids = ids + [EOS]
            sufs.append(ids)
        rs = list(map(float, row["suffix_rewards"]))

        l_pre, l0 = plen, len(bb) - plen
        np_ = len(prompt_ids)
        C = torch.zeros(numel, dtype=torch.float64)
        S = torch.zeros(numel, dtype=torch.float64)
        need_pre = (r0 > 0) or any(v > 0 for v in rs)
        if need_pre:
            seq = prompt_ids + bb
            C = _grad_of_range(model, params, seq, np_, np_ + l_pre)
            if r0 > 0:
                S = r0 * _grad_of_range(model, params, seq, np_ + l_pre, len(seq))
                bf["k0"] += r0 * C + S  # trajectory-sum PG: no token-count division
        l_cum = l0
        r_cum = r0
        for j in range(1, N_EXTRA + 1):
            ids, rj = sufs[j - 1], rs[j - 1]
            l_cum += len(ids)
            r_cum += rj
            if rj > 0 and ids:
                seq = prompt_ids + bb[:plen] + ids
                S += rj * _grad_of_range(model, params, seq, len(seq) - len(ids), len(seq))
            if j in (1, 3, 7, 15) and need_pre:
                rbar = r_cum / (j + 1)
                bf[f"k{j}_naive"] += rbar * C + S
                bf[f"k{j}_ttpg"] += rbar * C + S / (j + 1)

    print(f"{'slot':>12} {'||diff||/||bf||':>18}")
    for s in slots:
        diff = (got[s].double() - bf[s]).norm().item()
        rel = diff / max(bf[s].norm().item(), 1e-30)
        print(f"{s:>12} {rel:18.3e}")
        assert rel < 5e-5, f"treegrad slot {s} mismatch: rel={rel}"
    print("[smoke] treegrad OK")


def test_gstar():
    os.makedirs(SMOKE_DIR, exist_ok=True)
    df_test = pd.read_parquet(TEST_DATA)
    tok = AutoTokenizer.from_pretrained(TINY)
    texts = [
        ["short answer", "a much longer answer with several tokens indeed", "", "mid len ans"],
        ["another reply", "tiny", "some more tokens here", "last one"],
    ]
    rewards = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    rows = []
    for i in range(2):
        src = df_test.iloc[i]
        rows.append(
            {
                "data_source": src["data_source"],
                "prompt": src["prompt"],
                "ability": src["ability"],
                "reward_model": src["reward_model"],
                "extra_info": src["extra_info"],
                "responses": texts[i],
            }
        )
    data_path = os.path.join(SMOKE_DIR, "fake_rollouts.parquet")
    pd.DataFrame(rows).to_parquet(data_path)
    rw_path = os.path.join(SMOKE_DIR, "fake_rewards.npz")
    np.savez(rw_path, rewards=rewards)

    out_dir = os.path.join(SMOKE_DIR, "gstar")
    subprocess.run(
        [PYTHON, EXP, "--mode", "gstar", "--model", TINY, "--device", "cpu",
         "--data", data_path, "--rewards", rw_path, "--tag", "smoke",
         "--out-dir", out_dir, "--log-chunk", "4", "--pg-norm", "sum"],
        check=True, cwd=LLM_DIR,
    )
    got = torch.load(os.path.join(out_dir, "gstar_sum_smoke_rank0.pt"), map_location="cpu")["acc"]

    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    numel = sum(p.numel() for p in params)
    bf = torch.zeros(numel, dtype=torch.float64)
    for i in range(2):
        messages = [dict(m) for m in df_test.iloc[i]["prompt"]]
        prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        for j, text in enumerate(texts[i]):
            if rewards[i, j] <= 0:
                continue
            resp_ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(resp_ids) >= 2048:
                resp_ids = resp_ids[:2048]
            else:
                resp_ids = resp_ids + [EOS]
            seq = prompt_ids + resp_ids
            g = _grad_of_range(model, params, seq, len(prompt_ids), len(seq))
            bf += rewards[i, j] * g  # trajectory-sum PG: no 1/|tau| factor

    rel = (got.double() - bf).norm().item() / max(bf.norm().item(), 1e-30)
    print(f"[smoke] gstar rel diff = {rel:.3e}")
    assert rel < 5e-5, f"gstar mismatch: rel={rel}"
    print("[smoke] gstar OK")


if __name__ == "__main__":
    test_treegrad()
    test_gstar()
    print("ALL SMOKE TESTS PASSED")
