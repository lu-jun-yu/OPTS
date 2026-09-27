"""CPU checks for RQ3 calculations and the prompt-length limit."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.RQ3 import gradients as pg
from experiments.RQ3.allocation import build_masks
from experiments.RQ3.coefficients import build_tree_coefficients, RETURN_SCHEMA
from experiments.RQ3.credit import summarize
from experiments.RQ3.metrics import reduce


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(torch.arange(25).reshape(5, 5).float()/25)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.table[input_ids])


def tree_row():
    return pd.Series(dict(tree_uids=['u']*3, tree_seqs=[0]*3,
        tree_rids=['a','b','c'], tree_pids=[None,'a',None], tree_branch_pos=[-1,0,-1],
        response_ids=[[1,2,3],[3,4],[2]], global_indices=[0,1,2],
        tree_rounds=[0,1,3], tree_rewards=[1,0,1]), name=0)


def test_zero_value_weights_and_credit():
    row = tree_row()
    for backup in ('max','mean'):
        slots = build_tree_coefficients(row,1,backup)[0]['slots']
        first = 1. if backup == 'max' else .5
        np.testing.assert_allclose(dict(slots[0]['segs'])[0], [1,1,1])
        np.testing.assert_allclose(dict(slots[1]['segs'])[0], [first,.5,.5])
        np.testing.assert_allclose(dict(slots[7]['segs'])[0], [first/2,.25,.25])
        np.testing.assert_allclose(dict(slots[7]['segs'])[2], [.5])
    out = summarize([row.to_dict()])
    assert len(out['identity_check']['results']) == 6
    assert all(c['passed'] for c in out['identity_check']['results'].values())


def test_matched_allocation_skips_short_without_redraw():
    events = pd.DataFrame([dict(prompt_idx=0, group=g, round=r, score=1.)
                           for g in range(8) for r in range(1,8)])
    eligible = np.ones((8,2),bool); eligible[0,0] = False
    mask = build_masks(events, eligible)
    np.testing.assert_array_equal(mask['nominal_selection'][:,0].sum(0), [8]*7)
    np.testing.assert_array_equal(mask['realized_selection'][:,0].sum(0), [7]*7)
    np.testing.assert_array_equal(mask['k'][0], 0)
    assert 'k_p' not in mask


def test_fixed_gradient_matches_direct_autograd(tmp_path):
    bb, suffix = [1]*128+[2,3], [3,4]
    pd.DataFrame([dict(prompt_idx=0,group=0,backbone_ids=bb,prefix_len=128,
        backbone_reward=0.,suffix_ids=[suffix]*7,suffix_rewards=[1.]*7)]).to_parquet(tmp_path/'a.parquet')
    pd.DataFrame([dict(prompt='unused')]).to_parquet(tmp_path/'prompts.parquet')
    events = pd.DataFrame([dict(prompt_idx=0,group=g,round=r,score=1.)
                           for g in range(8) for r in range(1,8)])
    np.savez(tmp_path/'mask.npz',**build_masks(events,np.ones((8,1),bool)))
    model = TinyActor()
    args = SimpleNamespace(device='cpu',tree_data=[str(tmp_path/'a.parquet')],group=0,
        rank=0,world_size=1,max_trees=-1,backbone_only=False,mask_file=str(tmp_path/'mask.npz'),
        prompts=str(tmp_path/'prompts.parquet'),accs_on='cpu',log_chunk=64,out_dir=str(tmp_path),
        model='tiny',no_save=False)
    with patch.object(pg,'_load_actor',return_value=(model,SimpleNamespace(pad_token_id=0,eos_token_id=0),list(model.parameters()))), patch.object(pg,'_prompt_ids',return_value=[4]):
        pg.run_treegrad(args)
    payload = torch.load(tmp_path/'treegrad_g00_rank0.pt',weights_only=False)
    assert set(payload['accs']) == set(pg.EST_SLOTS)
    for slot, got in payload['accs'].items():
        if slot == 's0':
            assert got.abs().sum() == 0
            continue
        k = int(slot[1:].split('_')[0]); model.zero_grad(set_to_none=True)
        pre, _ = pg._response_terms_loss(model,[4]+bb,[(1,129),(129,131)],0,'cpu',64)
        (suf,) = pg._response_terms_loss(model,[4]+bb[:128]+suffix,[(129,131)],0,'cpu',64)
        loss = pre*k/(k+1) + suf*k*(1/(k+1) if slot.endswith('ttpg') else 1)
        loss.backward()
        # Allow FP32 rounding from separate backward accumulations.
        torch.testing.assert_close(got,model.table.grad.flatten(),atol=2e-4,rtol=2e-5)


def test_opts_gradient_matches_direct_autograd(tmp_path):
    row = tree_row(); trees = build_tree_coefficients(row,1)
    pd.DataFrame([dict(prompt='unused')]).to_parquet(tmp_path/'c.parquet')
    torch.save(dict(schema_version='exp1_v2',credit_mode='return',return_schema=RETURN_SCHEMA,
        shard=0,shard_size=1,gen_path=str(tmp_path/'c.parquet'),trees=trees),tmp_path/'coef.pt')
    np.savez(tmp_path/'mask.npz',allocation='prompt_round_v2')
    model = TinyActor()
    args = SimpleNamespace(device='cpu',tree_data=[str(tmp_path/'coef.pt')],group=0,rank=0,
        world_size=1,max_trees=-1,mask_file=str(tmp_path/'mask.npz'),accs_on='cpu',log_chunk=64,
        out_dir=str(tmp_path),model='tiny')
    with patch.object(pg,'_load_actor',return_value=(model,SimpleNamespace(pad_token_id=0),list(model.parameters()))), patch.object(pg,'_prompt_ids',return_value=[4]):
        pg.run_optsgrad(args)
    payload = torch.load(tmp_path/'optsgrad_g00_rank0.pt',weights_only=False)
    assert set(payload['accs']) == {f'max_s{s}' for s in (0,1,3,7)}
    for s, slot in trees[0]['slots'].items():
        model.zero_grad(set_to_none=True); loss = 0
        for j,coef in slot['segs']:
            ctx = pg.response_context(trees[0]['rids'],j,[4],{})
            ids = list(trees[0]['rids'][j][2])
            loss += pg._weighted_terms_losses(model,ctx+ids,len(ctx),len(ctx)+len(ids),[coef],0,'cpu',64)[0]
        loss.backward()
        torch.testing.assert_close(payload['accs'][f'max_s{s}'],model.table.grad.flatten())


def test_m1_bias_hand_computed():
    def group(i, slots):
        return dict(group=i,model='tiny',parameter_names=['w'],n_trees_total=2,
                    accs={k:torch.tensor(v,dtype=torch.float32) for k,v in slots.items()})
    fixed = ['s0']+[f's{s}_{m}' for s in (1,3,7) for m in ('naive','ttpg')]
    est = [group(i,{k:[float(i+1),2.] for k in fixed}) for i in range(8)]
    opts = [group(i,{f'max_s{s}':[float(i+2),2.] for s in (0,1,3,7)}) for i in range(8)]
    refs = [group(i,{'s0':[2.,2.]}) for i in range(32)]
    result = reduce(est,opts,refs,chunk=1)['aggregations']['theory']['metrics']
    assert result['A/s0']['1']['bias'] == pytest.approx(1.25/np.sqrt(2))
    assert result['C2/s7']['1']['bias'] == pytest.approx(1.75/np.sqrt(2))


def test_prompt_region_length_limit():
    from experiments.RQ3.controls import _request_batch
    requests = [dict(prompt_ids=[1]*1141,raw_prompt='test',raw_prompt_len=1013,seed=123)]
    batch = _request_batch(requests,SimpleNamespace(pad_token_id=0),1152)
    assert batch.batch['attention_mask'].sum().item() == 1141
    with pytest.raises(ValueError,match='invalid continuation prompt length'):
        _request_batch(requests,SimpleNamespace(pad_token_id=0),1140)
