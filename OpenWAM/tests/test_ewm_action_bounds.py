"""Full-range expert commands must survive training and inference normalization."""
import json
import numpy as np
import pytest
import torch
from openwam.dataloader.ewm_normalizer import EWMNormalizer


def normalizer(tmp_path, mode='minmax'):
    action=dict(type=mode,min=[-.06]*3+[-.3]*3+[-1.],max=[.06]*3+[.4]*3+[1.],
                q01=[-.05]*3+[-.12]*3+[-1.],q50=[0.]*7,q99=[.05]*3+[.15]*3+[1.])
    payload=dict(version=1,voxel=dict(count_clip=15,num_bins=1),
                 proprio=dict(mean=[0.]*8,std=[1.]*8),action=action)
    p=tmp_path/'stats.json';p.write_text(json.dumps(payload))
    return EWMNormalizer(p)


def test_tube_rotation_tail_roundtrips_without_changing_future_supervision(tmp_path):
    n=normalizer(tmp_path)
    raw=torch.tensor([[.059,-.059,0.,-.29135296,.33,.3,-1.],[0.,0.,0.,.4,-.3,0.,1.]])
    target=n.normalize_action(raw)
    assert torch.isfinite(target).all() and target.abs().max()<=1
    torch.testing.assert_close(n.denormalize_action(target),raw,rtol=0,atol=1e-7)
    # These actions saturated irreversibly under the old quantile mapping.
    old=normalizer(tmp_path,'quantile')
    assert abs(float(old.denormalize_action(old.normalize_action(raw))[0,3])+.12)<1e-6


def test_legacy_checkpoint_mapping_is_unchanged(tmp_path):
    n=normalizer(tmp_path,'quantile');raw=torch.linspace(-.4,.4,70).reshape(10,7)
    lo=torch.tensor(n.action_q01);hi=torch.tensor(n.action_q99)
    expected=(2*(raw-lo)/(hi-lo)-1).clamp(-1,1)
    torch.testing.assert_close(n.normalize_action(raw),expected,rtol=0,atol=0)
    torch.testing.assert_close(n.denormalize_action(expected),(expected+1)*.5*(hi-lo)+lo,rtol=0,atol=0)


def test_minmax_constant_dimensions_and_invalid_bounds(tmp_path):
    n=normalizer(tmp_path);p=n.stats_path;payload=json.loads(p.read_text())
    payload['action']['min'][3]=payload['action']['max'][3]=payload['action']['q50'][3]=.125
    p.write_text(json.dumps(payload));n=EWMNormalizer(p)
    assert n.degenerate_action_mask[3]
    assert n.normalize_action(torch.zeros(7))[3]==0
    assert n.denormalize_action(torch.zeros(7))[3]==.125
    payload['action']['min'][0]=1.;p.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='lower bound'):
        EWMNormalizer(p)
