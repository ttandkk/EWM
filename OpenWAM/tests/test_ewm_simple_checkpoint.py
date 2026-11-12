"""Retired features cannot silently change inference; active experts survive import."""
from types import SimpleNamespace

import pytest
import torch

from openwam.model.architectures.ewm import EWMModel
from openwam.model.architectures.ewm_checkpoint import normalize_policy_payload
from tests.ewm_helpers import options
from tests.test_ewm_model import small_batch


def test_legacy_action_only_import_keeps_predictions_and_removes_event_weights():
    torch.set_num_threads(2)
    legacy = EWMModel(**options(cameras=2))
    legacy.training_mode = 'action_only'  # Old layout retained an inactive visual expert.
    payload = dict(format='ewm_action_only_v1', horizon=4,
                   model_config=dict(type='ewm', training_mode='action_only', rgb_history_steps=0,
                                     alignment_weight=0., window_routing={'enabled': False}),
                   model=legacy.policy_state_dict())
    imported = normalize_policy_payload(payload)
    policy = EWMModel(**options(cameras=2, mode='action_only'))
    policy.load_policy_state_dict(imported['model'])
    assert len(imported['model']) < len(payload['model'])
    assert policy.event_input is None and all(block.event is None for block in policy.blocks)
    batch = small_batch(4, cameras=2)
    fixed = dict(t=torch.tensor([.4]), action_noise=torch.randn_like(batch['action']))
    torch.testing.assert_close(policy(batch, **fixed)['loss'], legacy(batch, **fixed)['loss'], rtol=0, atol=0)
    for cache in (False, True):
        kwargs = dict(horizon=4, num_steps=2, decode_future=False, cache_context_kv=cache,
                      normalizer=SimpleNamespace(denormalize_action=lambda x: x))
        a = policy.sample_policy(batch, generator=torch.Generator().manual_seed(14), **kwargs)
        b = legacy.sample_policy(batch, generator=torch.Generator().manual_seed(14), **kwargs)
        torch.testing.assert_close(a['action'], b['action'], rtol=0, atol=0)
    broken = dict(imported['model']); broken.pop('action_head.1.weight')
    with pytest.raises(RuntimeError, match='missing'):
        policy.load_policy_state_dict(broken)


def test_disabled_legacy_memory_import_keeps_core_strict():
    model = EWMModel(**options())
    core = model.policy_state_dict()
    payload = dict(format='ewm_rgb_event_memory_120hz_v2',
                   model_config=dict(type='ewm_rgb_event_memory', memory_enabled=False, memory_dim=256),
                   model={**core, 'rgb_event_memory.memory_spec': torch.zeros(3),
                          'memory_projectors.0.weight': torch.zeros(2, 2)})
    imported = normalize_policy_payload(payload)
    assert imported['model_config'] == {'type': 'ewm'}
    assert set(imported['model']) == set(core)
    model.load_policy_state_dict(imported['model'])
    imported['model']['unknown.weight'] = torch.zeros(1)
    with pytest.raises(RuntimeError, match='unexpected'):
        model.load_policy_state_dict(imported['model'])


@pytest.mark.parametrize('extra', [dict(memory_enabled=True), dict(rgb_history_steps=2),
                                   dict(alignment_weight=.01), dict(window_routing={'enabled': True})])
def test_enabled_retired_features_require_archived_source(extra):
    with pytest.raises(ValueError, match='removed'):
        normalize_policy_payload(dict(format='ewm_wan_latent_v2', model={},
                                      model_config=dict(type='ewm', **extra)))


def test_rgb_action_policy_load_keeps_visual_expert(tmp_path, monkeypatch):
    from scripts.eval_ewm import eval_ewm_policy as inference
    from tests.test_ewm_rgb_action import rgb_model
    model = rgb_model()
    path = tmp_path/'checkpoint.pt'
    torch.save(dict(format='ewm_rgb_action_latent_v1', model=model.policy_state_dict(), horizon=24,
                    model_config=dict(type='ewm', training_mode='rgb_action')), path)
    monkeypatch.setattr(EWMModel, 'from_config', classmethod(lambda cls, cfg: rgb_model()))
    monkeypatch.setattr(inference, 'EWMNormalizer', lambda _: SimpleNamespace())
    loaded, _, horizon = inference.load_policy(path)
    assert horizon == 24 and loaded.training_mode == 'rgb_action'
    assert loaded.event_codec is not None and all(block.event is not None for block in loaded.blocks)
    for key, value in model.policy_state_dict().items():
        torch.testing.assert_close(loaded.policy_state_dict()[key], value)


@pytest.mark.parametrize('flag', ['--event-guided-replan', '--event-replan-record-only',
                                  '--event-replan-threshold'])
def test_removed_replanning_flags_are_rejected(flag):
    from scripts.eval_ewm.eval_ewm_piper_tasks_sim import build_parser
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(['--task', 'conveyor', '--plan-only', flag])
    assert exc.value.code == 2
