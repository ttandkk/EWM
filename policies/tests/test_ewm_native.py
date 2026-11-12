"""Native EWM sampling with tiny frozen encoders, no downloaded assets or GPU."""
import json

import numpy as np
import pytest


def test_ewm_adapter_matches_native_sampler(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("openwam")
    from openwam.model.architectures.ewm import EWMModel
    from tests.ewm_helpers import options
    from piper_policies.ewm.adapter import EWMPolicy
    from piper_policies.contract import RGB_CONTRACT
    torch.set_num_threads(2)
    original = EWMModel(**options(cameras=2, mode="action_only"))
    config = dict(type="ewm", training_mode="action_only", event_camera_mode="dual", voxel_bins=1)
    torch.save(dict(format="ewm_action_only_v2", horizon=4, model_config=config, model=original.policy_state_dict()), tmp_path / "checkpoint_step_42.pt")
    (tmp_path / "config.yaml").write_text("model:\n  training_mode: action_only\n  event_camera_mode: dual\ndataloader:\n  chunk_size: 4\n")
    (tmp_path / "normalization.json").write_text(json.dumps(dict(version=1, voxel=dict(num_bins=1,count_clip=15),
        proprio=dict(mean=[1.] * 8,std=[2.] * 8),action=dict(type="minmax", min=[-.1]*6+[-1.],max=[.2]*6+[1.]))))
    monkeypatch.setattr(EWMModel, "from_config", classmethod(lambda cls, cfg: EWMModel(**options(cameras=2, mode="action_only"))))
    policy = EWMPolicy(tmp_path, device="cpu")
    from piper_policies.metadata import load_metadata
    assert policy.descriptor == load_metadata(tmp_path)[3]
    assert not policy.optimization["compile_core_active"]
    request = dict(op="predict", raw=True, observation_contract=RGB_CONTRACT,
                   timing=dict(obs_video_time_s=0.,proprio_time_s=0.), seed=6,
                   sampling_steps=2, action_chunk_size=4, prompt="move")
    arrays = dict(obs_video=np.zeros((2,3,256,256),np.uint8),proprio=np.arange(8,dtype=np.float32))
    reply, result = policy.handle(request,arrays)
    batch = dict(obs_video=torch.as_tensor(arrays["obs_video"])[None],
                 proprio=policy.normalizer.normalize_proprio(torch.as_tensor(arrays["proprio"]))[None],prompt=["move"])
    with torch.inference_mode():
        expected = policy.model.sample_policy(batch,horizon=4,normalizer=policy.normalizer,num_steps=2,
            generator=torch.Generator().manual_seed(6),decode_future=False,cache_context_kv=True)
    np.testing.assert_array_equal(result["executed_actions"],expected["action"][0].numpy())
    assert reply["inference_seconds"] > 0 and not reply["prediction_recorded"]
