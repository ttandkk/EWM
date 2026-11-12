"""Eight-task RGB/action contract and episode-tail behavior."""
import numpy as np
import pytest
import torch

from openwam.dataloader.ewm_rgb_action import PiperRGBActionDataset
from tests.test_ewm_tail_mask import toy_dataset


def test_rgb_tail_masks_do_not_cross_into_next_episode(monkeypatch, tmp_path):
    d = toy_dataset(monkeypatch, tmp_path)
    from openwam.dataloader import ewm_dataset, ewm_rgb_action
    monkeypatch.setattr(ewm_rgb_action, "_read_table", ewm_dataset._read_table)
    d.__class__ = PiperRGBActionDataset
    seen = []
    def rgb(row, indices):
        seen.append(indices.tolist())
        return torch.as_tensor(indices[:, None, None, None, None] + np.array([0, 50])[None, :, None, None, None], dtype=torch.uint8)
    d._rgb = rgb
    d._direct_event_frames = lambda *args: pytest.fail("RGB-action read events")
    tail = d[4]
    assert seen[-1] == [4, 5, 5, 5, 5]
    assert tail["action_valid_mask"].tolist() == [True, True, False, False]
    assert tail["future_valid_mask"].tolist() == [True, False, False, False]
    assert tail["action"][:, 0].tolist() == [5, 6, 0, 0]
    assert tail["future_rgb"][:, 0, 0, 0, 0].tolist() == [5, 5, 5, 5]
    last = d[5]
    assert last["action_valid_mask"].tolist() == [True, False, False, False]
    assert not last["future_valid_mask"].any()
    assert d[6]["action"][0, 0] == 7


def test_rgb_multitask_config_matches_protocol(monkeypatch):
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from openwam.dataloader.registry import DATASET_REGISTRY
    monkeypatch.setenv("NEW_PROJECT_ROOT", "/tmp/project")
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        cfg = compose(config_name="train_ewm_eight_rgb_action")
    assert cfg.dataloader.type in DATASET_REGISTRY
    assert cfg.model.training_mode == "rgb_action"
    assert cfg.model.event_camera_mode == cfg.dataloader.event_camera_mode == "dual"
    assert cfg.model.event_horizon == cfg.dataloader.chunk_size == 24
    assert cfg.model.event_temporal_downsample == 2
    assert cfg.dataloader.pad_tail and len(cfg.dataloader.tasks) == 8


def test_rgb_action_policy_request_sends_only_current_rgb_and_state(monkeypatch):
    from types import SimpleNamespace
    from scripts.eval_ewm import eval_ewm_piper_conveyor_sim as client
    from scripts.eval_ewm.ewm_rgb_action_protocol import OBSERVATION_CONTRACT
    seen = {}
    def request(conn, header, arrays):
        seen.update(header=header, arrays=arrays)
        return {"ok": True}, {}
    from piper_bench.evaluation import policy_client
    monkeypatch.setattr(policy_client, "request", request)
    args = SimpleNamespace(policy_mode="rgb_action")
    timing = dict(obs_video_time_s=1.0, proprio_time_s=1.0, obs_event_end_s=1.0)
    client.policy_request(None, args, dict(event_time_contract="event", defer_visualization=True, timing=timing),
                          dict(obs_video="rgb", obs_voxel="event", proprio="state"))
    assert seen["arrays"] == dict(obs_video="rgb", proprio="state")
    assert seen["header"]["observation_contract"] == OBSERVATION_CONTRACT
    assert seen["header"]["decode_future_rgb"] is False
    assert seen["header"]["timing"] == dict(obs_video_time_s=1.0, proprio_time_s=1.0)
    assert "event_time_contract" not in seen["header"]


def test_rgb_action_protocol_accepts_action_only_reply_without_video():
    from types import SimpleNamespace
    from scripts.eval_ewm.ewm_rgb_action_protocol import prepare_request, OBSERVATION_CONTRACT
    arrays = dict(obs_video=np.zeros((2, 3, 256, 256), dtype=np.uint8), proprio=np.zeros(8, dtype=np.float32))
    req = dict(observation_contract=OBSERVATION_CONTRACT, observation_row_offset=0,
               raw_observation=True, decode_future_rgb=False, seed=42, sampling_steps=4,
               action_chunk_size=24, prompt="catch", timing=dict(obs_video_time_s=0., proprio_time_s=0.))
    batch, seed, steps, chunk = prepare_request(req, arrays, SimpleNamespace(normalize_proprio=lambda x: x), horizon=24, device="cpu")
    assert (seed, steps, chunk) == (42, 4, 24)
    assert batch["obs_rgb"] is batch["obs_video"]
    assert "obs_voxel" not in batch
    with pytest.raises(ValueError, match="boolean"):
        prepare_request(dict(req, decode_future_rgb="false"), arrays, SimpleNamespace(normalize_proprio=lambda x: x), horizon=24, device="cpu")


def test_rgb_multitask_trainer_updates_both_heads(tmp_path, monkeypatch):
    from accelerate import Accelerator
    from openwam.dataloader.ewm_piper_rgb_multitask import PiperMultiTaskRGBActionDataset
    from scripts.train_ewm import train
    from tests.test_ewm_rgb_action import TinyRGBDataset, rgb_model, tiny_training_config

    class TinyMulti(PiperMultiTaskRGBActionDataset):
        normalizer = TinyRGBDataset.normalizer
        voxel_bins = 1
        event_camera_mode = "dual"
        tasks = ("a", "b")
        lengths = (8, 8)
        def __init__(self):
            pass
        def __getitem__(self, index):
            task, window = index
            return TinyRGBDataset()[window]

    torch.set_num_threads(2)
    model = rgb_model()
    video = model.event_head[-1].weight.detach().clone()
    action = model.action_head[-1].weight.detach().clone()
    cfg = tiny_training_config(tmp_path / "multi", steps=2)
    train(cfg, dataset=TinyMulti(), model=model, accelerator=Accelerator(cpu=True, mixed_precision="no"))
    assert not torch.equal(video, model.event_head[-1].weight)
    assert not torch.equal(action, model.action_head[-1].weight)
