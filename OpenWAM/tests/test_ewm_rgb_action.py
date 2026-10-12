"""RGB-action production entry: modality isolation, alignment, training and resume."""
from __future__ import annotations

import csv
import importlib
import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from accelerate import Accelerator
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from openwam.dataloader.ewm_dataset import EWMDataset, _read_table
from openwam.dataloader.ewm_rgb_action import PiperRGBActionDataset
from openwam.model.architectures.ewm import EWMModel
from openwam.model.architectures.lean_rgb_codec import LeanRGBCodec
from scripts.train_ewm import train
from tests.test_ewm_model import FakeLeanVAE, small_model

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/train_ewm/train_ewm_rgb_action.sh"
DATA = ROOT.parent / "datasets/piper_stack_fronttight_v3_50/job177683"


def rgb_config(overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="train_ewm_rgb_action_piper", overrides=list(overrides))


def launcher_command(output, **options):
    environment = {"PATH": os.environ["PATH"], "DRY_RUN": "true", "SLURM_JOB_ID": "991234",
                   "OUTPUT_DIR": str(output), **{key: str(value) for key, value in options.items()}}
    return subprocess.run(["bash", str(LAUNCHER)], env=environment, text=True, capture_output=True)


def test_rgb_launcher_matches_event_action_production_config(tmp_path):
    output = tmp_path / "must_not_be_created"
    result = launcher_command(output)
    assert result.returncode == 0, result.stderr
    assert not output.exists()
    args = shlex.split(result.stdout)
    assert "--nproc_per_node=2" in args
    assert args[args.index("--config-name") + 1] == "train_ewm_rgb_action_piper"
    cfg = rgb_config(args[args.index("--config-name") + 2:])
    assert cfg.model.training_mode == "rgb_action"
    assert cfg.model.event_codec_impl == "leanvae16_rgb_joint"
    assert cfg.model.event_camera_mode == cfg.dataloader.event_camera_mode == "dual"
    assert cfg.model.dino_injection_interval == 1
    assert cfg.dataloader.type == "ewm_piper_rgb_action"
    assert not cfg.dataloader.gpu_voxelization and cfg.dataloader.event_frame_cache_dir is None
    assert cfg.dataloader.chunk_size == 24
    assert cfg.training.compile_core and cfg.training.max_steps == 80000
    assert cfg.training.batch_size == 32 and cfg.training.mixed_precision == "bf16"
    assert cfg.project.wandb.project == "CVPR2027-EWM"
    assert "WANDB_MODE=${WANDB_MODE:-online}" in LAUNCHER.read_text()
    baseline = ROOT / "outputs/ewm_lean16_p4_d768_a512_dual_ada2_80k/job179463/config.yaml"
    assert baseline.is_file(), "the referenced production baseline must remain available"
    expected = OmegaConf.to_container(OmegaConf.load(baseline), resolve=True)
    actual = OmegaConf.to_container(cfg, resolve=True)
    # Only representation/data mode, event I/O and run identity may differ.
    del actual["model"]["training_mode"]
    for section, keys in {"model": ["event_codec_impl"],
                          "dataloader": ["type", "gpu_voxelization", "event_frame_cache_dir"],
                          "training": ["output_path"]}.items():
        for key in keys:
            actual[section][key] = expected[section][key]
    actual["project"]["wandb"]["run_name"] = expected["project"]["wandb"]["run_name"]
    assert actual == expected


@pytest.mark.parametrize("options", [
    {"TRAINING_MODE": "joint"}, {"EVENT_CODEC_IMPL": "leanvae16_joint"},
    {"RGB_CAMERA_MODE": "cam0_only"}, {"CHUNK_SIZE": "25"},
    {"PURE_NOISE_EVAL_ENABLED": "true"},
])
def test_launcher_invalid_mode_and_geometry_fail_before_training(tmp_path, options):
    result = launcher_command(tmp_path / "output", **options)
    assert result.returncode == 2
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("camera_mode", ["dual", "wrist_only"])
def test_rgb_dataset_projects_only_state_action_and_aligns_future_rows(tmp_path, monkeypatch, camera_mode):
    _read_table.cache_clear()
    raw_actions = np.arange(12 * 7, dtype=np.float32).reshape(12, 7)
    raw_state = np.arange(12 * 8, dtype=np.float32).reshape(12, 8)
    pq.write_table(pa.table({"action": raw_actions.tolist(), "observation.state": raw_state.tolist(),
                            "task_index": [0] * 12,
                            "observation.event_frame.cam0": [b"DO_NOT_READ"] * 12}), tmp_path / "data.parquet")
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/piper_temporal_contract.json").write_text(
        json.dumps({"temporal_contract": "aligned_event_frame_pre_action_v1"}))
    init_options, reads, frames_seen = [], [], []

    def fake_init(self, path, **kwargs):
        init_options.append(kwargs)
        self.dataset_dir = Path(path)
        self.first_anchor, self.window_stride, self.observation_row_offset, self.chunk_size = 0, 1, 0, 4
        self.direct_event_frames = False  # RGB/action does not require event metadata.
        self.event_camera_mode, self.data_path = camera_mode, "data.parquet"
        self._cumulative = np.array([0, 4])
        self.episodes = pd.DataFrame([{"data/chunk_index": 0, "data/file_index": 0, "_data_offset": 3}])
        self.prompts = {0: "stack"}
        self.normalizer = SimpleNamespace(normalize_action=lambda x: x, normalize_proprio=lambda x: x)

    def forbidden(*a, **kw):
        raise AssertionError("RGB-action accessed the event pipeline")

    original_read = pq.read_table

    def read_table(path, **kwargs):
        reads.append(kwargs["columns"])
        return original_read(path, **kwargs)

    def rgb(self, row, frames):
        frames_seen.append(frames.tolist())
        result = torch.tensor(frames[:, None] + np.array([0, 50])[None], dtype=torch.uint8)
        return result[:, :, None, None, None].expand(-1, -1, 3, 256, 256).clone()

    monkeypatch.setattr(EWMDataset, "__init__", fake_init)
    monkeypatch.setattr(EWMDataset, "_rgb", rgb)
    for name in ("_direct_event_frames", "_voxel", "_sparse_event_frames"):
        monkeypatch.setattr(EWMDataset, name, forbidden)
    monkeypatch.setattr(pq, "read_table", read_table)
    dataset = PiperRGBActionDataset(tmp_path, gpu_voxelization=True, event_frame_cache_dir="/no/cache")
    item = dataset[1]
    assert set(item) == {"obs_video", "obs_rgb", "future_rgb", "action", "proprio", "prompt"}
    assert frames_seen == [[1, 2, 3, 4, 5]]
    assert reads == [["task_index", "observation.state", "action"]]
    assert init_options[0]["event_frame_cache_dir"] is None and not init_options[0]["gpu_voxelization"]
    assert item["obs_video"][:, 0, 0, 0].tolist() == [1, 51]
    indices = [0, 1] if camera_mode == "dual" else [1]
    torch.testing.assert_close(item["obs_rgb"], item["obs_video"][indices])
    assert item["future_rgb"][:, -1, 0, 0, 0].tolist() == [52, 53, 54, 55]
    torch.testing.assert_close(item["action"], torch.from_numpy(raw_actions[4:8]))
    torch.testing.assert_close(item["proprio"], torch.from_numpy(raw_state[4]))
    with pytest.raises(IndexError):
        dataset[4]
    _read_table.cache_clear()


def test_real_piper_rgb_action_sample_matches_joint_actions_and_current_images(monkeypatch):
    assert DATA.is_dir()
    from openwam.dataloader.registry import build_dataset

    cfg = rgb_config(["dataloader.event_camera_mode=dual", "model.event_camera_mode=dual",
                      "dataloader.rgb_decode_backend=pyav"])
    rgb = build_dataset(cfg.dataloader)
    event = EWMDataset(DATA, chunk_size=24, voxel_bins=1, event_camera_mode="dual", rgb_decode_backend="pyav")

    def forbidden(*a, **kw):
        raise AssertionError("RGB-action read an event payload")

    monkeypatch.setattr(rgb, "_direct_event_frames", forbidden)
    for index in (0, 17, len(rgb) - 1):
        a, b = rgb[index], event[index]
        assert a["future_rgb"].shape == (24, 2, 3, 256, 256)
        assert a["future_rgb"].dtype == a["obs_video"].dtype == torch.uint8
        assert set(a) == {"obs_video", "obs_rgb", "future_rgb", "action", "proprio", "prompt"}
        for key in ("obs_video", "proprio", "action"):
            torch.testing.assert_close(a[key], b[key])
        assert a["prompt"] == b["prompt"]
        torch.testing.assert_close(a["obs_rgb"], a["obs_video"])


def test_rgb_codec_native_range_and_causal_current_slice(monkeypatch):
    codec = LeanRGBCodec(vae=FakeLeanVAE(), batch_size=1)
    seen = []
    original_encode = codec.vae.model.encode

    def encode(value):
        seen.append(value.detach().clone())
        return original_encode(value)

    monkeypatch.setattr(codec.vae.model, "encode", encode)
    rgb = torch.zeros(1, 25, 2, 3, 32, 32, dtype=torch.uint8)
    rgb[:, :, 1] = 255
    latent = codec.encode(rgb)
    assert latent.shape == (1, 7, 2, 16, 4, 4)
    assert all(x.dtype == torch.bfloat16 for x in seen)
    assert float(seen[0].min()) == -.5 and float(seen[1].max()) == .5
    changed = rgb.clone()
    changed[:, 1:] = 128
    torch.testing.assert_close(codec.encode(changed)[:, 0], latent[:, 0])
    torch.testing.assert_close(codec.encode(rgb[:, :1])[:, 0], latent[:, 0])
    reconstructed = codec.decode(latent)
    assert reconstructed.shape == rgb.shape
    assert 0 <= reconstructed.min() <= reconstructed.max() <= 1


def rgb_model():
    return small_model(training_mode="rgb_action", event_codec_impl="leanvae16_rgb_joint",
                       event_temporal_compression=4, event_spatial_compression=16, event_patch_size=2,
                       event_camera_mode="dual", event_codec=LeanRGBCodec(vae=FakeLeanVAE(), batch_size=1))


def test_rgb_action_compiled_core_preserves_both_losses_and_gradients():
    model = rgb_model()
    item = TinyRGBDataset()[0]
    batch = {key: [value] if isinstance(value, str) else value[None] for key, value in item.items()}
    fixed = dict(t=torch.tensor([.5]), event_noise=torch.zeros(1, 1, 2, 16, 4, 4),
                 action_noise=torch.zeros(1, 4, 7))
    before = model(batch, **fixed)
    model.predict_velocity = torch.compile(model.predict_velocity, backend="eager", fullgraph=True)
    after = model(batch, **fixed)
    for key in ("loss", "loss_rgb", "loss_action"):
        torch.testing.assert_close(before[key], after[key])
    after["loss"].backward()
    assert model.event_head[-1].weight.grad is not None
    assert model.action_head[-1].weight.grad is not None


class TinyRGBDataset(PiperRGBActionDataset):
    gpu_voxelization = False
    voxel_bins = 1
    event_camera_mode = "dual"
    normalizer = SimpleNamespace(count_clip=15, metadata={"count_clip": 15, "voxel_bins": 1})

    def __init__(self):
        pass

    def __len__(self):
        return 8

    def __getitem__(self, index):
        frames = torch.full((5, 2, 3, 32, 32), 20 + index * 3, dtype=torch.uint8)
        return {"obs_video": frames[0], "obs_rgb": frames[0], "future_rgb": frames[1:],
                "proprio": torch.full((8,), index / 10), "prompt": "stack",
                "action": torch.full((4, 7), index / 10)}


def tiny_training_config(output, steps=3):
    cfg = rgb_config(["model.event_camera_mode=dual", "dataloader.event_camera_mode=dual"])
    cfg.training.update({"output_path": str(output), "max_steps": steps,
                         "lr_schedule_steps": 6, "batch_size": 2, "dataset_num_workers": 0,
                         "mixed_precision": "no", "log_every": 1, "diagnostic_every": 2,
                         "compile_core": False, "save_checkpoints": False,
                         "save_full_states_for_resume": False, "save_steps": 0})
    cfg.dataloader.chunk_size = 4
    cfg.project.wandb.project = None
    return cfg


def test_rgb_trainer_updates_both_heads_logs_wandb_and_no_event_metrics(tmp_path, monkeypatch):
    module = importlib.import_module("scripts.train_ewm.train_ewm")
    logs = []
    finished = []
    fake_run = SimpleNamespace(log=lambda metrics, **kw: logs.append(metrics), finish=lambda: finished.append(True))
    monkeypatch.setattr(module, "init_wandb", lambda cfg: fake_run)
    cfg = tiny_training_config(tmp_path / "train")
    model = rgb_model()
    before_video = model.event_head[-1].weight.detach().clone()
    before_action = model.action_head[-1].weight.detach().clone()
    train(cfg, dataset=TinyRGBDataset(), model=model, accelerator=Accelerator(cpu=True, mixed_precision="no"))
    assert not torch.equal(before_video, model.event_head[-1].weight)
    assert not torch.equal(before_action, model.action_head[-1].weight)
    assert all(p.grad is None and not p.requires_grad for p in model.event_codec.parameters())
    output = Path(cfg.training.output_path)
    assert (output / "rgb_metrics.csv").is_file()
    assert not (output / "event_metrics.csv").exists() and not list(output.glob("*.pt"))
    with (output / "rgb_metrics.csv").open() as handle:
        metrics = list(csv.DictReader(handle))
    assert {"rgb_x0_mse", "rgb_x0_mae", "rgb_x0_psnr"}.issubset(metrics[0])
    assert "event_f1" not in metrics[0]
    assert logs and finished
    assert all("train/loss_rgb" in row and "train/loss_action" in row for row in logs)
    assert all("train/loss_event" not in row and "train/event_f1" not in row for row in logs)
    assert any("train/rgb_x0_psnr" in row for row in logs)


def test_rgb_full_state_resume_matches_continuous_training(tmp_path):
    def execute(output, steps, resume=False):
        cfg = tiny_training_config(output, steps)
        cfg.training.save_checkpoints = cfg.training.save_full_states_for_resume = True
        cfg.training.save_steps = 2
        if resume:
            cfg.training.resume_ckpt_path = str(output)
        torch.manual_seed(42)
        return train(cfg, dataset=TinyRGBDataset(), model=rgb_model(),
                     accelerator=Accelerator(cpu=True, mixed_precision="no"))

    full, split = tmp_path / "full", tmp_path / "split"
    execute(full, 4)
    execute(split, 2)
    execute(split, 4, resume=True)
    a = torch.load(full / "checkpoint_step_000004.pt", weights_only=True)
    b = torch.load(split / "checkpoint_step_000004.pt", weights_only=True)
    assert a["format"] == b["format"] == "ewm_rgb_action_latent_v1"
    assert not any(name.startswith(("event_codec.", "context_encoder.dino.", "context_encoder.text."))
                   for name in b["model"])
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
    restored = rgb_model()
    restored.load_policy_state_dict(b["model"])
    item = TinyRGBDataset()[0]
    batch = {key: [value] if isinstance(value, str) else value[None] for key, value in item.items()}
    prediction = restored.sample_policy(batch, horizon=4, num_steps=2,
        normalizer=SimpleNamespace(denormalize_action=lambda x: x), generator=torch.Generator().manual_seed(7))
    assert prediction["future_rgb"].shape == batch["future_rgb"].shape
    assert prediction["action"].shape == (1, 4, 7)
    assert "future_voxel" not in prediction
