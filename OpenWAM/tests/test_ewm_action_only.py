"""Action-only ablation: no event I/O, attention, loss, codec, or gradients."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from accelerate import Accelerator
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from openwam.dataloader.ewm_action_only import PiperActionOnlyDataset, _read_action_table
from openwam.dataloader.ewm_dataset import EWMDataset
from openwam.model.architectures.ewm import (
    EWMContextEncoder, EWMJointBlock, EWMModel, event_action_coordinates,
    rectified_action_flow_inputs,
)
from scripts.train_ewm import train


class FakeDino(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(6))
        self.properties = SimpleNamespace(z_dim=6)

    def batch_encode(self, pixels):
        return (pixels.mean((1, 2, 3, 4))[:, None] * self.weight)[..., None, None, None]


class FakeText(torch.nn.Module):
    hidden_size = 16

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(16))

    def forward(self, prompts):
        return self.weight.expand(len(prompts), 4, 16), torch.tensor(
            [[True, True, False, False]]).expand(len(prompts), -1)


def make_model(**kwargs):
    options = dict(dim=32, action_hidden_dim=16, heads=2, layers=2,
                   training_mode="action_only", event_vae_path="/no/vae/required",
                   context_encoder=EWMContextEncoder(FakeDino(), FakeText(), 32))
    options.update(kwargs)
    return EWMModel(**options)


def batch(horizon=3):
    return dict(obs_video=torch.randint(0, 256, (2, 2, 3, 16, 16), dtype=torch.uint8),
                action=torch.randn(2, horizon, 7), proprio=torch.randn(2, 8),
                prompt=["stack", "stack"])


def forbidden(*args, **kwargs):
    raise AssertionError("disabled event/future-video path was called")


def test_action_flow_math_and_validation():
    target = torch.randn(2, 3, 7)
    noise = torch.randn_like(target)
    t = torch.tensor([0., 1.])
    noisy, velocity, actual_t = rectified_action_flow_inputs(target, t=t, action_noise=noise)
    torch.testing.assert_close(noisy[0], target[0])
    torch.testing.assert_close(noisy[1], noise[1])
    torch.testing.assert_close(velocity, noise - target)
    torch.testing.assert_close(actual_t, t)
    with pytest.raises(ValueError, match="action"):
        rectified_action_flow_inputs(torch.zeros(2, 3, 8))
    with pytest.raises(ValueError, match="flow time"):
        rectified_action_flow_inputs(target, t=torch.tensor([float("nan"), .2]))


@pytest.mark.parametrize("checkpointing,skip", [(False, 0), (True, 0), (True, 1)])
def test_action_only_no_event_loss_attention_codec_or_gradients(monkeypatch, checkpointing, skip):
    monkeypatch.setattr("openwam.model.architectures.ewm.LeanEventCodec", forbidden)
    model = make_model(gradient_checkpointing=checkpointing, gradient_checkpointing_skip_first_blocks=skip)
    assert model.event_codec is None
    for module in (model.event_input, model.event_head, *(b.event for b in model.blocks)):
        monkeypatch.setattr(module, "forward", forbidden)
    for block in model.blocks:
        monkeypatch.setattr(block.event, "prepare", forbidden)
        monkeypatch.setattr(block.event, "finish", forbidden)
        monkeypatch.setattr(block.event.context, "project_kv", forbidden)
    monkeypatch.setattr(model, "_decode_future", forbidden)
    inputs = batch()
    fixed = dict(t=torch.tensor([.3, .8]), action_noise=torch.randn_like(inputs["action"]), diagnostics=True)
    first = model(inputs, **fixed)
    poisoned = dict(inputs, obs_voxel=object(), future_voxel=object(), obs_rgb=object(), future_rgb=object())
    second = model(poisoned, event_noise=object(), **fixed)
    assert set(first) == {"loss", "loss_action", "action_x0_mse", "action_x0_mae"}
    torch.testing.assert_close(first["loss"], first["loss_action"])
    torch.testing.assert_close(first["loss"], second["loss"])
    first["loss"].backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and parameter.grad.isfinite().all(), name
        else:
            assert parameter.grad is None, name
    assert not model.camera.requires_grad
    assert model.action_head[-1].weight.grad is not None
    assert model.context_encoder.rgb_projection.weight.grad is not None
    assert model.context_encoder.dino.weight.grad is None
    # Non-multiple-of-four horizon is valid when no VAE is involved.
    normalizer = SimpleNamespace(denormalize_action=lambda x: x * 2)
    before_training = model.training
    sampled = model.sample_policy(inputs, horizon=3, normalizer=normalizer, num_steps=2,
                                  generator=torch.Generator().manual_seed(91))
    assert set(sampled) == {"action", "action_normalized"}
    assert sampled["action"].shape == (2, 3, 7) and sampled["action"].isfinite().all()
    assert model.training == before_training
    uncached = model.sample_policy(poisoned, horizon=3, normalizer=normalizer, num_steps=2,
                                   cache_context_kv=False, generator=torch.Generator().manual_seed(91))
    torch.testing.assert_close(sampled["action"], uncached["action"])
    clone = make_model()
    clone.load_policy_state_dict(model.policy_state_dict())


def test_action_only_block_attention_has_only_state_and_action_tokens(monkeypatch):
    block = EWMJointBlock(32, 2, 16)
    monkeypatch.setattr(block.event, "prepare", forbidden)
    action = torch.randn(2, 4, 16, requires_grad=True)
    context = torch.randn(2, 3, 32)
    _, coords = event_action_coordinates(3, 1, 1, num_cameras=1)
    mask = torch.tensor([[True, True, False]]).expand(2, -1)
    original = torch.nn.functional.scaled_dot_product_attention
    calls = []

    def counted(q, k, v, **kwargs):
        calls.append((q.shape[-2], k.shape[-2]))
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", counted)
    first = block(object(), action, torch.tensor([.2, .8]), context, mask,
                  None, coords, action_only=True)
    assert first[0] is None and first[1].shape == action.shape
    assert calls[0] == (4, 4)  # Exactly one state token and three action tokens.
    changed = context.clone()
    changed[:, -1] += 10000
    second = block(None, action, torch.tensor([.2, .8]), changed, mask, None, coords, action_only=True)
    torch.testing.assert_close(first[1], second[1])
    first[1].square().mean().backward()
    assert all(p.grad is None for p in block.event.parameters())


def test_action_only_profile_has_no_vae_segment():
    model = make_model()
    inputs = batch()
    fixed = dict(t=torch.tensor([.2, .7]), action_noise=torch.zeros_like(inputs["action"]))
    plain, profiled = model(inputs, **fixed), model(inputs, profile_timing=True, **fixed)
    torch.testing.assert_close(plain["loss"], profiled["loss"])
    assert set(profiled["_profile"]) == {"flow_prepare", "context_encode", "dit_forward", "loss_and_diagnostics"}


def test_action_only_compiled_core_preserves_loss_and_gradients():
    model = make_model(dino_injection_interval=2)
    inputs = batch()
    fixed = dict(t=torch.tensor([.2, .7]), action_noise=torch.zeros_like(inputs["action"]))
    reference = model(inputs, **fixed)["loss"]
    model.predict_velocity = torch.compile(model.predict_velocity, backend="eager", fullgraph=True)
    actual = model(inputs, **fixed)["loss"]
    torch.testing.assert_close(actual, reference)
    actual.backward()
    assert model.action_head[-1].weight.grad is not None
    assert model.event_input.weight.grad is None


def test_action_only_checkpoint_roundtrip_and_format_guard(tmp_path, monkeypatch):
    from scripts.eval_ewm.eval_ewm_policy import load_policy

    model = make_model()
    stats = {"version": 1, "voxel": {"count_clip": 15, "num_bins": 1},
             "proprio": {"mean": [0] * 8, "std": [1] * 8},
             "action": {"q01": [-1] * 7, "q50": [0] * 7, "q99": [1] * 7}}
    (tmp_path / "normalization.json").write_text(json.dumps(stats))
    payload = dict(format="ewm_action_only_v1", model=model.policy_state_dict(),
                   model_config={"training_mode": "action_only"}, horizon=3, step=10)
    path = tmp_path / "policy.pt"
    torch.save(payload, path)
    monkeypatch.setattr(EWMModel, "from_config", lambda config: make_model())
    restored, normalizer, horizon = load_policy(path)
    assert horizon == 3 and restored.event_codec is None
    inputs = batch()
    fixed = dict(horizon=3, normalizer=normalizer, num_steps=2)
    first = model.sample_policy(inputs, generator=torch.Generator().manual_seed(99), **fixed)
    second = restored.sample_policy(inputs, generator=torch.Generator().manual_seed(99), **fixed)
    torch.testing.assert_close(first["action"], second["action"])
    payload["format"] = "ewm_wan_latent_v2"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="format and training_mode"):
        load_policy(path)


def test_action_only_dataset_projects_parquet_and_decodes_only_current_rgb(tmp_path, monkeypatch):
    import pandas as pd

    _read_action_table.cache_clear()
    data = tmp_path / "data.parquet"
    raw_actions = np.arange(6 * 7, dtype=np.float32).reshape(6, 7)
    pq.write_table(pa.table({"action": raw_actions.tolist(),
                            "observation.state": np.zeros((6, 8)).tolist(), "task_index": [0] * 6,
                            "observation.event_frame.cam0": [b"DO_NOT_READ"] * 6}), data)
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/piper_temporal_contract.json").write_text(
        json.dumps({"temporal_contract": "aligned_event_frame_pre_action_v1"}))
    init_options = []

    def fake_init(self, dataset_dir, **kwargs):
        init_options.append(kwargs)
        self.dataset_dir, self.first_anchor, self.window_stride = Path(dataset_dir), 0, 1
        self.observation_row_offset, self.chunk_size = 0, 3
        self.data_path = "data.parquet"
        self._cumulative = np.array([0, 3])
        self.episodes = pd.DataFrame([{"data/chunk_index": 0, "data/file_index": 0, "_data_offset": 0}])
        self.prompts = {0: "stack"}
        self.normalizer = SimpleNamespace(normalize_action=lambda x: x, normalize_proprio=lambda x: x)

    monkeypatch.setattr(EWMDataset, "__init__", fake_init)
    calls, columns = [], []
    original_read = pq.read_table

    def read_projected(path, **kwargs):
        columns.append(kwargs["columns"])
        return original_read(path, **kwargs)

    def rgb(self, row, frames):
        calls.append(frames.tolist())
        return torch.zeros(len(frames), 2, 3, 256, 256, dtype=torch.uint8)

    monkeypatch.setattr(pq, "read_table", read_projected)
    monkeypatch.setattr(EWMDataset, "_rgb", rgb)
    monkeypatch.setattr(EWMDataset, "_direct_event_frames", forbidden)
    dataset = PiperActionOnlyDataset(tmp_path, gpu_voxelization=True, event_frame_cache_dir="/no/cache")
    item = dataset[1]
    assert set(item) == {"obs_video", "proprio", "action", "prompt"}
    assert calls == [[1]]
    assert columns == [["observation.state", "action", "task_index"]]
    torch.testing.assert_close(item["action"], torch.from_numpy(raw_actions[1:4]))
    assert not init_options[0]["gpu_voxelization"] and init_options[0]["event_frame_cache_dir"] is None
    with pytest.raises(IndexError):
        dataset[3]
    _read_action_table.cache_clear()


def test_action_only_preset_and_training_smoke(tmp_path, monkeypatch):
    from openwam.dataloader.registry import build_dataset

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        preset = compose(config_name="train_ewm_action_only_piper")
    assert preset.model.training_mode == "action_only" and preset.model.event_vae_path is None
    assert preset.dataloader.type == "ewm_piper_action_only"
    assert not preset.dataloader.gpu_voxelization and not preset.training.pure_noise_eval.enabled
    assert preset.dataloader.chunk_size == 24
    require_path = Path(preset.dataloader.dataset_dir)
    if require_path.is_dir():
        dataset = build_dataset(preset.dataloader)
        real_item = dataset[0]
        assert set(real_item) == {"obs_video", "proprio", "action", "prompt"}
        assert real_item["action"].shape == (24, 7)

    class TinyActionDataset(PiperActionOnlyDataset):
        gpu_voxelization = False
        normalizer = SimpleNamespace(count_clip=15, metadata={"count_clip": 15, "voxel_bins": 1})

        def __init__(self):
            self.inputs = batch()

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {k: v[index % 2] for k, v in self.inputs.items()}

    cfg = preset.copy()
    cfg.training.update({"output_path": str(tmp_path / "train"), "max_steps": 2,
                         "batch_size": 2, "dataset_num_workers": 0, "mixed_precision": "no",
                         "log_every": 1, "diagnostic_every": 2, "compile_core": False,
                         "save_checkpoints": False, "save_full_states_for_resume": False,
                         "save_steps": 0})
    cfg.project.wandb.project = None
    model = make_model(gradient_checkpointing=True)
    before = model.action_head[-1].weight.detach().clone()
    train(cfg, dataset=TinyActionDataset(), model=model,
          accelerator=Accelerator(cpu=True, mixed_precision="no"))
    assert not torch.equal(before, model.action_head[-1].weight)
    output = Path(cfg.training.output_path)
    assert not list(output.glob("*.pt")) and not (output / "event_metrics.csv").exists()
    with (output / "action_metrics.csv").open() as handle:
        diagnostics = list(csv.DictReader(handle))
    assert set(diagnostics[0]) == {"step", "loss_action", "action_x0_mse", "action_x0_mae"}
    with (output / "debug_loss_history.csv").open() as handle:
        history = list(csv.DictReader(handle))
    assert len(history) == 2 and "loss_event" not in history[0]
    assert all(float(row["loss_video"]) == 0 for row in history)
