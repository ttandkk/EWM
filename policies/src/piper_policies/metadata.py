"""Read model metadata on the model side without importing torch or simulators."""
from __future__ import annotations

import hashlib
import json
import importlib.util
import re
from pathlib import Path

from omegaconf import OmegaConf

from .contract import ACTION_CONTRACT, EVENT_CONTRACT, RGB_CONTRACT, RGB_LATENT_CONTRACT, VERSION


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_identity(family):
    """Fingerprint shipped adapter and native inference code without loading it."""
    package = Path(__file__).parent
    files = {"piper_policies/" + p.relative_to(package).as_posix(): file_sha256(p)
             for p in sorted(package.glob("*.py"))}
    for p in sorted((package / family).glob("*.py")):
        files["piper_policies/" + p.relative_to(package).as_posix()] = file_sha256(p)
    native_package = "fastwam" if family == "fastwam" else "openwam"
    spec = importlib.util.find_spec(native_package)
    root = (Path(next(iter(spec.submodule_search_locations))) if spec and spec.submodule_search_locations
            else package.parents[2] / ("FastWAM/src/fastwam" if family == "fastwam" else "OpenWAM/openwam"))
    if root.is_dir():
        patterns = {
            "ewm": ("model/architectures/ewm*.py", "model/architectures/lean*.py", "model/architectures/wan_event_codec.py",
                    "model/architectures/temporal*.py", "dataloader/ewm_normalizer.py",
                    "model/text_encoder/t5_small.py", "model/video_backbone/encoder/dinov3.py",
                    "model/action_backbone/components.py", "model/action_backbone/separate_action_dit.py",
                    "model/architectures/event_codec.py", "model/video_backbone/cosmos_predict25/scheduler.py"),
            "openwam": ("deploy/engine.py", "deploy/model_loader.py", "deploy/denoise_schedule.py",
                        "model/architectures/dual_system/*.py", "dataloader/transforms/normalize.py",
                        "model/video_backbone/wan/*.py"),
            "fastwam": ("runtime.py", "models/wan22/fastwam*.py", "models/wan22/lora.py", "models/wan22/helpers/*.py",
                        "datasets/lerobot/processors/fastwam_processor.py", "datasets/lerobot/utils/normalizer.py"),
        }[family]
        for pattern in patterns:
            for p in sorted(root.glob(pattern)):
                files[native_package + "/" + p.relative_to(root).as_posix()] = file_sha256(p)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return digest, files


def select_checkpoint(directory, family, checkpoint=None, checkpoint_step=None):
    directory = Path(directory).expanduser().resolve()
    suffix = ".safetensors" if family == "openwam" else ".pt"
    if checkpoint:
        selected = Path(checkpoint).expanduser()
        if not selected.is_absolute():
            selected = directory / selected
        candidates = [selected]
    else:
        candidates = list(directory.glob("checkpoint_step_*" + suffix))
        if family == "fastwam":
            candidates += list(directory.glob("step_*" + suffix))
            candidates += list((directory / "checkpoints/weights").glob("step_*" + suffix))
        if not candidates:
            raise FileNotFoundError(f"no numbered {family} checkpoint in {directory}; use --checkpoint")
    numbered = []
    for path in candidates:
        match = re.search(r"(?:step_|step-|checkpoint-|ckpt_)(\d+)$", path.stem)
        if match:
            numbered.append((int(match.group(1)), path))
        elif checkpoint and checkpoint_step is not None:
            numbered.append((int(checkpoint_step), path))
    if not numbered:
        raise ValueError("checkpoint filename has no step number; supply --checkpoint-step")
    if checkpoint_step is not None:
        numbered = [item for item in numbered if item[0] == checkpoint_step]
        if not numbered:
            raise ValueError("requested checkpoint-step is unavailable or filename disagrees")
    step = max(item[0] for item in numbered)
    selected_paths = {path.resolve() for number, path in numbered if number == step}
    if len(selected_paths) != 1:
        raise ValueError(f"ambiguous checkpoint filenames for step {step}")
    selected = selected_paths.pop()
    if step < 0 or not selected.is_file():
        raise FileNotFoundError(selected)
    return selected.resolve(), step


def checkpoint_file(directory, step):
    """Resolve padded and legacy unpadded EWM names without ambiguity."""
    candidates = [path for path in Path(directory).glob("checkpoint_step_*.pt")
                  if path.stem.removeprefix("checkpoint_step_").isdigit()
                  and int(path.stem.removeprefix("checkpoint_step_")) == step]
    if len(candidates) != 1:
        raise ValueError(f"requested checkpoint-step is unavailable or ambiguous: {step}")
    return candidates[0]


def validate_training_contract(cfg, *, task=None, policy_mode=None):
    """Keep EWM training-mode/task guards on the model side of the boundary."""
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    mode = str(OmegaConf.select(cfg, "model.training_mode", default="joint"))
    if policy_mode is not None and mode != policy_mode:
        raise ValueError("Checkpoint training mode differs from the requested policy")
    tasks = OmegaConf.select(cfg, "dataloader.tasks")
    if task is not None and tasks is not None and task not in tasks:
        raise ValueError("evaluation task is not present in the checkpoint training tasks")
    dtype = OmegaConf.select(cfg, "dataloader.type")
    allowed = {"action_only": {"ewm_piper_action_only", "ewm_piper_multitask_action_only"},
               "rgb_action": {"ewm_piper_rgb_action", "ewm_piper_multitask_rgb_action"}}
    if mode in allowed and dtype is not None and dtype not in allowed[mode]:
        raise ValueError("checkpoint loader differs from the RGB/state training contract")
    return mode


def load_metadata(directory, family="ewm", *, checkpoint=None, checkpoint_step=None, config=None, normalization=None):
    directory = Path(directory).expanduser().resolve()
    path, step = select_checkpoint(directory, family, checkpoint, checkpoint_step)
    config_path = Path(config).resolve() if config else directory / "config.yaml"
    if not config and not config_path.is_file() and family == "fastwam":
        config_path = directory / ".hydra/config.yaml"
    cfg = OmegaConf.load(config_path)
    default_stats = {"ewm": "normalization.json", "openwam": "normalization_stats.npy", "fastwam": "fastwam_stats.json"}
    stats_path = Path(normalization).resolve() if normalization else directory / default_stats[family]
    if not normalization and not stats_path.is_file() and family == "fastwam":
        stats_path = directory / "dataset_stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"normalization must accompany the model: {stats_path}; use --normalization")
    select = lambda key, default=None: OmegaConf.select(cfg, key, default=default)
    mode = validate_training_contract(cfg) if family == "ewm" else "rgb_action"
    if mode not in {"joint", "rgb_action", "action_only"}:
        raise ValueError(f"unsupported action policy mode {mode}")
    if family == "ewm":
        horizon = int(select("dataloader.chunk_size"))
    else:
        horizon = int(select("dataloader.num_frames" if family == "openwam" else "data.train.num_frames")) - 1
    if horizon <= 0 or int(select("dataloader.observation_row_offset", 0)) != 0:
        raise ValueError("policy needs a positive horizon and current observations")
    event_horizon = select("model.event_horizon") if family == "ewm" else None
    event_horizon = horizon if event_horizon is None else int(event_horizon)
    camera_mode = str(select("model.event_camera_mode", "dual")) if family == "ewm" else "dual"
    if camera_mode not in {"dual", "global_only", "wrist_only"}:
        raise ValueError("unsupported policy camera selection")
    stats = json.loads(stats_path.read_text()) if family == "ewm" else {}
    training_data = {}
    for task, source in stats.get("sources", {}).items():
        training_data[task] = {k: source[k] for k in ("collection_sha256", "episodes", "frames", "windows") if k in source}
        if "input_signature" in source:
            training_data[task]["legacy_input_signature"] = source["input_signature"]
    manifest = directory / "training_data_manifest.json"
    if manifest.exists():
        training_data = {"manifest": json.loads(manifest.read_text()), "manifest_sha256": file_sha256(manifest)}
    source_sha256, source_files = implementation_identity(family)
    descriptor = dict(
        version=VERSION, family=family, mode=mode, action_horizon=horizon, event_horizon=event_horizon,
        action_dim=7, state_dim=8, control_hz=30, action_contract=ACTION_CONTRACT,
        rgb_shape=[2, 3, 256, 256], event_camera_mode=camera_mode,
        event_cameras=2 if camera_mode == "dual" else 1,
        voxel_bins=int(stats.get("voxel", {}).get("num_bins", select("model.voxel_bins", 1))),
        count_clip=float(stats.get("voxel", {}).get("count_clip", 15)),
        observation_contract=RGB_LATENT_CONTRACT if mode == "rgb_action" else RGB_CONTRACT,
        event_time_contract=EVENT_CONTRACT if mode == "joint" else None,
        prediction_kind=("events" if mode == "joint" else "rgb" if mode == "rgb_action" else None) if family == "ewm" else None,
        source_sha256=source_sha256, source_files_sha256=source_files,
        checkpoint_step=step, checkpoint_sha256=file_sha256(path),
        config_sha256=file_sha256(config_path), normalization_sha256=file_sha256(stats_path),
        training_data=training_data, post_trial_visualization_supported=family == "ewm" and mode != "action_only",
        training_observation_row_offset=0, inference_observation_row_offset=0,
    )
    if family == "openwam":
        from .openwam.normalization import normalization_payload
        descriptor["normalization_adapter"] = normalization_payload(stats_path)[0]
    if family != "ewm":
        stride = int(select("dataloader.video_stride" if family == "openwam" else "data.train.action_video_freq_ratio", 1))
        if stride <= 0 or horizon % stride or (horizon // stride) % 4:
            raise ValueError("training horizon/video stride is incompatible with Wan geometry")
        descriptor["video_frames"] = horizon // stride + 1
    return cfg, path, stats_path, descriptor


def hello_metadata(descriptor, optimization=None):
    # Keep old wire clients operational; canonical clients only read descriptor.
    return {**descriptor, "policy_descriptor": descriptor, "training_mode": descriptor["mode"],
            "horizon": descriptor["action_horizon"],
            "checkpoints": [descriptor["checkpoint_step"]], "deferred_visualization_supported": False,
            **(optimization or {}), "optimization": optimization or {}}


def resolve_evaluation_metadata(checkpoint_dir, task=None, dataset_dir_override=None, **kwargs):
    """Legacy checkpoint-to-descriptor bridge, without opening training datasets."""
    cfg, _, _, descriptor = load_metadata(checkpoint_dir, **kwargs)
    if descriptor["family"] == "ewm":
        validate_training_contract(cfg, task=task)
    return {"policy_descriptor": descriptor, "task": task,
            "dataset_dir": str(Path(dataset_dir_override).resolve()) if dataset_dir_override else None}
