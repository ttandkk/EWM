"""Isaac startup and asset binding, kept separate from task definitions."""
from __future__ import annotations

import os
from pathlib import Path

from piper_bench.assets.resolver import load_manifest, verify
from piper_bench.deployment.config import load_config, runtime_environment


PIPER_BUNDLE_PATH = ("Event-DataCollection/third_party/agilex-college/isaac_sim/agx_arm_IsaacLab/"
                    "IsaacLab_Data_Collection/source/agx_teleop/agx_description/agx_arm_urdf/piper/usd/piper.usd")


def launcher_class():
    """Apply configuration before AppLauncher imports Isaac Sim itself."""
    os.environ.update(runtime_environment())
    from isaaclab.app import AppLauncher
    return AppLauncher


def launch(args):
    """Create SimulationApp before importing simulator-dependent task modules."""
    return launcher_class()(args).app


def configure_asset_paths(cfg, *, config=None):
    """Bind a downloaded robot bundle and optional official asset mirror.

    The supplied configuration is fresh for this scene; installed third-party
    packages and historical configs are never rewritten.
    """
    local = config or load_config()
    asset_root = Path(os.environ.get("PIPER_ASSET_ROOT", local["assets_dir"]))
    robot = asset_root / PIPER_BUNDLE_PATH
    configured = "PIPER_ASSET_ROOT" in os.environ or bool(local.get("profiles", {}).get("simulation"))
    if robot.exists() or configured:
        report = verify(asset_root, load_manifest())
        if not report["passed"]:
            raise ValueError("Configured asset bundle failed verification; run `ewm assets verify`")
        cfg.scene.robot.spawn.usd_path = str(robot.resolve())
    source = Path(cfg.scene.robot.spawn.usd_path)
    if source.is_file():
        from .usd_compat import compatible_piper
        cfg.scene.robot.spawn.usd_path = str(compatible_piper(source))
    mirror = os.environ.get("PIPER_NVIDIA_ASSET_ROOT")
    if mirror:
        old = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0"
        for name in ("table", "ground", "ground_plane", "cube_1", "cube_2", "cube_3"):
            item = getattr(cfg.scene, name, None)
            spawn = getattr(item, "spawn", None)
            value = getattr(spawn, "usd_path", None)
            if value and value.startswith(old + "/"):
                spawn.usd_path = str(Path(mirror).expanduser().resolve()) + value[len(old):]
    return cfg
