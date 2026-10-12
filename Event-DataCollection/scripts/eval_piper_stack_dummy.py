#!/usr/bin/env python3
"""Evaluate an i.i.d. random-action baseline in the Piper cube-stack scene.

This is intentionally independent of the dataset collector: it creates no HDF5,
event stream, preview video, or project log artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trials", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=600)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--translation_scale", type=float, default=0.04)
parser.add_argument("--rotation_scale", type=float, default=0.20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.utils.math import quat_apply

from _isaacsim6_compat import HDR_DATA_TYPE, install_legacy_extension_utils, migrate_piper_stack_cfg, quat_from_wxyz
from piper_wrist_camera import (
    CAM0_EYE_WORLD,
    CAM0_TARGET_WORLD,
    CAM0_FOCAL_LENGTH_CM,
    CAM0_HORIZONTAL_APERTURE_CM,
    WRIST_EYE_LOCAL,
    WRIST_TARGET_LOCAL,
    WRIST_FOCAL_LENGTH_CM,
    WRIST_HORIZONTAL_APERTURE_CM,
)

install_legacy_extension_utils()
import agx_teleop.tasks  # noqa: F401


TASK = "Isaac-Stack-Cube-Piper-IK-Rel-v0"
CONTROL_HZ = 30
RESET_WARMUP_STEPS = 60


def external_camera() -> CameraCfg:
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam0", update_period=1.0 / CONTROL_HZ,
        height=256, width=256,
        data_types=["rgb", "distance_to_image_plane", "motion_vectors", HDR_DATA_TYPE],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAM0_FOCAL_LENGTH_CM, focus_distance=400.0,
            horizontal_aperture=CAM0_HORIZONTAL_APERTURE_CM, clipping_range=(0.1, 20.0)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -0.55, 0.75), rot=quat_from_wxyz((0.7071, 0.0, 0.7071, 0.0)), convention="world"
        ),
    )


def wrist_camera() -> CameraCfg:
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_wrist", update_period=1.0 / CONTROL_HZ,
        height=256, width=256,
        data_types=["rgb", "distance_to_image_plane", "motion_vectors", HDR_DATA_TYPE],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=WRIST_FOCAL_LENGTH_CM, focus_distance=400.0,
            horizontal_aperture=WRIST_HORIZONTAL_APERTURE_CM, clipping_range=(0.03, 20.0)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), rot=quat_from_wxyz((1.0, 0.0, 0.0, 0.0)), convention="world"
        ),
    )


def set_views(env: gym.Env) -> None:
    eye = torch.tensor([CAM0_EYE_WORLD], dtype=torch.float32, device=env.device)
    target = torch.tensor([CAM0_TARGET_WORLD], dtype=torch.float32, device=env.device)
    env.scene["cam0"].set_world_poses_from_view(eye, target)
    tcp_pos = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
    tcp_quat = env.scene["ee_frame"].data.target_quat_w[:, 0, :]
    eye_local = torch.tensor(WRIST_EYE_LOCAL, dtype=torch.float32, device=env.device).expand_as(tcp_pos)
    target_local = torch.tensor(WRIST_TARGET_LOCAL, dtype=torch.float32, device=env.device).expand_as(tcp_pos)
    env.scene["cam_wrist"].set_world_poses_from_view(
        tcp_pos + quat_apply(tcp_quat, eye_local), tcp_pos + quat_apply(tcp_quat, target_local)
    )


def configure_env():
    cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1, use_fabric=True)
    migrate_piper_stack_cfg(cfg)
    cfg.events.init_piper_arm_pose = None
    cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = 1.0 / 120.0, 4, 4
    cfg.episode_length_s = float(args_cli.max_steps) / CONTROL_HZ
    cfg.scene.cam0, cfg.scene.cam_wrist = external_camera(), wrist_camera()
    cfg.scene.light.spawn.color, cfg.scene.light.spawn.intensity = (1.0, 1.0, 1.0), 6000.0
    cfg.scene.table.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.16, 0.16), roughness=1.0, metallic=0.0)
    cfg.scene.table.spawn.visual_material_path = "visual_material_clean_evis"
    cfg.scene.clean_tabletop = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CleanTabletop",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0001)),
        spawn=sim_utils.CuboidCfg(size=(1.20, 0.90, 0.0002), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.16, 0.16), roughness=1.0, metallic=0.0)),
    )
    success = cfg.terminations.success
    cfg.terminations.success, cfg.terminations.time_out = None, None
    return cfg, success


def add_cube_patterns(env: gym.Env) -> None:
    from pxr import Sdf, UsdShade

    texture_dir = Path(__file__).resolve().parent.parent / "assets" / "piper_cube_patterns"
    stage = sim_utils.get_current_stage()
    for cube, texture_name in {"cube_1": "red_stripes.png", "cube_2": "blue_cross.png", "cube_3": "green_checker.png"}.items():
        material_path = f"/World/Looks/EvisCubePatterns/{cube}"
        material = UsdShade.Material.Define(stage, material_path)
        surface = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.72)
        reader = UsdShade.Shader.Define(stage, f"{material_path}/stReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        texture = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture_dir / texture_name)))
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        root = env.scene[cube].root_physx_view.prim_paths[0]
        binding = UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(root))
        binding.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)


def main() -> None:
    torch.manual_seed(args_cli.seed)
    cfg, success_term = configure_env()
    env = gym.make(TASK, cfg=cfg).unwrapped
    add_cube_patterns(env)
    zero = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=env.device)
    scales = torch.tensor([[args_cli.translation_scale] * 3 + [args_cli.rotation_scale] * 3], device=env.device)
    successes, lengths = 0, []
    try:
        for trial in range(args_cli.trials):
            env.sim.reset(); env.reset(); set_views(env)
            for _ in range(RESET_WARMUP_STEPS):
                set_views(env); env.step(zero)
            stable_success = 0
            for step in range(args_cli.max_steps):
                set_views(env)
                arm = (2 * torch.rand((1, 6), device=env.device) - 1) * scales
                grip = torch.where(torch.rand((1, 1), device=env.device) > 0.5, 1.0, -1.0)
                _, _, terminated, truncated, _ = env.step(torch.cat((arm, grip), dim=-1))
                stable_success = stable_success + 1 if bool(success_term.func(env, **success_term.params)[0]) else 0
                if stable_success >= 10 or bool(torch.logical_or(terminated, truncated).any()):
                    break
            success = stable_success >= 10
            successes += int(success); lengths.append(step + 1)
            print(f"[dummy-eval] trial={trial + 1}/{args_cli.trials} success={success} steps={step + 1}", flush=True)
        print(f"[dummy-eval] success_rate={successes / args_cli.trials:.3f} ({successes}/{args_cli.trials}) mean_steps={np.mean(lengths):.1f}", flush=True)
    finally:
        env.close(); simulation_app.close()

if __name__ == "__main__":
    main()
