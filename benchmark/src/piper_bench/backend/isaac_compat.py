"""Small compatibility shims for the Isaac Sim 6 migration."""

from __future__ import annotations

import sys
import types


try:
    from isaaclab.renderers.output_contract import RenderBufferKind

    HDR_DATA_TYPE = str(RenderBufferKind.RGB_HDR)
except ImportError:
    HDR_DATA_TYPE = "HdrColor"


def quat_from_wxyz(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert legacy WXYZ constants when running the XYZW Isaac Lab API."""
    if HDR_DATA_TYPE == "HdrColor":
        return quaternion
    w, x, y, z = quaternion
    return (x, y, z, w)


def migrate_piper_stack_cfg(cfg) -> None:
    """Migrate legacy Piper scene constants without editing its submodule."""
    if HDR_DATA_TYPE == "HdrColor":
        return
    cfg.scene.table.init_state.rot = quat_from_wxyz((0.707, 0.0, 0.0, 0.707))
    for cube_name in ("cube_1", "cube_2", "cube_3"):
        getattr(cfg.scene, cube_name).init_state.rot = quat_from_wxyz((1.0, 0.0, 0.0, 0.0))
    cfg.xr.anchor_rot = quat_from_wxyz((0.866, 0.0, 0.0, -0.5))


def alias_legacy_hdr_output(scene, camera_names: tuple[str, ...]) -> None:
    """Expose Isaac Lab 3's ``rgb_hdr`` buffer under EVIS's legacy key."""
    if HDR_DATA_TYPE == "HdrColor":
        return
    for name in camera_names:
        output = scene[name].data.output
        output["HdrColor"] = output[HDR_DATA_TYPE]


def install_legacy_extension_utils() -> None:
    """Provide Isaac Sim 5 / Isaac Lab 2 APIs used by the Piper task."""
    from isaaclab.sim import SimulationCfg

    if not hasattr(SimulationCfg, "physx"):
        from isaaclab_physx.physics import PhysxCfg

        def get_physx(cfg: SimulationCfg) -> PhysxCfg:
            if cfg.physics is None:
                cfg.physics = PhysxCfg()
            return cfg.physics

        def set_physx(cfg: SimulationCfg, value: PhysxCfg) -> None:
            cfg.physics = value

        SimulationCfg.physx = property(get_physx, set_physx)

    try:
        from isaacsim.core.utils.extensions import enable_extension as _  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    import isaacsim
    import omni.kit.app

    core_module = sys.modules.setdefault("isaacsim.core", types.ModuleType("isaacsim.core"))
    utils_module = sys.modules.setdefault("isaacsim.core.utils", types.ModuleType("isaacsim.core.utils"))
    extensions_module = types.ModuleType("isaacsim.core.utils.extensions")

    def enable_extension(extension_name: str) -> None:
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate(extension_name, True)

    extensions_module.enable_extension = enable_extension
    sys.modules["isaacsim.core.utils.extensions"] = extensions_module
    isaacsim.core = core_module
    core_module.utils = utils_module
    utils_module.extensions = extensions_module
