"""Fixed, physical scene-light OOD shared by RGB and HDR-derived EVIS."""
import math

VERSION = "piper_fixed_scene_lighting_v1"
BOUNDARIES = (0.4, 0.6, 1.0, 1.4, 1.6)
IMAGING_KEYS = (
    "/rtx/post/histogram/enabled", "/rtx/post/tonemap/op",
    "/rtx/post/tonemap/filmIso", "/rtx/post/tonemap/exposureTime",
    "/rtx/post/tonemap/fNumber", "/rtx/post/tonemap/responsivity",
    "/rtx/post/tonemap/whitepoint", "/rtx/post/tonemap/colorMode",
    "/rtx/post/tonemap/enableSrgbToGamma",
)


def spawn_light_without_wrist_shadows(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
    """Exclude end-effector shadows from RGB/HDR while retaining object shadows.

    Selected single-environment tasks opt in through their light spawner.
    The shared base scene, event threshold and event processing stay unchanged.
    """
    import isaaclab.sim as sim
    from pxr import Sdf
    from pxr import UsdLux

    light = sim.spawn_light(prim_path, spawn_cfg, translation, orientation, **kwargs)
    shadows = UsdLux.LightAPI(light).GetShadowLinkCollectionAPI()
    shadows.CreateIncludeRootAttr(True)
    shadows.CreateExcludesRel().SetTargets([
        Sdf.Path('/World/envs/env_0/Robot/'+link)
        for link in ('link6', 'gripper_base', 'gripper_link1', 'gripper_link2')])
    light.CreateAttribute('piper:shadow_profile', Sdf.ValueTypeNames.String).Set(
        'piper_end_effector_shadow_exclusion_v1')
    return light


def sample_plan(seed, trials=50):
    """Independent uniform light draws, balanced dark/bright and shuffled."""
    import numpy as np
    if seed < 0 or trials < 2 or trials % 2:
        raise ValueError("nonnegative seed and a positive even trial count required")
    rng = np.random.default_rng(np.random.SeedSequence([20261021, seed, 901]))
    rows = [dict(group=group, intensity_multiplier=float(v))
            for group, low, high in (("dark", .4, .6), ("bright", 1.4, 1.6))
            for v in rng.uniform(low, high, trials//2)]
    return dict(version=VERSION, seed=seed, trials=trials, fixed_within_episode=True,
                episodes=[dict(trial=i, **rows[j]) for i, j in enumerate(rng.permutation(trials))])


def scaled_intensities(baseline, multiplier):
    if not math.isfinite(multiplier) or multiplier <= 0 or not baseline:
        raise ValueError("positive finite light multiplier and nonempty baseline required")
    if any(not math.isfinite(v) or v < 0 for v in baseline.values()):
        raise ValueError("invalid baseline light intensity")
    return {path: value * multiplier for path, value in baseline.items()}


class FixedSceneLighting:
    """Snapshot L0 once; apply absolute L0*m before reset/settle and DVS reset.

    Only USD light intensity is authored. Light exposure/color/pose, materials,
    camera extrinsics and response settings remain fixed across conditions.
    """
    def __init__(self, stage):
        import carb
        from pxr import UsdLux
        self.stage = stage
        self.settings = carb.settings.get_settings()
        self.auto_exposure_before = self.settings.get("/rtx/post/histogram/enabled")
        self.settings.set("/rtx/post/histogram/enabled", False)
        self.imaging = self.imaging_state()
        self.baseline = {}
        self.attributes = {}
        self.other_inputs = {}
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdLux.LightAPI):
                continue
            api = UsdLux.LightAPI(prim)
            path = str(prim.GetPath())
            attr = api.GetIntensityAttr()
            self.baseline[path] = float(attr.Get())
            self.attributes[path] = attr
            self.other_inputs[path] = {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                                       if a.GetName() != "inputs:intensity"}
        scaled_intensities(self.baseline, 1.)

    def imaging_state(self):
        return {key: self.settings.get(key) for key in IMAGING_KEYS}

    def apply(self, multiplier):
        expected = scaled_intensities(self.baseline, multiplier)
        for path, value in expected.items():
            self.attributes[path].Set(value)
        self.verify(multiplier)
        return dict(version=VERSION, multiplier=multiplier, baseline=self.baseline,
                    actual={p: float(a.Get()) for p, a in self.attributes.items()},
                    imaging=self.imaging, auto_exposure_before=self.auto_exposure_before,
                    fixed_within_episode=True, rgb_postprocessing=False)

    def verify(self, multiplier):
        if self.imaging_state() != self.imaging:
            raise RuntimeError("camera response/auto exposure changed within lighting comparison")
        for path, value in scaled_intensities(self.baseline, multiplier).items():
            if not math.isclose(float(self.attributes[path].Get()), value, rel_tol=1e-6, abs_tol=1e-6):
                raise RuntimeError(f"light intensity changed: {path}")
            current = {a.GetName(): str(a.Get()) for a in self.stage.GetPrimAtPath(path).GetAttributes()
                       if a.GetName() != "inputs:intensity"}
            if current != self.other_inputs[path]:
                raise RuntimeError(f"non-intensity light attributes changed: {path}")
