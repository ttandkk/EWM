"""A circular surface on the existing Piper conveyor scene; objects remain dynamic."""
from dataclasses import dataclass
import math
import numpy as np

from piper_bench.tasks.conveyor.core import ConveyorConfig
from piper_bench.tasks.conveyor.core import ConveyorRuntime
from piper_bench.tasks.conveyor.core import CONTROL_HZ
from piper_bench.tasks.conveyor.core import make_env_cfg as conveyor_env_cfg
from piper_bench.tasks.conveyor.core import camera_metadata as conveyor_camera_metadata
from piper_bench.tasks.conveyor.core import object_in_basket
from piper_bench.tasks.conveyor.core import rotation_xyzw

TASK = "Isaac-Turntable-Pick-Piper-IK-Rel-v0"
PROMPT = "Pick up both objects on the rotating turntable and place them in the basket."


@dataclass(frozen=True)
class TurntableConfig(ConveyorConfig):
    # Preserve this task's default independently of conveyor grasp randomization.
    yaw_jitter: float = math.radians(10)
    object_size: float = .03
    disk_radius: float = .18
    orbit_radius: float = .095
    center_y: float = -.05
    direction: int = 1
    phase_jitter: float = .06
    radial_jitter: float = .003

    def __post_init__(self):
        super().__post_init__()
        values = (self.object_size, self.disk_radius, self.orbit_radius, self.center_y,
                  self.phase_jitter, self.radial_jitter)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("turntable geometry must be finite")
        if self.object_size <= 0:
            raise ValueError("object size must be positive")
        if self.direction not in (-1, 1) or min(self.phase_jitter, self.radial_jitter) < 0:
            raise ValueError("invalid direction or jitter")
        if not 0 < self.orbit_radius - self.radial_jitter:
            raise ValueError("orbit radius must be positive")
        if self.orbit_radius + self.radial_jitter + self.object_size / math.sqrt(2) + .005 >= self.disk_radius:
            raise ValueError("objects must fit completely on the disk")
        # Closest basket outer corner must remain outside the disk footprint.
        dx = max(abs(self.belt_x-self.basket_x)-self.basket_length/2-self.basket_taper, 0.)
        dy = max(abs(self.center_y-self.basket_y)-self.basket_width/2-self.basket_taper, 0.)
        if math.hypot(dx, dy) <= self.disk_radius + .005:
            raise ValueError("disk intersects the original basket")

    @property
    def omega(self):
        """Signed radians/second; PhysX surface API uses degrees/second."""
        return self.direction * self.speed / self.orbit_radius

    @property
    def center(self):
        return np.array([self.belt_x, self.center_y])

    def conveyor_config(self):
        # The unchanged basket spawner reconstructs a ConveyorConfig from metadata.
        return ConveyorConfig(**{key: getattr(self, key) for key in ConveyorConfig.__dataclass_fields__})

    def supply_positions(self, seed):
        rng = np.random.default_rng(seed)
        phase = np.array([-2.5, -.4]) + rng.uniform(-self.phase_jitter, self.phase_jitter, 2)
        radius = self.orbit_radius + rng.uniform(-self.radial_jitter, self.radial_jitter, 2)
        xy = self.center + radius[:, None]*np.stack((np.cos(phase), np.sin(phase)), axis=1)
        if np.linalg.norm(xy[0]-xy[1]) <= self.object_size*math.sqrt(2)+.01:
            raise ValueError("spawned objects overlap")
        return xy[:, 0], xy[:, 1], rng.uniform(-self.yaw_jitter, self.yaw_jitter, 2)


def predict_position(position, cfg, seconds):
    """Constant angular-speed geometric reference; only used by the oracle."""
    result = np.array(position, dtype=float, copy=True)
    angle = cfg.omega * seconds
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    result[:2] = cfg.center + rotation @ (result[:2]-cfg.center)
    return result


DISK_SEGMENTS = 512
FRONT_FOCAL_LENGTH = 22.0


def camera_metadata():
    calibration = conveyor_camera_metadata()
    calibration.update(calibration_id="piper_turntable_front_wrist_v1_wide",
                       cam0_focal_length_cm=FRONT_FOCAL_LENGTH)
    return calibration


def disk_visual_geometry(radius, height):
    """Circular rim with smooth side normals and flat caps, independent of USD tessellation."""
    angles = np.arange(DISK_SEGMENTS)*2*math.pi/DISK_SEGMENTS
    radial = np.stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=1)
    points = np.concatenate((radius*radial+[0, 0, -height/2],
                             radius*radial+[0, 0, height/2]))
    n = DISK_SEGMENTS
    faces, normals = [], []
    for i in range(n):
        j = (i+1) % n
        faces.append([i, j, n+j, n+i])
        normals.extend([radial[i], radial[j], radial[j], radial[i]])
    faces.extend([list(range(n-1, -1, -1)), list(range(n, 2*n))])
    normals.extend([[0, 0, -1]]*n+[[0, 0, 1]]*n)
    return points.tolist(), faces, np.asarray(normals).tolist()


def make_env_cfg(config=TurntableConfig(), *, device="cpu", seed=42):
    import isaaclab.sim as sim
    from isaaclab.assets import RigidObjectCfg
    from pxr import PhysxSchema
    from pxr import Gf
    from pxr import UsdGeom
    from pxr import UsdShade

    cfg = conveyor_env_cfg(config.conveyor_config(), device=device, seed=seed)
    # Fit the enlarged disk with a small margin; retain both camera poses and wrist optics.
    cfg.scene.cam0.spawn.focal_length = FRONT_FOCAL_LENGTH

    def spawn_disk(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
        prim = sim.spawn_cylinder(prim_path, spawn_cfg, translation, orientation, **kwargs)
        # Keep the original cylinder collision geometry but replace its coarse
        # rendered silhouette. A visual mesh adds no extra contact surfaces.
        stage = prim.GetStage()
        base = str(prim.GetPath())
        collider = stage.GetPrimAtPath(base+'/geometry/mesh')
        UsdGeom.Imageable(collider).CreatePurposeAttr('guide')
        mesh = UsdGeom.Mesh.Define(stage, base+'/geometry/smooth_disk')
        points, faces, normals = disk_visual_geometry(config.disk_radius, config.belt_top)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
        mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateNormalsAttr(normals)
        mesh.SetNormalsInterpolation('faceVarying')
        material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial()
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        surface = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(prim)
        surface.CreateSurfaceVelocityEnabledAttr().Set(True)
        surface.CreateSurfaceVelocityLocalSpaceAttr().Set(False)
        surface.CreateSurfaceVelocityAttr().Set(Gf.Vec3f(0., 0., 0.))
        surface.CreateSurfaceAngularVelocityAttr().Set(Gf.Vec3f(0., 0., math.degrees(config.omega)))
        return prim

    # Reuse the asset key so shared reset/inspection code still works. A circular
    # collider is rotationally invariant; contact velocity supplies its rotation.
    cfg.scene.belt = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Turntable",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(config.belt_x, config.center_y, config.belt_top/2)),
        spawn=sim.CylinderCfg(func=spawn_disk, radius=config.disk_radius, height=config.belt_top, axis="Z",
            rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0.),
            physics_material=sim.RigidBodyMaterialCfg(static_friction=.9, dynamic_friction=.8),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.10,.13,.17), roughness=1., metallic=0.)),
    )
    cfg.scene.belt_roller_0 = None
    cfg.scene.belt_roller_1 = None
    return cfg


class OffDiskGuard:
    """Conservative low/away guard, matching the mature conveyor protocol."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.counts = [0, 0]

    def update(self, poses, tcp, active):
        for i, pose in enumerate(poses):
            extent = np.abs(rotation_xyzw(pose[3:7])).sum(axis=1)*self.cfg.object_size/2
            nearest = np.maximum(np.abs(np.asarray(pose[:2])-self.cfg.center)-extent[:2], 0.)
            outside = np.linalg.norm(nearest) > self.cfg.disk_radius+.001
            low = pose[2] < self.cfg.belt_top+self.cfg.object_size/2-.005
            away = np.linalg.norm(np.asarray(pose[:3])-tcp) > .08
            failed = active[i] and outside and low and away and not object_in_basket(pose, self.cfg)
            self.counts[i] = self.counts[i]+1 if failed else 0
            if self.counts[i] >= 6:
                return f'object_{i+1}_off_disk'
        return None


class TurntableRuntime(ConveyorRuntime):
    def reset(self, seed):
        super().reset(seed)
        self.off_disk_guard = OffDiskGuard(self.cfg)

    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        from pxr import UsdGeom
        from pxr import Gf
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        # Visual-only marks rotate at the same speed as the contact surface.
        path = str(env.scene.env_prim_paths[0]) + "/TurntableMarks"
        marks = UsdGeom.Xform.Define(stage, path)
        marks.AddTranslateOp().Set(Gf.Vec3d(cfg.belt_x, cfg.center_y, cfg.belt_top+.0003))
        self.rotation_op = marks.AddRotateZOp()
        for i in range(3):
            angle = i*2*math.pi/3
            mark = UsdGeom.Cube.Define(stage, path+f"/Mark{i}")
            mark.CreateSizeAttr(1.)
            mark_radius = cfg.disk_radius-.013
            mark.AddTranslateOp().Set(Gf.Vec3d(mark_radius*math.cos(angle), mark_radius*math.sin(angle), 0.))
            mark.AddRotateZOp().Set(math.degrees(angle))
            mark.AddScaleOp().Set(Gf.Vec3f(.015, .004, .0004))
            mark.CreateDisplayColorAttr().Set([Gf.Vec3f(.35,.37,.40)])
        self.before_step()

    def before_step(self):
        self.rotation_op.Set(math.degrees(self.cfg.omega*(self.time+1/CONTROL_HZ)))

    def after_step(self):
        self.control_step += 1
        poses, velocities, tcp = self.state()
        success = self.tracker.update(poses, velocities, tcp, self.active)
        reason = None
        for i, pose in enumerate(poses):
            if self.active[i] and not object_in_basket(pose, self.cfg):
                if pose[2] < -.04:
                    reason = f"object_{i+1}_fell"
        if not success and reason is None:
            reason = self.off_disk_guard.update(poses, tcp, self.active)
        if self.time >= self.cfg.max_seconds and not success:
            reason = "timeout"
        return success, reason

    def metadata(self):
        data = super().metadata()
        data.update(task=TASK, prompt=PROMPT, angular_velocity_rad_s=self.cfg.omega,
                    surface_angular_velocity_deg_s=math.degrees(self.cfg.omega),
                    motion="physical_contact_angular_surface_velocity",
                    disk_visual_segments=DISK_SEGMENTS,
                    failure_protocol="off_disk_low_away_v1_six_ticks",
                    object_pose_override="spawn_and_reset_only")
        return data
