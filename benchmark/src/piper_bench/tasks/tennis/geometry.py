"""Open box with a horizontal pan-style front handle on the same dynamic body."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BoxPiece:
    name: str
    size: tuple
    position: tuple


def container_pieces(cfg):
    half, t, h = cfg.container_inner/2, cfg.wall, cfg.container_height
    return [BoxPiece('floor', (2*(half+t), 2*(half+t), t), (0., 0., t/2)),
            BoxPiece('left', (t, 2*(half+t), h), (-half-t/2, 0., t+h/2)),
            BoxPiece('right', (t, 2*(half+t), h), (half+t/2, 0., t+h/2)),
            BoxPiece('front', (2*half, t, h), (0., -half-t/2, t+h/2)),
            BoxPiece('back', (2*half, t, h), (0., half+t/2, t+h/2)),
            BoxPiece('handle_grip', (cfg.handle_length, cfg.handle_width, cfg.handle_thickness),
                     (-half-t-cfg.handle_length/2, 0., cfg.handle_offset[2]))]



def spawn_container(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim
    from piper_bench.tasks.tennis.config import ContainerConfig
    cfg = ContainerConfig(**spawn_cfg.tennis_config)
    prim = sim.spawn_cuboid(prim_path, spawn_cfg, translation, orientation, **kwargs)
    from pxr import PhysxSchema
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateEnableCCDAttr(True)
    base, stage = str(prim.GetPath()), prim.GetStage()
    stage.RemovePrim(base+'/geometry')
    from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg
    for piece in container_pieces(cfg):
        # No child rigid bodies: these six colliders form one free rigid object.
        collision = sim.CollisionPropertiesCfg(contact_offset=.0005, rest_offset=0.)
        if piece.name.startswith('handle_'):
            # Finite finger-pad contact resists rotation around the pinch axis.
            # A zero torsional patch behaves like a hinge under catch impacts.
            collision = PhysxCollisionPropertiesCfg(contact_offset=.0005, rest_offset=0.,
                torsional_patch_radius=.012, min_torsional_patch_radius=cfg.handle_thickness/2)
        sim.spawn_cuboid(base+'/'+piece.name, sim.CuboidCfg(size=piece.size,
            collision_props=collision,
            physics_material=sim.RigidBodyMaterialCfg(static_friction=1.1, dynamic_friction=.9, restitution=.05),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.08, .09, .10) if piece.name.startswith('handle_') else (.20, .42, .68), roughness=.75)),
            translation=piece.position)
    return prim


def spawn_ball(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim
    from pxr import PhysxSchema
    prim = sim.spawn_sphere(prim_path, spawn_cfg, translation, orientation, **kwargs)
    # The pinned IsaacLab rigid-body config omits this PhysX-specific field.
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateEnableCCDAttr(True)
    kind = getattr(spawn_cfg, 'ball_skin', 'tennis')
    if kind == 'tennis':
        from piper_bench.tasks.tennis.appearance import add_felt_visual
        add_felt_visual(prim, spawn_cfg.radius)
    else:
        from piper_bench.scenes.ball_props import attach_ball_skin
        attach_ball_skin(prim, spawn_cfg.radius, kind)
    return prim
