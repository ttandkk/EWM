"""Reuse Sektion's top drawer/handle USD references inside a tabletop housing.

Dimensions preserve clearance for the existing cup inside the smaller housing.
The prismatic joint is passive: the robot must physically pull and push it.
"""


def spawn_drawer(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from pxr import Gf
    from pxr import Sdf
    from pxr import UsdGeom
    from pxr import UsdPhysics
    from pxr import UsdShade
    from pxr import PhysxSchema
    from piper_bench.tasks.drawer_cup.config import DrawerCupConfig
    c = DrawerCupConfig(**spawn_cfg.drawer_config)
    prim = sim.spawn_cuboid(prim_path, spawn_cfg, translation, orientation, **kwargs)
    stage, base = prim.GetStage(), str(prim.GetPath())
    stage.RemovePrim(base+'/geometry')
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    art = PhysxSchema.PhysxArticulationAPI.Apply(prim)
    art.CreateEnabledSelfCollisionsAttr(False)
    art.CreateSolverPositionIterationCountAttr(32)
    art.CreateSolverVelocityIterationCountAttr(8)
    asset = ISAAC_NUCLEUS_DIR+'/Props/Sektion_Cabinet/'
    shell = UsdGeom.Xform.Define(stage, base+'/housing').GetPrim()
    drawer = UsdGeom.Xform.Define(stage, base+'/drawer').GetPrim()
    for body, mass in ((shell, 2.), (drawer, .18)):
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr(mass)
    # References retain the original materials, open cavity and convex decomposition.
    for name, source in (('tray', 'drawer_top'), ('handle', 'drawer_handle_top')):
        for kind in ('visuals', 'collisions'):
            ref = UsdGeom.Xform.Define(stage, base+f'/drawer/{name}_{kind}')
            ref.GetPrim().GetReferences().AddReference(asset+f'sektion_cabinet_{kind}.usd', f'/{source}_{kind}')
            if name == 'handle':
                ref.AddTranslateOp().Set(Gf.Vec3d(c.handle_extension, 0., 0.))
                if kind == 'collisions':
                    # A single convex hull fills the handle's opening. Preserve
                    # its concavity so a fingertip can hook behind the bar.
                    mesh = stage.GetPrimAtPath(str(ref.GetPath())+'/collisions')
                    UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr('convexDecomposition')
            ref.AddScaleOp().Set(Gf.Vec3d(c.scale_x, c.scale_y, c.scale_z))
    # Retain the existing handle, extending its mounts so open fingertips clear
    # the front panel instead of catching its top edge during the approach.
    for i, sign in enumerate((-1, 1)):
        sim.spawn_cuboid(base+f'/drawer/handle_mount_{i}', sim.CuboidCfg(
            size=(c.handle_extension+.004, .008, .008),
            collision_props=sim.CollisionPropertiesCfg(contact_offset=.0005, rest_offset=0.),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.45, .45, .44), roughness=.4, metallic=.5)),
            translation=(.273*c.scale_x+c.handle_extension/2, sign*.07*c.scale_y, .009*c.scale_z))
    material = sim.PreviewSurfaceCfg(diffuse_color=(.35, .32, .28), roughness=.6)
    pieces = [((-.004, 0., -.071), (.246, .296, .010)),
              ((-.004, 0., .075), (.246, .296, .010)),
              ((-.004, -.143, .002), (.246, .010, .136)),
              ((-.004, .143, .002), (.246, .010, .136)),
              ((-.124, 0., .002), (.008, .276, .136))]
    for i, (pos, size) in enumerate(pieces):
        ratios = (c.scale_x/.45, c.scale_y/.38, c.scale_z)
        pos = tuple(p*r for p, r in zip(pos, ratios))
        size = tuple(d*r for d, r in zip(size, ratios))
        sim.spawn_cuboid(base+f'/housing/panel_{i}', sim.CuboidCfg(size=size,
            collision_props=sim.CollisionPropertiesCfg(contact_offset=.0005, rest_offset=0.),
            visual_material=material), translation=pos)
    # The housing is fixed to the table. Only reset writes the drawer joint state.
    fixed = UsdPhysics.FixedJoint.Define(stage, base+'/fixed_base')
    fixed.CreateBody1Rel().SetTargets([shell.GetPath()])
    joint = UsdPhysics.PrismaticJoint.Define(stage, base+'/drawer_joint')
    joint.CreateBody0Rel().SetTargets([shell.GetPath()])
    joint.CreateBody1Rel().SetTargets([drawer.GetPath()])
    joint.CreateAxisAttr('X')
    joint.CreateLowerLimitAttr(0.)
    joint.CreateUpperLimitAttr(c.travel)
    for key in ('LocalPos0', 'LocalPos1'):
        getattr(joint, 'Create'+key+'Attr')().Set(Gf.Vec3f(0.))
    for key in ('LocalRot0', 'LocalRot1'):
        getattr(joint, 'Create'+key+'Attr')().Set(Gf.Quatf(1.))
    pj = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
    pj.CreateJointFrictionAttr(.12)
    pj.CreateMaxJointVelocityAttr(.12)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
    drive.CreateStiffnessAttr(0.)
    drive.CreateDampingAttr(2.)
    drive.CreateMaxForceAttr(15.)
    return prim
