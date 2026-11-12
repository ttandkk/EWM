"""Code-native open cardboard carton and raised conveyor side frames."""


def spawn_carton(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim
    from pxr import Gf
    from pxr import UsdGeom
    from pxr import UsdPhysics
    from pxr import UsdShade
    from pxr import PhysxSchema
    from piper_bench.tasks.conveyor.core import ConveyorConfig
    cfg = ConveyorConfig(**spawn_cfg.basket_config)
    stage = sim.get_current_stage()
    root = UsdGeom.Xform.Define(stage,prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.,0.,0.))))
    if orientation is not None:
        x,y,z,w = orientation
        root.AddOrientOp().Set(Gf.Quatf(w,Gf.Vec3f(x,y,z)))
    materials = {}
    for name,color in [('cardboard',(.48,.32,.16)),('cut_edge',(.63,.46,.27)),('seam',(.34,.22,.11))]:
        path=prim_path+'/'+name
        material=sim.PreviewSurfaceCfg(diffuse_color=color,roughness=.95,metallic=0.)
        material.func(path,material)
        materials[name]=UsdShade.Material(stage.GetPrimAtPath(path))

    def piece(name,position,size,material='cardboard',collision=False):
        cube=UsdGeom.Cube.Define(stage,prim_path+'/'+name)
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*position))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(materials[material])
        if collision:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
            api=PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            api.CreateContactOffsetAttr(.001)
            api.CreateRestOffsetAttr(0.)

    x,y,t,h=cfg.basket_length,cfg.basket_width,cfg.wall_thickness,cfg.basket_height
    piece('bottom',(0.,0.,t/2),(x+2*t,y+2*t,t),collision=True)
    for sign,label in [(-1,'minus'),(1,'plus')]:
        piece('wall_x_'+label,(sign*(x+t)/2,0.,t+h/2),(t,y+2*t,h),collision=True)
        piece('wall_y_'+label,(0.,sign*(y+t)/2,t+h/2),(x,t,h),collision=True)
        # Thin, exposed paper edges and corner seams; no extra collision surfaces.
        piece('edge_x_'+label,(sign*(x+t)/2,0.,t+h+.0001),(t,y+2*t,.0002),'cut_edge')
        piece('edge_y_'+label,(0.,sign*(y+t)/2,t+h+.0001),(x,t,.0002),'cut_edge')
        for j,other in enumerate((-1,1)):
            piece(f'fold_{label}_{j}',(sign*(x/2+t+.0001),other*(y/2-.001),t+h/2),
                  (.0002,.0005,h-.001),'seam')
    piece('floor_join',(0.,0.,t+.0001),(.001,y,.0002),'seam')
    return root.GetPrim()


def add_conveyor_frames(scene,cfg):
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg
    rail_width=.009
    rail_height=cfg.belt_top+.008
    green=sim.PreviewSurfaceCfg(diffuse_color=(.25,.30,.23),roughness=.65,metallic=.15)
    dark=sim.PreviewSurfaceCfg(diffuse_color=(.035,.037,.034),roughness=.8,metallic=.1)
    for i,sign in enumerate((-1,1)):
        setattr(scene,f'conveyor_side_{i}',AssetBaseCfg(
            prim_path=f'{{ENV_REGEX_NS}}/ConveyorSide_{i}',
            init_state=AssetBaseCfg.InitialStateCfg(pos=(cfg.basket_x+sign*(cfg.belt_width+rail_width)/2,
                                                       cfg.basket_y,rail_height/2)),
            spawn=sim.CuboidCfg(size=(rail_width,cfg.belt_length,rail_height),
                collision_props=sim.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.),
                visual_material=green)))
        for j,end in enumerate((-1,1)):
            setattr(scene,f'conveyor_endcap_{i}_{j}',AssetBaseCfg(
                prim_path=f'{{ENV_REGEX_NS}}/ConveyorEndcap_{i}_{j}',
                init_state=AssetBaseCfg.InitialStateCfg(pos=(cfg.basket_x+sign*(cfg.belt_width/2+rail_width+.002),
                    cfg.basket_y+end*cfg.belt_length/2,cfg.belt_top/2)),
                spawn=sim.CylinderCfg(radius=cfg.belt_top/2+.002,height=.005,axis='X',visual_material=dark)))
