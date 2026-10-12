"""Reproducible episode-constant belt speeds, independent of scene RNG."""
import hashlib
import math
import random
from dataclasses import replace


def uniform_speed_spec(bounds):
    if bounds is None:
        return None
    low, high = map(float, bounds)
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high <= .2):
        raise ValueError('speed range must satisfy 0 < min < max <= 0.2 m/s')
    return dict(distribution='uniform', min_m_s=low, max_m_s=high,
                sampling='episode_constant', sampler='sha256_python_random_v1')


def sample_episode_speed(spec, seed, episode):
    if spec != uniform_speed_spec([spec['min_m_s'], spec['max_m_s']]):
        raise ValueError('unsupported speed sampling contract')
    if episode < 0:
        raise ValueError('episode must be nonnegative')
    key=f'piper-belt-speed-v1:{int(seed)}:{int(episode)}'.encode()
    rng=random.Random(int.from_bytes(hashlib.sha256(key).digest(), 'big'))
    return rng.uniform(spec['min_m_s'], spec['max_m_s'])


def set_episode_speed(runtime, speed):
    """Apply speed before spawning; update physics and oracle metadata together."""
    if runtime.control_step != 0 or any(runtime.active):
        raise ValueError('belt speed can only change before episode supply')
    cfg=replace(runtime.cfg, speed=float(speed))
    from pxr import Gf, PhysxSchema
    import omni.usd
    stage=omni.usd.get_context().get_stage()
    paths=[f'{path}/Belt' for path in runtime.env.scene.env_prim_paths]
    prims=[stage.GetPrimAtPath(path) for path in paths]
    if not prims or not all(prim.IsValid() for prim in prims):
        raise RuntimeError('missing conveyor belt prim')
    for prim in prims:
        surface=PhysxSchema.PhysxSurfaceVelocityAPI(prim)
        if not surface.GetSurfaceVelocityEnabledAttr().Get():
            raise RuntimeError('belt surface velocity is not enabled')
        surface.GetSurfaceVelocityAttr().Set(Gf.Vec3f(0., cfg.speed, 0.))
        actual=surface.GetSurfaceVelocityAttr().Get()
        if not math.isclose(float(actual[1]), cfg.speed, rel_tol=1e-6, abs_tol=1e-8):
            raise RuntimeError('belt surface velocity update failed')
    runtime.cfg=cfg
    runtime.reset(runtime.seed)
    return cfg
