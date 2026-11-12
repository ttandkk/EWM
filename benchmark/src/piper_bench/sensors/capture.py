"""Shared RGB/event capture and local LeRobot IPC for the conveyor task."""
import pickle
import numpy as np
from piper_bench.tasks.conveyor.core import CAMERAS
from piper_bench.sensors.event_frames import EVENT_FRAME_KEY


def configure_renderer(spp=64):
    import carb
    import omni.replicator.core as rep
    settings = carb.settings.get_settings()
    for name in ('/rtx/dldenoiser/responsiveDenoising','/rtx/post/tvNoise/enabled',
                 '/rtx/post/tvNoise/enableFilmGrain'):
        settings.set(name,False)
    rep.settings.set_render_pathtraced(samples_per_pixel=spp)
    settings.set('/rtx/pathtracing/totalSpp',spp)


def snapshot(dvs, env):
    from piper_bench.backend.isaac_compat import alias_legacy_hdr_output
    alias_legacy_hdr_output(env.scene,CAMERAS)
    # Physics stays on the CPU; event interpolation and contrast processing
    # run on the same GPU that produces the RTX images.
    return {name:tuple(None if value is None else value.to('cuda:0') for value in data)
            for name,data in dvs.snapshot().items()}


def capture_frame(runtime, action, events):
    env=runtime.env
    robot=env.scene['robot'].data
    ee=env.scene['ee_frame'].data
    def array(t): return t.detach().cpu().numpy().astype(np.float32)
    poses,velocities,_=runtime.state()
    frame={'action':array(action[0]),'observation.state':array(robot.joint_pos[0]),
           'observation.joint_vel':array(robot.joint_vel[0]),
           'observation.ee_pose':np.concatenate((array(ee.target_pos_w[0,0]),array(ee.target_quat_w[0,0]))),
           'observation.object_pose':poses.astype(np.float32).reshape(-1),
           'observation.object_velocity':velocities.astype(np.float32).reshape(-1),
           'observation.object_active':np.asarray(runtime.active,dtype=np.float32),
           'observation.sim_time':np.array([runtime.time],dtype=np.float32)}
    for name in CAMERAS:
        frame['observation.images.'+name]=env.scene[name].data.output['rgb'][0,...,:3].detach().cpu().numpy().astype(np.uint8)
        frame[EVENT_FRAME_KEY+'.'+name]=events[name]
    return frame


def send_writer_message(writer,message):
    payload=pickle.dumps(message,protocol=pickle.HIGHEST_PROTOCOL)
    writer.stdin.write(len(payload).to_bytes(8,'little'))
    writer.stdin.write(payload)


def writer_reply(writer,expected):
    response=writer.stdout.readline().decode('utf-8',errors='replace').strip()
    if not response.startswith(expected+' '):
        raise RuntimeError(f'writer expected {expected}, got {response!r}, exit={writer.poll()}')
    return int(response.split()[1])


def refresh_spawn_views(env):
    from piper_bench.tasks.conveyor.core import set_views
    set_views(env)
    env.sim.forward()
    env.sim.render()
    for name in CAMERAS:
        env.scene[name].update(0.0,force_recompute=True)
