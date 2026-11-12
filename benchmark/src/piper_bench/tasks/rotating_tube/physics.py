"""Fast contact/expert debugging without rendering; does not collect training data."""
import argparse
import json
import traceback
from pathlib import Path
from piper_bench.backend.isaac import launcher_class, launch
AppLauncher = launcher_class()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--trials',type=int,default=3)
parser.add_argument('--seed',type=int,default=20261016)
AppLauncher.add_app_launcher_args(parser);parser.set_defaults(device='cpu')
args=parser.parse_args()
if args.trials<1:parser.error('trials must be positive')
if args.device!='cpu':parser.error('use --device cpu; RTX and EVIS still run on GPU')
app=launch(args)
import gymnasium as gym
import torch
from piper_bench.tasks.rotating_tube.scene import TASK
from piper_bench.tasks.rotating_tube.scene import TubeConfig
from piper_bench.tasks.rotating_tube.scene import TubeRuntime
from piper_bench.tasks.rotating_tube.scene import make_env_cfg
from piper_bench.tasks.rotating_tube.scene import insertion_metrics
from piper_bench.tasks.rotating_tube.oracle import TubeOracle

def main():
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    cfg=TubeConfig();envcfg=make_env_cfg(cfg,device=args.device,seed=args.seed)
    envcfg.scene.cam0=None;envcfg.scene.cam_wrist=None;envcfg.sim.render_interval=10000000
    env=gym.make(TASK,cfg=envcfg).unwrapped;results=[]
    try:
        runtime=TubeRuntime(env,cfg)
        zero=torch.tensor([[0.,0.,0.,0.,0.,0.,1.]],device=env.device)
        for trial in range(args.trials):
            runtime.motion_enabled=False;env.reset()
            for _ in range(60):env.step(zero)
            runtime.reset(args.seed+trial);runtime.supply();env.sim.forward()
            oracle=TubeOracle(runtime);rows=[]
            for step in range(900):
                action=oracle.action() if step else zero;env.step(action)
                joints=env.scene['robot'].data.joint_pos[0].detach().cpu().numpy()
                success,reason=runtime.after_step(abs(joints[-2]-joints[-1])>.045)
                poses,vel,tcp=runtime.state()
                row=dict(time=runtime.time,stage=oracle.stage,poses=poses.tolist(),velocities=vel.tolist(),
                         tcp=tcp.tolist(),joints=joints.tolist(),hole=runtime.hole_target()[0].tolist(),
                         tcp_quaternion=env.scene['ee_frame'].data.target_quat_w[0,0].detach().cpu().tolist(),
                         target=None if oracle.target is None else oracle.target.tolist(),
                         metrics=insertion_metrics(poses[0],vel[0],tcp,cfg,runtime.hole_target(),abs(joints[-2]-joints[-1])>.045),
                         success=bool(success),reason=reason)
                rows.append(row)
                if step%30==0 or success or reason:
                    (out/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
                    print(f'[fast] trial={trial} step={step+1} stage={oracle.stage} tube={poses[0,:3].round(4).tolist()} '
                          f'tcp={tcp.round(3).tolist()} tilt={row["metrics"]["tilt_deg"]:.1f} success={success}',flush=True)
                if success or reason:break
            (out/f'trial_{trial:03d}.json').write_text(json.dumps(rows,indent=2)+'\n')
            results.append(dict(seed=args.seed+trial,passed=bool(success and oracle.grasp_lifted),
                                final_stage=oracle.stage,retries=oracle.retries,last=rows[-1]))
        (out/'complete.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in results),results=results),indent=2)+'\n')
        if not all(r['passed'] for r in results):raise RuntimeError('some physical oracle trials failed')
    finally:env.close()
try:main()
except Exception:traceback.print_exc();raise
finally:app.close()
