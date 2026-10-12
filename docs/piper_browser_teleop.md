# Piper headless browser teleoperation

The separate collector is
`Event-DataCollection/scripts/collect_piper_stack_teleop.py`. It reuses the
existing Piper scene, two cameras, EVIS event-frame timing, and LeRobot writer.
The automatic oracle collector now shares the same default wrist camera mount;
its control trajectory is unchanged. No old sparse-event conversion is involved.

## Launch

For a one-command Pro 6000 batch launch, run from the login node:

```bash
bash /projects/hdd/myproject/CVPR2027/scripts/start_piper_teleop_pro6000.sh
```

The command prints a Slurm job ID. Check its node and read the private browser
URL from the log after Isaac Sim starts:

```bash
squeue -j <JOB_ID> -o '%.18i %.8T %.10M %.20R'
tail -f /projects/hdd/myproject/CVPR2027/logs/piper_teleop_pro6000_<JOB_ID>.log
```

The data is written to
`datasets/piper_stack_teleop_pathtracing_pro6000/job<JOB_ID>` with one successful episode,
2400 maximum steps and 64-SPP path tracing. The global camera now looks back
from the front of the table toward the robot: world eye `(1.25, 0, 0.62)` m,
target `(0.24, 0, 0.16)` m, focal length 30 cm. This tighter view removes the
white strip at the top and makes the cubes larger. The Pro 6000 launcher also pins
the wrist camera preset: eye `(-0.12, 0, -0.07)` m,
target `(0, 0, 0.17)` m in TCP coordinates, with an 8.5 cm focal length
(Isaac Lab camera units). This is intended to place the gripper nearer the
image bottom and widen the view.
The log contains a bearer token
and is owner-readable only; do not share it. On Windows, create the tunnel in
a separate PowerShell window, replacing `<NODE>` with the allocated node:

```powershell
ssh -N -o ExitOnForwardFailure=yes -J haotian012@10.97.216.128 -L 127.0.0.1:28765:127.0.0.1:8765 haotian012@<NODE>
```

Open the logged URL in your browser, changing only its local port from `8765`
to `28765`. Keep the tunnel running while teleoperating. A queued job cannot
serve the page; wait for `RUNNING` and the URL in the log.

For a manual launch, reserve a GPU node first. On that node:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=/projects/hdd/myproject/CVPR2027/isaacsim601/cache
export OMNI_USER_CONFIG_PATH=/projects/hdd/myproject/CVPR2027/isaacsim601/config
cd /projects/hdd/myproject/CVPR2027/Event-DataCollection
/projects/hdd/myproject/CVPR2027/isaacsim601/env/bin/python \
  scripts/collect_piper_stack_teleop.py \
  --output /projects/hdd/myproject/CVPR2027/datasets/piper_stack_teleop_pathtracing_v1 \
  --episodes 1 --max_steps 2400
```

The script prints a private URL containing a random token. The server binds
`127.0.0.1:8765` by default. Forward that port to your laptop through the
cluster's approved SSH route; if login-node forwarding cannot reach a
compute-node loopback port, use `--teleop-host <compute-node-interface>` with
an SSH tunnel to that interface. Do not publish the port or token publicly.
Only one browser operator can control a session.

The default renderer now matches the proven automatic collector: path tracing
at 64 SPP. The former teleop-only realtime/AA-off override caused full-frame
RGB speckle and false event activity. `--renderer_mode realtime` remains an
explicit diagnostic option, **not** a source for production event-frame data.
Path tracing is stable but may increase the delay between command and preview.
Use a fresh output directory when changing renderer or action-frame semantics;
`meta/teleop_capture.json` records both.

The global and wrist cameras now share one calibration across teleoperation,
automatic collection, and EWM simulation evaluation
(`piper_stack_front_global_tight_wrist_low_v3`). The front global view and wrist
view were visually checked on a Pro 6000; the static probe PNGs are under
`tmp/piper_wrist_calibration/probe_tight_v5/`. A complete automatic episode
with the tighter front view succeeded on its first attempt (454 frames). Its
four-panel cam0 RGB/cam0 event/wrist RGB/wrist event video is
`tmp/piper_wrist_calibration/auto_tight_v5_dual_event_h264.mp4`, and the
four-frame contact sheet is the adjacent `.png`. The video was fully decoded
after writing (H.264/YUV420P, 1024x256, 30 FPS); sampled frames show no white
top strip, larger cubes, and the arm and stacking workspace inside the global
view throughout. The event panels multiply raw OFF/ON counts by 17 for viewing
only; the stored event-frame data are unchanged. Earlier previews use the old
global framing. Do not mix new
episodes with old-camera training data
unless the camera distribution change is intentional. The automatic collector
records its full camera calibration in
`meta/piper_camera_calibration.json` and rejects `--resume` on a dataset with
a different or missing calibration. To save teleop static probe PNGs without
creating a dataset, add `--startup-smoke-test --camera-probe-output
/projects/hdd/myproject/CVPR2027/tmp/piper_wrist_probe` to a manual launch.
The eye, target and focal length can be tuned with `--wrist-eye-local X Y Z`,
`--wrist-target-local X Y Z`, and `--wrist-focal-length CM`; the selected
values are stored in `meta/teleop_capture.json` for every dataset.

To test scene/camera/EVIS initialization without creating a dataset, add
`--startup-smoke-test`. The browser page shows the table and wrist cameras.
W/S and A/D translate along the current wrist image's up/down and left/right
directions projected onto the tabletop; Q/E changes world height. As the wrist
turns, the image-aligned translation axes turn with it. Z/X, T/G and C/V still
rotate about world axes; K toggles the gripper. The click-to-jog buttons execute
1, 4 or 8 bounded control steps, then pause automatically. They are useful when
network and rendering delay makes continuous key holds hard to stop accurately.
The buttons pause, resume, reset, discard or save. The page pauses on focus
loss. The server also freezes simulation if input heartbeat expires or the
browser disconnects. Press Continue explicitly after reconnecting.

The Save button accepts only episodes with the task success predicate true
for 10 consecutive simulated steps and at least 25 frames (one EWM K=24
training window). It then writes the buffered aligned rows
through the existing LeRobot writer and resets. Reset and Discard never save.
The row contract remains: RGB, state and event-frame are observed at `t_i`;
the stored action is executed over `(t_i,t_(i+1)]`.
