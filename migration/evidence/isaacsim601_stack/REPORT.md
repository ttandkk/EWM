# Isaac Sim 6.0.1 Piper stack verification

Date: 2026-09-29 UTC

## Installed runtime

- Python: 3.12
- Isaac Sim: 6.0.1.0
- Isaac Lab: `v3.0.0-beta2.patch1` (`ffff603eafc6b74264a5261cc0183d6a65390d78`)
- Isaac Lab package: 6.1.14
- PyTorch: 2.11.0+cu128
- Torchvision: 0.26.0+cu128
- Runtime: `/projects/hdd/myproject/CVPR2027/isaacsim601/env`
- Isaac Lab checkout: `/projects/hdd/myproject/CVPR2027/isaacsim601/IsaacLab`
- Runtime disk usage: 29 GiB; Isaac Lab checkout: 111 MiB

This verification intentionally covers only `Isaac-Stack-Cube-Piper-IK-Rel-v0`.

## Compatibility changes

The top-level scripts now adapt the pinned Piper task without modifying its
third-party source:

- old extension-enable import mapped to Kit's extension manager;
- `SimulationCfg.physx` mapped to Isaac Lab 3's `SimulationCfg.physics`;
- WXYZ configuration quaternions converted to Isaac Lab 3 XYZW;
- redundant reset-time articulation-default mutation disabled;
- old `HdrColor` consumer mapped to the new linear `rgb_hdr` output;
- episode time is advanced deterministically from the 30 Hz control clock;
- the collection-loop index no longer aliases a Gym return value.

The three deterministic cube textures were generated with the repository's
`scripts/generate_cube_pattern_textures.py` utility.

## GPU results

All final jobs ran on RTX PRO 6000 Blackwell GPUs in `cluster02`.

1. Static dual-camera probe, job `176537`: `COMPLETED`, exit `0:0`.
   Four 256x256 RGB frames per camera were written; wrist mean absolute frame
   difference was 53.659697.
2. One successful RGB/depth/HDR-EVIS stack collection, job `176578`:
   `COMPLETED`, exit `0:0`, one attempt, 454 control frames.
3. One 600-step random-policy evaluation, job `176569`: `COMPLETED`, exit
   `0:0`; expected baseline result was 0/1 success.

Validated episode:

`/projects/hdd/myproject/CVPR2027/tmp/isaacsim601_stack/collection_verified/episode_00000.h5`

- file size: 327,789,879 bytes (313 MiB allocated directory size)
- SHA256: `4a6e1190309b8ac82a2bfb4b593a4af94a5f6806532b394594beb613bf28791c`
- control frames: 454 at 30 Hz
- cam0 events: 25,939,014
- cam_wrist events: 61,018,812
- event time range: 0.0041666667 to 15.1291666667 seconds
- event coordinates: x/y each cover 0 through 255
- polarities: -1 and +1
- observation times strictly increase; event times are nondecreasing
- state and depth arrays contain only finite values

The validation script is `tmp/isaacsim601_stack/validate_episode.py`.

## Rendering stability correction

The initial real-time/AA-off collection above is retained as migration
evidence, but must not be used as training data. Its RGB output contained
full-frame stochastic speckle, which the HDR event model converted into
25.9M/61.0M false events.

A four-frame no-motion probe compared mean absolute RGB frame differences:

| Renderer | cam0 | cam_wrist |
| --- | ---: | ---: |
| Realtime, AA off | 46.730 | 53.660 |
| Realtime, DLSS | 5.208 | 6.494 |
| Realtime, TAA | 59.709 | 56.720 |
| Realtime, DLAA | 3.944 | 4.351 |
| Realtime, FXAA | 58.349 | 56.048 |
| Path tracing, 64 SPP | **0.030** | **0.015** |

Path tracing at 64 SPP is now the collection script's default. The stable
one-episode rerun was Slurm job `176628`: `COMPLETED`, exit `0:0`, elapsed
`00:01:33`, one successful attempt.

Stable episode:

`/projects/hdd/myproject/CVPR2027/tmp/isaacsim601_stack/collection_stable/episode_00000.h5`

- file size: 80,117,593 bytes
- SHA256: `c3ad2f9796e14a22cf3a23a16a9d16c80257f47abe4e382da90dd1c3515ffb01`
- control frames: 453 at 30 Hz
- cam0 events: 1,205,430 (79,830/s, median 2,592/control frame)
- cam_wrist events: 1,580,070 (104,640/s, median 3,329/control frame)
- events are visually localized to moving robot and cube edges; the static
  background no longer generates full-frame activity

The preview writer now emits H.264 High, yuv420p MP4 with fast-start instead of
the less widely supported MPEG-4 Part 2 stream. Final video:

`/projects/hdd/myproject/CVPR2027/tmp/isaacsim601_stack/collection_stable/episode_00000_preview_h264.mp4`

- duration: 15.1 seconds; 453 frames at 30 FPS; 512x512
- file size: 2,167,924 bytes
- SHA256: `b68d38966694bdb998d0f6e5a1a7559ff3fccbb5bacd44fa5f0da90799fe2771`

## Re-run files

- `tmp/isaacsim601_stack/smoke_static_rgb.sbatch`
- `tmp/isaacsim601_stack/collect_one.sbatch`
- `tmp/isaacsim601_stack/probe_renderers.sbatch`
- `tmp/isaacsim601_stack/collect_one_stable.sbatch`
- `tmp/isaacsim601_stack/eval_one.sbatch`

## Known packaging notes

The stack-only environment deliberately omits Isaac Lab development, Newton,
and XR/retargeting extras that this task does not execute. Consequently `pip
check` reports those optional packages as absent. It also reports the EVIS
package's legacy `numpy<2` metadata, while Isaac Sim 6.0.1 requires NumPy
2.3.1; the tested EVIS stack path works with NumPy 2.3.1.

The repository's original `third_party/IsaacLab` checkout remains on the legacy
commit and is not used by this runtime. An interrupted v3 checkout left it
dirty; two recovery stashes named `codex-partial-checkout-*` are retained. The
verified runtime uses the clean isolated checkout listed above.
