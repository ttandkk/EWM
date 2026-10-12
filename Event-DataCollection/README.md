# Event-DataCollection

Piper stack/drawer demonstration collection with synchronized RGB, depth, state,
action, and EVIS event-camera streams. The repository also converts collected
HDF5 episodes into a local LeRobot dataset.

## Repository policy

- Third-party source is pinned as Git submodules; do not copy or modify it in
  the top-level repository.
- Isaac Sim runtimes, datasets, logs, videos, and checkpoints are not tracked.
- Top-level files larger than 25 MiB are rejected by `tools/check_large_files.sh`.
- Collection and conversion use separate Python environments. The verified
  Isaac Sim 6.0.1 runtime and the converter both use Python 3.12, but retain
  separate dependency sets.

See `versions.lock` and `THIRD_PARTY_NOTICES.md` for exact revisions and
upstream sources.

## Clone

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none \
  --recurse-submodules --shallow-submodules --also-filter-submodules \
  <GITHUB_URL>/Event-DataCollection.git
cd Event-DataCollection
```

`GIT_LFS_SKIP_SMUDGE=1` avoids downloading LeRobot test artifacts that are not
needed by the converter.

If a shallow submodule fetch cannot reach a pinned revision, use:

```bash
git submodule update --init --recursive
```

## 1. Simulation/collection environment

Requirements: Linux x86-64, NVIDIA GPU/driver, Isaac Sim 6.0.1, and Python
3.12. The verified project-local runtime is:

```bash
source /projects/hdd/myproject/CVPR2027/isaacsim601/env/bin/activate
python tools/check_install.py --mode sim
```

The optional full renderer smoke test writes only to a temporary directory:

```bash
bash tools/smoke_test.sh
```

### Collect stack episodes

```bash
python scripts/collect_piper_stack_evis.py \
  --episodes 50 \
  --output datasets/piper_stack_eventframe_lerobot \
  --headless --enable_cameras
```

Stack collection now writes LeRobot v3 directly. The Isaac Sim process sends
successful episodes to the existing `envs/lerobot/bin/python` environment
(`--writer-python` overrides it), so LeRobot is not installed into Isaac Sim.
Choose a new output directory; `--resume` appends only to a compatible
event-frame dataset. Each row contains pre-action RGB/state/action plus lossless
`observation.event_frame.cam0` and `observation.event_frame.cam_wrist` PNGs.
Their R/G channels are clipped OFF/ON counts for the previous 30 Hz interval
`(t_(i-1), t_i]`; B is zero. The first row's event-frame is zero. RGB videos
are H.264 MP4, and `meta/ewm_normalization_stats.json` is generated after
collection. `EWMDataset` reads these frames directly, without raw-event
voxelization; configure `observation_row_offset=0` and `voxel_bins=1`.

### Conveyor pick and basket task

`Isaac-Conveyor-Pick-Piper-IK-Rel-v0` uses the same Piper model, relative
7-D IK/gripper actions, 8-D joint state, dual 256×256 RGB cameras, and EVIS
OFF/ON event-frame contract as stack collection. Two 4 cm cubes start together
on a physical conveyor moving at 0.06 m/s; the robot places both in the compact
red basket. Initial positions and yaw are reproducible from the episode seed.
The wrist camera uses a fixed TCP mount, preserving the approved initial view
and avoiding world-up look-at flips when pointing straight down.
Only initial supply writes object poses; belt transport, gripping and release
use PhysX contacts. Physics runs on CPU, while RTX rendering and EVIS run on GPU.

From the project root, the RTX 5090 launchers are:

```bash
sinfo -o '%P %G %D %N'
sbatch scripts/test_piper_conveyor_rtx5090.sbatch --mode reach --max-steps 1050
sbatch scripts/eval_piper_conveyor_rtx5090.sbatch
sbatch scripts/collect_piper_conveyor_rtx5090.sbatch
```

The evaluation launcher runs 100 independently seeded trials and requires at
least 95% full-task success. Collection keeps 50 successful episodes from at
most 200 attempts, records rejected attempts separately, and validates every
RGB video and event payload before writing `.export_complete`. Collection
filtering is not an estimate of oracle success rate. Existing `_a40.sbatch`
launchers request A40; `sbatch --gres=gpu:a6000:1` overrides the GPU type when
A6000 is available. Slurm logs are under the project `logs/` directory.

For the randomized collection preset, pass `--yaw-jitter-deg 20 --spacing-jitter 0.01`
to both evaluation and collection. This samples each cube yaw independently in
±20°, longitudinal separation in 13–15 cm, and lateral positions in ±1 cm.
Both objects still spawn at simulation time zero. Seeds reproduce every initial
configuration. Generate 50 successful episodes sequentially on one GPU:

```bash
PIPER_CONVEYOR_EPISODES=50 \
PIPER_CONVEYOR_DATASET_ROOT=/projects/hdd/myproject/CVPR2027/datasets/piper_conveyor_front_wrist_v3_random_50 \
sbatch scripts/collect_piper_conveyor_rtx5090.sbatch \
  --seed 20272002 --yaw-jitter-deg 20 --spacing-jitter 0.01
```

`collect_piper_conveyor_pro6000.sbatch` is an alternative one-GPU launcher.
`PIPER_CONVEYOR_DATASET_ROOT` selects the parent; each new job writes its own
`job<job_id>` directory. To extend a completed 25-episode collection to 50,
set `PIPER_CONVEYOR_OUTPUT` to the existing dataset and the episode target to
50, then add `--resume` with exactly the same seed and randomization arguments.
Use `--dependency=afterok:<previous_job>` when the preceding collection is
still running. Resume continues the attempt seed sequence, invalidates old
completion/audit markers, and recomputes normalization over all episodes.

The current `simultaneous_fast_tracking_v31_random_placement` oracle passed
97/100 fresh randomized trials on RTX 5090 (job 181810, seeds
20276002–20276101), using the strict physical success predicate below.
The three failed trials remain in the evaluation report. The existing EWM
checkpoint passed the simulator interface smoke check; learned success on
this new task has not been measured.

The completed single-GPU randomized export is
`/projects/hdd/myproject/CVPR2027/datasets/piper_conveyor_front_wrist_v3_random_50/shards/job181776`.
Jobs 181776 and 181863 ran sequentially with one RTX 5090 each and saved
50 successful episodes (13,675 frames) from 52 attempts. Both existing training
loaders passed 150 sampled windows across all 50 episodes; all 100 RGB videos
decoded successfully. Per-episode physical stability and initial randomized
poses were checked from the exported data, and the first 25 episodes remained
unchanged through resume. See `meta/piper_completion_audit.json`,
`meta/piper_resume_preservation.json`, and `meta/ewm_reader_smoke.json` in the export.
Representative four-view videos for episodes 0, 24 and 49 are under
`tmp/piper_conveyor_v1/preview/random50_episode{0,24,49}_four_view_256_h264.mp4`
at the project root, with 256×256 pixels per panel.

Successful collection is written to
`datasets/piper_conveyor_front_wrist_v3_50/job<job_id>/`. To collect one preview,
set `PIPER_CONVEYOR_EPISODES=1` and optionally `PIPER_CONVEYOR_OUTPUT` before
submission. Resume with the same output path and `--resume`; the collector
rejects changed scene, camera, seed, or controller profiles.

Full success requires all eight corners of both cubes inside the tapered
basket, low linear/angular velocities, and the hand at least 10 cm away for
30 consecutive 30 Hz ticks. The simulation clock advances only with physics
steps and remains paused during policy inference. The oracle evaluation is
separate from the existing-checkpoint EWM interface smoke test; compatibility
does not establish learned-policy task success.

After validation, render real RGB and measured events in the order cam0 RGB,
cam0 events, wrist RGB, wrist events (four 256×256 panels):

```bash
/projects/hdd/myproject/CVPR2027/envs/lerobot/bin/python \
  Event-DataCollection/scripts/render_piper_lerobot_four_view.py \
  datasets/piper_conveyor_front_wrist_v3_50/job<job_id> \
  --panel-size 256 --output tmp/conveyor_preview.mp4
```

Scene configuration and strict success checks live in `scripts/piper_conveyor.py`;
feedback motion is in `scripts/piper_conveyor_oracle.py`. Geometry, motion and
event-frame tests are in `tests/test_piper_conveyor*.py` and
`tests/test_piper_event_frames.py`. Generated data and previews stay outside Git.

### Collect drawer episodes

```bash
python scripts/collect_piper_drawer_evis.py \
  --episodes 50 \
  --output datasets/piper_drawer \
  --headless --enable_cameras
```

Resume an interrupted collection by adding `--resume`. The HDF5 preview
script below applies only to legacy HDF5 collections, not direct LeRobot data.

## 2. Legacy HDF5 conversion environment

Do not install LeRobot into the Isaac Sim environment: their Python/NumPy
requirements differ. Create and activate a separate Python 3.12 environment:

```bash
python3.12 -m venv .venv-convert
source .venv-convert/bin/activate
bash tools/bootstrap_convert.sh
```

For older HDF5 collections only, convert an episode or directory containing
`episode_*.h5`:

```bash
python scripts/convert_piper_hdf5_to_lerobot.py \
  --input datasets/piper_stack \
  --output datasets/piper_stack_lerobot \
  --repo-id local/piper_stack_evis
```

The converter stores RGB observations in LeRobot's video/tabular layout and
keeps compact event arrays plus past/future frame offsets under `events/`.

## Development checks

Enable the local commit guard once per clone:

```bash
python -m pip install pre-commit
pre-commit install
```


Before every commit or push:

```bash
bash tools/check_large_files.sh
git status --short
git diff --check
```

Generated data belongs outside Git. Publish any required large artifact via a
release or object store and document its SHA256 checksum and destination path.
