# Reproducible Piper deployment

This deployment covers separate simulation and LeRobot writer environments.
Optional EWM/OpenWAM and FastWAM policy profiles remain available in the installer,
but fresh model environments are outside the current scope.
`ewm setup` writes only a new managed prefix; it refuses a nonempty
unmanaged directory and never repairs or resets an existing third-party worktree.
The canonical release lock is
[`simulation.lock.json`](../benchmark/src/piper_bench/resources/deployment/simulation.lock.json).
It includes exact package versions, Python minors, Git commits, the FastWAM patch
hash and the Isaac compatibility-code hash. The four requirements files and both
locks ship inside the benchmark wheel.

## Native installation

Use Linux x86_64 and Python 3.12 for simulation/writer. Install Git, a C/C++
compiler, FFmpeg, OpenGL/EGL/Vulkan libraries,
and a supported NVIDIA driver. Rendering requires an NVIDIA GPU even though the
task's physics device is CPU. Check NVIDIA's current
[Python installation requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html)
for the target machine. This release explicitly pins Isaac Sim **6.0.1.0** and
Isaac Lab commit **ffff603eafc6b74264a5261cc0183d6a65390d78**; do not silently use
the newest version shown in the rolling documentation.

From a source checkout, the bootstrap commands need only the Python standard
library. Review NVIDIA's license terms before providing `--accept-eula`.

```bash
export PIPER_HOME="$HOME/.local/share/piper-bench"
./ewm setup --profile simulation --prefix "$PIPER_HOME" --dry-run
./ewm setup --profile simulation --prefix "$PIPER_HOME" --accept-eula
export PIPER_CONFIG="$PIPER_HOME/deployment.json"
./ewm assets download
./ewm assets verify
./ewm doctor --output deployment_report.json
./ewm doctor --render --events --output deployment_report.json
./ewm demo --task tennis_rolling --oracle --output-dir demo_tennis
```

`deployment/install.sh` invokes the same setup module. `--python /path/to/python`
selects the interpreter used to create a new virtualenv. Each profile writes its
own environment below `$PIPER_HOME/envs`, fixed sources below `sources`, and an
installation report. First-party build inputs are copied into content-hashed
directories below `first-party`; pip never builds inside the original checkout.
Resuming an existing managed install checks source revisions
and refuses unfamiliar tracked changes. No command merges the environments.

```bash
./ewm setup --profile writer --prefix "$PIPER_HOME" --python python3.12
```

The writer Python is selected from `PIPER_WRITER_PYTHON`, `--writer-python`, or the
generated local config. `PIPER_SIM_PYTHON` can select an existing compatible
simulation environment for diagnostics without reinstalling it. Such a check is
an **existing-environment test**, not proof of a fresh installation. EWM and
FastWAM inference also require their own weights and normalization artifacts;
these are not needed for an Oracle simulator demo.

The lock reconstructs the tested runtime with `--no-deps`; dependency metadata
is checked after installation. Exactly 15 known simulation exceptions are
enumerated with owner/version, exact `pip check` error and functional impact.
They include EVIS 0.1.0's stale `numpy<2` requirement (runtime NumPy 2.3.1),
unused upstream teleoperation/IK/export/test extras and a coverage development
pin. Any other error fails setup. This does not certify those omitted upstream
features; EVIS still requires a successful GPU event test. The writer lock has
68 runtime pins verified against its actual environment with
`PYTHONNOUSERSITE=1`, plus the separately installed LeRobot source and installer
bootstrap tools. User-site-only packages were excluded. The writer uses its
PyAV path; five real I/O checks passed in the original isolated writer
environment. A completely new writer prefix then installed successfully in
337.86 seconds with a clean `pip check` and no exceptions. Outside the checkout,
its installed package passed all 11 task schemas, two-episode save/resume,
60 Parquet rows, 120 event PNGs with exact values, and decoding all four videos.
All 225 package files and the clean LeRobot source matched the frozen inputs.
Actual simulator-to-writer collection job **1734780** also passed: two episodes,
170 frames, eight decoded videos, and raw event output, with zero validation
failures. The temporary fresh simulation environment was retired after its
reports, wheels and Kit logs were archived. The temporary acceptance writer was
also removed during the user-requested cleanup on 2026-11-12; original project
environments remain available. Its small acceptance reports and logs are kept
under `results/storage_cleanup_20261112/evidence/`.
Policy consumers of TorchCodec must separately
verify their native FFmpeg decode path; TorchCodec is not a writer dependency.
TorchCodec needs the matching FFmpeg **shared libraries**, not merely an executable
named `ffmpeg` on `PATH`. The pinned TorchCodec 0.4/0.5 loaders try FFmpeg majors
7, 6, 5 and 4. Configure `LD_LIBRARY_PATH` to a compatible library directory before
starting the policy interpreter. Fresh policy CPU checks were stopped when
model environments were removed from the requested scope. No fresh policy TorchCodec validation or FFmpeg reinstall
is claimed.

## Assets and licensing

[`assets.lock.json`](../benchmark/src/piper_bench/resources/deployment/assets.lock.json)
locks **53 runtime files (37,022,403 bytes)** by size and SHA256. Small generated
props/textures are packaged with the benchmark. The Piper USD plus its four
configuration layers and two license files are fetched from the public
[Agilex source at the fixed commit](https://github.com/agilexrobotics/Agilex-College/tree/c2688be41e1bc99a9addd555f237a16bc839d936/isaac_sim/agx_arm_IsaacLab/IsaacLab_Data_Collection).
Their contents match the archived experimental bundle byte for byte; robot
geometry has not been substituted. Downloads are atomic and differing local
files are never overwritten. MIT and BSD upstream license files remain in the
bundle. There is no blanket new license assignment to mixed third-party assets.

The historical `kritokronos/ewm_assets` release was private. It remains archival
provenance in the manifest; the default download does not require access to that
repository or change its visibility. A custom immutable Hub manifest can use an
authorized user's `HF_TOKEN`, which is not forwarded to a different redirect
host. Credentials are never stored in the manifest.

NVIDIA's SeattleLabTable, Grid environment and their transitive USD/material
references are resolved by the installed Isaac asset service. Kit supplies
`OmniPBR.mdl`. Their exact known roots and provider are recorded in the manifest.
They are governed by NVIDIA's terms and are **not a redistributed, offline,
hash-locked mirror**. Network access to the official service is required for a
fresh cache. `PIPER_NVIDIA_ASSET_ROOT` may point at an authorized mirror retaining
the official `Assets/Isaac/6.0` relative hierarchy; the user must supply its full
dependency closure. `assets verify` checks the 53 locked files, while the real
renderer checks loaded USD composition; neither claims every external texture
has been recursively hashed.

`PIPER_ASSET_ROOT` selects a relocated bundle. First-party scene assets resolve
through this root or packaged resources, independent of the current directory.
The robot's scene configuration is rebound to the verified bundle without
editing the installed Agilex source.

The fixed upstream Piper layer contains one dangling reference at
`/piper/world/visuals` to the absent `/visuals/world`. The runtime creates an
audited, content-addressed five-layer derivative under its cache: four files
are copied byte for byte and only that empty reference is removed from the
base layer. It accepts only the five original locked hashes. Before binding
the derived robot, it checks all 761 composed prims (including instance proxies),
20 meshes, 6,921 attributes and 387 relationships using native USD value equality,
including metadata, time samples, connections and targets. No value tolerance
or field exclusion is used. The original bundle remains unchanged, and the
strict composition check remains enabled. `derivation.json` records source and
derived hashes, the pinned recipe hash, the operation and the semantic proof;
runtime audit JSON includes that record. Unknown custom robot sources are
rejected by this compatibility path rather than silently modified.

## What doctor verifies

The report distinguishes `passed`, `passed_with_exceptions`, `failed` and
`not_run` for each layer. Missing GPU access or an unrequested test never counts
as rendering/event success. `deployment_accepted` becomes true only when all
layers have passed.

1. Environment: all pinned simulation package versions, the three editable
   source commits and tracked cleanliness, compatibility code hash, and exact
   dependency-metadata exceptions. This layer does not import the renderer.
2. Assets: all 53 local sizes/hashes, with external dependencies disclosed.
3. `--render`: two short Oracle episodes in one real Isaac process, joint
   motion, both 256×256 RGB streams, changing images, the moving wrist camera's
   measured rigid transform to the physical hand, loaded USD composition, and
   FFmpeg encode/decode of the recordings.
4. `--events`: actual linear-HDR EVIS output, OFF/ON activity on both cameras,
   finite nonnegative event counts, contiguous 30 Hz timestamps and empty
   initial event frames after **both** episode resets.
5. Oracle: actual task success from the existing runtime metrics, separate from
   camera/event health. Successful rendering cannot mask a failed task.

The report contains the precise command, child return code, log, per-episode
audit JSON and video paths. Defaults are two tennis_rolling episodes with at
most 600 control steps each; use `--timeout` to bound the child. No policy,
training dataset or pretrained model is used.

## Environment boundaries

Each profile has its own Python executable and dependency set. The shared
managed prefix stores their paths in `deployment.json`; it does not merge
`site-packages` or require activating an environment in the parent shell.

| Profile | Python | Environment directory |
|---|---|---|
| `simulation` | 3.12 | `$PIPER_HOME/envs/simulation` |
| `writer` | 3.12 | `$PIPER_HOME/envs/writer` |

The CLI starts the configured interpreter and passes only the required project
packages across environment boundaries. User-site packages are disabled.
Source editable installs explicitly use PEP 517 with build isolation disabled,
so a legacy `setup.py develop` subprocess cannot silently download a different
build toolchain. Policy setup installs all locked binary dependencies first and
checks that Torch imports with the exact locked version before building the two
known source distributions, ANTLR and DeepSpeed. Normal setup retains the
installed environments. The optional `policy-ewm` and `policy-fastwam` profiles require Python 3.10 and separate environments; they
are not required steps or freshly verified environments for this deployment.
Temporary acceptance jobs may archive their evidence and retire only their own
disposable prefix after all checks pass.

Container delivery is outside the requested scope. Native installation,
asset verification and real runtime checks are the deployment acceptance path.

## Validation record

On 2026-11-11 a fresh anonymous download verified all 53 public/packaged assets.
The existing simulation environment passed 198 package/version entries, three
source revision/cleanliness checks, compatibility hash and the 15 documented
metadata exceptions. This CPU result deliberately reports rendering/events as
`not_run` and `deployment_accepted: false`.

A full native installation also completed in a **new independent prefix** on
2026-11-11: all 195 pinned noneditable dependencies, the three newly cloned fixed
source repositories, and the benchmark built from an isolated first-party
snapshot. Its installed `site-packages/piper_bench` then ran from outside the
checkout with `PYTHONPATH` and `PYTHONHOME` cleared. A second public asset download
verified 53/53 files, and CPU doctor passed all 198 version entries, source
commits, compatibility hash and exact metadata exceptions. The local evidence
is `setup-simulation.json` and `deployment-cpu-report.json` in that installation
prefix; copies are retained with the deployment smoke artifacts.

The first temporary-SSD attempt exhausted its write quota. That initial attempt
used pip's default user cache before explicit prefix-local cache isolation was
added. Its installation logs remain; the failed temporary directory was later
removed at the user's storage-cleanup request. No user cache was cleaned during
that installation. The independent HDD prefix completed
successfully. Subsequent installation caches, temporary downloads and build inputs
are confined to the chosen prefix. The official Kit extension-cache
wheel alone expands to 77,681 files / 11,660,291,364 bytes; network-filesystem
metadata latency can make installation slow. Installation does not create
`build` or `egg-info` directories in the source checkout.

The first existing-environment GPU doctor ran two successful Oracle episodes
(84 and 86 frames). RGB, wrist attachment, EVIS, timestamps, reset and video checks
passed, but strict USD composition failed on the upstream empty reference
described above. The check was retained; the derived-asset fix passed independent
CPU native-value comparisons. A separate fixed-action replay completed with
all 100 recorded actions, the same terminal state and original failure label.
Isaac automatically wrote runtime logs/cache below the existing Kit installation;
no dependency installation or upgrade was performed in that environment.
The USD-fixed benchmark wheel was installed into the fresh prefix with no dependency
changes: all 224 installed package files match that wheel and its frozen source,
and a new CPU doctor passed all versions, source revisions, assets and the USD
recipe hash. Six real CPU USD regression cases plus 14 deployment checks passed.
The fresh-environment GPU doctor **1734674 completed successfully** on a PRO6000
in 10 minutes 11 seconds. Both Oracle episodes succeeded (84/86 frames), with
RGB, moving wrist attachment, EVIS, reset/timestamps and video checks passing.
Strict USD composition reported zero errors in both episodes;
`deployment_accepted` is true for this frozen installation.
The prefix remained frozen throughout both GPU validations; later CLI/doctor package bridging, writer bootstrap
and the corrected 68-pin writer lock belong to the newer release wheel.
Its script and input hashes are frozen under
`results/deployment_refactor_smoke_20261111/fresh-env-gpu/`.
The fresh GPU report and per-episode evidence are retained separately from the
earlier failed existing-environment doctor; no checks were waived. Source-mode and direct module
entrypoints apply saved runtime settings before importing Isaac's AppLauncher;
explicit asset/cache/FFmpeg environment overrides remain effective.
The full collection/export/validate job **1734780** also passed: two accepted
episodes, 170 frames, eight decoded videos and verified raw events. After these
checks, the user requested storage cleanup. This task-owned temporary simulation
environment was archived and removed; the original project environment remains.
Its reports, frozen wheels and Kit logs are retained under
`results/storage_cleanup_20261111/retired-native-environment/`. The separate GPU
doctor outputs remain in the project results directory. On 2026-11-12 the user
also requested deletion of the disposable collection dataset/videos, temporary
writer and external preparation directories. Their small reports, logs and
launch scripts were hash-verified and archived under
`results/storage_cleanup_20261112/evidence/`; the exact deletion list and
completion status are in that cleanup directory. Formal datasets and results
were not among the deletion targets.
Historical prefix paths in those reports should not be treated as a current
installed runtime after cleanup.

Development container probes exposed two installer defects: an untouched
`--no-checkout` clone was incorrectly treated as populated when HEAD already
matched the pin, and a legacy editable build started a nested pip process that
dropped `--no-build-isolation`. Both fixes apply to native setup. The current
25 deployment tests include real temporary Git repositories and an offline
`setup.py`-only editable installation with the pinned pip/setuptools/wheel tools.
The failed probe logs and small metadata remain archived; container recovery,
image export and container GPU acceptance were cancelled at the user's request.
The sole task-owned container root filesystem and base image were then removed,
reclaiming 26.053 GiB after the eight useful metadata archives were hash-verified.
Original environments and the external frozen evidence were retained.

Fresh policy environment validation was stopped at the user's request because
OpenWAM and FastWAM environments are outside this deployment scope. EWM job
**1734885** was cancelled after 6 minutes 38 seconds. It had downloaded its
locked wheels and was installing them when the stop was requested; setup and
runtime checks had not passed. FastWAM was never
submitted. The interrupted installation is not successful fresh-policy evidence.
Its logs and small metadata are retained for auditing; its task-owned temporary
prefix was removed after the job stopped. The remaining preparation directories
were removed in the 2026-11-12 cleanup, with small records kept under
`results/storage_cleanup_20261112/evidence/`. Original model environments remain
untouched.
