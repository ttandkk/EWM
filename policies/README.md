# Piper policy services

This package owns model loading, saved normalization, model resource relocation,
and raw-observation inference. The benchmark imports none of the model packages
and never reads a Hydra checkpoint config. Model dependencies belong to separate
OpenWAM or FastWAM environments; the base package can describe a checkpoint and
validate its wire protocol without importing torch or a simulator.

Install the two lightweight local packages into the **model** environment (use
its existing lock/install procedure for torch and native model dependencies):

```bash
python -m pip install -e ./benchmark -e ./policies
python -m pip install --no-deps -e ./OpenWAM   # EWM / OpenWAM environment
# FastWAM uses its own environment and installed ./FastWAM package instead.
```

No checkpoint/config/statistics files are rewritten. The examples assume the
current directory is the project root; all CLI paths can be absolute.

```bash
python -m piper_policies describe --family ewm --checkpoint-dir /models/ewm-run
python -m piper_policies serve --family ewm --checkpoint-dir /models/ewm-run \
  --checkpoint checkpoint_step_80000.pt --pretrained-root /models/pretrained \
  --ready-file /tmp/ewm-policy.json
python -m piper_policies serve --family openwam --checkpoint-dir /models/wam-run \
  --pretrained-root /models/pretrained --ready-file /tmp/wam-policy.json
python -m piper_policies serve --family fastwam --checkpoint-dir /models/fastwam-run \
  --checkpoint checkpoints/weights/step_030000.pt \
  --config /models/fastwam-run/config.yaml --normalization /models/fastwam-run/dataset_stats.json \
  --project-root /workspace/EWM/FastWAM --pretrained-root /models/pretrained \
  --ready-file /tmp/fastwam-policy.json
```

Use the saved **composed training config**, not an uncomposed Hydra preset, for
`--config`. For an unnumbered checkpoint supply `--checkpoint-step`. OpenWAM
requires its native `normalization_stats.npy` next to `config.yaml`. EWM uses
`normalization.json`. FastWAM accepts its native run layout: `config.yaml`,
`dataset_stats.json`, and `checkpoints/weights/step_030000.pt`. Automatic selection
also preserves top-level `step_*.pt` / `checkpoint_step_*.pt` compatibility;
`--checkpoint-step` selects a specific step and duplicate names for that step
are rejected. Existing top-level `fastwam_stats.json` takes priority over
`dataset_stats.json`; `--normalization` overrides both. The legacy checkpoint
suite launcher also forwards `--checkpoint`, `--config`, `--normalization`, and
repeated `--path-map` options unchanged to both describe and serve. The benchmark needs only the URL in the
ready file, its own task/suite definition and evaluation assets.

Piper OpenWAM exports with the known `{"eef": 7D, "state": 8D}` statistics
schema are adapted to the native `eef_state` key in a temporary model-side
file, using the native normalizer without changing any statistical value. Native
`eef_state` files are passed through unchanged. Unknown/ambiguous schemas fail
explicitly. `policy_descriptor.normalization_adapter` records the mapping and
both original/effective file SHA256 values; the checkpoint statistics are never
rewritten. Temporary files are removed after the native factory has loaded its
owned arrays, including when the factory fails.

The service binds localhost by default. `--host` and `--port` support a private
remote deployment or an SSH tunnel. This protocol has no authentication; network
access control belongs to deployment. Inference is serialized on the thread
which created the model, including post-trial decoding.

## Relocation and provenance

`--path-map /old/root=/new/root` maps path components, with the most specific root
winning. `--pretrained-root` replaces the suffix following an existing
`pretrained/` component. `--project-root` anchors relative model paths. The
allowlists in `resources.py` are explicit; prompts, model IDs and training dataset
locations are never blindly rewritten. FastWAM honors the native
`DIFFSYNTH_MODEL_BASE_PATH`; its automatic downloads default to disabled while
serving. Prepare the weights before starting a service.

Active EWM training presets accept `EWM_PROJECT_ROOT`, `EWM_PRETRAINED_ROOT`, the
existing `EWM_DINO_PATH`/`EWM_T5_PATH`/`EWM_VAE_PATH`, and explicit data variables.
`NEW_PROJECT_ROOT` remains a compatible override for eight-task training. The
normal training entrypoint supplies the repository root independent of cwd.
Historical job-specific launch scripts remain historical and should be replaced
with a current resource profile for new runs.

Every descriptor contains byte-level SHA256 hashes of the checkpoint, original
config, normalization and the selected adapter/native inference source files.
Computing the checkpoint hash reads the entire file. External pretrained content
(such as DINO, T5, VAE, and a FastWAM LoRA base) is **not included** in these hashes.
Matching descriptors therefore cannot establish that all model assets are
identical. A separately validated resource manifest would be needed for that
stronger claim.
Legacy `collection_sha256` hashes the raw task collection JSON. Legacy
`input_signature` is only a size/mtime summary and is exposed as
`legacy_input_signature`, never as a content identity. Optional
`training_data_manifest.json` is included with its own hash. Its supported
structure is `{"version":"piper_training_data_v1","tasks":{"conveyor":
{"collection_sha256":"...","metadata_identity":{"version":
"piper_dataset_metadata_v1","sha256":{"piper_collection_config.json":"...",
"piper_camera_calibration.json":"...","piper_temporal_contract.json":"..."}}}}}`.
The optional metadata_identity uses the benchmark's canonical JSON hashes;
collection_sha256 remains the original file-byte hash. Missing full data
manifests must not be described as verification of all training videos/actions.

## Protocol

`hello` returns `policy_descriptor.version = piper_policy_v1` and compatible
legacy top-level metadata. A complete deterministic example is
[`tests/fixtures/hello_joint.json`](tests/fixtures/hello_joint.json); its hashes
are synthetic fixture values. `predict` uses the benchmark's pickle-free JSON /
NumPy wire frames:

- uint8 `obs_video[2,3,256,256]`, ordered cam0 then cam_wrist;
- float32 physical `proprio[8]` and a nonempty prompt;
- joint EWM only: nonnegative float32 event counts
  `obs_voxel[event_cameras,2*voxel_bins,256,256]`;
- current RGB/state timestamps, and joint EWM's one-control-interval event
  start/end timestamps, with matching `observation_step` and `control_hz=30`;
- seed, positive sampling_steps, and action_chunk_size within the trained horizon.

Replies contain physical float32 `executed_actions[N,7]`, inference_seconds,
observation_time_s, observation_step, control_hz and the action contract
`piper_delta_eef_gripper_v1`. One service is pinned to one checkpoint and rejects
requests for a different step. Timing/shape/nonfinite/future-label mismatches
fail before inference. The service does not schedule or execute robot actions.

EWM reuses its native compatibility loader, normalizer and sampler. Its service
CLI defaults to compilation on, CUDA Graphs off, context K/V and BF16 core
caches on, compile mode `default`, three warmups, and three seeded eager/compiled comparisons with flow4.
Action-only uses its original predict_velocity compiler without a BF16 snapshot.
Compilation failures do not publish readiness. Action deviations above .02 are
recorded by default, matching the current formal configuration;
`--no-allow-compile-action-error` opts into a strict startup gate. Use
`--no-compile-core --warmup-iters 0 --benchmark-iters 0` for CPU diagnostics.
Actual optimization state and comparison timings live in hello.optimization
(and compatible top-level fields), separate from the immutable descriptor. Joint and
RGB-action EWM support bounded latent recording and **post-trial** decoding.
OpenWAM uses the native JointInferenceEngine (including normalization); FastWAM
uses its native infer_action and processor. Their Piper layout is the training
layout: horizontal cam0 + cam_wrist. WAM adapters advertise no prediction-video
recording support. Full-size model inference and GPU latency must be validated
in the respective model environment; the CPU suite does not establish that.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=policies/src:benchmark/src:OpenWAM \
python -m pytest policies/tests -q -p no:cacheprovider
```
