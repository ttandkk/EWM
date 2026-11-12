# Real model adapter validation preparation

**Current scope: deferred, not part of the deployment acceptance criteria.**
The user removed OpenWAM/FastWAM model-environment setup and further model
execution validation from this task. OpenWAM full-model execution remains
unverified. Existing adapters and the completed FastWAM 1734847 evidence are
retained; the preparations and job-status notes below are historical records.

The user requested removal of the external `piper-model-validation` and policy
preparation directories on 2026-11-12. Small reports, logs and launch scripts were
archived under `results/storage_cleanup_20261112/evidence/`; the deletion and
archive mapping is in `results/storage_cleanup_20261112/manifest.json`. The
external preparation paths below are historical, not available runtime inputs.

An independent CPU reproduction found an unresolved **v3 orphan-process bug**:
a short-lived leader can leave a child in a separate session while the guard
returns success and misses that child. The v3 cleanup guarantee is therefore
incomplete. An unfinished subreaper change was **not tested or accepted**; its
source/diff and the v3 failure evidence were archived under
`results/storage_cleanup_20261111/deferred-model-validation/`, explicitly outside
this delivery. Only that unfinished increment was withdrawn, restoring the
project guard to the frozen v3 bytes. This issue must not be described as fixed,
and no v4 execution snapshot was created.

This directory contains explicit preparation tools. No GPU inference or training
is performed by `service_gpu_probe.py prepare` or by the OpenWAM exporter without
`--execute-initialization`. `run` starts a real policy service and returns one
physical float32 `[1,7]` action; it never commands a robot or starts a simulator.

The reviewed source snapshot and plans for this session are in
`/tmp/piper-real-policy-final-gwgaxooh/` (549 files, approximately 5 MB).
The source-manifest SHA256 is
`d7e6ca0e703dfbba1fbe949e9ef20001782b028b5e865c3bd76602e7cdcffc7a`.
The snapshot includes policy/benchmark/OpenWAM/FastWAM Python source and a narrow
websockets 16.1.1 overlay copied from the existing OpenWAM environment. The overlay
contains only websockets and its distribution metadata, with no `.so` or other
site-packages. Existing environments and FastWAM's working tree are unchanged.
This is not a fresh policy environment installation test.

## Available inputs

All paths below are relative to `/projects/_hdd/isaachdd/EWM`.

| Case | Existing run directory | Selected weight | Evidence label |
| --- | --- | --- | --- |
| EWM joint | `OpenWAM/outputs/ewm_joint_stride1_ada_20261106` | `checkpoint_step_080000.pt` (1,061,163,146 bytes) | Trained 80k; formal inference startup |
| EWM action-only | `tmp/ewm_formal_ada_serial_compile_20261104/runs/action_only` | `checkpoint_step_080000.pt` (378,990,195 bytes) | Trained 80k; eager load/contract probe |
| EWM RGB-action | `tmp/ewm_formal_ada_serial_compile_20261104/runs/rgb_action` | `checkpoint_step_080000.pt` (1,061,163,146 bytes) | Trained 80k; eager load/contract probe |
| FastWAM | `tmp/fastwam_b32_formal_20261111/gpu_checkpoint_probe` | `checkpoints/weights/step_000002.pt` (2,122,653,061 bytes) | Two-step training probe; not completed formal training |

Each EWM run has its composed `config.yaml` and `normalization.json`; FastWAM has
`config.yaml` and `dataset_stats.json`. The FastWAM payload was read with CPU mmap:
step=2, `mot` has 1,424 tensors, and `video_lora`/`proprio_encoder` are present.
The separate formal FastWAM output `FastWAM/outputs/piper_rgb_lora_b32_30k_20261111`
had config/statistics but no numbered policy weight at the inventory time.

EWM symmetric stride2 and asymmetric 80k weights also exist at
`tmp/ewm_formal_ada_serial_compile_20261104/runs/joint` and
`OpenWAM/outputs/ewm_eight_no_memory_no_alignment_243038`; they are not additional
cases in this minimal adapter check.

No trained OpenWAM `checkpoint_step_*.safetensors` was found under `tmp`,
`OpenWAM/outputs`, `FastWAM/outputs`, or `ckpts`. Existing OpenWAM smoke reports
state `checkpoint_saving=false`. A zero-training initialization export is feasible:

1. Load the local pretrained Wan2.2 backbone with the native full-size Piper config.
2. Keep action/proprio and LoRA initialization at seed 42; perform zero optimizer updates.
3. Use native `save_assets_for_deployment` and `save_checkpoint` in a new HDD directory.
4. Reuse the real pooled `openwam_normalization_stats.npy` from
   `tmp/training_data_cache/piper_rgb_dynamic_v1/`.
5. Load that export via the unchanged adapter/native strict loader and issue one predict.

This OpenWAM result would prove initialization/export/adapter engineering only.
It would not represent a Piper-trained model or a policy-quality result.

## Environments and external assets

EWM/OpenWAM: `envs/openwam/bin/python`, Python 3.10.22, torch 2.7.1+cu128,
OmegaConf 2.3.1, transformers 5.17.0, websockets 16.1.1.
FastWAM: `envs/fastwam/bin/python`, Python 3.10.22, torch 2.7.1+cu128,
OmegaConf 2.3.0, transformers 4.49.0. Its original environment lacks websockets;
the isolated overlay supplies that one dependency. Venv Python symlinks are
preserved rather than resolved to the base interpreter.

Local DINOv3, T5-small, LeanVAE, Wan2.2 video/VAE/UMT5, tokenizer, and FastWAM
ActionDiT weights exist under `pretrained/`. Automatic downloads are disabled.
The four plan files contain newly computed checkpoint/config/normalization and
selected implementation SHA256 values. All frozen source/overlay files have a
separate manifest hash. External pretrained content was checked for existence
and size, **not freshly rehashed**; the snapshot's `external_assets_inventory.json`
references the historical `pretrained/training_weights_manifest.json` separately.
Matching policy descriptors do not prove identity of all external model assets.

## Request and checks

Every request uses seed 42, flow4, two deterministic synthetic RGB images,
timestamp 1/30 s, observation step 1, and real training state mean from pooled
statistics. Joint EWM adds a one-control-interval deterministic event array.
These inputs exercise the actual model path without claiming an evaluation score.

The joint case retains formal compilation/default mode, graphs off, K/V cache,
BF16 snapshot, three warmups, and three eager/compiled comparisons. Actual hello
metadata must satisfy the benchmark's `formal_inference_settings`; the probe also
checks BF16 cache, flow4 warmup, finite measured comparison times and action error.
Action error above .02 is recorded, matching the formal policy. It is not silently
turned into a stricter rejection. Action-only and RGB-action explicitly use eager
startup and do not claim formal timing validation.

The client verifies descriptor equality, action dtype/shape/finiteness, physical
contract, timestamp/step/control rate, and nonnegative measured inference time.
It saves hello, request, raw observation, reply, action, timing and optimization
metadata, sends shutdown, and only terminates its own child process group if needed.
Ready/startup timeout is 1,200 s per service; one reply timeout is 600 s.

## Resource review and current block

`serial_gpu_probe.sbatch` in the review directory requests one Pro6000, 128 GiB
host RAM, eight CPUs and 60 minutes, with all families run serially. The OpenWAM
native full-state export is expected to write about 25 GB; reserve at least 40 GB
of **HDD user quota** for the export and small probe outputs. The script writes
large outputs to `/projects/_hdd/isaachdd/piper-model-validation/$SLURM_JOB_ID`,
not `/tmp`. Filesystem free space is not evidence of remaining user quota.

Only `sbatch --test-only` was executed. The site hook changed that request to
**four CPUs and 33,792 MiB host RAM per single Pro6000**, so the requested 128 GiB
was not granted. The proposed combined script must not be treated as resource
ready: resolve host memory allocation or separate/qualify the OpenWAM export
before an actual submission. No real Slurm job was submitted during that preparation-only stage.

CPU checks executed: `13 passed` in `test_service_gpu_probe.py`; four real
model-side `describe` commands passed. Full model GPU loading, OpenWAM export,
compiled startup and predict remain unexecuted.


## Subsequently authorized four-case submission

After the resource review, the parent task explicitly authorized a separate
four-existing-checkpoint job. It excludes OpenWAM initialization and uses
1 Pro6000, 4 CPUs, 33,792 MiB and a 45-minute limit. Its script records each case's
exit code, continues to independent later cases after failures, and returns
nonzero overall if any case fails.

Job **1734723** was submitted at `2026-11-11T14:23:34Z`. At the subsequent status
check it was `PENDING (Priority)`; this is not GPU inference evidence yet.

Frozen submission directory:
`/projects/_hdd/isaachdd/piper-model-validation/prepared-four-case-k4h_8d6q/`.
The source-manifest SHA is unchanged from the preparation snapshot. Batch SHA256:
`ad914867109cf7466a4d59e7b5afafa2a0b4a6ff6dc3b042f93d0397181dca29`.
Plans and source were copied to this HDD location before submission.

Expected outputs:
`/projects/_hdd/isaachdd/piper-model-validation/four-case-1734723/`.
Log: `/projects/_hdd/isaachdd/EWM/logs/policy_four_real_1734723.log`.
The subsequent OpenWAM submission is described below.

## OpenWAM native initialization and public normalization compatibility

The real Piper statistics contain `eef` (7D action) and `state` (8D proprio),
whereas the native normalizer expects `eef_state`. The public policy boundary now
translates that known schema without changing any statistic, shape or dtype.
Already canonical input keeps its original bytes. Unknown or ambiguous schemas
fail explicitly. The descriptor records the mapping and both source/effective
content hashes. Native normalization code and the original writer/statistics are
unchanged; a temporary canonical file lives until the native factory owns its
arrays and is then removed, including on failure.

The checked-in real aggregate-statistics fixture has source SHA256
`fda2ea9f370452305659f831e93a18ca162ea84a14ec11ecfa8cef6e74fcb7a6`;
the canonical bytes have SHA256
`fd421d5dd121878f7ece902658c29feb7df908506153829f5b602f1efcd1a197`.
The public constructor tests use the real native normalizer while doubling only
the heavyweight model loader. They verify physical 8D normalization, 7D action
denormalization, describe/serve equality, source preservation, schema rejection,
mutation detection and temporary-file cleanup.

Executed together with the protocol and preparation checks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=policies/src:benchmark/src:OpenWAM \
envs/openwam/bin/python -m pytest \
  policies/tests/test_openwam_normalization.py \
  policies/tests/test_policy_contract.py \
  policies/validation/test_service_gpu_probe.py -q -p no:cacheprovider
```

Result: **49 passed** (9 normalization, 23 protocol, 17 probe checks). These are
CPU tests and do not establish successful full-model GPU inference.

Native source and metadata sizing showed that constructing/casting UMT5 beside
the video backbone can exceed the single-GPU allocation's 33 GiB host limit.
The reviewed estimate was 36,247,552,384 bytes for initialization and
49,824,954,368 bytes for strict reload; these are estimates, not measured full-model
peak RSS. The separately authorized job requests **2 Pro6000, 8 CPUs, 67,584 MiB
host RAM, 45 minutes**, with inference exclusively on `cuda:0`. The second GPU
reservation supplies the required host-memory quota; it does not shard or reduce
the native model.

Job **1734785** was submitted at `2026-11-11T15:02:06Z` after shell syntax,
17 required-input existence checks, frozen-source verification, an actual
model-side metadata check and a successful `sbatch --test-only` review.
Frozen directory:
`/projects/_hdd/isaachdd/piper-model-validation/prepared-openwam-legacy-myyxc1qh/`.
Batch SHA256:
`3cfa20d511de9fc6111f03fc19e60ee23fbdce521c3c44a36035eee8325dd55e`.
Source-manifest SHA256:
`d47561200fe3684b9a55ee5ac92a755e6c06d4eb383bada8389a0116aa321ddc`.

Its dependent stages are full native initialization/export (18-minute timeout),
descriptor preparation (6 minutes), and strict native adapter reload plus one
physical action response (18 minutes). Export preserves the original legacy
statistics for the public adapter and writes a separate canonical audit copy.
Outputs are new HDD files under
`/projects/_hdd/isaachdd/piper-model-validation/openwam-1734785/`;
log: `/projects/_hdd/isaachdd/EWM/logs/openwam_native_probe_1734785.log`.
GNU time records stage peak RSS; the service wrapper records exact PyTorch
allocated/reserved CUDA peaks, and a separate monitor samples allocated-device
memory. A failed dependent stage stops the job and preserves its evidence.

At the status check after submission, **both 1734723 and 1734785 were
`PENDING (QOSMaxJobsPerUserLimit)` with unknown start time**. No initialization,
strict GPU reload or GPU action result is claimed here. The OpenWAM artifact is
explicitly **pretrained backbone plus untrained Piper action/proprio modules,
zero optimizer updates**; even a successful probe is engineering validation,
not a trained-policy or success-rate result.

The submitted scripts are immutable. They disable downloads and write model
outputs to their own HDD directories, but do **not** fully isolate caches:
service Inductor/Triton caches are local; general XDG/Hugging Face/torch/CUDA
cache variables and `TMPDIR` were not set. The compatibility helper therefore
uses the system temporary directory (normally `/tmp`) for a small statistics
file. Existing local Wan paths bypass downloading and the tokenizer loads from
a local directory, but dependency defaults can still point to home caches.
There is no observed full-model GPU execution yet to establish which of those
locations are actually written. Future submissions should set their common
cache and temporary directories beneath the new output directory without
changing `HOME`.

## Completed four-case GPU evidence and storage hold

Job **1734723** ran from `2026-11-11T15:15:25Z` to `15:29:05Z` (13m40s).
Slurm reports `OUT_OF_MEMORY`, exit `0:125`, while the recorded case exits are
`0, 0, 0, 1` respectively:

| Case | Actual result | Service inference | Request round trip |
| --- | --- | ---: | ---: |
| Joint EWM, formal compilation | Passed | 37.579 ms | 46.271 ms |
| Action-only EWM, eager | Passed | 470.529 ms | 474.143 ms |
| RGB-action EWM, eager | Passed | 289.547 ms | 293.069 ms |
| FastWAM step-2 checkpoint | Failed before ready/hello | Unavailable | Unavailable |

Each passed case loaded its actual 80k checkpoint, returned one finite float32
`[1,7]` physical action, preserved observation time `1/30`, step `1`, and `30 Hz`,
and shut down successfully. Action-only/RGB timings include their first eager
request without warmup; these are not comparable formal performance benchmarks.

Joint hello verified actual compile enabled, default mode, CUDA graphs disabled,
K/V and BF16 caching enabled, flow4, three warmups and three benchmark iterations.
Startup took 270.115 s. The measured benchmark means were 80.386 ms eager without
K/V, 74.842 ms eager with K/V, and 30.116 ms optimized. Maximum action difference
was `0.00732421875`, below the configured `0.02` recording threshold.

The FastWAM service received SIGKILL (`-9`), and the original Slurm stderr records
one batch-step `oom_kill` event. This is evidence of **host/cgroup OOM**, not a
reported CUDA allocator OOM. Its server log is empty, so the precise native
loading phase cannot be established. No hello or action was produced. Slurm
did not supply MaxRSS and this older frozen probe did not record exact GPU peaks;
neither peak is invented here. Source inspection shows the native loader creates
a default CPU FP32 module and loads CPU weights before moving it to CUDA/BF16;
the public adapter also enables the full text encoder for arbitrary prompts.
These identify a host-memory risk, not proof of the exact interrupted operation.

The small `four-case-1734723/audit_summary.json` preserves action values/hashes,
replies, optimization metadata, the original Slurm error, limitations and the
resource recommendation. Before adding this small report, outputs occupied
28,453,888 bytes (27.14 MiB): 24.58 MiB reusable Inductor/Triton caches and
2.56 MiB other evidence. No new checkpoint was saved. Caches/results are retained.

If later authorized, the focused remaining retry needs only the existing FastWAM
step-2 checkpoint and existing pretrained files: load, hello, one predict. A
2-Pro6000 / 8-CPU / 67,584-MiB / 30-minute allocation gives host-memory headroom
under this site's per-GPU quota; inference stays on `cuda:0`. This recommendation
has not been submitted or validated and requires no export, training, download,
or new large weights. Unbuffered loading logs and host/GPU peak metrics should
be enabled for that retry.

Job **1734785** was explicitly held before starting in response to the storage
priority: `PENDING (JobHeldUser)`, priority zero, runtime zero. Its output directory
does not exist. The hold receipt is `storage_hold_20261111.json` beside its
unchanged frozen script. These probes are engineering evidence from synthetic
observations, not task success rates or background-OOD evaluation.

## Prepared FastWAM-only retry with bounded storage

The focused retry was prepared for review at
`/projects/_hdd/isaachdd/piper-model-validation/prepared-fastwam-retry-meb9zyun/`.
It references the existing step-2 checkpoint and existing pretrained assets;
there is no initialization export, training, new model environment or copied
weight. The prepared directory occupies approximately 8.74 MiB including frozen
sources, one pure-Python websocket overlay, metadata and CPU evidence. All ten
required external asset files exist; their large content has not been freshly
hashed. The checkpoint itself matches its saved SHA256.

`fastwam_existing_probe.sbatch` requests 2 Pro6000, 8 CPUs, 67,584 MiB and 30
minutes. Only `cuda:0` is used; the second allocation supplies host-memory quota.
Its SHA256 is
`58f7243e1278315a3828053f7ee243ad533c6cee792dd5a9530f8ad1b3b2ce77`;
the source-manifest SHA is
`f7b834e0688ce2602dd13eb3db00620b606c55107137b9be7a38079b3336aecd`.
`bash -n` and `sbatch --test-only` succeeded. After review of those exact inputs,
job **1734847** was submitted at `2026-11-11T16:00:29Z`; its receipt is
`submission.json` in the preparation directory. Output is
`/projects/_hdd/isaachdd/piper-model-validation/fastwam-retry-1734847/`, and the
Slurm log is `EWM/logs/fastwam_existing_probe_1734847.log`. The plan SHA256 is
`0faa0fd3609f951b8e56f55ab37a959ab4225f4e6881fcf803ece73c632a5189`.
A number printed by test-only is not the submitted job. OpenWAM job 1734785
remains held. Submission alone is not a successful GPU result.

The batch places common temporary, XDG, Hugging Face, torch, Inductor, Triton,
CUDA, ModelScope and related caches under its own new output directory, preserves
`HOME`, and disables automatic downloads. `bounded_probe.py` samples combined
preparation/output storage and the owned process tree every two seconds. It stops
the owned client/service processes at 512 MiB and imposes a 64 MiB per-child-file
write limit, leaving headroom within the 1 GiB task budget. This is a sampling
guard plus a per-file limit, **not a filesystem quota**. It records CPU RSS,
available cgroup memory counters/events, termination cause and final directory
sizes. The native wrapper enables INFO startup logs and records allocator peaks
when it exits normally; allocated-GPU sampling remains separate. OOM evidence
must distinguish cgroup events, CUDA errors and unavailable metrics.

Four actual CPU tests cover cache placement, avoiding external symlink traversal,
successful subprocess evidence, and budget-triggered termination of a child
service in a separate session. Three native-wrapper instrumentation tests pass.
No GPU work was done during this preparation.

The formerly skipped FastWAM processor test was also executed successfully using
the existing `envs/fastwam/bin/python -I -B`, an isolated stdlib-only test harness,
and explicitly frozen first-party source paths. There was no installation into
the existing environment and no pytest skip. The real native processor,
min/max normalizer and policy adapter verified horizontal camera order,
normalized 8D proprio, prompt formatting and 7D physical action denormalization;
only the heavy sampler was doubled. Evidence is
`cpu_processor/result/result.json` in the preparation directory. This establishes
the CPU boundary, not a full-model GPU result or a fresh-environment installation.

### FastWAM retry completed

Job **1734847 completed successfully**, exit `0:0`, in 6m49s
(`2026-11-11T16:02:41Z`–`16:09:30Z`). The actual step-2 checkpoint loaded, hello
matched the descriptor, one finite float32 `[1,7]` physical action was returned,
timestamps/step/control rate matched, and the model service shut down normally.
Startup was 371.462 s, model inference 8.610 s, and request round trip 8.916 s.
This is a first request on a two-step engineering checkpoint, not a formal
trained-policy latency or task-success result.

The model process recorded peak RSS **37.668 GiB**, CUDA allocated peak
**25.028 GiB**, and reserved peak **25.631 GiB** on `cuda:0`. The sampled owned
process sum peaked at 37.013 GiB; cgroup memory including filesystem cache peaked
at 65.997 GiB with zero reported OOM kills. These are different measurements.
Outputs occupied 638,976 bytes before adding the small audit JSON; preparation
plus output remained about 9.35 MiB. No new checkpoint or environment was created.

The frozen GPU sampler could not use Slurm's global IDs `5,6` in the process's
visible NVML namespace `0,1`; its errors are retained. Actual allocator peaks
come from `prediction/server_resources.json`. A small allocation-local query
verified the visible UUIDs. Its supplemental sampling step `1734847.1` was
intentionally stopped after 57 s; that cancelled diagnostic step does not change
the main job/batch success and is not a full-run memory measurement. The new
OpenWAM wrapper must use allocation-visible UUIDs without editing this frozen job.

`fastwam-retry-1734847/audit_summary.json` records the exact action/hash, reply,
resources, Slurm success and these monitoring limits. OpenWAM job 1734785 remains
held; its separate temporary-checkpoint proposal has not executed.

## OpenWAM disposable-checkpoint wrapper: prepared only

The small reviewed-wrapper candidate is under
`/projects/_hdd/isaachdd/piper-model-validation/planned-openwam-ephemeral-20261111/`
and occupies 39,424 bytes. It references the unchanged earlier OpenWAM source
freeze rather than copying it. The script SHA256 is
`c1e2ee2ba6a7f73e1bb3df0bf02a670d9ec17393c9703611fdf6bcc533199c9c`.
It has **not** run or been submitted, and the old job 1734785 remains held.

Container GPU acceptance is no longer a prerequisite or project goal. Execution
instead requires a root-authored storage-release receipt confirming that useful
evidence was archived, the task-owned approximately 26 GiB unpacked rootfs was
removed, the separate policy installation is idle, and the 26 GiB temporary
budget is available. Only a deliberately false example receipt currently exists.
Without `--execute`, the wrapper reports the plan and creates no output.

The unchanged full native model initializes from existing bases, writes exactly
one untrained checkpoint, strictly reloads it through the public legacy-stats
adapter, and requests one physical action. Native deployment assets add about
20.5 MiB of tokenizer files, not another weight copy. The estimate is about
23.8 GiB peak disk with bounded caches; it is not a measured export size.
The proposed allocation is 2 Pro6000 / 8 CPUs / 66 GiB / 45 minutes, using
`cuda:0`. Export has a 25 GiB per-file limit; metadata/inference use 64 MiB.
Total output stops at a sampled 25.5 GiB and caches at 512 MiB, including the
historical helper's separate plan/prediction compiler-cache directories.

After all recorded model processes exit, the wrapper archives checkpoint
SHA256/header/size, config/stats hashes, action and timing evidence, resource
samples and logs, verifies that evidence, and deletes only its exact generated
checkpoint plus explicitly owned caches. The same rule handles failed untrained
exports, marking partial files honestly. Original weights/statistics are never
cleanup targets. Failure to archive or retire produces a failure record and
nonzero completion; a batch killed before cleanup requires root audit.
Descriptors/plans retain their original bytes and paths as historical evidence;
`checkpoint_lifecycle.json` marks `retained=false` and
`plan_is_executable=false` after retirement.

Fifteen small CPU tests passed: nine lifecycle/identity/device-mapping checks and
six resource-guard checks. They include real subprocess budget termination,
symbolic-link protection, live-process refusal, partial-file retirement and
normal plan-only behavior. GPU sampling now resolves allocation-visible UUIDs;
this fix exists only in the new helper snapshot and does not alter prior jobs.

The final **v3 review** is separately preserved at
`/projects/_hdd/isaachdd/piper-model-validation/planned-openwam-ephemeral-20261111-v3/`
(41,472 bytes); the earlier preparation remains unchanged. Its script SHA256 is
`eb5baa0ae1eacdb0f31afbc635a7ca1549521dcd74b856829a2abad0041d3094`.
The wrapper resolves every checkpoint/cache path and requires it to remain under
the owned output, including when an intermediate `plan` or `prediction` directory
is a symlink. It checks again immediately before deletion. Cleanup ignores
repeated TERM/INT while stopping processes, archiving and retiring, then restores
the previous handlers. SIGKILL still requires root post-audit.
Eight directly related lifecycle tests passed, including external parent-symlink
rejection and actual repeated TERM/INT delivery during cleanup. Shell syntax and
frozen v3 plan-only checks passed. No execution receipt or new job was created.
