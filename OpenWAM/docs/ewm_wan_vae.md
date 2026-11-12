# EWM configurable event VAE latent flow

> The defaults below describe the base codec configuration. Eight-task training presets and saved checkpoint configs override camera mode, temporal stride and compilation. Use the [current evaluation protocol](../../docs/evaluation_protocol.md) for formal runs; do not infer their settings from the historical base defaults here.

## Defaults and codec selection

Production defaults: frozen **LeanVAE-16 BF16**, joint current1+future24 encoding, latent **4×4 patch**, Event D768 / Action D512, H8/L12/K24, wrist-only events with dual-view RGB context. Both experts project Q/K/V into the same 8×96 attention space, following OpenWAM MoT's heterogeneous residual-width layout. Gradient checkpointing and torch.compile are **off**. Online encoding uses codec_batch_size16 videos per call. Ordinary training steps do not decode.

| Selection | VAE latent C,T,H,W at256px | VAE stride | DiT patch | Total stride | Event tokens |
|---|---|---:|---:|---:|---:|
| `model=ewm` (default), or `model=ewm_leanvae16` | 16,7,32,32 | 8 | 4×4 | 32 | 448 |
| `model=ewm_wan22` | 48,7,16,16 | 16 | 2×2 | 32 | 448 |

Token counts include the observation slice, assume one event camera and K24; dual events double them. Action/proprio still contributes25 tokens. Set `model.event_patch_size` to 2, 4 or 8 and keep `model.event_spatial_compression` equal to VAE stride times patch size. Old checkpoint configs without `event_patch_size` infer it from their saved total stride.

```bash
cd /projects/hdd/myproject/CVPR2027/OpenWAM
# Current defaults
python scripts/train.py --config-name train_ewm
# Wan alternative
python scripts/train.py --config-name train_ewm model=ewm_wan22
# Piper joint two-card launch (explicit dual events)
sbatch scripts/train_ewm/train_ewm.sh
# Mixed REASSEMBLE/DVLA event-only entry
python scripts/train.py --config-name train_ewm_event_only
# Piper action-only ablation (no event/VAE path)
python scripts/train.py --config-name train_ewm_action_only_piper
# Same two-card Piper launcher, with a separate action-only output directory
sbatch --export=ALL,TRAINING_MODE=action_only scripts/train_ewm/train_ewm.sh
```

Both joint codecs use temporal stride4. `EWM_VAE_PATH` overrides the selected weight directory; unset a previous codec's override before switching. Defaults are `pretrained/leanvae` (LeanVAE-dim16.ckpt) and `pretrained/wan2.2-vae`. The Python Lean codec also accepts an exact checkpoint file path. The Piper Slurm launcher explicitly overrides camera geometry and compilation; it is not a mixed event-only launcher.

### Optional LeanVAE temporal input sampling

`model.event_temporal_downsample=1|2|3` selects every Nth future frame **before**
the frozen LeanVAE encoder. The default is 1, including old saved configs without
this field. `event_temporal_compression` remains 4, the native pretrained VAE
stride; this option does not modify the VAE architecture or weights. Both
`leanvae16_joint` and `leanvae16_rgb_joint` support it. Wan keeps stride 1 here.

For the current 24-step horizon and 256×256 images with patch4:

| Input sampling stride | Input frames including observation | Latent slices including observation | Wrist / dual visual tokens | Dual visual + state/action tokens |
|---|---:|---:|---:|---:|
| 1 (default) | 25 | 7 | 448 / 896 | 921 |
| 2 | 13 | 4 | 256 / 512 | 537 |
| 3 | 9 | 3 | 192 / 384 | 409 |

The observation stays at offset 0. Future offsets are `[2,4,...,24]` at stride2
and `[3,6,...,24]` at stride3. Frames are selected, not averaged or accumulated;
short event bursts between selected frames can be lost. All 24 action targets,
their control rate and current RGB conditioning stay intact.
Visual RoPE timestamps use the original frame/action time units.

**Stride4 is not supported. No frames are added to make a sampling stride fit.**
The horizon must be divisible by `4 * event_temporal_downsample`; incompatible
values fail before VAE encoding. The pre-existing dataset tail-padding policy
is unchanged: only complete groups of real source frames receive latent supervision.

```bash
# Example override; this starts training when executed.
python scripts/train.py --config-name train_ewm_eight \
  model.event_temporal_downsample=2
# The joint and RGB-action Slurm launchers also accept this environment variable.
sbatch --export=ALL,EVENT_TEMPORAL_DOWNSAMPLE=2 scripts/train_ewm/train_ewm.sh
```

Prediction APIs continue returning 24 actions and 24 display frames: each decoded
sampled frame is repeated across its original interval. These repeated display
frames do not recover missing temporal detail and are never fed back as added
training frames. Decoded diagnostics score only the sampled timestamps.
The selected stride is saved in `model_config`; use the same setting for training
and evaluation. An existing stride1 policy is not a validated stride2/3 policy:
train or fine-tune a separate run when changing it. Existing source frames and
normalization formats need no changes for this option. This change alone does
not establish a quality improvement or a measured GPU speedup.

### Action-only mode

`model.training_mode=action_only` pairs with `dataloader.type=ewm_piper_action_only`.
The preset uses the same Piper export and normalization as joint training:
current dual RGB, 8-D state and text condition `action[k:k+24]`. Only the
current RGB frames are decoded; Parquet column projection reads state,
action and task index, without loading event PNGs or future RGB targets.

The event expert is not constructed: no event input/head/camera parameters,
per-layer event branches or VAE are allocated. Each action layer attends only
the state token plus the action sequence and retains its RGB/text cross
attention. The action expert retains D512 residuals and D768 attention geometry.
New policy checkpoints contain only the active core and context projections.

The objective is only action velocity MSE (`loss == loss_action`). Low-frequency
diagnostics write normalized-action `action_x0_mse`/`action_x0_mae` to
`action_metrics.csv`; these are teacher-noised diagnostics, not task success.
No event/latent diagnostics are computed. The shared debug CSV retains a
legacy `loss_video=0` placeholder, which is not an event loss.

New action-only policy payloads use `ewm_action_only_v2`, record their mode in
`model_config`, and load through `eval_ewm_policy.load_policy`. `sample_policy`
requires RGB/state/text and a normalizer, and returns only `action` and
`action_normalized` (no future event/RGB/latent output). Since no VAE is
involved, action horizon need not be divisible by four. The existing joint
pure-noise callback does not apply to this mode. Event-guided replanning has been removed.
Full-state training resume retains the existing model/data compatibility
checks. Defaults for the other three modes are unchanged. The action-only
batch launcher defaults compilation off; enable it only after profiling.

#### Independent Piper action-only evaluation

Use `scripts/eval_ewm/run_ewm_action_only_piper_eval.sh`, not the joint/event
evaluation launcher. It starts its own `ewm_action_only_eval_server.py` and
`eval_ewm_action_only_piper_sim.py`. The server accepts `ewm_action_only_v2`
and compatible single-frame `ewm_action_only_v1` policies from
Piper action-only runs, and sends current dual RGB, normalized 8-D state
and instruction to `sample_policy`. It rejects event arrays and event
prediction/replan flags. No EVIS generation or event VAE decoding runs.

```bash
cd /projects/hdd/myproject/CVPR2027/OpenWAM
# Run after the selected policy checkpoint has been saved; omit the step for latest.
bash scripts/eval_ewm/run_ewm_action_only_piper_eval.sh \
  --checkpoint-dir outputs/ewm_action_only_d768_a512_ada2_80k/job180957 \
  --checkpoint-step 80000
```

Defaults match the joint fixed-horizon launcher: 50 trials, 500 control
steps, 4 flow-sampling steps, execute at most 16 actions per replan, model
seed 10000, scene seed 20260930 in stream mode, 30 Hz control, path tracing
at 64 SPP, the same cameras/materials/table/cube resets, 60 warmup steps
plus one initial capture, and success sustained for 10 consecutive steps.
Action normalization, model sampling seed rule, optional RGB-only OOD,
watchdog/restart/resume, timing and horizon statistics also match.
Compilation remains off and context K/V caching on by default.

Outputs are isolated under `<run>/eval_piper_sim_action_only/job<id>/`:
`manifest.json`, `summary.txt`, `complete.json`, per-trial initial state,
step/result JSON, scene RNG snapshots and H.264 videos. Videos contain
only the two RGB views (512×256, 30 FPS); event and predicted-event rows
are intentionally absent. The manifest identifies action-only mode and
records policy/config/normalization hashes. `--output-dir` can override
the destination without overwriting an already completed run.

For strictly paired initial scenes, optionally add
`--scene-rng-reference-dir /path/to/trusted/old/eval/jobNNN`. This restores
the old `scene_rng_before_trial_NNN.pt` before each reset; all requested
snapshots and the matching stream-mode scene seed are required. Scene
RNG files use PyTorch serialization: use only trusted project outputs.
Initial cube poses/joints are recorded for comparison. Without replay,
the original seed/stream protocol is retained; the same seed alone does
not guarantee bitwise-identical trajectories across different policies
or simulator/GPU versions. The action-only noise stream has no event
noise, so it intentionally does not match the joint model's full noise stream.

## Representation and time alignment

Input voxels are `[B,T,camera,2,H,W]`. Piper uses one-bin OFF/ON counts normalized with count_clip15. Mixed event-only uses native `count > 0`, area resizing and letterboxing to floating occupancy `[0,1]`, without count/15; DVLA blue/red disk channels are swapped to canonical OFF/ON. See `../../docs/dvla_reassemble_joint_pretraining.md` for polarity evidence and remaining distribution differences. Pseudo RGB is `[OFF,(OFF+ON)/2,ON]`; use floating values, no polarity cancellation, uint8 quantization or adaptive contrast. Lean expects[-0.5,0.5], Wan[-1,1]. Both codecs are frozen/eval. Lean weights and compute stay BF16 even under a different outer autocast setting; mapping and output restoration use FP32. The previous FP32 Lean integration was superseded by the requested BF16 default.

Encode the full25-frame window once. Keep first latent as observation condition and add Gaussian noise only to six future slices. Flow coordinates use each codec's native convention (raw deterministic Lean latents; pretrained normalized Wan latents), without old voxel event_scale. Actions retain24 steps. Event/action loss weights are equal; latent loss magnitudes need not match across codecs. Temporal RoPE places future latent groups at2.5,6.5,...,22.5. Horizon must be divisible by4.

At sampling, encode only the observation, integrate future latent/action flow, concatenate observed+future latent slices, decode25 frames, discard the observed frame. Do not decode future latents in isolation. Causal future-perturbation checks found no future leakage. Lean's BF16 joint/single-frame observation latents have previously shown finite-precision differences; this is a known numerical difference, not a claim of bitwise alignment. Do not apply the old FP32 observation tolerance to BF16.

`wan22_frame` remains an explicit temporal-stride1 option, requiring matching configuration. Do not mix frame and joint checkpoints/caches.

## Checkpoints, dependencies and metrics

Event-predicting policy payloads retain historical format string `ewm_wan_latent_v2`; RGB-action uses `ewm_rgb_action_latent_v1`, and action-only uses `ewm_action_only_v2`. Resolved model_config records the training mode, codec, path and patch geometry. VAE/DINO/T5 are external dependencies excluded from policy state (action-only does not load a VAE). Wan/Lean or patch2/patch4 trainable projections have different shapes: use the saved configuration to restore a policy, not the latest default. Strict policy loading rejects mismatched keys/shapes. Policy files do not include optimizer/RNG resume state; old dense ewm_trainable_v1 checkpoints are incompatible.

For new runs, set `training.save_full_states_for_resume=true` (batch launcher:
`SAVE_FULL_STATES_FOR_RESUME=true`). Each policy save also writes
`resume_state_step_N/` with optimizer, scheduler, per-rank RNG and the
current epoch's batch position. The default
`training.keep_last_k_resume_states=2` keeps the two newest complete states.
These files are substantially larger than policy checkpoints. Resume with
`training.resume_ckpt_path=/path/to/original/run` (batch launcher:
`RESUME_CKPT_PATH=/path/to/original/run`); the trainer reuses that run
directory and selects the latest complete state. Keep model/data settings,
GPU count, batch size, gradient accumulation and `lr_schedule_steps`
unchanged. `max_steps` may increase when the original learning-rate horizon
was already long enough.

For the two-Ada launcher:

```bash
SAVE_FULL_STATES_FOR_RESUME=true sbatch scripts/train_ewm_lean16_p4_d768_a512_dual_80k_ada2_online.sbatch
SAVE_FULL_STATES_FOR_RESUME=true RESUME_CKPT_PATH=/path/to/original/run-dir \
  sbatch scripts/train_ewm_lean16_p4_d768_a512_dual_80k_ada2_online.sbatch
```

Older policy-only runs (including job178351) cannot be resumed exactly;
they were saved without optimizer and RNG state.

Lean uses strict weights_only checkpoint loading and validates16 channels. MIT-licensed official inference code is vendored under `openwam/model/video_backbone/encoder/leanvae_src/`, with pinned upstream commit and local-change notes. No temporary scripts, Lightning, LPIPS or extra training dependencies are required. Weight provenance/checksums are in each pretrained directory's source.json.

`loss_event`, latent_flow_mse and latent_x0_mse are latent metrics. Low-frequency diagnostics (one sample/rank, every200 steps and final step) decode to normalized voxels, using occupancy threshold1/30. These teacher-noised x0 diagnostics are not full forecasting evaluation. Online encoding remains active; no latent cache is implemented. No dense codec is reintroduced.

## Event and observation time alignment

The Piper collector calls `env.step(action[k])` before saving RGB and robot
state to row `k`, while assigning that row the pre-step timestamp `t_k`.
The RGB/state in row `k` therefore describe the scene at approximately
`t_{k+1}`. The raw event stream and action[k] describe the transition
`[t_k, t_{k+1})`. The EWM event loader uses a 30 Hz causal interval; for
ordinary frames, voxel frame `k` matches the exporter's future-event offset
`k-1`, and target frame 1 matches future-event offset `k`.

Run 175042 was trained before this collection-time offset was identified. Its
saved config has no `observation_row_offset`, which resolves to legacy `0`:
its RGB/state are one 30 Hz control interval ahead of its observation event
voxel, and overlap the first target event/action interval. The historical
training and evaluation artifacts remain unchanged.

Inference now always uses physically aligned observations, including with
legacy checkpoints: RGB and proprioception are taken at the end of the
observed event interval. Offline evaluators override the saved training
offset to `-1`; their default indices preserve the historical episode and
target anchors. Their newly measured F1 may differ from the old 175042
numbers because the observation inputs have changed. This is an inference
time correction, not a repair of the weights trained on shifted inputs.

Training defaults remain at `dataloader.observation_row_offset=0` for now.
Aligned inference sets it to `-1`: RGB/state come from row `k-1`, while
action[k], observation event voxel `k`, and future event voxels
`k+1..k+24` keep their original intervals. The aligned loader starts at
anchor `k=2` to avoid an export-time-zero boundary exception. The client/server evaluator checks each requested event
window against saved `event_future_offsets.npy`, requires aligned inputs
in its handshake and prediction requests, and records both the historical
training offset and the inference offset in its manifest.

The standalone `scripts/eval_ewm/eval_ewm_policy.py --dataset` path uses the same
aligned loader and audits event offsets. Its live `--env-factory` path
requires `obs_video_time_s`, `proprio_time_s`,
`obs_event_start_s`, and `obs_event_end_s` in every observation.
RGB and proprioception timestamps must coincide with the event interval
end, and the event interval must span one 30 Hz period. Missing or
mismatched timestamps raise an error before inference.

## Optional pure-noise rollout evaluation

Set `training.pure_noise_eval.enabled=true` (or `PURE_NOISE_EVAL_ENABLED=true`
with the batch launcher) to evaluate generated 24-frame events after each
5,000 optimizer steps and at the final step. It is disabled by default.
The callback first saves an immutable policy checkpoint, then submits a
separate single-L40 Slurm job. Training waits only for `sbatch` to return,
not for sampling or decoding. When enabled, the evaluation interval also
becomes a checkpoint-save interval, even if `training.save_steps` is larger.
`training.save_checkpoints` must remain true.

```bash
sbatch --export=ALL,PURE_NOISE_EVAL_ENABLED=true,PURE_NOISE_EVAL_EVERY=5000 \
  scripts/train_ewm/train_ewm_d512_l12_k24_t2_wrist_ada2_80k.sh
```

Defaults are `partition=cluster02`, `gres=gpu:l40:1`, and
`time=00:30:00`; override with `PURE_NOISE_EVAL_PARTITION`,
`PURE_NOISE_EVAL_GRES`, and `PURE_NOISE_EVAL_TIME`. Direct Hydra launches
can override the matching `training.pure_noise_eval.*` fields. Evaluation
uses the fixed training windows and paired seeds in
`scripts/eval_ewm/eval_ewm_pure_noise.py`, 20 reverse Euler steps, and the same
F1/precision/recall/occupancy/voxel metrics as the four-checkpoint audit.
These scores track training progress, not held-out generalization.

For manual evaluation of saved checkpoints, use the same entry point. It
loads the run's saved `config.yaml` and `normalization.json`, constructs the
training EWMDataset, and calls the deployment `sample_policy` method. It
checks that horizon, event camera mode, voxel bins and normalization match
before sampling. The `dataset_dir` in the saved config points to the LeRobot
export produced by Event-DataCollection; this command does not consume raw
HDF5 episodes directly. If `--steps` is omitted, all saved policy checkpoints
are evaluated.

For one-command client/server evaluation, use `scripts/eval_ewm/run_ewm_client_server_eval.sh`:

```bash
bash /projects/hdd/myproject/CVPR2027/OpenWAM/scripts/eval_ewm/run_ewm_client_server_eval.sh \
  --checkpoint-dir /projects/hdd/myproject/CVPR2027/OpenWAM/outputs/ewm_lean16_p4_d512_80k/job175042 \
  --steps 80000
```

The launcher reserves one L40 through Slurm, starts a local EWM WebSocket
server, waits for readiness, runs the dataset client, and stops the server.
Without `--steps`, the client evaluates the latest saved checkpoint. Results
go under `<run>/eval_client_server/<UTC timestamp>/`; process logs go under
the project `logs/` directory. The client constructs the same EWMDataset from
the run's saved config, uses its saved normalization, sends only current RGB,
current OFF/ON voxel, proprio and prompt for generation, and scores against
future targets locally. Server and client compare config and normalization
hashes before evaluation. The server calls the same `sample_policy` path as
deployment, with BF16 autocast and FP32 flow state. Clean-codec oracle requests
carry future targets separately and are excluded from generation requests.
This EWM server is separate from OpenWAM's general WAMPolicy deployment server,
whose WebSocket protocol returns actions from camera PNGs.

Inference can use `model.eval()` followed by
`model.compile_inference(cache_bf16_weights=True)` and sampling inside
`torch.autocast("cuda", dtype=torch.bfloat16)`. This keeps a separate BF16 copy
of the velocity core's Linear weights at fixed addresses for CUDA Graph reuse;
FP32 training parameters, norms, VAE and DiT structure remain unchanged. The
cache adds GPU memory and is excluded from policy checkpoints. Warm up the
requested sampling length before timing; compilation/capture is lazy. Entering
training, loading a state dict, or moving device/dtype disables this cached
compile. Recompile afterwards, including after direct in-place weight edits.
The eval server enables this cache with `--compile-core` by default and rebuilds
it after checkpoint switches; `--no-cache-bf16-weights` selects ordinary compiled
inference. The generic API defaults the cache off for compatibility. Sampling an
already-eval model no longer recursively switches mode twice per request.

```bash
/projects/hdd/myproject/CVPR2027/envs/openwam/bin/python \
  /projects/hdd/myproject/CVPR2027/OpenWAM/scripts/eval_ewm/eval_ewm_pure_noise.py \
  --checkpoint-dir /projects/hdd/myproject/CVPR2027/OpenWAM/outputs/ewm_lean16_p4_d512_80k/job175042 \
  --steps 80000 \
  --output-dir /projects/hdd/myproject/CVPR2027/OpenWAM/outputs/ewm_lean16_p4_d512_80k/job175042/eval_manual_80k
```

Each result goes to `<run>/pure_noise_eval/step_XXXXXX/`. For the first two
visualized windows, `comparison_<index>.png` shows binary OFF/ON occupancy on
white, `comparison_<index>_counts.png` shades quantized count levels, and
`comparison_<index>_counts.npz` stores exact two-channel integer count grids
`[24, 2, 256, 256]` for GT, persistence, clean codec and the checkpoint's
first sampled seed. Predicted voxels are thresholded, then rounded to the
GT count grid `0..count_clip` for display only. Metrics continue to use the
unmodified continuous decoder output.

The main process records the submitted job ID, or a Slurm submission error, in
`<run>/pure_noise_eval/submissions/step_XXXXXX.json`. Submission failure
does not stop training; rerun the standalone evaluator for any missed
checkpoint. Slurm logs go under the project `logs/` directory.

## Validation history and limits

Historical Wan speed: job174328 approximately3.90s/update (GC on, compile off); controlled bottleneck report is `tmp/ewm_joint_bottleneck/REPORT.md`. Historical Lean integration: job174462 verified real-weight reconstruction, forward/backward and sampling at both patch sizes with FP32 codec;0.360s was an encoder-only FP32 batch32 measurement. BF16 codec-only0.144s from job174446 was also not full training throughput. These measurements predate the current default change.

Current BF16 Lean patch4, GC off, compile off real-data D512 training benchmark and regression results are recorded in `tmp/ewm_lean_speed/REPORT.md`. Short speed checks do not establish forecasting convergence. Checkpoint saving is disabled for diagnostics (`SAVE_CHECKPOINTS=false SAVE_STEPS=0`).
