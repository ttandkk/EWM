# Piper RGB-action EWM mode

Use `train_ewm_rgb_action_piper.yaml` to train a joint flow model on the Piper 30 Hz dataset at
`/projects/hdd/myproject/CVPR2027/datasets/piper_stack_fronttight_v3_50/job177683`.
This mode does not read event-frame payloads per training sample.
It retains the same frozen DINOv3 dual-RGB context, T5 instruction context,
proprioception, heterogeneous video/action experts and DINO injection frequency
as event-action. Only the video branch's observed/target modality changes.

For an anchor row `k`, the input is RGB from both cameras at `k`, the 8-D state at `k`,
and the instruction. The video target is wrist RGB at `k+1...k+24`; the action target is
the normalized 7-D recorded command at `k...k+23`. Piper's
`meta/piper_temporal_contract.json` records that `action[k]` acts after observation `k`
and before observation `k+1`. RGB targets use the original 256 x 256 resolution.
Set both `model.event_camera_mode=dual` and `dataloader.event_camera_mode=dual` to
predict both RGB cameras; the action target stays unchanged.

The frozen local LeanVAE-16 encodes real RGB in its native input range. The video expert
predicts RGB video latents and the action expert predicts the 7-D action sequence.
The total loss is `loss_rgb + loss_action`. `rgb_metrics.csv` reports decoded
teacher-noised x0 MSE, MAE, and PSNR; these are diagnostics, not sampled rollout scores.
Checkpoints use `ewm_rgb_action_latent_v1` so event-only evaluators cannot mistake
them for event-latent checkpoints.

Launch from `OpenWAM/` using the existing environment and entry point:

```bash
LD_LIBRARY_PATH=/projects/hdd/myproject/CVPR2027/envs/torchcodec_ffmpeg7/lib /projects/hdd/myproject/CVPR2027/envs/openwam/bin/python scripts/train.py --config-name train_ewm_rgb_action_piper
```

The RGB loader supports exact indexed PyAV or TorchCodec decoding. The preset uses
TorchCodec; the aligned production launcher uses PyAV like the historical event-action
run. The dataset normalizer continues to normalize Piper state and action; RGB stays
uint8 until the frozen VAE and DINO paths. Real RGB maps to Lean's native [-0.5,0.5],
without event count normalization or pseudo-RGB polarity mapping. Sampled `future_rgb`
is floating [0,1], not uint8. Selecting a config alone does not submit a job.

## Aligned production Slurm entry

The independent `scripts/train_ewm/train_ewm_rgb_action.sh` does not change the
existing joint/action-only launcher. Its defaults match the event-action launch and
job179463: two RTX6000 Ada GPUs, video D768 / action D512, H8/L12/K24, latent patch4,
dual-camera prediction, BF16, batch32 per GPU (global64), 80k optimizer steps,
AdamW 1e-4 / betas0.9,0.95 / weight decay0.01, clip1, seed42, cosine schedule with
4000-step warmup, DINO injection every layer, and compiled denoising core.
LeanVAE, DINOv3 and T5 remain frozen. VAE batch size8 is also unchanged.

```bash
cd /projects/hdd/myproject/CVPR2027/OpenWAM
sbatch scripts/train_ewm/train_ewm_rgb_action.sh

# Inspect the command without creating outputs, allocating GPUs or training.
DRY_RUN=true bash scripts/train_ewm/train_ewm_rgb_action.sh

# Optional wrist-only video target; current DINO context remains dual RGB.
sbatch --export=ALL,RGB_CAMERA_MODE=wrist_only scripts/train_ewm/train_ewm_rgb_action.sh

# Same model/data/optimization on two L40 GPUs when Ada quota is occupied.
# Multi-GPU compilation is only validated on Ada; use eager core on L40.
sbatch --gres=gpu:l40:2 --export=ALL,COMPILE_CORE=false scripts/train_ewm/train_ewm_rgb_action.sh
```

The script uses the same 50-episode Piper export, action/state normalization,
non-overlapping control intervals and overlapping stride1 windows as event-action.
At anchor k it jointly encodes current RGB plus future24 per camera into7 causal
latent slices, keeps the first as condition, and supervises the six future slices
and24 aligned actions using shared flow time and post-encoding Gaussian noise.
Dual-camera256x256 prediction therefore retains896 video tokens plus25 state/action
tokens, matching the dual-event model. No event payloads, caches or voxelization run.

Online W&B upload is enabled by default (`WANDB_MODE=online`, project `CVPR2027-EWM`);
use the existing login or `WANDB_API_KEY`, and optionally set `WANDB_ENTITY`.
Logs are `logs/ewm_rgb_p4_dual_ada2_80k_<job>.log`; outputs are
`OpenWAM/outputs/ewm_rgb_action_lean16_p4_d768_a512_dual_ada2_80k/job<job>/`.
The trainer reports `loss_rgb`, `loss_action`, latent diagnostics and decoded RGB
MSE/MAE/PSNR, not event precision/recall/F1. Latent losses across modalities need
not have the same numerical scale; use matched closed-loop evaluation for task success.

Policies and full resume states save every10k steps and at the final step, with4
retained snapshots by default. `RESUME_CKPT_PATH=/path/to/run` restores the matching
RGB-action policy, optimizer, scheduler, RNG and sampler position. Event/RGB/action-only
checkpoint formats cannot be mixed on resume. Set
`SAVE_CHECKPOINTS=false SAVE_FULL_STATES_FOR_RESUME=false SAVE_STEPS=0` for diagnostics.
The event-specific pure-noise evaluation callback is intentionally unsupported.

## Post-trial prediction videos (2026-11-07)

The eight-task ID/OOD suite and shared conveyor evaluator default to
`--record-predictions` for both `rgb_action` and joint event-action policies.
Action-only does not allocate a prediction archive. Use `--no-record-predictions`
to disable this diagnostic explicitly. Controller, flow-step, and latency options
remain independent of recording.

During a trial the server samples actions and visual latents without decoding.
It copies the exact observed latent and future latent into a bounded CPU archive
(at most 1 GiB), so subsequent CUDA Graph requests cannot overwrite them.
These copies are included in the measured action reply round trip; VAE decoding
and prediction-video encoding are not. No visual latent is sent with action replies.

After control stops, only plans actually used for actions are decoded, on the
same inference thread. The saved trial video has three rows: real dual RGB,
observed dual events, and predicted RGB/events. Prediction index `j` from
observation `k` is displayed at frame `k+j+1`; skipped stale action prefixes stay
skipped, and exhausted asymmetric event horizons display an unavailable panel.
Unused plans are discarded and the archive is cleared before the next trial.
The temporary real-only video is replaced only after successful video validation,
without retaining a duplicate full video or latent files on disk.

Manifest/summary fields record `record_predictions`, `prediction_decode_phase`,
and `prediction_decode_in_equivalent_latency`. Per-trial `prediction_recording`
records decoded anchors/seeds, the archive peak, and post-trial processing time.
Resume checks reject mixing these settings with a different saved protocol.
The older independent stack RGB diagnostic is a separate evaluator; it is not
the eight-task ID/OOD entry point.

## Validation

The RGB/model/attention/resume/action-only/event-only regression suite passes77 tests.
A two-process CPU DDP check also verifies finite losses, gradients on both trainable
experts, frozen encoder weights and synchronized optimizer updates.

L40 smoke job181053 completed successfully on `gpu-l40-1` in3m12s
(2026-10-03 00:35:02–00:38:14 China time). It used the real50-episode Piper export,
local frozen LeanVAE/DINO/T5 and the full D768/A512/H8/L12/K24 dual-view BF16 model.
Three training steps produced finite RGB/action losses and decoded RGB diagnostics.
This bounded check used one GPU, batch1, an eager core, disabled W&B and no checkpoint
saving; it does not measure production batch32 throughput or validate two-L40 GPU
memory capacity. Configuration, metrics and timings are under
`tmp/ewm_rgb_action_smoke/results/job181053/`; the log is
`logs/ewm_rgb_action_smoke_181053.log`.
