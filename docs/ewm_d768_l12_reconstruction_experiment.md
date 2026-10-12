# D768 / L12 reconstruction trial

Trial configuration: `OpenWAM/configs/train_ewm_d768_l12.yaml` inherits the
standard training config and overrides D=768, heads=12, L=12, base_dim=32, and
its output directory. The standard D512/L8 defaults are unchanged.

## What the zero-baseline comparison measures

The earlier failing reconstruction experiment instantiated only
`EventVoxelCodec(dim=32, base_dim=2)`. It did not instantiate EWMJointBlock,
DINO/T5 context, or a rectified-flow sampler. Consequently L never entered
that experiment. Increasing L cannot directly change this particular result.

To test codec capacity fairly, the new comparison uses D512 versus D768,
both with production base_dim=32. Each run uses the same dataset sample 0,
future index 3 (V4), both cameras, two polarities, four bins and native 256x256
resolution. Each runs 100 Adam updates at learning rate 0.003 with seed 103,
unweighted voxel reconstruction MSE, and no clamp during scoring. Inputs are
clean voxels; this is a codec diagnostic, not EWM flow training.

Report global MSE, active-region MSE (target > 0), inactive-region MSE, active
fraction, and the all-zero predictor baseline. Lower error than random
initialization is not sufficient; compare against the baseline explicitly.
The resulting image uses the same display scale as the target. Different D
also changes the weight shapes, so identical seeds do not imply identical
initial weights.

## Reproduction

From the project root:

```bash
OMP_NUM_THREADS=3 PYTHONPATH=OpenWAM \
EWM_RECON_DIM=768 EWM_RECON_BASE_DIM=32 EWM_RECON_STEPS=100 EWM_RECON_LR=0.003 \
EWM_DIAGNOSTICS_DIR="$PWD/OpenWAM/outputs/ewm_validation/d768_l12_codec" \
envs/openwam/bin/python -u -m pytest \
  OpenWAM/tests/test_ewm_model.py::test_real_event_reconstruction_sanity -q -s
```

Repeat with `EWM_RECON_DIM=512` and a different output directory for the
matched production-width reference. The test defaults remain D32/base_dim2
when these experimental overrides are absent.

The isolated large codec has 26,507,782 parameters. The full D768/L12 EWM
has 170,118,144 parameters in each expert's joint branches and 367,946,765
parameters excluding external context modules (not including DINO/T5 or their
projections/embeddings). The trial's meta-device construction verified the
requested width, number of heads and layer count, without running its forward.

## Environment and interpretation limits

CUDA is unavailable. The session has a three-CPU quota despite twelve visible
logical CPUs. Tests use native CPU Conv3D kernels because earlier oneDNN
backward checks were inconsistent. These runs evaluate a single training
interval and cannot establish reconstruction generalization, policy quality,
or the benefit of twelve joint layers. Full D768/L12/K8 flow overfit and GPU
throughput remain separate production gates.

Artifacts are under `OpenWAM/outputs/ewm_validation/d512_l8_codec` and
`d768_l12_codec`; progress logs hold pre-update errors, and the final JSON
contains an additional post-update reconstruction evaluation.

## Result

Both 100-step runs completed. The D768 trial used 26,507,782 codec parameters
and took 2,137.58 seconds; its final global MSE was 0.000409306, **6.42 times
the all-zero baseline** (0.0000637775). Its active-region MSE was 0.00447747,
which is lower than the zero predictor's 0.00627645, but the predicted
background incurred 0.000367544 inactive-region MSE. It recovers some event
support while producing enough background/grid error to lose globally.

The matched D512/base_dim=32 codec used 13,857,798 parameters and took
4,569.74 seconds in this CPU-quota-constrained run. Its final global MSE was
0.000189966, **2.98 times the zero baseline**, and active-region MSE was
0.00639686, slightly worse than zero. Thus D512 was better on the global
metric here, whereas D768 was better only on active events; neither beats the
zero baseline globally.

This does not show that D=768/L=12 fixes the reconstruction issue. The
comparison exercises only the codec, so L=12 and the joint Event/Action
experts are not on the computation path. It does show that increasing codec
embedding width alone improves active-event fitting but does not suppress
background artifacts. The next useful controlled ablation is decoder output
regularization or interpolation-upsample+Conv3D, evaluated with the same
active/inactive and zero-baseline metrics; changing EWM depth is a separate
flow-model experiment.


## One-bin follow-up

The D768/L12 trial configuration exposes `voxel_bins`. The one-bin experiment keeps the causal window at 33.333 ms and merges only its four intra-window bins. It uses a dedicated codec branch with temporal stride 1 in every stage; it does not apply the v1 4→1 temporal compression to a length-one axis.

On the same real target (sample 0, future index 3, both cameras, D=768, base_dim=32, Adam lr=0.003, 100 steps), the CUDA run on an RTX 6000 Ada obtained global MSE **0.000471754**, below the all-zero baseline **0.000488637** by **3.46%**. Active-region MSE improved from **0.0175591** to **0.0150309** and inactive-region MSE was **0.0000550**. Thus merging bins resolves the specific codec-only global-zero-baseline failure for this target, though the margin is small and does not establish end-to-end flow quality. The codec has 23,656,966 parameters in this one-bin branch; L=12 does not participate in a codec-only experiment.

The reproducible report is `OpenWAM/outputs/ewm_validation/d768_l12_bin1_cuda/real_event_reconstruction.json`.


## Flow training diagnostics

The training objective remains `loss_event + loss_action`. Each train step now logs `flow_mse`, `flow_active_mse`, `flow_inactive_mse`, `x0_mse`, `x0_active_mse`, `x0_inactive_mse`, `event_precision`, `event_recall`, `event_f1`, `pred_nonzero_ratio`, and `target_nonzero_ratio` to the console, `event_metrics.csv`, and W&B when enabled. The active mask is `clean x0 > 0`; flow targets are deliberately not used to define it because they include dense Gaussian noise. `x0_hat = xt - t * v_hat` is diagnostic-only and does not change backpropagation. The occupancy threshold is `model.diagnostic_occupancy_threshold` (default `1/30`, half a clipped event count).

For a controlled probe, use a fixed group of eight sample indices, a fixed noise seed, and evaluate at `t = [0.1, 0.3, 0.5, 0.7, 0.9]` every 500 optimizer steps. Compare 1-bin and 4-bin runs only with the same indices, seed, and checkpoints. A healthy run should lower `x0_active_mse` while recall rises without a large increase in `x0_inactive_mse` or `pred_nonzero_ratio`.


## Event-coordinate scale

The D768/L12 one-bin configuration uses the default `model.event_scale: 10.0`. Before flow matching, the model multiplies current and future normalized event voxels by this value while retaining terminal noise `N(0,1)`. It divides sampled voxels by the same value before clamping and reports x0/occupancy diagnostics in original normalized voxel units. This is a coordinate transform intended to make a one-count event (`1/15`) comparable to flow noise; it does not alter the dataset or count normalization.
