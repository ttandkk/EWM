# EWM v1 implementation and validation

Implementation follows `ewm_v1_architecture_spec.md` and `implement.md`.
EWM-specific model code remains in `OpenWAM/openwam/model/architectures/ewm.py`.
The existing dataset, normalizer, DINO, T5, and video architectures are reused without changing their implementations.

## Implemented

- Four-bin, two-polarity 3D codec; 4→1 temporal and 16× spatial compression.
- p-major voxel conversion; clean current event/state prefixes; future-only decoding.
- Shared multimodal RoPE: 16/24/24 dimensions at head_dim=64. Actions only rotate the time subspace; `a_k` and `V_{k+1}` both use coordinate 1.
- Independent expert AdaLN/QKV/output/FFN modules and full bidirectional Event/Action mixed SDPA in every layer.
- Frozen OpenWAM DINOv3 and T5-small external context; learned modality/camera embeddings; actual RGB token count and text padding mask.
- One shared uniform flow time per sample, voxel/action Gaussian noise, independently averaged velocity losses summed with equal weights.
- Joint backward Euler sampling using `CosmosFlowSchedulerAdapter(shift_video=1)`; final event/action clamping and action denormalization.
- Arbitrary K; activation checkpointing for codec and joint blocks; production codec microbatch default 16 (CPU diagnostic tests use 2).
- `configs/model/ewm.yaml` supplies base defaults; `configs/train_ewm.yaml` centrally overrides model size (D=512, heads=8, L=8), codec width, memory settings, pretrained paths, and dataset horizon (K=8).
- Existing `scripts/train.py` dispatches EWM to `scripts/train_ewm.py`. This path reuses Hydra, Accelerate, dataset registry, seeding, cosine scheduler and CSV/W&B helpers without pretending events are video latents.
- Independent `scripts/eval_ewm_policy.py`: checkpoint loading, offline dataset inference, or Gym-style environment loop executing only the first physical action and reobserving each step.

## Run

From `/projects/hdd/myproject/CVPR2027/OpenWAM`, use `../envs/openwam/bin/python`.
Provide local pretrained DINOv3 and T5-small directories (including tokenizer files for T5):

```bash
EWM_DINO_PATH=/projects/hdd/myproject/CVPR2027/pretrained/dinov3-vitl16 \
EWM_T5_PATH=/projects/hdd/myproject/CVPR2027/pretrained/t5-small \
  ../envs/openwam/bin/python scripts/train.py --config-name train_ewm
```

The standalone `scripts/train_ewm.py` accepts the same Hydra overrides. CPU diagnostic runs use `training.mixed_precision=no model.codec_batch_size=2` and small model dimensions; production training should use CUDA. Increase `dataloader.chunk_size` to 16 or 32 only after K=8 validation on the target hardware.

Outputs are `checkpoint.pt`, resolved `config.yaml`, `normalization.json`, `debug_loss_history.csv`, and `event_metrics.csv` under `training.output_path`. New `ewm_trainable_v1` checkpoints contain only EWM weights/buffers, context projections/embeddings, and model configuration for policy inference. Frozen DINO/T5 state is excluded; old full-state checkpoints remain readable. All EWM keys are checked strictly. They do not contain optimizer/RNG state for exact training resume. Frozen encoder paths must remain accessible when rebuilding a policy.

Offline inference:

```bash
../envs/openwam/bin/python scripts/eval_ewm_policy.py \
  --checkpoint outputs/ewm/checkpoint.pt \
  --dataset ../datasets/lerobot_piper_stack_evis_uv_patterned_50_v3 \
  --horizon 8 --num-steps 20 --output outputs/ewm_policy
```

For simulation, replace `--dataset ...` with `--env-factory package.module:make_env`.
The factory must return a single Gym-style environment with `reset()` and `step(action)`;
`step` accepts a raw physical 7-D numpy action and returns the standard five values.
Each raw observation contains:

- `obs_video`: uint8 `[2,3,H,W]`, camera order cam0, cam_wrist.
- `obs_voxel`: nonnegative counts `[2,8,H,W]`, p-major bins for `(t[k-1],t[k]]`.
- `proprio`: raw float `[8]`.
- `prompt`: string.

The script normalizes observations using the checkpoint's stats and records observations/predictions plus episode metrics. Event predictions remain continuous normalized counts; use `EWMNormalizer.denormalize_voxel` for visualization. A simulator-specific RGB/event sensor adapter must provide this observation contract; the generic loop does not synthesize event streams or initialize Isaac Sim itself.

## Validation evidence

Post-review regression result: **80 passed, 2 skipped in 95.86 seconds**. Both skipped cases require CUDA (real pretrained context GPU path and production D512/L8/K8 smoke). Ruff checks, Hydra config composition, policy CLI help, and `git diff --check` also passed.

The test files cover codec geometry/finite gradients/axis separation, full mixed visibility, RoPE phases and transition alignment, context freezing/masking, flow endpoints and loss reduction, joint integration direction, arbitrary horizons, checkpoint gradients, training/logging/checkpoint loading, and receding-horizon inference.

- Real exported dataset: normalized RGB/event/state/action contract and exact `V_k → a_k,V_{k+1}` alignment passed.
- Real dataset K=8, 256×256: small-width model with full 4608-event + 9-action mixed attention completed forward/backward with finite gradients. External encoders use test doubles in this diagnostic.
- K=1/8/16/32: small 16×16-input models completed training/backward and joint sampling. This validates dynamic K, not production K=32 throughput or memory.
- Moving-edge reconstruction: 64×64 synthetic dual-polarity/four-bin clip, exact 4→1 and 16× compression, 120 optimization steps. MSE 0.151214 → 0.002518; reconstructed foreground mean 0.9444, absolute background mean 0.0264. All four bins preserve rightward motion. The figure shows some residual background/edge blur, so this is a codec sanity check rather than a quality claim.
- K=8 fixed-noise single-batch flow overfit, 80 steps: event loss 1.671891 → 1.008894; action loss 1.251854 → 0.001559. Both objectives decrease; this does not establish generalization or sampling quality.
- Evidence artifacts: `OpenWAM/outputs/ewm_validation/{pytest_review.xml,codec_reconstruction.json,codec_reconstruction.png,flow_overfit.json}`. These outputs are ignored by Git.

Reproduce from the project root:

```bash
OMP_NUM_THREADS=2 PYTHONPATH=OpenWAM \
EWM_DIAGNOSTICS_DIR="$PWD/OpenWAM/outputs/ewm_validation" \
  envs/openwam/bin/python -m pytest \
  OpenWAM/tests/test_ewm_event_codec.py \
  OpenWAM/tests/test_ewm_attention.py \
  OpenWAM/tests/test_ewm_model.py \
  OpenWAM/tests/test_ewm_dataset.py \
  OpenWAM/tests/test_t5_small_text_encoder.py \
  OpenWAM/tests/test_cosmos_predict25_scheduler.py \
  OpenWAM/tests/test_dinov3_encoder.py -q \
  --junitxml=OpenWAM/outputs/ewm_validation/pytest_review.xml
```

## Environment limits

This host reports `torch.cuda.is_available() == False`. Local weights are now available: `pretrained/dinov3-vitl16` (user-selected camenduru snapshot, 303,129,600 parameters) and `pretrained/t5-small` (full ~60M checkpoint, loaded encoder 35,330,816 parameters). Source revisions are recorded in each `download_source.json`. Real pretrained context forward/backward has passed on CPU with finite `[1,576,512]` output and frozen encoders. Production D=512/L=8 CUDA training, pretrained GPU execution, policy quality, FlashAttention performance, K=16/32 GPU profiling and simulator success remain unvalidated.

The host's PyTorch 2.7.1 oneDNN CPU Conv3D backward produced inconsistent gradients even between identical ordinary models. CPU training and numerical model tests disable oneDNN for the complete forward/backward operation; CUDA behavior is unchanged. Checkpointed versus ordinary gradients match with native CPU kernels.


## Review adjustments and new evidence

- RGB preprocessing uses tensor operations on the DINO device/dtype, reusing OpenWAM ImageNet constants. No `.cpu()`, NumPy or PIL occurs in the context forward path. FP32 and BF16 preprocessing exactly match the original PIL reference.
- Mixed attention concatenates both query streams for one SDPA call; reference tests compare both outputs and all Q/K/V gradients against the former two-call formula.
- Oracle sampling uses known clean data/noise and returns `epsilon-x0` for both modalities. At 1/7/20 steps, scheduler outputs **before clamping** recover clean targets within absolute tolerance 2e-6.
- Codec microbatch output/gradient equivalence is tested across 1 and 16. The production setting 16 is a profiling starting point, not an established optimum; measure 16/32/64 on the target GPU.
- Active/inactive diagnostics use clean future voxel > 0 and velocity residual squared errors. They are detached and do not change the loss. `event_metrics.csv` and W&B record both MSEs and occupancy; empty masks report zero. Aggregation averages per-batch diagnostics across microbatches/ranks, not global region pixel-weighted statistics.
- Checkpoint tests cover frozen weight exclusion, retained projections/embeddings, strict missing/unexpected EWM keys, and train/save/load/sample integration.
- Decoder remains non-causal `ConvTranspose3d(kernel_size=stride)` plus residual Conv3D refinement. Interpolation upsampling is not currently implemented.

### Real-event reconstruction is not yet a production quality pass

A small D=32/base_dim=2 codec was optimized for 100 steps on dataset sample 0,
future index 3, both cameras and all four bins at native 256x256 (no resize).
Active fraction was 0.0101614. Global reconstruction MSE fell from 0.165158 to
0.000328887; active MSE fell from 0.984102 to 0.00618494; inactive MSE ended at
0.000268771. The **all-zero baseline global MSE is lower: 0.0000637775**;
its active MSE is 0.00627645. Thus the reduction from random initialization
does not demonstrate useful real-event reconstruction. Visual inspection
shows poor edge recovery and visible background grid artifacts. This is a
numerical sanity experiment, not a claim that the reconstruction quality gate
has passed, nor evidence isolating the decoder as the sole cause.

Artifacts: `OpenWAM/outputs/ewm_validation/real_event_reconstruction.json` and
`real_event_reconstruction.png`. Follow up with a properly sized codec and
longer overfit, then compare interpolation+Conv3D if artifacts persist; do not
silently change the frozen flow loss or decoder based on this small experiment.

### D768/L12 capacity follow-up

`configs/train_ewm_d768_l12.yaml` provides the requested full-model trial
configuration: D=768, 12 heads, L=12, and codec base_dim=32. Its full
D768/L12/K8 output geometry was checked on the meta device. CUDA is required
for a meaningful numerical full-model flow test on this host.

For the isolated real-event codec diagnostic, matching D512/base_dim=32 and
D768/base_dim=32 runs completed on the same sample. D768 achieved lower
active-region MSE (0.00447747 vs zero baseline 0.00627645) but global MSE
0.000409306, 6.42× the zero baseline because of inactive-region artifacts.
D512 achieved global MSE 0.000189966, 2.98× the baseline, but did not improve
active-region MSE. Thus D768/L12 does not solve the global zero-baseline issue;
L does not participate in the codec-only experiment. Full results, scope and
reproduction instructions are in `ewm_d768_l12_reconstruction_experiment.md`.

GPU gates are executable tests `test_real_pretrained_tensor_context[cuda]`
and `test_production_d512_l8_k8_gpu_smoke`; they skip explicitly without CUDA.
The latter covers real data/pretrained encoders, bf16 forward/backward and two
joint sampling steps. It is a smoke test, not convergence or throughput proof.

## Production model parameter count

`test_production_model_parameter_counts` composes `configs/train_ewm.yaml` through Hydra (including model-group defaults and training-file overrides), then loads the real local pretrained encoders. Current configuration: D=512, heads=8, layers=8,
codec base_dim=32. The test passed and checks exhaustive, disjoint parameter
groups and their trainable/frozen subtotals. Counts include unique
`nn.Parameter` tensors, not buffers or optimizer state.

| Group | Parameters | Trainable |
| --- | ---: | ---: |
| T5-small encoder actually loaded | 35,330,816 | 0 |
| DINOv3 ViT-L/16 | 303,129,600 | 0 |
| Event expert | 64,301,062 | 64,301,062 |
| Action expert | 50,980,871 | 50,980,871 |
| External context projections/embeddings | 526,848 | 526,848 |
| **Total** | **454,269,197** | **115,808,781** |

Frozen total: 338,460,416. The T5-small download contains the full ~60M model,
but EWM instantiates only its encoder; the unused decoder is not counted.
Event expert includes encoder 12,035,588 + decoder 1,822,210 + camera embedding
1,024 + eight Event joint branches 50,442,240. Action expert includes state MLP
267,264 + action MLP 266,752 + head 4,615 + eight Action joint branches
50,442,240. Thus the two experts have identical joint-branch parameter counts;
their total difference comes from the input/output modules.

Full machine-readable report: `OpenWAM/outputs/ewm_validation/parameter_counts.json`.
Run from the project root with:

```bash
OMP_NUM_THREADS=2 PYTHONPATH=OpenWAM HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
EWM_DIAGNOSTICS_DIR="$PWD/OpenWAM/outputs/ewm_validation" \
envs/openwam/bin/python -m pytest \
  OpenWAM/tests/test_ewm_model.py::test_production_model_parameter_counts -q -s
```


Model size can be changed directly in `OpenWAM/configs/train_ewm.yaml` through
`model.dim` (D), `model.layers` (L), `model.heads`, and `model.base_dim`. Prefer
`D/heads=64` to retain the production 16/24/24 RoPE split. For example:

```bash
../envs/openwam/bin/python scripts/train.py --config-name train_ewm \
  model.dim=256 model.layers=4 model.heads=4 model.base_dim=16
```

The training config now defaults to the downloaded local pretrained directories;
`EWM_DINO_PATH` and `EWM_T5_PATH` still override them. Changing EWM D/L does not
resize the frozen DINO/T5 encoders.
