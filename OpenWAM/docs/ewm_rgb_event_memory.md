# RGB/event hierarchical short-term memory

Implemented model: `ewm_rgb_event_memory`; policy format:
`ewm_rgb_event_memory_latent_v1`. The two levels are **RGB semantics and event
motion**, covering the same window, rather than additional temporal tiers.

## Observation contract

At anchor k, RGB frames are `[k-9,k-6,k-3,k]`, without RGB differences. Event
bins are `[k-9,...,k]`. At 30 Hz, endpoint span is 0.3s and the event accumulation
support is approximately 0.333s. The dedicated Piper reader uses aligned direct
event-frame exports, one bin, OFF/ON, count_clip15 and same-row RGB/state/event.
Episode-start history is zero padded and masked; zeros in a valid event bin
remain valid observations. The current event bin must exactly match obs_voxel.

RGB: reuse the existing frozen DINOv3 ViT-L/16, pool each patch grid to 8x8,
project to D256, and attend across four frames independently at each spatial
slot using the current frame as query. Current DINO grids are encoded once
and shared with the original full-resolution RGB context.

Event: area resize normalized OFF/ON to 128x128 and float-pack
`[OFF,(OFF+ON)/2,ON]`. Read `[E9,E1-E0,...,E9-E8]`, retaining signed values.
This equals the pixel-difference probe's normalized 2x2 sum for Piper count_clip15,
without its byte quantization. Use the probe's shared convolution stem
3->16->32->64 (kernels5/3/3, stride2), reading at its 4x4 pool. Remove the global
flatten/projection and speed regression head. Per spatial cell, concatenate
the ten encoded slots in order and apply an MLP to produce D256 motion tokens.
Interval midpoint, duration and anchor/difference type are encoded explicitly.
No speed-probe checkpoint is selected or loaded automatically.

Fusion: 64 RGB semantic tokens query 16 event motion tokens per camera; gated
residual fusion produces 64 D256 tokens per camera. Learned spatial and camera
embeddings distinguish slots. The dual-view preset has128 memory tokens,
921 joint event/action queries and1049 K/V tokens at K24. Dense future patches
and action prediction are preserved. Each DiT layer projects memory to shared
K/V; event and action queries both read it. Memory stays clean, is not a
prediction target, and is prepared once per request outside denoising.

## Historical DINO cache

Cache frozen encoder outputs **before trainable projections**. Storage uses
BF16 bit patterns in uint16 `.npy` arrays `[frames,2,1024,8,8]`, one per episode.
The manifest verifies encoder file metadata fingerprint, fixed preprocessing,
source video paths/sizes/mtimes/episode offsets, episode lengths and array shape.
Use the same frozen encoder and deterministic preprocessing; rebuild the cache
if these change. Only three historical frames are loaded per training sample;
the current frame comes from the existing online context encoding.

For the Piper dataset's22,685 frames, pooled dual-camera storage is5.54GiB,
excluding small headers. VAE latent caching is unrelated and remains unimplemented.
Cache generation is implemented; no complete dataset cache has been generated
as part of the implementation task.

From the project root, generate once:

```bash
export LD_LIBRARY_PATH=/projects/hdd/myproject/CVPR2027/envs/torchcodec_ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY=/projects/hdd/myproject/CVPR2027/envs/openwam/bin/python
"$PY" OpenWAM/scripts/train_ewm/cache_dino_memory.py \
  --dataset /projects/hdd/myproject/CVPR2027/datasets/piper_stack_fronttight_v3_50/job177683 \
  --dino-path /projects/hdd/myproject/CVPR2027/pretrained/dinov3-vitl16 \
  --output /projects/hdd/myproject/CVPR2027/datasets/piper_dino_history8 \
  --device cuda --batch-size 16
```

Use a GPU allocation for generation. The script refuses to overwrite completed
caches or existing episode arrays. Set batch size lower if necessary.

Train from `OpenWAM/`:

```bash
export EWM_DATASET_DIR=/projects/hdd/myproject/CVPR2027/datasets/piper_stack_fronttight_v3_50/job177683
export EWM_MEMORY_DINO_CACHE=/projects/hdd/myproject/CVPR2027/datasets/piper_dino_history8
"$PY" scripts/train.py --config-name train_ewm_rgb_event_memory_piper
```

Without EWM_MEMORY_DINO_CACHE, the reader supplies raw historical RGB and the
model encodes it online in chunks; this incurs three extra historical images
per camera/sample. Cache storage is not an additional copy of DINO weights.
The preset inherits batch32/GPU, K24 and the existing frozen joint BF16 codec.
For initialization from an existing dense policy, add
`training.init_policy_checkpoint=/absolute/path/checkpoint.pt`.
Set initialization to null when resuming a full optimizer state.

For diagnostics use `training.save_checkpoints=false training.save_steps=0`.
The old pure-noise evaluation launcher is disabled in this preset because it
supplies single observations. Model sampling and the generic policy evaluator
accept history. The conveyor client/server now exchange continuous physical
history and reset it per episode. Other custom clients must supply history;
never fill an unobserved gap by repeating current frames.

## Budget and validation

Memory-only K/V tensor payload at B32, L12, BF16 is144MiB; this is not total
extra training memory. With dual event forecasts, QxK attention work increases
by128/921=13.9%, while query FFNs are unchanged. Complete training overhead
requires measuring encoders, gradients and data/cache I/O. The earlier5–15%
extra GPU-time allowance was provisional, not a measured result.

Unit/regression gates cover signed-difference reconstruction, boundary masks,
causality, both-modal history effects, memory gradients, frozen encoders,
current DINO reuse, cached/online equality, checkpoint contracts, dense fallback,
continuous history snapshots, and fullgraph velocity compilation.
GPU profile scripts/results and the sparse-script deletion manifest live in
`tmp/ewm_rgb_event_memory/`. GPU profiles use synthetic observations and exclude
loader/cache I/O; they do not establish learned task accuracy.

48 obsolete sparse-token training/benchmark/prototype scripts were removed.
Their experiment metrics, logs and checkpoints remain. The production sparse
model and compatibility tests remain solely to read/verify existing policies;
the shared trainer rejects new sparse-token training.

## Measured GPU computation (Slurm200343)

RTX PRO6000 Blackwell Max-Q, BF16, B32, dual views, K24, eager training:

| Case | Median step | Peak allocated | Time increase |
| --- | ---: | ---: | ---: |
| Dense | 0.674s | 33.407GiB | baseline |
| Cached-history memory | 0.764s | 34.714GiB | 13.34% |
| Online-history memory | 1.005s | 34.714GiB | 49.16% |

Three timed steps after two warmups, synthetic inputs, real frozen encoders, no
checkpoint. Includes forward/backward/AdamW; excludes loader/cache I/O and H2D.
Inputs and precomputed cache are resident in all cases. Full workload overhead
and learned success rates are not established. Details: ../../migration/evidence/ewm_rgb_event_memory/REPORT.md.
