# Mixed event-only preprocessing and polarity audit

Audited 2026-10-02 (Asia/Shanghai). Final selected model: Event D768 / Action D512, H8/L12/K24, LeanVAE BF16, patch4, spatial stride32, temporal stride4, 448 single-camera event tokens. Raw exports are unchanged.

## Polarity evidence

REASSEMBLE official source explicitly calls polarity1 a brightness increase and all other values a decrease. The port preserves the original event array; the formal exporter indexes channels by polarity0/1. Thus training uses disk indices [0,1] as [OFF,ON].

- [Official visualization, pinned commit432cc15](https://github.com/TUWIEN-ASL/REASSEMBLE/blob/432cc15ce3e028edc2f98a786f28bf6baf31ac6f/scripts/visualization/vizualize_data.py#L82).
- [Lossless port, pinned commit54aba4d](https://github.com/lvjonok/reassemble-lerobot-port/blob/54aba4d2f688a3ffaf4afab7e2a045f13cab12c7/scripts/port_reassemble.py#L281).

DVLA disk order is blue/red. The local source dataset card says events come from v2e, but its linked DynamicVLA/mujoco generation repository returns HTTP404. v2e alone does not define the custom DVLA color renderer. Instead, check the actual paired RGB and exported events: eight episodes spread across the dataset, all three cameras, up to160 frames per episode at25Hz. On pixels with |delta log(mean RGB+1)|>0.03 and only one active event channel, blue tracks increasing brightness and red decreasing brightness. This is strong sample-based polarity evidence, not source-code proof or exhaustive per-episode verification.

| Camera | Blue positive brightness | Red negative brightness | Pixels, blue / red |
| --- | ---: | ---: | ---: |
| opst_cam | 80.09% | 81.26% | 624,366 / 474,962 |
| wrist_cam | 87.82% | 88.36% | 2,700,059 / 3,057,415 |
| side_cam | 79.98% | 80.35% | 793,109 / 567,449 |

The loader now swaps DVLA channels online with disk indices [1,0] before occupancy conversion. Both sources supply [OFF,ON]. All24 camera/episode checks support this direction. The raw export metadata retains its historical unverified wording; this report records the new evidence.

## Common numerical contract

1. Threshold each native channel independently at count>0, preserving both polarities.
2. Use PyTorch area interpolation to192x256, then pad32 zero rows above and below.
3. Produce FP32 occupancy in[0,1] with shape[T,1,2,256,256]. Do not divide by15, quantize to bytes, cancel signs, or apply histogram matching.
4. Form floating pseudo-RGB[OFF,(OFF+ON)/2,ON]; Lean maps this to[-0.5,0.5], with BF16 codec weights/compute.
5. Encode current1+future24 jointly; add Gaussian noise only after encoding. Native latents/velocity are not bounded by occupancy range.
6. RGB uses Lanczos image resize, uint8[0,255], and the same padding geometry. Valid masks exclude padded rows from latent error and decoded metrics, without claiming VAE boundary receptive-field independence.

Disk REASSEMBLE counts are uint8 clipped to15. DVLA counts are uint8 sums of ten binary4ms masks, so at most10 although the declared clip ceiling is15. These count magnitudes do not have the same physical meaning.

## Distribution estimates after canonical polarity

Sample24 windows of25 frames per source, seed20261002, episodes drawn proportional to valid starts. Each source contributes600 sampled frames and29,491,200 valid pixels per channel. Default DVLA event view: opst_cam. Padding is excluded. Estimates are not a full export scan.

| Source | OFF mean | ON mean | OFF nonzero | ON nonzero | OFF std | ON std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reassemble | 0.01285 | 0.01188 | 3.181% | 3.114% | 0.08509 | 0.07958 |
| dvla | 0.02642 | 0.02866 | 3.317% | 3.613% | 0.15177 | 0.15768 |

The processing convention and numeric range match, but the distributions remain different: mean DVLA coverage is about2.06x (OFF) / 2.41x (ON) that of REASSEMBLE. Resized nonzero ratios are closer because REASSEMBLE activity is more fractional within active output cells. Real/synthetic sensor response, camera content, background filtering, 33.3ms versus40ms windows and lossy H.264 rendering remain domain gaps. No arbitrary scaling is introduced to hide them. Retain source-specific prompt/FPS and evaluate separately.

Native rates remain30/25Hz; future24 frames span0.80/0.96s. Source-homogeneous batching is enabled; held-out evaluation is still not implemented in the mixed loader. Current random-window sampling proportions are about58% REASSEMBLE /42% DVLA.

## Reproduction and artifacts

- `audit.py`: exact sampling and brightness checks.
- `results/audit.json`: corrected occupancy summaries, raw histograms in original disk order, per-window samples and all camera polarity checks.
- `results/audit_before_channel_fix.json`: initial evidence before online DVLA swap.
- `sources/`: pinned official source snapshots and GitHub tree metadata.
- Existing configuration/event-only tests and `OpenWAM/tests/test_ewm_event_only_dataset.py` protect patch geometry, branch disabling, mask behavior, polarity swapping, fractional occupancy and Lean numeric mapping.

Run from the workspace root:

```bash
PYTHONPATH=OpenWAM OMP_NUM_THREADS=2 LD_LIBRARY_PATH=/projects/hdd/myproject/CVPR2027/envs/torchcodec_ffmpeg7/lib \
  envs/openwam/bin/python tmp/event_only_contract_audit/audit.py
```

Final validation: `tests/test_ewm_model.py` and `tests/test_ewm_event_only_dataset.py`: **41 passed**, including final patch4 configuration, event-only action exclusion, valid-region masking, canonical polarity swapping and numeric mapping. The run took299.95s, including shared-filesystem startup waits. The stale evaluation-test import was updated to the existing `scripts.eval_ewm.eval_ewm_policy` location. No GPU training job was submitted.

## Temporal overlap contract shared with Piper

All three sources use disjoint event accumulation intervals. Piper and REASSEMBLE use consecutive33.3ms intervals; DVLA uses disjoint groups of ten4ms rendered frames (40ms). REASSEMBLE frame i covers `(t0+(i-1)/30, t0+i/30]`; DVLA frame j covers source indices `[10j,10j+9]`. Neither loader extends an event accumulation window across adjacent frames. REASSEMBLE's10ms background-activity filter support is not its accumulation-window length.

Event-only explicitly matches Piper training-window sampling: `chunk_size=24`, `window_stride=1`, current event frame k and future event frames k+1..k+24, with current RGB at the anchor. Adjacent anchors share23 future targets and24 frames when including the observation. Samples stay within an episode and are shuffled for training. Thus event accumulation is non-overlapping while training clips overlap. Native30/25Hz rates are retained, so physical accumulation length and forecast duration remain source-specific (33.3/40ms and0.80/0.96s). This is alignment of window-overlap rules and frame counts, not equal physical timing.

## Source-homogeneous batch implementation

`dataloader.homogeneous_source_batches=true` is the event-only default. The sampler independently shuffles every source's valid window indices each epoch, forms groups of `training.batch_size * world_size`, and randomly interleaves source groups. It yields each group's per-GPU batches consecutively, matching Accelerate's non-split sharding. Every GPU therefore receives the same source at a micro-step, with the configured per-GPU batch size unchanged. Each source's final group is padded only from that source; all original windows are visited and fewer than one global batch of indices per source are repeated per epoch. Sampling remains approximately58:42 rather than forcing50:50.

Shuffling uses the shared run seed plus epoch and is replayable for full-state resume. The effective batching option is stored in the resolved config/resume contract. `homogeneous_source_batches=false` selects the previous mixed-source random batches. Accelerate `split_batches=true` is rejected for homogeneous grouping, since rank-sized batches are already emitted. Gradient accumulation may combine different source-homogeneous micro-batches within one optimizer update. Native30/25Hz timing and occupancy preprocessing are unchanged.

Homogeneous-batch validation: **18 checks passed** across sampler, resume and preprocessing tests. Accelerate BatchSamplerShard was tested at world sizes1/2/3/8, including sources shorter than one global batch, source-local padding, complete index coverage and deterministic epoch replay. Full-state CPU training/resume produced exactly equal final weights at gradient accumulation1/2; the accumulation2 case resumed from epoch1. Tests use small synthetic datasets/models and simulated rank sharding; a live multi-GPU training smoke was not part of this validation.
