# RGB/event short-term memory implementation and validation

## Implemented

- Frozen shared DINO: RGB frames [-9,-6,-3,0], no RGB differences, semantic spatial memory.
- Probe CNN stem: continuous ten event bins, current anchor + nine signed pixel differences,
  with separate OFF/ON information, spatial motion readout and ordered temporal MLP.
- RGB queries event motion through gated fusion; dual view outputs128 clean memory K/V tokens.
- Current DINO output is shared with original RGB context. Memory grids are canonical BF16
  in both online and cache paths. Frozen full-resolution current RGB context remains unchanged.
- Historical DINO cache utility and validated BF16 uint16-bit mmap reader.
- Dedicated dual-view Piper preset and optional dense-policy initialization.
- Sampling prepares memory once; conveyor physical-frame history resets per episode,
  including missing-frame masks and bool-mask wire support.
- Removed48 legacy sparse-token dedicated scripts/prototypes; historical results remain.
  Deletion inventory: deleted_sparse_scripts.json. Production compatibility model/tests remain;
  new sparse-token training is rejected.

## GPU verification: Slurm200343

GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition. B32, K24, dual RGB/events, D768/A512, H8/L12, BF16,
compile off, real local frozen DINO/T5/LeanVAE. No checkpoints saved.
Synthetic normalized observations. Each case used2 warmup and3 timed optimizer steps.
Online/cache memory K/V equality and BF16 bit-storage roundtrip passed.
Trainable gradients were finite and present; all three cases completed with finite loss.

| Case | Median compute step | Peak allocated GPU memory | Time vs dense |
| --- | ---: | ---: | ---: |
| Dense | 0.673970s | 33.407GiB | baseline |
| Memory, cached history | 0.763881s | 34.714GiB | +13.34% |
| Memory, online history | 1.005322s | 34.714GiB | +49.16% |

Peak increase: 1.307GiB in this harness.
All cases share resident synthetic inputs, including history tensors and the precomputed
history cache, so this does not include the incremental dataset-input allocation of a real
loader. Data loading, video decoding, cache disk I/O and H2D are excluded. The result measures
training computation, not learned task accuracy or multi-GPU training throughput.
The short sequential comparison is a budget estimate, not a statistically repeated benchmark.
Raw results: results/profile_200343.json.

The first GPU attempt200316 failed because the validation script assumed raw DINO grids
were BF16 and viewed FP32 values as uint16, changing the last axis. Production memory features
and the validation script were corrected to canonical BF16; the successful rerun is200343.
The cache generator retains native frozen weight precision and uses the same CUDA BF16
activation autocast as training, then stores pooled grids as BF16.

## CPU checks

Candidate gates:8 passed. Real Piper anchors0 and10 were read with correct causal windows
and episode-start masks. Final relevant regression suite: **204 passed,0 failed in78.48s**.
This includes16 dedicated RGB/event-memory tests; static Python and batch-shell
syntax checks also passed. The shared trainer completed an actual one-step
CPU memory update with checkpoint saving disabled.

## Usage and scope

See ../../OpenWAM/docs/ewm_rgb_event_memory.md for configuration and cache commands.
No complete Piper DINO cache or long training/closed-loop evaluation was started.
