# DROID processed RGB → 1-bin event-frame conversion plan

Date: 2026-09-29. Source dataset:
`datasets/droid-processed`, Hugging Face revision
`7bad681fc20752682dcd1e0ef497efbb7745f1a8`.

## Scope and geometry

Convert only `observation.images.wrist` in the first full run, matching EWM's
wrist-event camera mode. Keep the source's native 320×180 grid and all RGB video
files untouched. The result is a sidecar under `datasets/droid-processed/events/`.
The external cameras may be converted in a separate run using the same contract.

Each event-frame is `uint8[2,180,320]`, channels `[OFF,ON]`, with independent
integer counts clipped at 15. Normalize to floating `[0,1]` only when read for
training (`count / 15`). Never subtract OFF from ON. No RGB resize, crop,
letterbox, or spatial binning is part of the conversion.

## Time and causality

For each episode, let `t_i = i/15` seconds and define event-frame `E_i` from
`(t_{i-1},t_i]`. Pair `E_i` with the observation RGB and state in row `i`.
The first event-frame `E_0` is zero because no previous RGB frame exists.
Reset the event simulator at every episode boundary, including boundaries
inside a shared MP4 file.

DROID collection reads observation before generating/executing action, then
records them together. A processed-data audit on 99,517 transitions across
four data files found that action `a_i` best explains state change
`s_i→s_{i+1}`; see `tmp/droid_action_alignment/REPORT.md`.

For EWM with current row `k` and horizon `K`:

- Observation: `RGB_k, state_k, E_k`.
- Conditioning actions: `a_k, ..., a_{k+K-1}`.
- Prediction targets: `E_{k+1}, ..., E_{k+K}`.

The terminal action has no within-episode future event target. Do not apply
Piper's observation-row offset -1 to this dataset.

## Event generation

Stream-decode one source video shard at a time with the original 15 Hz frame
order. Convert each BGR frame to grayscale at the same 320×180 pixel grid.
Feed frames at episode-local times `i/15` to the installed v2e
`EventEmulator.generate_events`. Use direct frame-to-frame simulation without SuperSloMo interpolation. The
conversion stores one temporal bin per 15 Hz interval; this choice avoids
hallucinating unobserved intermediate RGB frames and keeps the full run
tractable. Rapid motion can still alias, so interpolated conversion can be
evaluated later as a separate quality ablation.

Use a deterministic clean v2e setting: ON/OFF log thresholds 0.2/0.2,
threshold mismatch sigma 0.02, zero cutoff, zero leak, zero shot noise,
zero refractory period, and a fixed camera seed 1. These are the values set
by v2e's `set_dvs_params("clean")` in the current checkout. For each
`generate_events(frame_i,t_i)` result, histogram events by polarity, y,
and x; clip each count at 15 and write `E_i`. Sparse event lists live only
in memory during that frame and are never saved.

## Visual QA of external cameras (episode 0)

The first 255-frame pilot shows a strong 15 Hz alternating-density pattern in
both external-camera event streams. This is already present in the saved raw
count frames: active-pixel occupancy has lag-1 correlation -0.73 for
`external_0` and -0.83 for `external_1`. The source RGB frame-to-frame difference
also alternates in phase; its correlation with raw event occupancy is 0.85
and 0.95 respectively. The RGB image itself does not visibly flicker in
brightness: for example, `external_0` frames 20–23 have mean gray levels
69.60, 69.80, 70.05, 70.36, while raw event active-pixel fractions are
0.228, 0.094, 0.237, 0.093. The event view shows temporal changes only,
so this uneven change becomes conspicuous. Decoded MP4 occupancy follows raw
event occupancy with correlation above 0.9999; MP4 encoding is not the cause.
This pilot alone does not establish the underlying acquisition or resampling
mechanism for the source frame cadence. A simple interpolation within each
15 Hz interval leaves its endpoints unchanged and may not fix the alternating
per-bin count; audit source timing before choosing a conversion remedy.

Background check: pixels with no event anywhere in the `external_0` episode
stay near pure white after H.264 decoding (mean 254.2–255 across frames).
A separate FFV1 video decodes with zero channel mismatches against all
44,064,000 raw palette values. The visibly flashing white *area* is the
complement of the event mask: for frames 20–23 the event occupied 22.8%,
9.4%, 23.7%, 9.3% of pixels. A causal two-frame persistence preview makes
those fractions 25.3%, 25.3%, 26.1%, 26.0%, but is display-only and must not
replace the per-interval training targets.
See `tmp/droid_event_visualize/results/flicker_timeseries.png` and
`flicker_diagnostic.json`.

Do not extrapolate the wrist-only no-interpolation decision to full external
camera conversion without a source-cadence audit and a visual/temporal quality
comparison. Any temporal merging or persistence used for visualization must
remain separate from training event-frame targets and their 15 Hz alignment.

## Follow-up interpolation pilot and guide check

The saved DROID source MP4 is H.264/yuv420p at 15 fps with uniform 1/15 s
presentation timestamps. The original direct v2e pilot did **not** include
2× interpolation or a 30 Hz intermediate event-frame stream. In eight
sampled `external_0` episodes, the raw RGB mean absolute frame difference
was 1.8–6.4× higher for even than odd episode-local transitions. The
alternation is visible before our event-generation code; its exact upstream
acquisition or processing cause is still unknown. In the same eight wrist
samples, the even/odd RGB-difference ratio was only 0.96–1.29, with positive
lag-1 correlations, so the strict parity artifact is specific to the sampled
external views and should not be assumed for the wrist-only conversion.

A 75-frame GPU pilot fed 8 linearly interpolated intensity frames into v2e
per 15 Hz interval, then accumulated events back into that same interval.
For `external_0`, only 9 of 8,640,000 count entries and 5 of 4,320,000
binary mask pixels changed from direct simulation; for wrist these numbers
were 57 and 28. The external occupancy lag-1 correlation stayed -0.835.
Thus higher *within-interval* timestamp resolution alone did not repair the
15 Hz 1-bin alternating target. This does not test motion-aware FILM.

A separate 8× interpolation + 6 Hz photoreceptor cutoff pilot reduced the
alternation (occupancy lag-1 -0.835 → +0.782) but cut total clipped counts
from 1,438,858 to 527,582 in 75 frames. Spatial F1 against the direct mask
was highest with a one-frame delay. Treat this as an experimental display
candidate, not an accepted training target: it changes event magnitude and
timing, potentially affecting action-to-event causality. See
`tmp/droid_interpolation_pilot/results/` and the five-second preview there.

v2e's advice about finer timestamps addresses burst timing in sparse event
streams. Our training target discards within-bin timestamps by summing all
events in `(t_{i-1},t_i]`; any proposed fix must be judged after that same
15 Hz accumulation. A 30 Hz event target would change the dataset/model time
contract and cannot be silently substituted for the action-aligned 15 Hz
proposal.

## Static-background failure in the processed external videos

A fixed left-background ROI (`x=0:140, y=30:175`) in `external_0` proves that
the alternating pattern is not ordinary moving-object event activity. In the
first 255 frames, the source H.264 alternates I/P types exactly; the first
shard has ~296 kb/s video bitrate. At frames 150–153, 99.2% of background
pixels are identical from I frame 150 to P frame 151, whereas only 13.7%
match from I frame 150 to I frame 152. Phase-correlation estimates near-zero
background translation. The mean background grayscale stays ~82, while the
mean absolute pixel difference is 0.027 for 150→151 and 3.46 for 150→152.
Both OFF and ON events appear on the latter static background. Thus this
processed RGB stream contains a codec/acquisition-cadence-dependent
photometric change that v2e legitimately responds to but that is **not a
physical scene event**. The original pre-encoding frames are unavailable
locally, so the respective contributions of capture and lossy encoding cannot
be separated conclusively.

Over the 180-frame diagnostic, the static ROI's raw event active fraction is
6.23% (even I frames 12.07%, odd P frames 0.54%). Spatial NLM denoising
with `h=8` lowers this to 3.08% while reducing active pixels in a moving-arm
ROI from 22.09% to 17.44%; `h=12` lowers background to 2.11% but moving ROI
to 15.88%. A causal Farneback motion gate lowers background to 1.57% but
moving ROI to 13.53%. None of these is a faithful drop-in conversion fix;
all delete potentially valid motion events and residual background artifacts
remain. See `tmp/droid_background_artifact/results/` and the frame 150–153
RGB/difference/event montage in `tmp/droid_event_visualize/results/`.

Do not generate training targets for the external cameras from the current
processed MP4s with the direct v2e method. Prefer a higher-quality source or
an explicitly validated reconstruction method. Re-encoding already decoded
MP4 frames losslessly cannot restore the missing clean signal. Keep the
wrist-only decision separate; it did not show the same strict I/P parity in
the eight sampled episodes.

## Output and restartability

Use one HDF5 file per source wrist MP4, preserving its chunk/file indices:

`events/observation.images.wrist/chunk-000/file-000.h5`

Each file has a compressed, chunked `frames` dataset with shape
`[N,2,180,320]`, dtype `uint8`, temporal chunks of 16 frames and gzip
level 4. Frame index within the HDF5 file is identical to frame index within
the corresponding MP4. Read `meta/episodes/*.parquet` to identify episode
boundaries and offsets; verify episode lengths, frame counts and 15 Hz
timestamps. Store source revision, v2e commit, parameters and polarity
order in HDF5 attributes and a sidecar manifest.

Schedule one independent Slurm array task per video shard (110 wrist MP4
files). Write to a temporary file and atomically rename after all episodes
in that shard pass count/shape/boundary checks. Reruns skip verified shards.

## Capacity

Dataset has 11,936,559 frames. Each uncompressed frame is
`2 × 180 × 320 = 115,200` bytes.

| Scope | Uncompressed count-frame bytes | Decimal TB | Binary TiB |
|---|---:|---:|---:|
| Wrist only | 1,375,091,596,800 | 1.375 | 1.251 |
| All three cameras | 4,125,274,790,400 | 4.125 | 3.752 |

Native-resolution v2e pilot: 8 wrist episodes spread across the dataset,
1,396 frames, 160.8 MB uncompressed. Whole-episode zlib level 3 used
16.8 MB (9.58× reduction), corresponding to about 144 GB for wrist if the
sample ratio held corpus-wide. A separate five-episode subset compressed
in 16-frame chunks at zlib level 4 by 12.64×. The sample is small and
scene-dependent, so **plan for 150–300 GB of final wrist event files**.
Together with the existing ~80 GB of MP4 video, the nominal on-disk total
is ~230–380 GB, plus a small in-progress shard. This is a planning estimate,
not a guaranteed bound; the uncompressed wrist reference is 1.375 TB.
The full three-camera compressed size has not been sampled; its uncompressed
size is 4.125 TB.

Pilot code and measurements are under `tmp/droid_eventframe_capacity/`.
For a like-for-like uncompressed frame comparison, one wrist RGB stream is
`11,936,559 × 3 × 180 × 320 = 2.063 TB`; the two-channel event stream is
exactly 2/3 of that (1.375 TB). The on-disk wrist MP4 files occupy only
20.72 GB because H.264 uses lossy inter-frame video compression. All three
camera MP4 streams together occupy 80.25 GB. The event-frame estimate is
larger because the plan stores exact integer counts with lossless compression;
frame count and channel count alone do not predict compressed bytes.

## Streaming and disk use

Conversion decodes MP4 frames sequentially, immediately accumulates each
interval's events and writes a compressed HDF5 chunk. It never saves a fully
decoded RGB video or uncompressed event-frame array. Atomic rename changes
the name of the current compressed shard; it does not require a second copy
of the whole dataset. Monitor actual output size during the first video
shards and stop scheduling new shards if the available quota approaches
exhaustion.

Training reads only requested HDF5 chunks and MP4 frames for each sample or
batch. Full-dataset decoding is unnecessary. On-the-fly v2e generation from
MP4 alone would save the event sidecar space, but v2e's state depends on
previous frames within an episode; random-access training would need to
replay from the episode start or maintain state snapshots. That option needs
a throughput benchmark before use.

## Training integration

The current Piper EWM loader requires 30 Hz, 256×256 and Piper camera keys.
It cannot directly read these 15 Hz, 180×320 sidecars. A DROID-specific loader
or a generalized loader must preserve the indexing above and create matching
one-bin `count_clip=15` normalization metadata. Any model-input geometry
adaptation belongs in that training path; the saved event frames retain
the original 320×180 resolution.
