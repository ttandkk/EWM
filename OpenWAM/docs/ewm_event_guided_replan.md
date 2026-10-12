# Optional event-guided Piper replanning

The default evaluation still executes fixed 16-action chunks. Enable adaptive
execution with the existing launcher:

```bash
sbatch OpenWAM/scripts/eval_ewm/run_ewm_piper_eval.sh \
  --checkpoint-dir /projects/hdd/myproject/CVPR2027/OpenWAM/outputs/<run>/job<id> \
  --event-guided-replan
```

For threshold calibration, replace the last flag with
`--event-replan-record-only`. This records errors without interrupting chunks.
Both modes automatically enable future event decoding and six-panel videos.
Neither mode changes checkpoint weights. Existing running jobs retain their
submitted script; the new options apply to future evaluations.

After action j, compare its resulting observed events with predicted future
event j from that same plan (observation k, action k, future event k+1).
Discard unexecuted actions and predict again when the error exceeds the
threshold for the configured number of consecutive frames, after the minimum
execution horizon. Success, termination and the step budget take precedence.
At the maximum execution horizon, replan even without an event trigger.

| Option | Default |
| --- | --- |
| `--event-guided-replan` | false |
| `--event-replan-record-only` | false |
| `--event-replan-threshold` | 0.6 |
| `--event-replan-patience` | 2 |
| `--event-replan-min-horizon` | 4 |
| `--event-replan-min-pixels` | 4 |
| `--execution-horizon` (maximum in adaptive mode) | 16 |

Real counts >0 and predicted occupancy (existing 1/30 threshold and neighborhood
filter) are max-pooled from 256x256 to 64x64. OFF and ON remain separate.
Precision and recall allow a one-cell Chebyshev displacement within the same
polarity; each camera's error is 1-F1, then camera errors are averaged. Both
empty frames score zero; a one-sided nonempty frame scores one. If both pooled
occupancy counts are below min-pixels, that camera scores zero. The pixel gate
counts polarity cells, not original-resolution pixels. The minimum-horizon
gate counts consecutive mismatches from the beginning of each plan.

## Exact score and threshold interpretation

For camera c, let R_c and P_c be real and predicted binary occupancy masks,
including both polarity channels. Let D expand each channel independently
with a 3x3 neighborhood on the 64x64 grid. Counts below refer to occupied
polarity cells, not event count intensity:

```text
precision_c = |P_c AND D(R_c)| / |P_c|
recall_c    = |R_c AND D(P_c)| / |R_c|
F1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)
error_c     = 1 - F1_c
event_error = mean(error_c across cameras)
```

When precision + recall is zero, error_c is 1. The sparse/empty rules take
precedence: if max(|R_c|, |P_c|) < 4, error_c is 0; otherwise, if exactly one
mask is empty, error_c is 1. Matching allows reuse of neighboring cells; it
is not a one-to-one correspondence between individual events. Because masks
are pooled into blocks, the one-cell tolerance is not an exact four-pixel
distance cutoff on the original image.

The threshold is manually configured, not estimated online. In the dual-camera
setting, threshold 0.3 means average tolerant F1 < 0.7, **not** that 30% of
image pixels differ. A low-error camera can dilute a large error in the other
camera because aggregation is an arithmetic mean, not a maximum.

The trigger uses a strict comparison (`event_error > threshold`): increment
the streak on exceedance and reset it to zero otherwise. Trigger only after
at least four actions have executed and the streak reaches two frames. Reset
the streak for each newly generated plan. A score exactly equal to 0.3 does
not exceed a 0.3 threshold. Success, termination, step-budget exhaustion and
maximum-horizon completion retain precedence over an event trigger.

## Threshold 0.3 comparison (2026-10-02)

Job 180622 evaluates job179463 checkpoint step70000 with threshold0.6.
At the initial inspection, 2,613 recorded steps had mean error0.1142,
maximum0.4571 and p95≈0.2408; none exceeded0.6, so no event-triggered replanning
occurred in that inspected prefix. These are partial-run observations, not
final results or a calibration guarantee.

Job180653 was submitted using the same checkpoint and 50-trial protocol with
`--event-guided-replan --event-replan-threshold 0.3`; patience2, minimum horizon4
and maximum horizon16 remain unchanged. The implementation default is still
0.6; 0.3 is an explicit experiment override:

```bash
sbatch OpenWAM/scripts/eval_ewm/run_ewm_piper_eval.sh \
  --checkpoint-dir /projects/hdd/myproject/CVPR2027/OpenWAM/outputs/ewm_lean16_p4_d768_a512_dual_ada2_80k/job179463 \
  --checkpoint-step 70000 --event-guided-replan --event-replan-threshold 0.3
```

Check both event_mismatch counts and replan_chunks horizon statistics to
verify adaptive execution. Short all_chunks horizons alone do not establish
an event trigger: successful trials and budget-limited trials truncate their
last chunk even in fixed-horizon mode. Assess whether high scores coincide
with meaningful trajectory deviation before attributing benefit to a lower
threshold.

Each step JSON records camera errors, mean error, mismatch streak, trigger and
chunk end reason. Videos overlay error, executed horizon and end reason.
Each trial result records execution_chunks and execution_horizon_statistics;
summary.txt aggregates completed trials, and complete.json includes global
structured statistics. Means weight chunks equally, not trials equally.
all_chunks includes terminal tails; replan_chunks includes only event_mismatch
and max_horizon endings. Reason counts expose event-trigger frequency.

Threshold 0.6 is an initial tuning value, not a calibrated divergence detector.
Codec blur and event sparsity can cause false triggers; compare record-only
scores on normal and failing trajectories before interpreting adaptive results.
Decoder and monitoring costs should be included when comparing inference and
trial timing. This heuristic identifies disagreement, not its cause.
