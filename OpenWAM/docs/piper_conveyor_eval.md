# Piper conveyor evaluation clocks

The dynamic evaluator is `scripts/eval_ewm/eval_ewm_piper_conveyor_sim.py`, submitted through `scripts/eval_ewm/run_ewm_piper_conveyor_eval.sh`. The default scene, RGB cameras, 64-sample path tracing, EVIS interpolation, 30 Hz event intervals, physical controller, randomized supply and two-object success criterion match the training export. The default remains in distribution; the optional speed presets below change only the conveyor surface velocity.

## Paused inference (default)

The simulator stops while the policy request runs, then executes the action chunk at 30 Hz **simulation time**. This is the existing protocol. A 30 FPS rendered video does not mean the environment ran at 30 Hz wall time.

## Latency-equivalent evaluation: `--latency-aware`

The evaluator measures wall-clock round-trip time through validation of the **action reply**. The server sends actions first; optional future-event decoding, transfer and display happen afterward and are not charged to control latency. Delivery is emulated at observation step `s + ceil(round_trip_seconds * 30)`. Physics, RGB and EVIS still advance for every simulation tick. Quantization adds less than one 33.33 ms tick. The observation is not refreshed before delivering the prediction.

### Buffered controller (default when latency is enabled)

`--latency-controller buffered` requests all 24 trained action steps. It requests another forecast after 16 simulation ticks from the active observation anchor, consuming the previous forecast's remaining eight steps while the new request is pending. When the result becomes available, it selects the prediction index corresponding to the current simulation step and discards elapsed actions. It does not repeat a stale action prefix. For a 128 ms request, four cached actions execute during the delay and the new plan starts at prediction index four. If latency exceeds the available forecast tail, only the actual buffer underflow produces hold commands.

The first request has no cached actions: it incurs an initial hold, retaining the last gripper command. Its relative action sequence starts at index zero after that wait, rather than dropping approach commands for motion that never occurred. This bootstrap offset is explicitly recorded. A terminal condition discards any pending result. An expired forecast is discarded and a fresh observation is requested.

This changes the execution controller from serial planning to planning with an action buffer. It does not make the hardware renderer run at 30 Hz. Predictions remain conditioned on the earlier state; the consumed old prefix may differ from the new forecast's predicted prefix. There is no prefix-conditioned retraining or guaranteed smooth stitching. Actual task success and plan-boundary motion must therefore be validated in addition to queue timing. The measured inference runs with simulation/rendering paused; simultaneous GPU contention, sensor/actuator hardware delays and external disturbances are not modeled.

### Serial controller for comparison

`--latency-controller serial` retains observe → wait → execute: it holds zero relative EE motion and the last commanded gripper for every charged request delay, then executes the first 16 actions. At a roughly 0.2 s request duration this visibly pauses between chunks. This behavior is a property of serial control, not a 30 FPS video-encoding failure. Older evaluations included predicted-event visualization in latency; new serial evaluations use action-only round-trip timing and cannot resume into those older runs.

The launcher defaults to one server warmup for this mode, before trial clocks start. `WARMUP_ITERS` can override this. Paused mode keeps its original zero-warmup default.

```bash
cd /projects/hdd/myproject/CVPR2027/OpenWAM
sbatch --gres=gpu:rtx5090:1 scripts/eval_ewm/run_ewm_piper_conveyor_eval.sh \
  --checkpoint-dir outputs/conveyor_event_action_182549 --latency-aware
```

Per-trial `latency_timing` records measured latency, scheduled delivery step, quantization, actual first action age, hold steps and offline throughput. Step rows distinguish `bootstrap_wait_hold`, `buffer_underflow_hold` and `policy` in buffered mode, or `policy_wait_hold` and `policy` in serial mode. Records include discarded prediction prefixes, executed actions, bootstrap/underflow counts and diagnostic visualization duration. Summary/manifest/complete files name the offline clock explicitly.

## Resource and output conventions

The submission wrapper requests one GPU. Policy inference and Isaac rendering/EVIS share that card. Default hardware is one Pro 6000; an explicit `sbatch --gres=gpu:rtx5090:1` request selects one RTX 5090. The original wall-clock `--realtime` option and two-GPU layout options have been removed.

Default evaluation remains paused inference. Add `--latency-aware` to enable the offline delay model. Resume verifies the selected clock, controller and latency source and cannot mix modes in an output directory; existing paused manifests remain compatible. A 30 FPS output video represents simulation time in both modes.

Each trial's JSON records hold actions, latency, observation age, terminal object poses and task success. Short `--interface-only` smoke trials are excluded from formal success rates. Without a speed preset, the scene and data distribution remain those of the training export.

## Action-only conveyor evaluation

Use `scripts/eval_ewm/eval_ewm_action_only_piper_conveyor_sim.py` and `scripts/eval_ewm/run_ewm_action_only_piper_conveyor_eval.sh` for the dynamic RGB/state-only policy. The entry point reuses the same conveyor physics, cameras, seed rules, success tracker, simulation clocks and action-buffer controller. It connects to the independent `ewm_action_only_eval_server.py` and verifies `ewm_action_only_v1` / `rgb_state_pre_action_v1`.

The action-only wire payload contains exactly `obs_video` and `proprio`; prompt and same-row RGB/state timestamps are supplied in the header. Event arrays and predicted-event decoding are excluded. Observed events are still captured for the diagnostic video (two RGB panels plus two observed-event panels, each 256x256) and never used by the policy. Keeping the same capture pipeline preserves scene rendering and diagnostics across the comparison.

For comparison with buffered event-action job 185001, use the 80k action-only checkpoint, one RTX 5090, the same 50 scene seeds starting at 20280002, model seed 10000, sampling4/execution16/trained24, and a 1050-step budget. Each policy is charged its own measured action-reply round trip; the faster policy is not artificially assigned the event-action policy's delay.

```bash
cd /projects/hdd/myproject/CVPR2027/OpenWAM
sbatch --gres=gpu:rtx5090:1 --cpus-per-task=4 --mem=48G \
  scripts/eval_ewm/run_ewm_action_only_piper_conveyor_eval.sh \
  --checkpoint-dir outputs/conveyor_action_only_l40_184023 \
  --latency-aware --latency-controller buffered
```

Default inference remains paused unless `--latency-aware` is supplied. The optional speed presets use this same entry point and shared task. Output metadata records `training_mode=action_only`, `event_input=false` and the RGB/state observation contract; resume cannot mix policy modes. Results are stored under the action-only training run's `eval_piper_conveyor_action_only/job<id>/` directory. The existing static action-only evaluator remains separate.

## Conveyor speed OOD

Both launchers accept `--speed-ood none|slow|fast`. The reference is the checkpoint training export speed: `none` keeps it, `slow` multiplies it by 2/3, and `fast` by 4/3. For the current 0.06 m/s training export, these are 0.06, 0.04 and 0.08 m/s. The override sets the actual PhysX belt surface velocity. It preserves object spacing, yaw/position randomization, paired scene seeds, cameras, lighting, control rates and the 1050-step budget. Slower speed does not imply an easier distribution shift for a policy trained at one fixed speed.

```bash
cd /projects/hdd/myproject/CVPR2027/OpenWAM
sbatch --gres=gpu:rtx5090:1 scripts/eval_ewm/run_ewm_piper_conveyor_eval.sh \
  --checkpoint-dir outputs/conveyor_event_action_182549 \
  --latency-aware --speed-ood fast
sbatch --gres=gpu:pro6000:1 scripts/eval_ewm/run_ewm_action_only_piper_conveyor_eval.sh \
  --checkpoint-dir outputs/conveyor_action_only_l40_184023 \
  --latency-aware --speed-ood slow
```

Both examples submit 50 trials. OOD defaults are stored under the existing evaluation group in `speed_slow/job<id>/` or `speed_fast/job<id>/`; default ID paths are unchanged. Summary, manifest, complete and trial records store the speed preset, training speed, evaluation speed and multiplier. OOD videos label the evaluated speed. Resume rejects mixing presets or geometry tolerances; legacy ID speed metadata defaults to its saved scene speed. Completed historical metrics are retained.

Validation used 39 CPU contract/recovery/controller tests and four single-GPU end-to-end interface trials (both policy modes and both speeds) in job 186203. Those short `--interface-only` runs verify the pipeline and produce rendered videos; they are not formal success-rate measurements. In particular, the 300-step slow trials ended at the diagnostic step limit, and both fast trials missed the second object. The GPU capture hook checked the authored PhysX surface velocity on every mode/preset. Additional free-cube-speed and grading replay diagnostics are isolated in `tmp/piper_conveyor_success_audit/`.

## Basket success criterion (`clipped_basket_v2`)

The basket is shallow relative to the cubes. Successful placement permits tilting, stacking and protrusion above its rim. The cube must intersect the basket interior below the rim; a cube entirely above the opening or merely touching its rim is rejected. Every vertex of the cube clipped at the rim plane, including cube-edge/plane intersections, must fit the tapered basket interior within 0.1 mm numerical tolerance. Its center must lie within the opening footprint and the cube must remain above the floor limit. Wall crossing below the rim remains invalid.

Both active cubes must also satisfy the unchanged linear/angular velocity thresholds, TCP retreat distance and 30 consecutive control ticks (one simulation second). No time budget, robot motion, camera, belt geometry or randomization setting changes. Older `full_volume_v1` required the entire cube below the rim and falsely rejected legitimate tilted or stacked placements. Evaluations record the criterion version and tolerance in summary, manifest, trial and complete files; resume cannot mix them. Historical reports and videos remain intact. Saved-action replay diagnostics live in `tmp/piper_conveyor_success_audit/` and do not invoke a new policy prediction.

## Exact speed override

Both dynamic launchers also accept `--conveyor-speed 0.07` (m/s). Use it instead of `--speed-ood slow|fast`; combining them is rejected. Only the actual belt surface speed changes. Exact-speed output groups use `speed_0p07/job<id>/` for0.07m/s, and metadata records `speed_ood=custom`, both training/evaluation speeds and their ratio. Setting the exact training speed is classified as in distribution. Resume checks the recorded exact speed and scene configuration. Existing presets and default output paths remain unchanged.

```bash
sbatch --gres=gpu:rtx5090:1 --cpus-per-task=4 --mem=48G \
  scripts/eval_ewm/run_ewm_piper_conveyor_eval.sh \
  --checkpoint-dir outputs/conveyor_event_action_182549 --checkpoint-step 80000 \
  --latency-aware --latency-controller buffered --conveyor-speed 0.07
```

The override passed seven isolated CPU tests covering speed-only scene changes and paired randomness in both modes, preset compatibility, invalid/conflicting input, metadata, resume rejection, launcher forwarding and unchanged control/sensor/grading code. The submitted evaluation supplies the rendered GPU trajectories.

## Event-guided replanning with latency

`--event-guided-replan` uses the static task's polarity-preserving tolerant
occupancy F1 gate. `--event-replan-record-only` computes the same scores without
requesting earlier plans. Both automatically enable future-event decoding;
these options require the joint event-action policy. With latency enabled,
use `--latency-controller buffered`.

The default threshold remains 0.6. For this exploratory speed comparison use
an explicit threshold of 0.3, patience2, minimum4 actually executed commands
from the delivered plan, and a fixed request interval of16 ticks:

```bash
sbatch --gres=gpu:rtx5090:1 --cpus-per-task=4 --mem=48G \
  OpenWAM/scripts/eval_ewm/run_ewm_piper_conveyor_eval.sh \
  --checkpoint-dir /projects/hdd/myproject/CVPR2027/OpenWAM/outputs/conveyor_event_action_182549 \
  --checkpoint-step 80000 --conveyor-speed 0.065 --trials 50 \
  --latency-aware --latency-controller buffered \
  --event-guided-replan --event-replan-threshold 0.3
```

An error-triggered request uses the observation after the triggering command.
It does not stop the arm: the old plan continues during measured action-reply
latency. On arrival, the new plan starts at `current_step - observation_step`,
dropping commands for elapsed time. Only one request is allowed at a time;
a mismatch cannot cancel a pending request or restart its latency clock. If
the old forecast runs out, the existing buffer-underflow hold is recorded.
Success, physical failure and the step budget take precedence over a trigger.

Action delivery and forecast availability are distinct. The action-reply clock
ends after validating the action response. The forecast-ready clock additionally
includes deferred decoding, transfer and occupancy conversion. Each is rounded
up to a30Hz tick. Actions can start at the first deadline, but event scoring
cannot use the forecast before the second deadline. With adaptive control, a
new request also waits for the previous request's decoding to finish; this
models the single server's sequential inference/decoding pipeline. Diagnostic
rendering and video encoding are not charged as policy latency. CPU event-score
runtime is recorded separately under `step_timing.event_monitor`.

Compare observed events after the command with `future[index]`, whose target
is `observation_step + index + 1`. The initial bootstrap keeps its relative
action prefix after the startup hold, so its forecast targets do not align with
simulation time; these frames are excluded from error scoring. An unready
forecast and a hold command also cannot trigger. Skipped action prefixes do
not count toward the minimum execution horizon, and the mismatch streak resets
when a newly delivered plan takes over.

Step JSON records event error, per-camera error, forecast readiness/alignment,
target step, eligibility and whether an early request was actually accepted.
Request metadata records both latency clocks/deadlines, dropped prefixes, and
the old plan's trigger step. `event_mismatch` execution-chunk endings occur
when the replacement actually takes over, after the waiting tail, rather than
at the moment the early request is launched. Result/summary/complete files
record configuration and accepted-trigger counts; resume rejects changed
settings or a changed latency contract.

This is still an offline timing emulation. It preserves the existing sensor,
renderer and physics rates, and does not measure concurrent simulation/inference
GPU contention or make new plans conditional on the old commands executed
while waiting. A lower event mismatch or more frequent replanning is not by
itself evidence of a task-success improvement.
