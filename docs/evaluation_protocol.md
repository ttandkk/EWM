# Eight-task evaluation protocol

Use [`scripts/eval_piper_final.sh`](../scripts/eval_piper_final.sh) for the current
protocol. Its checked-in configuration is
[`scripts/eval_profiles/piper_eight_final.json`](../scripts/eval_profiles/piper_eight_final.json),
version `piper_eight_continuous_min8_v1`. It makes the settings previously supplied
by experiment-specific launchers explicit. It does not modify existing results,
running jobs, or the general runner's historical defaults.

## Fixed settings

| Setting | Value |
| --- | --- |
| Tasks | conveyor, moving_carton, turntable_cups, rotating_tube, tennis_rolling, tennis_falling, tennis_parabolic, tennis_container |
| Trials | 50 per task and condition; 400 per condition |
| Policy seed / scene seed | 10000 / 20280002 |
| Control / inference | 30 Hz control; continuous inference; discard obsolete action prefixes |
| Minimum execution | Execute at least 8 actions from the active plan before switching, subject to termination and step budget |
| Equivalent latency | Enabled; measured action-reply delay advances simulation time |
| Denoising | 4 flow steps |
| Compilation | Enabled, `default` mode, CUDA Graphs disabled |
| Caches | Context K/V and BF16 weight caching enabled |
| Compile error policy | Record action differences; do not abort solely because their tolerance is exceeded |
| Startup | Policy probe enabled; 3 warmup and 3 benchmark iterations |
| Action-only forecasts | Disabled |
| EWM / RGB-action forecasts | Recorded, decoded after each trial; decode time excluded from equivalent latency |

`execution_horizon=10` is passed for compatibility with recorded manifests. In
continuous mode it is not a fixed replanning period: `min_execution_steps=8`
governs switching. The checkpoint retains its own action/visual prediction
horizons, normalization and temporal stride. This entrypoint does not convert
the asymmetric, stride-1 or stride-2 models into the same architecture.

The formal entrypoint reads `model.training_mode` from the checkpoint's
`config.yaml`. Supported modes are `action_only`, `joint` and `rgb_action`.
It sets recording accordingly, without requiring a second modality flag. It
never enables inline event decoding.

## ID and OOD conditions

Choose one or more of `id`, `lighting_ood`, `blur_ood`, `background_ood`,
`speed_ood`. The default is `id`; all eight tasks are always included. Conditions
are executed serially within each task by the existing suite runner.

The scene definitions remain in
[`dynamic_evaluation.py`](../Event-DataCollection/scripts/piper_tasks/dynamic_evaluation.py)
and its existing imaging/background helpers. The wrapper does not duplicate or
alter success criteria, task limits, seeds, rendering, or perturbation strength.

- Background OOD uses the current grey wall, pale herringbone floor and grey-green
  tabletop profile; the task objects, physics and cameras retain ID settings.
- Speed OOD contains 25 slow and 25 fast episodes per task in one 50-episode
  cell. Speeds are sampled outside the ID interval: `[0.8L, 0.9L]` and
  `[1.1H, 1.2H]` for ID boundaries `L,H`; the container task uses the existing
  inverse flight-duration conversion. Other appearance settings remain ID.
- Light and blur definitions are unchanged from the current source. Each run's
  `planned_scenes.json`, `manifest.json` and `suite_manifest.json` are the precise
  record of the resolved protocol and source hashes.

## Usage

First inspect a command without writing results or allocating a GPU:

```bash
bash scripts/eval_piper_final.sh \
  --checkpoint-dir /path/to/model_run \
  --output-dir /path/to/new_results \
  --conditions id --dry-run
```

`--checkpoint-step` defaults to `80000`; `--dataset-root` defaults to the project's
`datasets/`. A checkpoint directory needs its config, normalization and selected
policy checkpoint for execution. `--dry-run` reads only the config; it does not
claim that checkpoint/data validation has passed. `--preflight` performs the
existing CPU validation and writes plans; `--plan-only` uses the suite planner.

For one condition on one RTX 5090:

```bash
EWM_EVAL_GRES=gpu:rtx5090:1 EWM_EVAL_TIME=12:00:00 \
  bash scripts/eval_piper_final.sh \
  --checkpoint-dir /path/to/model_run \
  --output-dir /path/to/new_results --conditions id
```

For multiple conditions, choose an appropriate allocation, for example:

```bash
EWM_EVAL_GRES=gpu:rtx5090:1 EWM_EVAL_TIME=3-00:00:00 \
  bash scripts/eval_piper_final.sh \
  --checkpoint-dir /path/to/model_run \
  --output-dir /path/to/new_results \
  --conditions lighting_ood blur_ood background_ood speed_ood
```

Outside an allocation the formal wrapper first validates through the existing
CPU preflight, then submits a Slurm job using an absolute `--wrap` command;
inside one it uses the current allocation. `--dry-run`, `--preflight` and
`--plan-only` do not submit jobs. GPU type, partition and time are resource
choices, not fixed protocol fields. Set `EWM_EVAL_GRES` and `EWM_EVAL_TIME`
explicitly: without them the existing general launcher defaults to a 6000 Ada
and 8 hours, which may be insufficient for a complete condition.

Profile environment values override conflicting inherited `COMPILE_*`, caching,
probe and warmup/benchmark settings. Fixed protocol flags are not accepted by
this wrapper, so an accidental `--min-execution-steps 4` fails instead of silently
changing the comparison. To run a deliberate ablation or diagnostic subset, use
`scripts/eval_piper_eight_dynamic.sh` with all desired options explicitly, in a
different output directory. That general interface now shares the formal
`continuous`/minimum-8 defaults. Historical buffered/min4 is selected explicitly with `--execution-profile legacy_buffered_min4_v1`.

## Historical results and archived launchers

The profile follows the continuous/minimum-8 execution settings of the final
ID/OOD evaluations and job 1732584. Earlier ID runs without forecast recording
keep their original manifests; enabling post-trial recording here does not
retroactively add decoded videos or change those measured results. Compilation
and actual wall-clock inference still depend on the selected environment/GPU.

Fifteen fixed-checkpoint launchers were moved to
[`archive/launchers/`](../archive/launchers/README.md), with original bytes and
hashes retained. Their old names and options are historical evidence, not current
defaults. Other experiment scripts remain available for reproducing their
recorded settings.
