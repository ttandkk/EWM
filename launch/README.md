# Source snapshots and cluster resources

`freeze.py` copies the canonical benchmark (including resources), policy adapters,
model source/configs and compatibility entrypoints. It excludes environments,
weights, datasets and existing results. Every copied file is hashed. A snapshot
is never overwritten; verification rejects modified files.

```bash
python3 launch/freeze.py --output /path/to/new-source-snapshot
python3 launch/freeze.py --output /path/to/new-source-snapshot --verify
```

Slurm resources are selected independently of the benchmark protocol. The submit
helper first freezes source and completes CPU preflight; then it submits one
allocation. `--dry-run` prints the command without writing or submitting.

```bash
export PIPER_POLICY_PYTHON=/path/to/policy-env/bin/python
export PIPER_SIM_PYTHON=/path/to/simulation-env/bin/python
export PIPER_CONFIG=/path/to/runtime/deployment.json
python3 launch/slurm/submit_evaluation.py \
  --snapshot /path/to/new-source-snapshot \
  --partition YOUR_PARTITION --gres gpu:1 --time 08:00:00 \
  --log /path/to/logs/piper-%j.log --dry-run -- \
  --checkpoint-dir /path/to/checkpoint --checkpoint-step 80000 \
  --conditions background_ood --output-dir /path/to/new-results
```

Remove `--dry-run` to submit. The checkpoint compatibility launcher owns policy
processes, starts one service per cell, and always terminates its own process on
exit. It uses the independent model interpreter to describe/load weights; no
Hydra parsing occurs in the benchmark. `--pretrained-root` and `--project-root`
relocate model assets. Existing dataset/model artifacts remain external and are
identified by the recorded hashes. A preflight does not prove GPU inference.

Alternatively supply `--policy-url`/`--policy-map` for services managed separately.
They must be reachable from the allocated host. `--tasks` restricts the suite;
the default is eight tasks × five conditions × 50 trials. Protocol defaults are
continuous/min8, 30 Hz and v5 grey-green background. Hardware names appear only
in the launch arguments, not in the suite definition.

Existing collection launchers now freeze the canonical package too. Old frozen
runs retain their original hash lists and paths; resume never upgrades a snapshot.
Historical job scripts in `scripts/` may still require the original local layout;
the local `archive/` is excluded from the source release. New generic deployment
uses the installed CLI and this resource layer. Snapshots include the environment
selection helpers in `deployment/` and do not copy the old collection tree.
