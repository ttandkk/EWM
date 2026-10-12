#!/usr/bin/env bash
set -euo pipefail

root=/projects/hdd/myproject/CVPR2027
mkdir -p "$root/logs"
"$root/envs/openwam/bin/python" "$root/scripts/export_reassemble_training.py" prepare
job=$(sbatch --parsable "$root/scripts/reassemble_training_export.sbatch")
printf 'single_export_job=%s\n' "$job"
printf '%s\n' "Final export path: $root/datasets/reassemble_training_eventframes_10ms"
