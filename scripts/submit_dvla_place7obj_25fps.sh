#!/usr/bin/env bash
set -euo pipefail

root=/projects/hdd/myproject/CVPR2027
source_dir="$root/datasets/dvla-place7obj-roll-250hz-events-250fps-delta-trim"
test -f "$source_dir/.download_complete"
mkdir -p "$root/logs"
job=$(sbatch --parsable "$root/scripts/dvla_place7obj_25fps_training.sbatch")
printf 'dvla_25hz_export_job=%s\n' "$job"
printf '%s\n' "Final export path: $root/datasets/dvla-place7obj-roll-25fps-events-40ms-delta-trim"
