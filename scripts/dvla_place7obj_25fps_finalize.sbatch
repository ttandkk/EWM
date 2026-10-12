#!/usr/bin/env bash
#SBATCH --job-name=dvla25-finalize
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/dvla25_finalize_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/dvla25_finalize_%j.log
set -euo pipefail
ROOT=/projects/hdd/myproject/CVPR2027
SOURCE="$ROOT/datasets/dvla-place7obj-roll-250hz-events-250fps-delta-trim"
DEST="$ROOT/datasets/dvla-place7obj-roll-25fps-events-40ms-delta-trim"
test -f "$SOURCE/.download_complete"
"$ROOT/envs/lerobot/bin/python" "$ROOT/scripts/convert_dvla_place7obj_25fps.py" --source "$SOURCE" --dest "$DEST" finalize
