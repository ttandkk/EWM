#!/usr/bin/env bash
#SBATCH --job-name=piper-tennis-check
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:l40:1
#SBATCH --time=00:10:00
#SBATCH --output=/projects/_hdd/isaachdd/EWM/logs/piper_tennis_%j.log
#SBATCH --error=/projects/_hdd/isaachdd/EWM/logs/piper_tennis_%j.log
# Run from the EWM project root; optional first argument selects one tennis task.
set -euo pipefail
root="${PIPER_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the EWM project root}}"
selected="${1:-tennis}"
if (( $# > 0 )); then shift; fi
case "$selected" in
    tennis|tennis_rolling|tennis_falling|tennis_parabolic|tennis_container) ;;
    *) echo "Unknown tennis task: $selected" >&2; exit 2 ;;
esac
exec bash "$root/scripts/test_piper_dynamic_pro6000.sbatch" "$selected" "$@"
