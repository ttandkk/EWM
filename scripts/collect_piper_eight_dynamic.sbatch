#!/usr/bin/env bash
#SBATCH --job-name=piper-eight-collect
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:6000ada:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/projects/_hdd/isaachdd/EWM
#SBATCH --output=/projects/_hdd/isaachdd/EWM/logs/piper_eight_collect_%j.log
#SBATCH --error=/projects/_hdd/isaachdd/EWM/logs/piper_eight_collect_%j.log
# Submit from the project root; arguments pass through to the serial runner.
set -euo pipefail
root=/projects/_hdd/isaachdd/EWM
exec srun --ntasks=1 bash "$root/scripts/collect_piper_eight_dynamic.sh" "$@"
