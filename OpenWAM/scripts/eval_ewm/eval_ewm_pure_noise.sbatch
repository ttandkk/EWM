#!/bin/bash
#SBATCH --job-name=ewm-pure-noise-eval
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/ewm_pure_noise_eval_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/ewm_pure_noise_eval_%j.log
set -euo pipefail
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
root=/projects/hdd/myproject/CVPR2027
"$root/envs/openwam/bin/python" "$root/OpenWAM/scripts/eval_ewm/eval_ewm_pure_noise.py" \
  --checkpoint-dir "$1" --steps "$2" --output-dir "$3"
