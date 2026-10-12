#!/usr/bin/env bash
#SBATCH --job-name=piper-stack-regression
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:10:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/piper_stack_regression_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/piper_stack_regression_%j.log
set -euo pipefail
project_root=/projects/hdd/myproject/CVPR2027
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$project_root/isaacsim601/cache" OMNI_USER_CONFIG_PATH="$project_root/isaacsim601/config"
export PIPER_REGRESSION_OUT="$project_root/tmp/piper_conveyor_v1/stack_regression/job${SLURM_JOB_ID}"
"$project_root/isaacsim601/env/bin/python" "$project_root/tmp/piper_conveyor_v1/stack_regression.py" --trials 1 --max_steps 30 --headless --enable_cameras --device cpu
"$project_root/envs/openwam/bin/python" - "$PIPER_REGRESSION_OUT/result.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))['passed']
PY
