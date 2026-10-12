#!/usr/bin/env bash
#SBATCH --job-name=piper-conveyor-eval
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/piper_conveyor_eval_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/piper_conveyor_eval_%j.log
set -euo pipefail
project_root=/projects/hdd/myproject/CVPR2027
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$project_root/isaacsim601/cache" OMNI_USER_CONFIG_PATH="$project_root/isaacsim601/config"
output_dir="$project_root/tmp/piper_conveyor_v1/oracle_eval/job${SLURM_JOB_ID}"
cd "$project_root/Event-DataCollection"
"$project_root/isaacsim601/env/bin/python" scripts/eval_piper_conveyor.py \
  --output-dir "$output_dir" --mode oracle_eval --trials 100 --max-steps 1050 --seed 20264002 \
  --device cpu --headless --enable_cameras "$@"
"$project_root/isaacsim601/env/bin/python" - "$output_dir/result.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));assert r['passed'] and r['completed_trials']==r['requested_trials']
PY
