#!/usr/bin/env bash
#SBATCH --job-name=piper-conveyor-test
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/piper_conveyor_test_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/piper_conveyor_test_%j.log
set -euo pipefail
project_root=/projects/hdd/myproject/CVPR2027
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$project_root/isaacsim601/cache"
export OMNI_USER_CONFIG_PATH="$project_root/isaacsim601/config"
cd "$project_root/Event-DataCollection"
"$project_root/isaacsim601/env/bin/python" scripts/eval_piper_conveyor.py \
  --output-dir "$project_root/tmp/piper_conveyor_v1/scene_transport/job${SLURM_JOB_ID}" \
  --mode transport --max-steps 180 --seed 20261002 --headless --enable_cameras --device cpu "$@"

# A Slurm exit code alone is insufficient: Kit may mask a Python exception.
"$project_root/isaacsim601/env/bin/python" - "$project_root/tmp/piper_conveyor_v1/scene_transport/job${SLURM_JOB_ID}/result.json" <<'PYTEST'
import json, sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
if r['mode'] == 'oracle_eval':
    assert r['passed']
elif r['mode'] == 'jacobian_check':
    assert r['jacobian_passed']
else:
    assert r['video_decoded']
    assert r['task_success'] if r['mode'] == 'reach' else r['transport_passed']
PYTEST
