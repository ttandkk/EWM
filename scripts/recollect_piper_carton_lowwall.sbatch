#!/usr/bin/env bash
#SBATCH --job-name=piper-carton-lowwall
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=33G
#SBATCH --time=24:00:00
#SBATCH --chdir=/projects/_hdd/isaachdd/EWM
#SBATCH --output=/projects/_hdd/isaachdd/EWM/logs/piper_carton_lowwall_%j.log
#SBATCH --error=/projects/_hdd/isaachdd/EWM/logs/piper_carton_lowwall_%j.log
set -euo pipefail
source /projects/_hdd/isaachdd/EWM/deployment/runtime_env.sh
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PXR_WORK_THREAD_LIMIT=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH="$NEW_PROJECT_ROOT/envs/torchcodec_ffmpeg7/bin:$PATH"
run="${1:?prepared run directory required}"
export PYTHONPATH="$run/source/scripts${PYTHONPATH:+:$PYTHONPATH}"
ulimit -c 0
nvidia-smi --query-gpu=name,uuid --format=csv
"$OWPY" - "$run" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]);manifest=json.loads((r/'source_sha256.json').read_text())
for name,expected in manifest.items():
    assert hashlib.sha256((r/'source/scripts'/name).read_bytes()).hexdigest()==expected,name
PY
"$SIMPY" "$run/validate.py" --project-root "$NEW_PROJECT_ROOT" \
    --output-dir "$run/smoke_${SLURM_JOB_ID}" --collection-config "$run/collection_config.json" \
    --slots 0 24 49 --headless --enable_cameras --device cpu \
    --kit_args=--/plugins/carb.tasking.plugin/threadCount=4
"$OWPY" - "$run/smoke_${SLURM_JOB_ID}/summary.json" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['complete'] and r['all_quality_passed'], 'Smoke/placement quality failed; inspect videos before collecting'
PY
resume=()
if [[ -f "$run/audit/collection_config.json" ]]; then resume=(--resume); fi
dataset="$NEW_PROJECT_ROOT/datasets/piper_moving_carton_lowwall_v4_20261029"
"$SIMPY" "$run/source/scripts/collect_piper_tasks_evis.py" \
    --task moving_carton --collection-protocol piper_carton_lowwall_v4 \
    --output "$dataset" --audit-dir "$run/audit" \
    --episodes 50 --max-attempts 150 --acceptance-trials 20 --acceptance-rate .7 \
    --seed 20261029 --pathtracing-spp 64 --save-raw-events \
    --writer-python "$LRPY" --device cpu --headless --enable_cameras \
    --kit_args=--/plugins/carb.tasking.plugin/threadCount=4 "${resume[@]}"
"$LRPY" "$run/source/scripts/validate_piper_tasks_lerobot.py" "$dataset" --episodes 50
"$OWPY" - "$run" "$dataset" <<'PY'
import json,sys
from pathlib import Path
r,d=map(Path,sys.argv[1:]);v=json.loads((d/'meta/piper_validation.json').read_text())
assert v['passed'] and v['episodes']==50 and v['raw_events_validated']
(r/'complete.json').write_text(json.dumps(dict(passed=True,dataset=str(d),episodes=50,
    production_dataset_replaced=False,production_caches_modified=False),indent=2)+'\n')
PY
