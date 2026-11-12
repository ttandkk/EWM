#!/usr/bin/env bash
#SBATCH --job-name=piper-wrist-render
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=00:10:00
#SBATCH --output=/projects/_hdd/isaachdd/EWM/logs/piper_wrist_render_%j.log
#SBATCH --error=/projects/_hdd/isaachdd/EWM/logs/piper_wrist_render_%j.log
# Submit from EWM: sbatch scripts/render_piper_wrist_rtx5090.sbatch /absolute/render_bundle
set -euo pipefail
root="${PIPER_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from EWM}}"
bundle="${1:?Provide a reviewed bundle containing scripts/ and inputs/}"
source "$root/deployment/runtime_env.sh"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PATH="$root/envs/torchcodec_ffmpeg7/bin:$PATH"
export PYTHONPATH="$bundle/scripts${PYTHONPATH:+:$PYTHONPATH}"
run="$bundle/results/${SLURM_JOB_ID:?}"
mkdir -p "$(dirname "$run")"
mkdir "$run"
"$OWPY" - "$bundle" <<'PY'
import hashlib,json,sys
from pathlib import Path
bundle=Path(sys.argv[1])
for name,digest in json.loads((bundle/'sha256.json').read_text()).items():
    if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError('reviewed render input changed: '+name)
PY
failed=0
for task in square_nut rotating_tube drawer_cup; do
    if "$SIMPY" "$bundle/scripts/render_piper_wrist_calibration.py" \
        --task "$task" --trajectory-dir "$bundle/inputs/$task/oracle_000" \
        --output-dir "$run/$task" --stride 2 --spp 32 --device cpu --headless --enable_cameras \
        > "$run/$task.log" 2>&1 && test -f "$run/$task/calibration.json"; then
        echo "$task rendered and calibration checked"
    else
        failed=1
        echo "$task rendering failed; log: $run/$task.log"
        tail -25 "$run/$task.log"
    fi
done
"$OWPY" - "$run" "$failed" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]);reports={}
for task in ('square_nut','rotating_tube','drawer_cup'):
    path=run/task/'calibration.json'
    reports[task]=json.loads(path.read_text()) if path.exists() else {'passed':False,'error':'missing calibration report'}
passed=not int(sys.argv[2]) and all(r['passed'] for r in reports.values())
(run/'complete.json').write_text(json.dumps(dict(complete=True,passed=passed,tasks=reports),indent=2)+'\n')
if not passed:raise SystemExit(1)
PY
