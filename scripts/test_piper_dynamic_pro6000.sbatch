#!/usr/bin/env bash
#SBATCH --job-name=piper-dynamic-check
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/projects/_hdd/isaachdd/EWM/logs/piper_dynamic_%j.log
#SBATCH --error=/projects/_hdd/isaachdd/EWM/logs/piper_dynamic_%j.log
# Run from the project root: sbatch scripts/test_piper_dynamic_pro6000.sbatch [all|TASK] [preview options]
set -euo pipefail
root="${PIPER_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}}"
source "$root/deployment/runtime_env.sh"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PATH="$root/envs/torchcodec_ffmpeg7/bin:$PATH"
selected="${1:-all}"
if (( $# > 0 )); then shift; fi
case "$selected" in
    all) tasks=(moving_carton turntable_cups rotating_tube) ;;
    static) tasks=(push_t drawer_cup square_nut) ;;
    tennis) tasks=(tennis_rolling tennis_falling tennis_parabolic tennis_container) ;;
    moving_carton|turntable_cups|rotating_tube|push_t|drawer_cup|square_nut|tennis_rolling|tennis_falling|tennis_parabolic|tennis_container) tasks=("$selected") ;;
    *) echo "Unknown task: $selected" >&2; exit 2 ;;
esac
run="${PIPER_DYNAMIC_OUTPUT_ROOT:-$root/tmp/piper_dynamic_validation}/results/${SLURM_JOB_ID:?}"
"$OWPY" - "$root" "$run" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
root,run=map(Path,sys.argv[1:]);run.mkdir(parents=True,exist_ok=False)
scripts=root/'Event-DataCollection/scripts';tests=root/'Event-DataCollection/tests'
files=list(scripts.glob('piper_conveyor*.py'))
files += [scripts/name for name in ('piper_scene_background.py','piper_wrist_camera.py','piper_wrist_sensor.py','piper_event_frames.py','_isaacsim6_compat.py','eval_piper_dynamic.py')]
files += list((scripts/'piper_tasks').rglob('*.py'))
files += [tests/name for name in ('test_piper_moving_carton.py','test_piper_turntable_cups.py','test_piper_rotating_tube.py','test_piper_turntable.py','test_piper_tasks.py','test_piper_static_tasks.py','test_piper_square_nut.py','test_piper_tennis.py')]
manifest={}
for source in files:
    relative=source.relative_to(root/'Event-DataCollection');target=run/'source'/relative
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    manifest[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
shutil.copy2(root/'scripts/test_piper_dynamic_pro6000.sbatch',run/'source/launcher.sbatch')
manifest['launcher.sbatch']=hashlib.sha256((run/'source/launcher.sbatch').read_bytes()).hexdigest()
(run/'source_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
export PYTHONPATH="$run/source/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$SIMPY" -m unittest discover -s "$run/source/tests" -p 'test_piper_*.py' > "$run/cpu_tests.log" 2>&1
cat "$run/cpu_tests.log"
for task in "${tasks[@]}"; do
    "$SIMPY" "$run/source/scripts/eval_piper_dynamic.py" --task "$task" \
        --output-dir "$run/$task" --mode oracle --trials 1 --events --enable_cameras --device cpu "$@"
    "$OWPY" - "$run/$task/complete.json" <<'PY'
import json,sys
with open(sys.argv[1]) as f:report=json.load(f)
assert report['complete'] and report['passed'],report
print('Validated:',sys.argv[1],flush=True)
PY
done
"$OWPY" - "$run" "${tasks[@]}" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1])
report=dict(complete=True,passed=True,tasks={})
for name in sys.argv[2:]:
    result=json.loads((run/name/'complete.json').read_text())
    assert result['complete'] and result['passed'],result
    report['tasks'][name]=result
(run/'complete.json').write_text(json.dumps(report,indent=2)+'\n')
PY
