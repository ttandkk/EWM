#!/usr/bin/env bash
#SBATCH --job-name=ewm276490-id-ood-pro6k
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=33G
#SBATCH --time=3-00:00:00
#SBATCH --dependency=afterok:276490
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --chdir=/projects/_hdd/isaachdd/EWM
#SBATCH --output=/projects/_hdd/isaachdd/EWM/logs/piper_ewm_276490_id_ood_pro6000_%j.log
#SBATCH --error=/projects/_hdd/isaachdd/EWM/logs/piper_ewm_276490_id_ood_pro6000_%j.log
# One allocation: eight-task ID -> lighting OOD -> blur OOD, 400 trials each.
# --preflight validates the existing 20k checkpoint in separate CPU-only plans.
# Actual rollouts require the successful 276490 completion marker and 80k weights.
set -euo pipefail
root=/projects/_hdd/isaachdd/EWM
batch="$root/tmp/ewm_276490_id_ood_pro6000_20261107"
source_root="$batch/source"
checkpoint="$root/OpenWAM/outputs/ewm_joint_stride1_ada_20261106"
out="$root/results/piper_ewm_eval_276490_step080000_continuous_min8_decode_pro6000_20261107"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --preflight ) ]]; then
    echo 'Usage: eval_piper_ewm_276490_id_ood_pro6000.sbatch [--preflight]' >&2
    exit 2
fi
if [[ "${1:-}" != --preflight && -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch; --preflight only validates on CPU.' >&2
    exit 2
fi
cd "$root"
sha256sum --quiet -c "$batch/SHA256SUMS"
export EWM_PROJECT_ROOT="$source_root"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export COMPILE_CORE=true COMPILE_MODE=default COMPILE_CUDA_GRAPHS=false
export POLICY_SERVER_PROBE=true CACHE_CONTEXT_KV=true CACHE_BF16_WEIGHTS=true
export ALLOW_COMPILE_ACTION_ERROR=true WARMUP_ITERS=3 BENCHMARK_ITERS=3
export TORCHINDUCTOR_COMPILE_THREADS=2
export CUDA_HOME=/cluster/apps/software/CUDA/12.8.0
export PATH="$CUDA_HOME/bin:$PATH"
test -x "$CUDA_HOME/bin/nvcc"
python="$root/envs/openwam/bin/python"
"$python" -B - "$checkpoint" "${1:-}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from omegaconf import OmegaConf
directory = Path(sys.argv[1])
cfg = OmegaConf.load(directory / 'config.yaml')
assert cfg.model.get('training_mode', 'joint') == 'joint'
assert cfg.model.event_temporal_downsample == 1
assert cfg.model.event_horizon == cfg.dataloader.chunk_size == 24
if sys.argv[2] != '--preflight':
    done = json.loads((directory / 'formal_complete.json').read_text())
    assert done['mode'] == 'joint' and done['step'] == 80000 and str(done['job_id']) == '276490', done
    assert done['event_temporal_downsample'] == 1
    assert done['experiment_sha256'] == hashlib.sha256((directory / 'experiment.json').read_bytes()).hexdigest()
    assert (directory / 'checkpoint_step_080000.pt').is_file()
    print('TRAINING_COMPLETE', json.dumps(done), flush=True)
PY
tasks=(conveyor moving_carton turntable_cups rotating_tube tennis_rolling tennis_falling tennis_parabolic tennis_container)
conditions=(id lighting_ood blur_ood)
common=(--dataset-root "$root/datasets" --checkpoint-dir "$checkpoint"
        --latency-aware --latency-controller continuous --min-execution-steps 8
        --execution-horizon 10 --sampling-steps 4 --seed 10000 --scene-seed 20280002
        --record-predictions)
if [[ "${1:-}" == --preflight ]]; then
    for condition in "${conditions[@]}"; do
        bash "$source_root/scripts/eval_piper_eight_dynamic.sh" "${common[@]}" \
            --checkpoint-step 20000 --conditions "$condition" --tasks "${tasks[@]}" \
            --trials 50 --output-dir "$batch/preflight_step020000/$condition" --preflight
    done
    exit 0
fi
"$python" -B - <<'PY'
import torch
assert torch.cuda.device_count() == 1, 'Expected one allocated GPU'
name = torch.cuda.get_device_name(0)
assert 'PRO 6000' in name.upper(), name
print('GPU_VERIFIED', name, flush=True)
PY
echo "[ewm-id-ood] job=$SLURM_JOB_ID checkpoint=276490/80000 start=$(date -u +%FT%TZ)"
bash "$source_root/scripts/eval_piper_eight_dynamic.sh" "${common[@]}" \
    --checkpoint-step 80000 --conditions "${conditions[@]}" --tasks conveyor \
    --trials 2 --interface-only --max-steps 32 --output-dir "$out/smoke"
"$python" -B - "$out/smoke" <<'PY'
import json
import sys
from pathlib import Path
directory = Path(sys.argv[1])
suite = json.loads((directory / 'suite_summary.json').read_text())
assert suite['complete'] and suite['completed_trials'] == 6
for condition in ('id', 'lighting_ood', 'blur_ood'):
    for trial in range(2):
        result = json.loads((directory / 'conveyor' / condition / f'trial_{trial:03d}_result.json').read_text())
        recording = result['prediction_recording']
        assert result['video_decoded'] and result['latency_timing']['controller'] == 'continuous'
        assert recording['prediction_decode_phase'] == 'post_trial'
        assert recording['prediction_decode_in_equivalent_latency'] is False
        assert recording['prediction_kind'] == 'events' and recording['decoded_predictions'] > 0
print('ID_OOD_SMOKE_VERIFIED', flush=True)
PY
rc=0
for condition in "${conditions[@]}"; do
    echo "[ewm-id-ood] condition=$condition start=$(date -u +%FT%TZ)"
    bash "$source_root/scripts/eval_piper_eight_dynamic.sh" "${common[@]}" \
        --checkpoint-step 80000 --conditions "$condition" --tasks "${tasks[@]}" \
        --trials 50 --output-dir "$out/$condition" || rc=1
done
echo "[ewm-id-ood] finished job=$SLURM_JOB_ID exit=$rc time=$(date -u +%FT%TZ)"
exit "$rc"
