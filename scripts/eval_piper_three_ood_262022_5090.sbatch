#!/usr/bin/env bash
#SBATCH --job-name=piper-three-ood-262022
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/projects/_hdd/isaachdd/EWM
#SBATCH --output=/projects/_hdd/isaachdd/EWM/logs/piper_three_ood_262022_5090_%j.log
#SBATCH --error=/projects/_hdd/isaachdd/EWM/logs/piper_three_ood_262022_5090_%j.log
# One allocation: action-only -> joint EWM -> RGB-action, 800 formal trials each.
# The submission snapshot freezes all first-party code; weights/data are reused.
# CPU validation: bash scripts/eval_piper_three_ood_262022_5090.sbatch --preflight
set -euo pipefail
root=/projects/_hdd/isaachdd/EWM
batch="$root/tmp/ewm_three_ood_262022_5090_20261107"
source_root="$batch/source"
checkpoints="$root/tmp/ewm_formal_ada_serial_compile_20261104/runs"
out="$root/results/piper_three_ood_262022_step080000_continuous_min8_decode_5090_20261107"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --preflight ) ]]; then
    echo 'Usage: eval_piper_three_ood_262022_5090.sbatch [--preflight]' >&2
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
"$python" -B - "$checkpoints" <<'PY'
import json
import sys
from pathlib import Path
from omegaconf import OmegaConf
for mode in ('action_only', 'joint', 'rgb_action'):
    directory = Path(sys.argv[1]) / mode
    done = json.loads((directory / 'formal_complete.json').read_text())
    assert done['mode'] == mode and done['step'] == 80000 and str(done['job_id']) == '262022', done
    assert (directory / 'checkpoint_step_080000.pt').is_file()
    cfg = OmegaConf.load(directory / 'config.yaml')
    assert cfg.model.get('training_mode', 'joint') == mode and cfg.dataloader.chunk_size == 24
    if mode != 'action_only':
        assert cfg.model.event_horizon == 24 and cfg.model.event_temporal_downsample == 2
    print('TRAINING_COMPLETE', mode, json.dumps(done), flush=True)
PY
tasks=(conveyor moving_carton turntable_cups rotating_tube tennis_rolling tennis_falling tennis_parabolic tennis_container)
models=(action_only joint rgb_action)
common=(--checkpoint-step 80000 --dataset-root "$root/datasets"
        --conditions lighting_ood blur_ood --latency-aware --latency-controller continuous
        --min-execution-steps 8 --execution-horizon 10 --sampling-steps 4
        --seed 10000 --scene-seed 20280002)
run_suite() {
    local mode="$1" phase="$2"
    local recording=(--record-predictions)
    local options=(--tasks "${tasks[@]}" --trials 50)
    local destination="$out/$mode/ood"
    if [[ "$mode" == action_only ]]; then recording=(--no-record-predictions); fi
    if [[ "$phase" == smoke ]]; then
        options=(--tasks conveyor --trials 2 --interface-only --max-steps 32)
        destination="$out/$mode/smoke"
    elif [[ "$phase" == preflight ]]; then
        options+=(--preflight)
    fi
    echo "[ood-serial] mode=$mode phase=$phase start=$(date -u +%FT%TZ)"
    bash "$source_root/scripts/eval_piper_eight_dynamic.sh" "${common[@]}" \
        --checkpoint-dir "$checkpoints/$mode" "${recording[@]}" "${options[@]}" \
        --output-dir "$destination" || return $?
    echo "[ood-serial] mode=$mode phase=$phase complete=$(date -u +%FT%TZ)"
}
if [[ "${1:-}" == --preflight ]]; then
    for mode in "${models[@]}"; do run_suite "$mode" preflight; done
    exit 0
fi
"$python" -B - <<'PY'
import torch
assert torch.cuda.device_count() == 1, 'Expected one allocated GPU'
name = torch.cuda.get_device_name(0)
assert '5090' in name, name
print('GPU_VERIFIED', name, flush=True)
PY
# Verify both imaging conditions and two successive recording sessions before
# spending the allocation on formal trials. No success threshold is imposed.
for mode in "${models[@]}"; do run_suite "$mode" smoke; done
"$python" -B - "$out" <<'PY'
import json
import sys
from pathlib import Path
for mode in ('action_only', 'joint', 'rgb_action'):
    directory = Path(sys.argv[1]) / mode / 'smoke'
    suite = json.loads((directory / 'suite_summary.json').read_text())
    assert suite['complete'] and suite['completed_trials'] == 4, (mode, suite)
    for condition in ('lighting_ood', 'blur_ood'):
        cell = directory / 'conveyor' / condition
        for trial in range(2):
            result = json.loads((cell / f'trial_{trial:03d}_result.json').read_text())
            assert result['video_decoded'] and result['latency_timing']['controller'] == 'continuous'
            recording = result['prediction_recording']
            if mode == 'action_only':
                assert recording is None
            else:
                assert recording['prediction_decode_phase'] == 'post_trial'
                assert recording['prediction_decode_in_equivalent_latency'] is False
                assert recording['prediction_kind'] == ('rgb' if mode == 'rgb_action' else 'events')
                assert recording['decoded_predictions'] > 0
    print('OOD_SMOKE_VERIFIED', mode, flush=True)
PY
# A failed cell is reported by the suite and does not prevent the other models
# from being evaluated. Preserve a nonzero final exit status for incomplete work.
rc=0
for mode in "${models[@]}"; do
    run_suite "$mode" formal || rc=1
done
echo "[ood-serial] finished job=$SLURM_JOB_ID exit=$rc time=$(date -u +%FT%TZ)"
exit "$rc"
