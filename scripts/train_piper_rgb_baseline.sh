#!/usr/bin/env bash
# Run on an allocated compute node. This launcher does not submit jobs itself.
set -euo pipefail
baseline="${1:-help}"
[[ $# -eq 0 ]] || shift
PIPER_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PIPER_RGB_DATA_ROOT="${PIPER_RGB_DATA_ROOT:-$PIPER_PROJECT_ROOT/tmp/training_data_cache/piper_rgb_dynamic_v1}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export WANDB_MODE="${WANDB_MODE:-disabled}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
case "$baseline" in
  prepare)
    exec "$PIPER_PROJECT_ROOT/envs/lerobot/bin/python" "$PIPER_PROJECT_ROOT/scripts/prepare_piper_rgb_baselines.py" --output "$PIPER_RGB_DATA_ROOT" "$@"
    ;;
  openwam)
    test -f "$PIPER_RGB_DATA_ROOT/manifest.json"
    export DIFFSYNTH_MODEL_BASE_PATH="$PIPER_PROJECT_ROOT/pretrained"
    export LD_LIBRARY_PATH="$PIPER_PROJECT_ROOT/envs/torchcodec_ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    cd "$PIPER_PROJECT_ROOT/OpenWAM"
    exec "$PIPER_PROJECT_ROOT/envs/openwam/bin/torchrun" --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
      scripts/train.py --config-name train_openwam_piper_rgb "$@"
    ;;
  fastwam|fastwam-text-cache)
    test -f "$PIPER_RGB_DATA_ROOT/manifest.json"
    source "$PIPER_PROJECT_ROOT/deployment/activate_training.sh" fastwam
    cd "$PIPER_PROJECT_ROOT/FastWAM"
    if [[ "$baseline" == fastwam-text-cache ]]; then
      exec python scripts/precompute_text_embeds.py task=piper_rgb +overwrite=false "$@"
    fi
    # Fail before loading model weights when the required text cache is absent.
    python - "$PIPER_RGB_DATA_ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
root = Path(sys.argv[1])
for dataset in json.loads((root / 'manifest.json').read_text())['tasks']:
    for line in (root / dataset['task'] / 'meta/tasks.jsonl').read_text().splitlines():
        prompt = DEFAULT_PROMPT.format(task=json.loads(line)['task'])
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        path = root / 'fastwam_text' / f'{digest}.t5_len128.wan22ti2v5b.pt'
        if not path.is_file():
            raise SystemExit('Missing text cache; run train_piper_rgb_baseline.sh fastwam-text-cache first')
PY
    exec bash scripts/train_zero1.sh "${NPROC_PER_NODE:-1}" task=piper_rgb "$@"
    ;;
  groot)
    test -f "$PIPER_RGB_DATA_ROOT/manifest.json"
    source "$PIPER_PROJECT_ROOT/deployment/activate_training.sh" groot
    cd "$PIPER_PROJECT_ROOT/Isaac-GR00T"
    dataset_paths="$(python - "$PIPER_RGB_DATA_ROOT" <<'PY'
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
print(os.pathsep.join(str(root / task['task']) for task in json.loads((root / 'manifest.json').read_text())['tasks']))
PY
)"
    exec python gr00t/experiment/launch_finetune.py \
      --base-model-path "$GROOT_BASE_MODEL_PATH" --dataset-path "$dataset_paths" \
      --embodiment-tag NEW_EMBODIMENT --modality-config-path examples/Piper/piper_config.py \
      --num-gpus "${NPROC_PER_NODE:-1}" --global-batch-size "${NPROC_PER_NODE:-1}" --gradient-accumulation-steps 16 \
      --dataloader-num-workers 2 --max-steps 10000 --save-steps 2000 --save-total-limit 2 \
      --no-use-wandb --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
      --shortest-image-edge 256 --crop-fraction 1.0 --no-use-percentiles \
      --output-dir "$PIPER_PROJECT_ROOT/Isaac-GR00T/outputs/piper_rgb" "$@"
    ;;
  *)
    printf '%s\n' 'Usage: bash scripts/train_piper_rgb_baseline.sh prepare|openwam|fastwam-text-cache|fastwam|groot [native CLI overrides]'
    [[ "$baseline" == help ]] || exit 2
    ;;
esac
