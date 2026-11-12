#!/usr/bin/env bash
# Source this script with: source deployment/activate_training.sh groot|fastwam
training_env_name="${1:-}"
case "$training_env_name" in
    groot|fastwam) ;;
    *) printf 'Usage: source deployment/activate_training.sh groot|fastwam\n' >&2; return 2 ;;
esac
export TRAINING_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$TRAINING_PROJECT_ROOT/envs/$training_env_name/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$TRAINING_PROJECT_ROOT/tmp/training_cache/$training_env_name"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$TRAINING_PROJECT_ROOT/pretrained/.hf-cache/hub"
export GROOT_BASE_MODEL_PATH="$TRAINING_PROJECT_ROOT/pretrained/nvidia/GR00T-N1.7-3B"
export GROOT_BACKBONE_PATH="$TRAINING_PROJECT_ROOT/pretrained/nvidia/Cosmos-Reason2-2B"
export DIFFSYNTH_MODEL_BASE_PATH="$TRAINING_PROJECT_ROOT/pretrained"
export FASTWAM_ACTION_DIT_PATH="$TRAINING_PROJECT_ROOT/pretrained/FastWAM/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
export TEST_CACHE_PATH="$XDG_CACHE_HOME/tests"
export UV_CACHE_DIR="$TRAINING_PROJECT_ROOT/tmp/training_setup/uv-cache"
export NO_ALBUMENTATIONS_UPDATE=1
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
case ":${LD_LIBRARY_PATH:-}:" in
    *":$TRAINING_PROJECT_ROOT/envs/torchcodec_ffmpeg7/lib:"*) ;;
    *) export LD_LIBRARY_PATH="$TRAINING_PROJECT_ROOT/envs/torchcodec_ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
if [[ -z "${CUDA_HOME:-}" && -x /cluster/apps/software/CUDA/12.8.0/bin/nvcc ]]; then
    export CUDA_HOME=/cluster/apps/software/CUDA/12.8.0
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    case ":$PATH:" in
        *":$CUDA_HOME/bin:"*) ;;
        *) export PATH="$CUDA_HOME/bin:$PATH" ;;
    esac
fi
unset training_env_name
