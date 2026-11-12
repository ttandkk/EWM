#!/usr/bin/env bash
# Source this file to select destination-account EWM assets and native libraries.
# Python environments remain independent; use OWPY, LRPY, or SIMPY explicitly.
export NEW_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OWPY="${PIPER_POLICY_PYTHON:-${OWPY:-$NEW_PROJECT_ROOT/envs/openwam/bin/python}}"
export LRPY="${PIPER_WRITER_PYTHON:-${LRPY:-$NEW_PROJECT_ROOT/envs/lerobot/bin/python}}"
export SIMPY="${PIPER_SIM_PYTHON:-${SIMPY:-$NEW_PROJECT_ROOT/isaacsim601/env/bin/python}}"
export EWM_DATASET_DIR="${EWM_DATASET_DIR:-$NEW_PROJECT_ROOT/datasets/piper_stack}"
export EWM_DINO_PATH="${EWM_DINO_PATH:-$NEW_PROJECT_ROOT/pretrained/dinov3-vitl16}"
export EWM_T5_PATH="${EWM_T5_PATH:-$NEW_PROJECT_ROOT/pretrained/t5-small}"
export EWM_VAE_PATH="${EWM_VAE_PATH:-$NEW_PROJECT_ROOT/pretrained/leanvae}"
export XDG_CACHE_HOME="$NEW_PROJECT_ROOT/isaacsim601/cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export OMNI_USER_CONFIG_PATH="$NEW_PROJECT_ROOT/isaacsim601/config"
export PIP_CACHE_DIR="$NEW_PROJECT_ROOT/tmp/deployment/pip-cache"
export TMPDIR="$NEW_PROJECT_ROOT/tmp/deployment/work"
export CONDA_PKGS_DIRS="$NEW_PROJECT_ROOT/tmp/deployment/conda-pkgs"
case ":${LD_LIBRARY_PATH:-}:" in
  *":$NEW_PROJECT_ROOT/envs/torchcodec_ffmpeg7/lib:"*) ;;
  *) export LD_LIBRARY_PATH="$NEW_PROJECT_ROOT/envs/torchcodec_ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
