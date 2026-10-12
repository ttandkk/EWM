#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/event-data-collection-smoke.XXXXXX")"
trap 'rm -rf "${smoke_root}"' EXIT

python "${repo_root}/tools/check_install.py" --mode sim
python "${repo_root}/scripts/collect_piper_stack_evis.py" \
    --output "${smoke_root}/stack" \
    --static_rgb_steps 4 \
    --headless \
    --enable_cameras

test -f "${smoke_root}/stack/static_rgb_probe.h5"
printf 'Simulation smoke test passed.\n'
