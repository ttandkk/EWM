#!/usr/bin/env bash
set -euo pipefail
DEPLOY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$DEPLOY_ROOT/benchmark/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PIPER_BOOTSTRAP_PYTHON:-python3}" -B -m piper_bench.deployment.setup --project-root "$DEPLOY_ROOT" "$@"
