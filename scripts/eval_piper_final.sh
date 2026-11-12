#!/usr/bin/env bash
# Formal settings are fixed by eval_profiles/piper_eight_final.json.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$project_root/deployment/runtime_env.sh"
exec "$OWPY" -B "$project_root/scripts/eval_piper_final.py" "$@"
