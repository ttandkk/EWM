#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

if [[ ! -f "${repo_root}/third_party/lerobot/pyproject.toml" ]]; then
    printf 'LeRobot submodule is missing. Run: git submodule update --init third_party/lerobot\n' >&2
    exit 1
fi

"${python_bin}" - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit(f"LeRobot conversion requires Python >=3.12; got {sys.version.split()[0]}")
PY

"${python_bin}" -m pip install --upgrade pip
"${python_bin}" -m pip install -e "${repo_root}/third_party/lerobot[dataset]"
"${python_bin}" -m pip install h5py
"${python_bin}" "${repo_root}/tools/check_install.py" --mode convert
