#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys


def require_modules(names: tuple[str, ...]) -> None:
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit("Missing Python modules: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sim", "convert"), required=True)
    args = parser.parse_args()

    if args.mode == "sim":
        if sys.version_info[:2] != (3, 12):
            raise SystemExit("Simulation checks require Python 3.12")
        require_modules(("isaacsim", "isaaclab", "isaaclab_tasks", "agx_teleop", "dvs_gen", "h5py"))
    else:
        if sys.version_info < (3, 12):
            raise SystemExit("Conversion checks require Python >=3.12")
        require_modules(("lerobot", "h5py", "numpy"))

    print(f"{args.mode} environment check passed with Python {sys.version.split()[0]}")


if __name__ == "__main__":
    main()
