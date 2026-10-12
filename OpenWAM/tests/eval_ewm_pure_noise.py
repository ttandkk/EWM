"""Compatibility entry point for the EWM pure-noise evaluator."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_ewm.eval_ewm_pure_noise import main

if __name__ == "__main__":
    main()
