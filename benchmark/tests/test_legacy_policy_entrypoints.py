"""Old evaluator names share the package implementation and current protocol."""
import importlib
from pathlib import Path
import subprocess
import sys


def test_old_modules_alias_one_runtime(monkeypatch):
    root=Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "OpenWAM"))
    from piper_bench.evaluation import runtime, runner
    old=importlib.import_module("scripts.eval_ewm.eval_ewm_piper_conveyor_sim")
    tasks=importlib.import_module("scripts.eval_ewm.eval_ewm_piper_tasks_sim")
    assert old is runtime and tasks is runner
    args=old.build_parser().parse_args(["--plan-only"])
    old.load_collection(args)
    assert args.task=="conveyor" and args.latency_controller=="continuous" and args.min_execution_steps==8
    args=old.build_parser().parse_args(["--plan-only","--execution-profile","legacy_buffered_min4_v1"])
    old.load_collection(args)
    assert args.latency_controller=="buffered" and args.min_execution_steps==4


def test_old_cli_help_never_imports_isaac_or_models():
    root=Path(__file__).resolve().parents[2]
    for name in ("eval_ewm_piper_conveyor_sim.py","eval_ewm_piper_tasks_sim.py"):
        result=subprocess.run([sys.executable,str(root/"OpenWAM/scripts/eval_ewm"/name),"--help"],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        assert "--policy-url" in result.stdout and "--execution-profile" in result.stdout
