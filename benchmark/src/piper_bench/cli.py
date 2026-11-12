"""Public command line; model, simulation and writer processes stay isolated."""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys

COMMANDS = {
    "setup": "Install a pinned runtime in a new prefix",
    "assets": "Download or verify versioned simulation assets",
    "doctor": "Check dependencies, render cameras and run EVIS/Oracle",
    "demo": "Run an Oracle task and save a short recording",
    "collect": "Collect and export reproducible demonstrations",
    "evaluate": "Evaluate a task (--task) or a suite (--tasks)",
    "validate": "Validate an exported demonstration dataset",
    "replay": "Rejudge saved terminal states or replay fixed actions",
    "serve": "Start a policy in its configured model environment",
}

# These task entrypoints share the --mode/--trials RGB+event preview interface.
DEMO_TASKS = ('conveyor', 'push_t', 'drawer_cup', 'square_nut', 'tennis_rolling',
              'tennis_falling', 'tennis_parabolic', 'tennis_container')


def _simulation(command, argv):
    from .deployment.config import runtime_environment, simulation_python
    python = simulation_python()
    environment = runtime_environment()
    # Preserve the virtualenv executable spelling: resolving a symlink may
    # bypass its pyvenv.cfg and silently use the system environment instead.
    if os.path.abspath(python) == os.path.abspath(sys.executable):
        os.environ.update(environment)
        return None
    from .deployment.process import package_environment
    with package_environment(environment) as environment:
        return subprocess.run([str(python), "-B", "-m", "piper_bench", command, *argv],
                              env=environment, check=False).returncode


def demo(argv):
    from .deployment.config import runtime_environment, simulation_python
    from .deployment.doctor import preview_command
    parser = argparse.ArgumentParser(description="Run one Oracle trial and save RGB/event evidence",
        epilog="For acceptance checks on all tasks, use ewm collect --task TASK --output DIR --acceptance-only.")
    parser.add_argument("--task", choices=DEMO_TASKS, default="tennis_rolling")
    parser.add_argument("--oracle", action="store_true", help="Use the task's scripted Oracle (default)")
    parser.add_argument("--events", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("demo_output"))
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args(argv)
    environment = runtime_environment()
    command = preview_command(simulation_python(), args.output_dir.resolve(), task=args.task,
                              events=args.events, trials=1, max_steps=args.max_steps, audit=False)
    from .deployment.process import package_environment
    with package_environment(environment) as environment:
        return subprocess.run(command, env=environment, check=False).returncode


def serve(argv):
    from .deployment.config import load_config
    from .deployment.process import package_environment
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--family", choices=("ewm", "openwam", "fastwam"), default="ewm")
    args, _ = parser.parse_known_args(argv)
    profile = "policy-fastwam" if args.family == "fastwam" else "policy-ewm"
    value = (os.environ.get("PIPER_POLICY_PYTHON") or
             load_config().get("profiles", {}).get(profile, {}).get("python") or sys.executable)
    python = Path(value).expanduser().absolute()
    if not python.is_file():
        raise FileNotFoundError(f"policy Python not found: {python}")
    if os.path.abspath(python) == os.path.abspath(sys.executable):
        from piper_policies.cli import main as policy_main
        return policy_main(["serve", *argv])
    with package_environment(include_policies=True) as environment:
        return subprocess.run([str(python), "-B", "-m", "piper_policies", "serve", *argv],
                              env=environment, check=False).returncode


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, epilog="Run COMMAND --help for its options.")
    parser.add_argument("command", choices=COMMANDS, nargs="?", help="; ".join(f"{k}: {v}" for k,v in COMMANDS.items()))
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    command = parser.parse_args(argv[:1]).command
    rest = argv[1:]
    help_only = "--help" in rest or "-h" in rest
    cpu_only = help_only or "--plan-only" in rest or "--preflight" in rest or "--rejudge-only" in rest
    routes = dict(setup="deployment.setup", assets="assets.resolver", doctor="deployment.doctor",
                  collect="collection.runner", validate="collection.validation", replay="evaluation.replay")
    try:
        if command in ("collect", "evaluate", "replay") and not cpu_only:
            result = _simulation(command, rest)
            if result is not None:
                return result
        if command == "validate" and not help_only:
            from .collection.runner import resolve_writer_python
            writer = resolve_writer_python()
            if os.path.abspath(writer) != os.path.abspath(sys.executable):
                from .deployment.process import package_environment
                with package_environment() as environment:
                    return subprocess.run([str(writer), "-B", "-m", "piper_bench", command, *rest],
                                          env=environment, check=False).returncode
        if command == "demo":
            return demo(rest)
        if command == "serve":
            return serve(rest)
        if command == "evaluate":
            target = "evaluation.runner" if any(a == "--task" or a.startswith("--task=") for a in rest) else "evaluation.suite"
        else:
            target = routes[command]
        return importlib.import_module("piper_bench." + target).main(rest)
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as exc:
        parser.exit(2, f"ewm {command}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
