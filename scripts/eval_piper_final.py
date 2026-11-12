"""Launch the fixed eight-task evaluation profile using the existing suite runner."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "scripts/eval_profiles/piper_eight_final.json"
CONDITIONS = ("id", "lighting_ood", "blur_ood", "background_ood", "speed_ood")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--checkpoint-step", type=int, default=80000)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["id"])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Print the resolved command and profile without writing output or requesting a GPU")
    mode.add_argument("--preflight", action="store_true",
                      help="Check model/data and write scene manifests on CPU; do not request a GPU")
    mode.add_argument("--plan-only", action="store_true",
                      help="Write scene plans on CPU using the existing suite planner")
    return p


def checkpoint_mode(directory):
    # Inspect only the configuration: neither load weights nor initialize CUDA.
    from omegaconf import OmegaConf
    config = OmegaConf.load(directory / "config.yaml")
    return config.model.get("training_mode", "joint")


def build_command(args, profile, mode):
    if mode not in profile["record_predictions"]:
        raise ValueError(f"Unsupported formal evaluation training_mode: {mode}")
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("Duplicate evaluation conditions")
    if args.checkpoint_step <= 0:
        raise ValueError("checkpoint-step must be positive")
    recording = profile["record_predictions"][mode]
    command = ["bash", str(ROOT / "scripts/eval_piper_eight_dynamic.sh"),
               "--checkpoint-dir", str(args.checkpoint_dir.expanduser().resolve()),
               "--checkpoint-step", str(args.checkpoint_step),
               "--dataset-root", str(args.dataset_root.expanduser().resolve()),
               "--output-dir", str(args.output_dir.expanduser().resolve()),
               "--conditions", *args.conditions, "--tasks", *profile["tasks"]]
    for option in ("trials", "seed", "scene_seed", "sampling_steps", "latency_controller",
                   "min_execution_steps", "execution_horizon"):
        command.extend(["--" + option.replace("_", "-"), str(profile[option])])
    command.append("--latency-aware" if profile["latency_aware"] else "--no-latency-aware")
    command.append("--record-predictions" if recording else "--no-record-predictions")
    # Do not pass --decode-predicted-events: forecasts are decoded after a trial.
    if args.preflight:
        command.append("--preflight")
    elif args.plan_only:
        command.append("--plan-only")
    return command


def execute(command, environment, *, cpu_only):
    if cpu_only or environment.get("SLURM_JOB_ID") or environment.get("EWM_EVAL_DIRECT") == "1":
        return subprocess.run(command, env=environment, check=False).returncode
    checked = subprocess.run(command + ["--preflight"], env=environment, check=False)
    if checked.returncode:
        return checked.returncode
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    # Run the original shell path from --wrap. Submitting it as the batch script
    # would make BASH_SOURCE point at Slurm's spool copy instead of this project.
    submission = ["sbatch", "--partition=" + environment.get("EWM_EVAL_PARTITION", "cluster02"),
                  "--gres=" + environment.get("EWM_EVAL_GRES", "gpu:6000ada:1"),
                  "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G",
                  "--time=" + environment.get("EWM_EVAL_TIME", "08:00:00"),
                  "--job-name=piper-final-eval",
                  "--output=" + str(logs / "piper_final_eval_%j.log"),
                  "--error=" + str(logs / "piper_final_eval_%j.log"),
                  "--wrap", shlex.join(command)]
    return subprocess.run(submission, env=environment, check=False).returncode


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    profile = json.loads(PROFILE.read_text())
    try:
        mode = checkpoint_mode(args.checkpoint_dir.expanduser().resolve())
        command = build_command(args, profile, mode)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    # Protocol settings take precedence over a shell's historical environment.
    # Scheduling settings (GPU, time, partition) remain under caller control.
    environment = dict(os.environ)
    environment.update(profile["environment"])
    report = dict(profile=profile["version"], training_mode=mode, command=command,
                  environment=profile["environment"],
                  prediction_decode_phase=profile["prediction_decode_phase"]
                  if profile["record_predictions"][mode] else "disabled",
                  prediction_decode_in_equivalent_latency=profile["prediction_decode_in_equivalent_latency"])
    print(json.dumps(report, indent=2), flush=True)
    if args.dry_run:
        return 0
    return execute(command, environment, cpu_only=args.preflight or args.plan_only)


if __name__ == "__main__":
    raise SystemExit(main())
