"""Optional, nonblocking Slurm submission for EWM pure-noise rollout checks."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def eval_due(step: int, max_steps: int, options) -> bool:
    """Evaluate at the configured interval and once at the final checkpoint."""
    return bool(options.enabled) and (
        step % int(options.every_steps) == 0 or step == max_steps
    )


def submit_pure_noise_eval(output: Path, step: int, options) -> str | None:
    """Queue one immutable checkpoint; submission failure never stops training."""
    output = output.absolute()
    checkpoint = output / f"checkpoint_step_{step:06d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"pure-noise eval requires saved checkpoint {checkpoint}")
    submission_dir = output / "pure_noise_eval" / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)
    record_path = submission_dir / f"step_{step:06d}.json"
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous.get("status") == "submitted":
            logger.info("Pure-noise evaluation already submitted: %s", record_path)
            return previous["job_id"]

    project_root = Path("/projects/hdd/myproject/CVPR2027")
    script = project_root / "OpenWAM/scripts/eval_ewm/eval_ewm_pure_noise.sbatch"
    result_dir = output / "pure_noise_eval" / f"step_{step:06d}"
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9_-]", "_", output.name)
    log_path = log_dir / f"ewm_pure_noise_{tag}_step{step}_%j.log"
    command = [
        "sbatch", "--parsable",
        f"--partition={options.partition}",
        f"--gres={options.gres}",
        f"--time={options.time}",
        f"--output={log_path}", f"--error={log_path}",
        str(script), str(output), str(step), str(result_dir),
    ]
    record = {
        "step": step, "checkpoint": str(checkpoint),
        "result_dir": str(result_dir), "log_pattern": str(log_path),
        "command": command,
    }
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=30)
        if result.returncode:
            record.update(status="submission_failed", error=result.stderr.strip())
            logger.warning("Pure-noise eval submission failed at step %d: %s",
                           step, result.stderr.strip())
        else:
            job_id = result.stdout.strip().split(";", 1)[0]
            record.update(status="submitted", job_id=job_id)
            logger.info("Pure-noise eval step %d submitted as Slurm job %s", step, job_id)
    except (OSError, subprocess.TimeoutExpired) as exc:
        record.update(status="submission_failed", error=str(exc))
        logger.warning("Pure-noise eval submission failed at step %d: %s", step, exc)
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return record.get("job_id")
