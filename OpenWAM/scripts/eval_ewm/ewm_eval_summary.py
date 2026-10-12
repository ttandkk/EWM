"""Human-readable, compact summary for Piper EWM simulator evaluations."""
from __future__ import annotations

from pathlib import Path
from scripts.eval_ewm.ewm_event_replan import horizon_statistics


def write_summary(output: Path, *, max_steps: int, sampling_steps: int,
                  requested_trials: int, attempted_trials: int,
                  results: list[dict], task_name: str,
                  decode_predicted_events: bool = False,
                  rgb_ood: dict | None = None,
                  error: str | None = None) -> None:
    """Write current run status; keep detailed timing in each trial result JSON."""
    successful = sum(bool(result["success"]) for result in results)
    lines = [
        f"status: {'error' if error else 'complete' if len(results) == requested_trials else 'running'}",
        f"max_step: {max_steps}",
        f"sampling_steps: {sampling_steps}",
        f"decode_predicted_events: {str(decode_predicted_events).lower()}",
        f"task_name: {task_name}",
        f"trial_count: {requested_trials}",
        f"attempted_trials: {attempted_trials}",
        f"completed_trials: {len(results)}",
        f"full_length_trials: {sum(result['steps'] >= max_steps for result in results)}",
        f"successful_trials: {successful}",
        f"success_rate: {successful / requested_trials:.2%}",
        "success_rate_denominator: requested_trials",
    ]
    if rgb_ood is not None:
        lines.extend([
            f"rgb_ood_enabled: {str(rgb_ood['enabled']).lower()}",
            f"rgb_lighting_ood: {str(rgb_ood['lighting']).lower()}",
            f"rgb_motion_blur_ood: {str(rgb_ood['motion_blur']).lower()}",
            f"rgb_ood_severity: {rgb_ood['severity']}",
            f"rgb_ood_seed: {rgb_ood['seed']}",
            f"rgb_ood_scope: {rgb_ood['scope']}",
        ])
        if "mode" in rgb_ood:
            lines.extend([
                f"rgb_ood_mode: {rgb_ood['mode']}",
                f"rgb_ood_parameters_fixed_across_trials: {str(rgb_ood['parameters_fixed_across_trials']).lower()}",
                f"rgb_ood_exposure_gain: {rgb_ood['exposure_gain']}",
                f"rgb_ood_blur_kernel_length_px: {rgb_ood['blur_kernel_length_px']}",
                f"rgb_ood_blur_angle_degrees: {rgb_ood['blur_angle_degrees']}",
            ])
    for result in results:
        lines.extend([
            "",
            f"trial_{result['trial']:03d}: "
            f"{'success' if result['success'] else 'failure'}, "
            f"steps={result['steps']}, seed={result.get('sim_seed', 'n/a')}, "
            f"video={Path(result['video']).name}",
        ])
        if "execution_horizon_statistics" in result:
            stats = result["execution_horizon_statistics"]["all_chunks"]
            lines.append(f"execution_horizon: mean={stats['mean']}, min={stats['min']}, max={stats['max']}")
    chunks = [chunk for result in results for chunk in result.get("execution_chunks", [])]
    if chunks:
        stats = horizon_statistics(chunks)
        for name in ("all_chunks", "replan_chunks"):
            for key, value in stats[name].items():
                lines.append(f"execution_horizon_{name}_{key}: {value}")
        lines.append(f"execution_chunk_end_reasons: {stats['end_reasons']}")
    inference_calls = sum(result["inference_timing"]["calls"] for result in results)
    inference_seconds = sum(result["inference_timing"]["total_seconds"] for result in results)
    lines.extend([
        "",
        f"average_inference_ms_per_call: {1000 * inference_seconds / inference_calls:.2f}"
        if inference_calls else "average_inference_ms_per_call: n/a",
    ])
    if error is not None:
        lines.extend(["", f"error: {error}"])
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
