"""Independent RGB/state-only EWM server; event evaluation is unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

print("[ewm-rgb-server] importing numpy and torch", flush=True)
import numpy as np
import torch
print("[ewm-rgb-server] numpy and torch ready", flush=True)
from omegaconf import OmegaConf
from scripts.eval_ewm.ewm_eval_wire import encode_message
from scripts.eval_ewm.ewm_prediction_recording import PredictionArchive
from scripts.eval_ewm.ewm_eval_server import configure_inference_compile, serve_on_inference_thread
from scripts.eval_ewm.ewm_rgb_action_protocol import OBSERVATION_CONTRACT, prepare_request, prediction_array
from openwam.dataloader.ewm_normalizer import EWMNormalizer
from openwam.model.architectures.ewm import EWMModel


def load_policy(checkpoint, device="cuda"):
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != "ewm_rgb_action_latent_v1" or payload["model_config"].get("training_mode") != "rgb_action":
        raise ValueError("requires an RGB-action policy")
    if int(payload["step"]) != int(checkpoint.stem.removeprefix("checkpoint_step_")):
        raise ValueError("checkpoint filename and saved step disagree")
    model = EWMModel.from_config(payload["model_config"])
    model.load_policy_state_dict(payload["model"])
    model.to(device).eval()
    return model, EWMNormalizer(checkpoint.parent / "normalization.json"), int(payload["horizon"])


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--ready-file", required=True, type=Path)
    parser.add_argument("--compile-core", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache-context-kv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-bf16-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--benchmark-iters", type=int, default=None,
                        help="Startup comparison iterations (default: 1 with compilation, otherwise 0)")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--max-compile-action-error", type=float, default=0.02)
    parser.add_argument("--allow-compile-action-error", action=argparse.BooleanOptionalAction, default=True,
                        help="Record above-threshold compile deviations; use --no-allow-compile-action-error for strict fallback")
    parser.add_argument(
        "--warmup-prompt",
        default="Stack the blue-and-yellow patterned block on the red-and-white block, then stack the green-and-white block on top.",
    )
    args = parser.parse_args()
    if args.benchmark_iters is None:
        args.benchmark_iters = 1 if args.compile_core else 0
    if min(args.warmup_iters, args.benchmark_iters) < 0 or args.max_compile_action_error <= 0:
        parser.error("warmup/benchmark iterations must be nonnegative and max compile error positive")
    torch.set_num_threads(4)
    directory = args.checkpoint_dir.resolve()
    checkpoints = {int(p.stem.removeprefix("checkpoint_step_")): p
                   for p in directory.glob("checkpoint_step_*.pt")
                   if p.stem.removeprefix("checkpoint_step_").isdigit()}
    if not checkpoints:
        raise FileNotFoundError(f"no policy checkpoints in {directory}")
    cfg = OmegaConf.load(directory / "config.yaml")
    if cfg.model.get("training_mode") != "rgb_action":
        raise ValueError("this server accepts only rgb_action training runs")
    current_step = max(checkpoints)
    print(f"[ewm-rgb-server] loading checkpoint step {current_step}: {checkpoints[current_step]}", flush=True)
    model, normalizer, horizon = load_policy(checkpoints[current_step], device="cuda")
    print(f"[ewm-rgb-server] checkpoint loaded; horizon={horizon}", flush=True)
    if model.training_mode != "rgb_action" or cfg.model.get("training_mode") != "rgb_action":
        raise ValueError("this server accepts only rgb_action checkpoints/configs")
    if cfg.dataloader.type not in ("ewm_piper_rgb_action", "ewm_piper_multitask_rgb_action"):
        raise ValueError("RGB-action evaluation requires the Piper RGB/state training contract")
    if int(cfg.dataloader.get("observation_row_offset", 0)) != 0:
        raise ValueError("RGB-action evaluation requires training observation_row_offset=0")
    if int(cfg.dataloader.chunk_size) != horizon:
        raise ValueError("saved training config and policy horizon differ")
    if model.event_codec is None or model.event_camera_mode != "dual":
        raise ValueError("this temporary RGB evaluator requires the dual-view RGB codec")
    metadata = {
        "horizon": horizon, "training_mode": "rgb_action",
        "checkpoint_format": "ewm_rgb_action_latent_v1",
        "checkpoint_dir": str(directory),
        "config_sha256": file_hash(directory / "config.yaml"),
        "normalization_sha256": file_hash(directory / "normalization.json"),
        "checkpoint_sha256": {str(step): file_hash(path) for step, path in checkpoints.items()},
        "checkpoints": sorted(checkpoints),
        "observation_contract": OBSERVATION_CONTRACT,
        "training_observation_row_offset": 0,
        "inference_observation_row_offset": 0,
        "input_keys": ["obs_video", "obs_rgb", "proprio", "prompt"],
        "rgb_cameras": model.num_event_cameras, "decode_predicted_rgb": True,
        "prediction_wire_dtype": "uint8",
    }
    ready_file = args.ready_file.resolve()
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    if ready_file.exists():
        raise FileExistsError(f"refusing to replace readiness file: {ready_file}")

    compile_active = False
    cuda_graphs_enabled = False
    startup_benchmark = None

    def synthetic_sample(seed: int, *, cache_context_kv: bool):
        batch = {
            "obs_video": torch.zeros((1, 2, 3, 256, 256), device="cuda", dtype=torch.uint8),
            "proprio": torch.zeros((1, 8), device="cuda", dtype=torch.float32),
            "prompt": [args.warmup_prompt],
        }
        batch["obs_rgb"] = batch["obs_video"]
        generator = torch.Generator(device="cuda").manual_seed(seed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = model.sample_policy(
                batch, horizon=horizon, normalizer=normalizer, num_steps=4,
                generator=generator, decode_future=False,
                cache_context_kv=cache_context_kv)
        torch.cuda.synchronize()
        return result, time.perf_counter() - started

    def result_snapshot(result):
        return {
            "action": result["action_normalized"].detach().float().cpu(),
        }

    baseline_outputs = []
    baseline_times = []
    eager_kv_outputs = []
    eager_kv_times = []
    if args.benchmark_iters:
        print("[ewm-rgb-server] warming eager baseline without context K/V cache", flush=True)
        synthetic_sample(9100, cache_context_kv=False)
        for index in range(args.benchmark_iters):
            result, elapsed = synthetic_sample(9200 + index, cache_context_kv=False)
            baseline_outputs.append(result_snapshot(result))
            baseline_times.append(elapsed)
        print("[ewm-rgb-server] benchmarking eager inference with context K/V cache", flush=True)
        synthetic_sample(9150, cache_context_kv=True)
        for index in range(args.benchmark_iters):
            result, elapsed = synthetic_sample(9200 + index, cache_context_kv=True)
            eager_kv_outputs.append(result_snapshot(result))
            eager_kv_times.append(elapsed)

    if args.compile_core:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile-core requires torch.compile support")
        print(f"[ewm-rgb-server] enabling torch.compile mode={args.compile_mode}", flush=True)
        cuda_graphs_enabled = configure_inference_compile(
            model, mode=args.compile_mode, cache_bf16_weights=args.cache_bf16_weights,
            cuda_graphs=args.cuda_graphs)
        compile_active = True

    optimized_times = []
    optimized_outputs = []
    effective_warmup_iters = max(args.warmup_iters, int(args.compile_core))
    try:
        for index in range(effective_warmup_iters):
            _, elapsed = synthetic_sample(9300 + index, cache_context_kv=args.cache_context_kv)
            print(f"[ewm-rgb-server] optimized warmup {index + 1}/{effective_warmup_iters}: {elapsed:.3f}s", flush=True)
    except Exception as exc:
        if not compile_active:
            raise
        print(f"[ewm-rgb-server] compiled core failed; falling back to eager: {type(exc).__name__}: {exc}", flush=True)
        model.disable_inference_compile()
        compile_active = False
        torch.cuda.empty_cache()
        for index in range(max(args.warmup_iters, 1)):
            synthetic_sample(9400 + index, cache_context_kv=args.cache_context_kv)

    compiled_benchmark_active = compile_active
    if args.benchmark_iters:
        for index in range(args.benchmark_iters):
            result, elapsed = synthetic_sample(9200 + index, cache_context_kv=args.cache_context_kv)
            optimized_outputs.append(result_snapshot(result))
            optimized_times.append(elapsed)

        def error_metrics(candidate_outputs):
            action_values = torch.cat([
                (candidate["action"] - baseline["action"]).abs().flatten()
                for candidate, baseline in zip(candidate_outputs, baseline_outputs)
            ])
            return {
                "action_max_abs_error": float(action_values.max()),
                "action_mean_abs_error": float(action_values.mean()),
            }

        eager_kv_errors = error_metrics(eager_kv_outputs)
        candidate_errors = error_metrics(optimized_outputs)
        compiled_times = optimized_times if compiled_benchmark_active else []
        compiled_errors = candidate_errors if compiled_benchmark_active else None
        compile_error_exceeded = bool(
            compiled_benchmark_active
            and candidate_errors["action_max_abs_error"] > args.max_compile_action_error)
        compile_rejected_for_error = compile_error_exceeded and not args.allow_compile_action_error
        if compile_error_exceeded and args.allow_compile_action_error:
            print(f"[ewm-rgb-server] retaining compiled core: action max abs error "
                  f"{candidate_errors['action_max_abs_error']} exceeds {args.max_compile_action_error}", flush=True)
        if compile_rejected_for_error:
            print(
                "[ewm-rgb-server] rejecting compiled core and falling back to eager: "
                f"action max abs error {candidate_errors['action_max_abs_error']} exceeds "
                f"{args.max_compile_action_error}",
                flush=True)
            model.disable_inference_compile()
            compile_active = False
            torch.cuda.empty_cache()
            if args.cache_context_kv:
                optimized_times = eager_kv_times
                optimized_outputs = eager_kv_outputs
                candidate_errors = eager_kv_errors
            else:
                optimized_times = baseline_times
                optimized_outputs = baseline_outputs
                candidate_errors = error_metrics(baseline_outputs)

        startup_benchmark = {
            "iterations": args.benchmark_iters,
            "baseline_eager_no_kv_cache_mean_seconds": float(np.mean(baseline_times)),
            "baseline_eager_no_kv_cache_p50_seconds": float(np.percentile(baseline_times, 50)),
            "eager_kv_cache_mean_seconds": float(np.mean(eager_kv_times)),
            "eager_kv_cache_p50_seconds": float(np.percentile(eager_kv_times, 50)),
            "eager_kv_cache_speedup": float(np.mean(baseline_times) / np.mean(eager_kv_times)),
            **{f"eager_kv_cache_{name}": value for name, value in eager_kv_errors.items()},
            "compile_attempt_succeeded": compiled_benchmark_active,
            "compile_accepted": compile_active,
            "compile_rejected_for_error": compile_rejected_for_error,
            "compile_error_exceeded": compile_error_exceeded,
            "allow_compile_action_error": args.allow_compile_action_error,
            "max_compile_action_error": args.max_compile_action_error,
            "compiled_kv_cache_mean_seconds": (
                float(np.mean(compiled_times)) if compiled_times else None),
            "compiled_kv_cache_p50_seconds": (
                float(np.percentile(compiled_times, 50)) if compiled_times else None),
            "compiled_kv_cache_speedup": (
                float(np.mean(baseline_times) / np.mean(compiled_times)) if compiled_times else None),
            **({f"compiled_kv_cache_{name}": value for name, value in compiled_errors.items()}
               if compiled_errors is not None else {}),
            "optimized_mean_seconds": float(np.mean(optimized_times)),
            "optimized_p50_seconds": float(np.percentile(optimized_times, 50)),
            "speedup": float(np.mean(baseline_times) / np.mean(optimized_times)),
            **candidate_errors,
        }
        print(f"[ewm-rgb-server] startup benchmark: {json.dumps(startup_benchmark)}", flush=True)

    metadata.update({
        "compile_core_requested": args.compile_core,
        "compile_core_active": compile_active,
        "compile_mode": args.compile_mode if compile_active else None,
        "cuda_graphs_enabled": compile_active and cuda_graphs_enabled,
        "cache_context_kv": args.cache_context_kv,
        "cache_bf16_weights": bool(getattr(model, "_inference_bf16_weights", False)),
        "allow_compile_action_error": args.allow_compile_action_error,
        "max_compile_action_error": args.max_compile_action_error,
        "warmup_iters": effective_warmup_iters,
        "startup_benchmark": startup_benchmark,
    })

    archive = PredictionArchive()
    metadata["post_trial_visualization_supported"] = True

    def process_request(request, arrays):
        nonlocal current_step
        with torch.inference_mode():
            operation = request.get("op")
            if operation == "hello":
                yield encode_message({"ok": True, **metadata})
            elif operation in ("visualization_begin", "visualization_decode", "visualization_end"):
                reply, values = archive.handle(request, model)
                yield encode_message(reply, values)
            elif operation == "predict":
                batch, seed, steps, action_chunk_size = prepare_request(
                    request, arrays, normalizer, horizon=horizon, device="cuda")
                step = int(request["step"])
                if step not in checkpoints:
                    raise ValueError(f"checkpoint step {step} unavailable")
                if step != current_step:
                    payload = torch.load(checkpoints[step], map_location="cpu", weights_only=True)
                    if payload.get("format") != "ewm_rgb_action_latent_v1" or payload["model_config"].get("training_mode") != "rgb_action":
                        raise ValueError("checkpoint is not an RGB-action policy")
                    if (int(payload["horizon"]) != horizon
                            or payload["model_config"].get("event_horizon") != model.event_horizon
                            or payload["model_config"].get("event_temporal_downsample", 1) != model.event_temporal_downsample):
                        raise ValueError("checkpoint prediction horizons or RGB sampling differ")
                    if int(payload["step"]) != step:
                        raise ValueError("checkpoint step disagrees with filename")
                    model.load_policy_state_dict(payload["model"])
                    if compile_active and getattr(model, "_inference_velocity", None) is None:
                        configure_inference_compile(model, mode=args.compile_mode,
                            cache_bf16_weights=args.cache_bf16_weights, cuda_graphs=args.cuda_graphs)
                    del payload
                    current_step = step
                generator = torch.Generator(device="cuda").manual_seed(seed)
                recording = request.get("record_prediction") is not None
                if recording and request.get("decode_future_rgb", True):
                    raise ValueError("prediction recording requires post-trial decoding")
                torch.cuda.synchronize()
                inference_started = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model.sample_policy(batch, horizon=horizon,
                                                 normalizer=normalizer, num_steps=steps,
                                                 generator=generator,
                                                 decode_future=request.get("decode_future_rgb", True),
                                                 cache_context_kv=args.cache_context_kv,
                                                 return_observed_latent=recording)
                torch.cuda.synchronize()
                inference_seconds = time.perf_counter() - inference_started
                recorded = archive.record(request, result)
                executed_actions = result["action"][0, :action_chunk_size].float().cpu().numpy()
                if executed_actions.shape != (action_chunk_size, 7) or not np.isfinite(executed_actions).all():
                    raise FloatingPointError("invalid physical action chunk")
                arrays_out = {"executed_actions": executed_actions}
                if request.get("decode_future_rgb", True):
                    arrays_out["future_rgb"] = prediction_array(result, horizon, model.num_event_cameras)
                if action_chunk_size == 1:
                    arrays_out["executed_action"] = executed_actions[0]
                yield encode_message({
                    "ok": True, "step": step, "seed": seed,
                    "sampling_steps": steps, "action_chunk_size": action_chunk_size,
                    "training_mode": "rgb_action", "future_decoded": request.get("decode_future_rgb", True),
                    "prediction_recorded": recorded,
                    "inference_seconds": inference_seconds,
                }, arrays_out)
            else:
                raise ValueError(f"unknown operation: {operation}")

    serve_on_inference_thread(process_request, metadata, ready_file)


if __name__ == "__main__":
    main()
