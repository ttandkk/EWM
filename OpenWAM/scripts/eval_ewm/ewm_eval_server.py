"""GPU EWM inference server using OpenWAM's WebSocket evaluation pattern."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = next(root / "OpenWAM" for root in Path(__file__).resolve().parents
            if (root / "OpenWAM/openwam/model/architectures/ewm.py").is_file())
sys.path.insert(0, str(ROOT))

print("[ewm-server] importing numpy and torch", flush=True)
import numpy as np
import torch
print("[ewm-server] numpy and torch ready", flush=True)
from omegaconf import OmegaConf
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from scripts.eval_ewm.ewm_eval_wire import MAX_MESSAGE, decode_message, encode_message
from scripts.eval_ewm.ewm_eval_time_audit import EVENT_TIME_CONTRACT
from scripts.eval_ewm.eval_ewm_policy import load_policy, validate_observation_timing


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_array(arrays: dict, name: str, shape: tuple, dtype: str) -> torch.Tensor:
    value = arrays.get(name)
    if value is None or value.shape != shape or value.dtype.name != dtype:
        raise ValueError(f"{name} must be {dtype} {shape}")
    if dtype == "float32" and not np.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")
    return torch.from_numpy(value).cuda(non_blocking=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--ready-file", required=True, type=Path)
    parser.add_argument("--compile-core", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache-bf16-weights", action=argparse.BooleanOptionalAction, default=True,
                        help="With --compile-core, cache BF16 core Linear weights for inference")
    parser.add_argument("--cache-context-kv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--benchmark-iters", type=int, default=0)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--max-compile-action-error", type=float, default=0.02)
    parser.add_argument(
        "--warmup-prompt",
        default="Stack the blue-and-yellow patterned block on the red-and-white block, then stack the green-and-white block on top.",
    )
    args = parser.parse_args()
    if min(args.warmup_iters, args.benchmark_iters) < 0 or args.max_compile_action_error <= 0:
        parser.error("warmup/benchmark iterations must be nonnegative and max compile error positive")
    torch.set_num_threads(4)
    directory = args.checkpoint_dir.resolve()
    checkpoints = {int(p.stem.removeprefix("checkpoint_step_")): p
                   for p in directory.glob("checkpoint_step_*.pt")
                   if p.stem.removeprefix("checkpoint_step_").isdigit()}
    if not checkpoints:
        raise FileNotFoundError(f"no policy checkpoints in {directory}")
    current_step = max(checkpoints)
    print(f"[ewm-server] loading checkpoint step {current_step}: {checkpoints[current_step]}", flush=True)
    model, normalizer, horizon = load_policy(checkpoints[current_step], device="cuda")
    print(f"[ewm-server] checkpoint loaded; horizon={horizon}", flush=True)
    cfg = OmegaConf.load(directory / "config.yaml")
    if int(cfg.dataloader.get("observation_row_offset", 0)) != 0:
        raise ValueError("aligned event-frame evaluation requires training observation_row_offset=0")
    if int(cfg.dataloader.chunk_size) != horizon or int(cfg.model.voxel_bins) != model.voxel_bins:
        raise ValueError("saved training config and policy horizon/voxel bins differ")
    if cfg.dataloader.event_camera_mode != model.event_camera_mode or cfg.model.event_camera_mode != model.event_camera_mode:
        raise ValueError("saved training config and policy event camera mode differ")
    if int(cfg.dataloader.count_clip) != normalizer.count_clip or int(cfg.dataloader.voxel_bins) != normalizer.voxel_bins:
        raise ValueError("saved training config and normalization differ")
    camera_count = model.num_event_cameras
    metadata = {
        "horizon": horizon, "event_cameras": camera_count,
        "event_camera_mode": model.event_camera_mode,
        "voxel_bins": model.voxel_bins, "count_clip": normalizer.count_clip,
        "config_sha256": file_hash(directory / "config.yaml"),
        "normalization_sha256": file_hash(directory / "normalization.json"),
        "checkpoints": sorted(checkpoints),
        "event_time_contract": EVENT_TIME_CONTRACT,
        "training_observation_row_offset": int(cfg.dataloader.get("observation_row_offset", 0)),
        "inference_observation_row_offset": 0,
        "deferred_visualization_supported": True,
        "memory_history_required": getattr(model, "checkpoint_format", "") == "ewm_rgb_event_memory_latent_v1",
    }
    ready_file = args.ready_file.resolve()
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    if ready_file.exists():
        raise FileExistsError(f"refusing to replace readiness file: {ready_file}")

    compile_active = False
    startup_benchmark = None

    def synthetic_sample(seed: int, *, cache_context_kv: bool):
        batch = {
            "obs_video": torch.zeros((1, 2, 3, 256, 256), device="cuda", dtype=torch.uint8),
            "obs_voxel": torch.zeros(
                (1, camera_count, 2 * model.voxel_bins, 256, 256),
                device="cuda", dtype=torch.float32),
            "proprio": torch.zeros((1, 8), device="cuda", dtype=torch.float32),
            "prompt": [args.warmup_prompt],
        }
        if getattr(model, "checkpoint_format", "") == "ewm_rgb_event_memory_latent_v1":
            batch.update(memory_rgb=torch.zeros((1,3,2,3,256,256),device="cuda",dtype=torch.uint8),
                memory_event=batch["obs_voxel"][:,None].expand(-1,10,-1,-1,-1,-1),
                memory_rgb_valid=torch.tensor([[False,False,False,True]],device="cuda"),
                memory_event_valid=torch.tensor([[False]*9+[True]],device="cuda"))
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
            "latent": result["future_latent"].detach().float().cpu(),
        }

    baseline_outputs = []
    baseline_times = []
    eager_kv_outputs = []
    eager_kv_times = []
    if args.benchmark_iters:
        print("[ewm-server] warming eager baseline without context K/V cache", flush=True)
        synthetic_sample(9100, cache_context_kv=False)
        for index in range(args.benchmark_iters):
            result, elapsed = synthetic_sample(9200 + index, cache_context_kv=False)
            baseline_outputs.append(result_snapshot(result))
            baseline_times.append(elapsed)
        print("[ewm-server] benchmarking eager inference with context K/V cache", flush=True)
        synthetic_sample(9150, cache_context_kv=True)
        for index in range(args.benchmark_iters):
            result, elapsed = synthetic_sample(9200 + index, cache_context_kv=True)
            eager_kv_outputs.append(result_snapshot(result))
            eager_kv_times.append(elapsed)

    if args.compile_core:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile-core requires torch.compile support")
        print(f"[ewm-server] enabling torch.compile mode={args.compile_mode} "
              f"cache_bf16_weights={args.cache_bf16_weights}", flush=True)
        model.compile_inference(mode=args.compile_mode, cache_bf16_weights=args.cache_bf16_weights)
        compile_active = True

    optimized_times = []
    optimized_outputs = []
    effective_warmup_iters = max(args.warmup_iters, int(args.compile_core))
    try:
        for index in range(effective_warmup_iters):
            _, elapsed = synthetic_sample(9300 + index, cache_context_kv=args.cache_context_kv)
            print(f"[ewm-server] optimized warmup {index + 1}/{effective_warmup_iters}: {elapsed:.3f}s", flush=True)
    except Exception as exc:
        if not compile_active:
            raise
        print(f"[ewm-server] compiled core failed; falling back to eager: {type(exc).__name__}: {exc}", flush=True)
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
            latent_values = torch.cat([
                (candidate["latent"] - baseline["latent"]).abs().flatten()
                for candidate, baseline in zip(candidate_outputs, baseline_outputs)
            ])
            return {
                "action_max_abs_error": float(action_values.max()),
                "action_mean_abs_error": float(action_values.mean()),
                "latent_max_abs_error": float(latent_values.max()),
                "latent_mean_abs_error": float(latent_values.mean()),
            }

        eager_kv_errors = error_metrics(eager_kv_outputs)
        candidate_errors = error_metrics(optimized_outputs)
        compiled_times = optimized_times if compiled_benchmark_active else []
        compiled_errors = candidate_errors if compiled_benchmark_active else None
        compile_rejected_for_error = bool(
            compiled_benchmark_active
            and candidate_errors["action_max_abs_error"] > args.max_compile_action_error)
        if compile_rejected_for_error:
            print(
                "[ewm-server] rejecting compiled core and falling back to eager: "
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
        print(f"[ewm-server] startup benchmark: {json.dumps(startup_benchmark)}", flush=True)

    metadata.update({
        "compile_core_requested": args.compile_core,
        "compile_core_active": compile_active,
        "compile_mode": args.compile_mode if compile_active else None,
        "cache_context_kv": args.cache_context_kv,
        "warmup_iters": effective_warmup_iters,
        "startup_benchmark": startup_benchmark,
    })

    def handler(conn) -> None:
        nonlocal current_step
        with torch.inference_mode():
            while True:
                try:
                    request, arrays = decode_message(conn.recv(timeout=600))
                except ConnectionClosed:
                    break
                try:
                    operation = request.get("op")
                    if operation == "hello":
                        conn.send(encode_message({"ok": True, **metadata}))
                    elif operation == "shutdown":
                        conn.send(encode_message({"ok": True}))
                        server.shutdown()
                        break
                    elif operation == "predict":
                        if request.get("event_time_contract") != EVENT_TIME_CONTRACT:
                            raise ValueError("missing or incompatible event time contract")
                        if request.get("observation_row_offset") != 0:
                            raise ValueError("inference requires RGB/state aligned to the observed event interval end")
                        step = int(request["step"])
                        if step not in checkpoints:
                            raise ValueError(f"checkpoint step {step} unavailable")
                        if step != current_step:
                            payload = torch.load(checkpoints[step], map_location="cpu", weights_only=True)
                            if int(payload["step"]) != step:
                                raise ValueError("checkpoint step disagrees with filename")
                            model.load_policy_state_dict(payload["model"])
                            if compile_active and getattr(model, "_inference_velocity", None) is None:
                                model.compile_inference(
                                    mode=args.compile_mode, cache_bf16_weights=args.cache_bf16_weights)
                            del payload
                            current_step = step
                        rgb = require_array(arrays, "obs_video", (2, 3, 256, 256), "uint8")
                        voxel = require_array(arrays, "obs_voxel", (camera_count, 2 * model.voxel_bins, 256, 256), "float32")
                        proprio = require_array(arrays, "proprio", (8,), "float32")
                        raw_observation = request.get("raw_observation") is True
                        if raw_observation:
                            validate_observation_timing(request.get("timing", {}))
                            voxel = normalizer.normalize_voxel(voxel)
                            proprio = normalizer.normalize_proprio(proprio)
                        seed = int(request["seed"])
                        steps = int(request["sampling_steps"])
                        action_chunk_size = int(request.get("action_chunk_size", 1))
                        if steps <= 0 or not 0 <= seed < 2**63 or not 1 <= action_chunk_size <= horizon:
                            raise ValueError("invalid seed, sampling_steps or action_chunk_size")
                        prompt = request["prompt"]
                        if not isinstance(prompt, str) or not prompt:
                            raise ValueError("prompt must be a nonempty string")
                        batch = {"obs_video": rgb[None], "obs_voxel": voxel[None],
                                 "proprio": proprio[None], "prompt": [prompt]}
                        if getattr(model, "checkpoint_format", "") == "ewm_rgb_event_memory_latent_v1":
                            event_history = require_array(arrays, "memory_event", (10,camera_count,2,256,256), "float32")
                            if raw_observation: event_history = normalizer.normalize_voxel(event_history)
                            batch.update(memory_event=event_history[None],
                                memory_rgb=require_array(arrays,"memory_rgb",(3,2,3,256,256),"uint8")[None],
                                memory_rgb_valid=require_array(arrays,"memory_rgb_valid",(4,),"bool")[None],
                                memory_event_valid=require_array(arrays,"memory_event_valid",(10,),"bool")[None])
                        generator = torch.Generator(device="cuda").manual_seed(seed)
                        torch.cuda.synchronize()
                        inference_started = time.perf_counter()
                        decode_future_events = request.get("decode_future_events", False)
                        if not isinstance(decode_future_events, bool):
                            raise ValueError("decode_future_events must be a boolean")
                        defer_visualization = request.get("defer_visualization", False)
                        if not isinstance(defer_visualization, bool) or (defer_visualization and not raw_observation):
                            raise ValueError("defer_visualization requires a raw-observation request")
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            result = model.sample_policy(batch, horizon=horizon,
                                                         normalizer=normalizer, num_steps=steps,
                                                         generator=generator,
                                                         decode_future=not raw_observation or (decode_future_events and not defer_visualization),
                                                         cache_context_kv=args.cache_context_kv)
                        torch.cuda.synchronize()
                        inference_seconds = time.perf_counter() - inference_started
                        if raw_observation:
                            executed_actions = result["action"][0, :action_chunk_size].float().cpu().numpy()
                            if not np.isfinite(executed_actions).all():
                                raise FloatingPointError("nonfinite action")
                            arrays_out = {"executed_actions": executed_actions}
                            if decode_future_events and not defer_visualization:
                                prediction = result["future_voxel"][0, :action_chunk_size].float().cpu().numpy()
                                if not np.isfinite(prediction).all():
                                    raise FloatingPointError("nonfinite decoded future events")
                                arrays_out["future_voxel"] = prediction
                            # Preserve the old one-action wire response for
                            # clients that omit action_chunk_size.
                            if action_chunk_size == 1:
                                arrays_out["executed_action"] = executed_actions[0]
                            conn.send(encode_message({
                                "ok": True, "step": step, "seed": seed,
                                "sampling_steps": steps,
                                "action_chunk_size": action_chunk_size,
                                "future_decoded": decode_future_events and not defer_visualization,
                                "inference_seconds": inference_seconds,
                                "visualization_pending": decode_future_events and defer_visualization,
                            }, arrays_out))
                            # The action reply is sent before diagnostics. The
                            # client's control clock stops at that first reply.
                            if decode_future_events and defer_visualization:
                                with torch.autocast("cuda", dtype=torch.bfloat16):
                                    observed = model.event_codec.encode(batch["obs_voxel"][:, None])[:, 0]
                                    prediction = model._decode_future(observed, result["future_latent"])
                                prediction = prediction[0, :action_chunk_size].float().cpu().numpy()
                                if not np.isfinite(prediction).all():
                                    raise FloatingPointError("nonfinite diagnostic future events")
                                conn.send(encode_message({"ok": True, "op": "predict_visualization",
                                                          "step": step, "seed": seed, "future_decoded": True},
                                                         {"future_voxel": prediction}))
                        else:
                            prediction = result["future_voxel"][0].float().cpu().numpy()
                            action = result["action_normalized"][0].float().cpu().numpy()
                            if not np.isfinite(prediction).all() or not np.isfinite(action).all():
                                raise FloatingPointError("nonfinite rollout")
                            conn.send(encode_message({"ok": True, "step": step, "seed": seed},
                                                     {"future_voxel": prediction, "action_normalized": action}))
                    elif operation == "clean_codec":
                        obs = require_array(arrays, "obs_voxel", (camera_count, 2 * model.voxel_bins, 256, 256), "float32")
                        future = require_array(arrays, "future_voxel", (horizon, camera_count, 2 * model.voxel_bins, 256, 256), "float32")
                        full = torch.cat((obs[None, None], future[None]), 1)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            decoded = model.event_codec.decode(model.event_codec.encode(full))[0, 1:]
                        conn.send(encode_message({"ok": True}, {"future_voxel": decoded.float().cpu().numpy()}))
                    else:
                        raise ValueError(f"unknown operation: {operation}")
                except Exception as exc:
                    traceback.print_exc()
                    conn.send(encode_message({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))

    with serve(handler, "127.0.0.1", 0, max_size=MAX_MESSAGE,
               compression=None, ping_interval=None) as server:
        port = server.socket.getsockname()[1]
        ready_file.write_text(json.dumps({"url": f"ws://127.0.0.1:{port}", **metadata}) + "\n")
        print(f"EWM eval server ready: ws://127.0.0.1:{port}", flush=True)
        try:
            server.serve_forever()
        finally:
            ready_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
