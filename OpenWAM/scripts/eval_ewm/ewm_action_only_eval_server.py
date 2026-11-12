"""Independent RGB/state-only EWM server; event evaluation is unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

print("[ewm-action-server] importing numpy and torch", flush=True)
import numpy as np
import torch
from openwam.model.architectures.ewm_checkpoint import normalize_policy_payload
print("[ewm-action-server] numpy and torch ready", flush=True)
from omegaconf import OmegaConf
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from scripts.eval_ewm.ewm_eval_wire import MAX_MESSAGE, decode_message, encode_message
from scripts.eval_ewm.ewm_action_only_protocol import OBSERVATION_CONTRACT, prepare_request
from scripts.eval_ewm.eval_ewm_policy import load_policy


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


from openwam.model.architectures.ewm_inference import configure_action_compile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--ready-file", required=True, type=Path)
    parser.add_argument("--compile-core", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache-context-kv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--benchmark-iters", type=int, default=None,
                        help="Startup comparison iterations (default: 1 with compilation, otherwise 0)")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
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
    if cfg.model.get("training_mode") != "action_only":
        raise ValueError("this server accepts only action_only training runs")
    current_step = max(checkpoints)
    print(f"[ewm-action-server] loading checkpoint step {current_step}: {checkpoints[current_step]}", flush=True)
    model, normalizer, horizon = load_policy(checkpoints[current_step], device="cuda")
    print(f"[ewm-action-server] checkpoint loaded; horizon={horizon}", flush=True)
    if model.training_mode != "action_only" or cfg.model.get("training_mode") != "action_only":
        raise ValueError("this server accepts only action_only checkpoints/configs")
    if cfg.dataloader.type not in ("ewm_piper_action_only", "ewm_piper_multitask_action_only"):
        raise ValueError("action-only evaluation requires the Piper RGB/state training contract")
    if int(cfg.dataloader.get("observation_row_offset", 0)) != 0:
        raise ValueError("action-only evaluation requires training observation_row_offset=0")
    if int(cfg.dataloader.chunk_size) != horizon:
        raise ValueError("saved training config and policy horizon differ")
    if model.event_codec is not None:
        raise ValueError("action-only policy unexpectedly instantiated an event codec")
    metadata = {
        "horizon": horizon, "training_mode": "action_only",
        "checkpoint_format": "ewm_action_only_v2",
        "checkpoint_dir": str(directory),
        "config_sha256": file_hash(directory / "config.yaml"),
        "normalization_sha256": file_hash(directory / "normalization.json"),
        "checkpoint_sha256": {str(step): file_hash(path) for step, path in checkpoints.items()},
        "checkpoints": sorted(checkpoints),
        "observation_contract": OBSERVATION_CONTRACT,
        "training_observation_row_offset": 0,
        "inference_observation_row_offset": 0,
        "input_keys": ["obs_video", "proprio", "prompt"],
    }
    ready_file = args.ready_file.resolve()
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    if ready_file.exists():
        raise FileExistsError(f"refusing to replace readiness file: {ready_file}")

    eager_predict_velocity = model.predict_velocity
    compile_active = False
    cuda_graphs_enabled = False
    startup_benchmark = None

    def synthetic_sample(seed: int, *, cache_context_kv: bool):
        batch = {
            "obs_video": torch.zeros((1, 2, 3, 256, 256), device="cuda", dtype=torch.uint8),
            "proprio": torch.zeros((1, 8), device="cuda", dtype=torch.float32),
            "prompt": [args.warmup_prompt],
        }
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
        print("[ewm-action-server] warming eager baseline without context K/V cache", flush=True)
        synthetic_sample(9100, cache_context_kv=False)
        for index in range(args.benchmark_iters):
            result, elapsed = synthetic_sample(9200 + index, cache_context_kv=False)
            baseline_outputs.append(result_snapshot(result))
            baseline_times.append(elapsed)
        print("[ewm-action-server] benchmarking eager inference with context K/V cache", flush=True)
        synthetic_sample(9150, cache_context_kv=True)
        for index in range(args.benchmark_iters):
            result, elapsed = synthetic_sample(9200 + index, cache_context_kv=True)
            eager_kv_outputs.append(result_snapshot(result))
            eager_kv_times.append(elapsed)

    if args.compile_core:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile-core requires torch.compile support")
        print(f"[ewm-action-server] enabling torch.compile mode={args.compile_mode}", flush=True)
        cuda_graphs_enabled = configure_action_compile(
            model, mode=args.compile_mode, cuda_graphs=args.cuda_graphs)
        compile_active = True

    optimized_times = []
    optimized_outputs = []
    effective_warmup_iters = max(args.warmup_iters, int(args.compile_core))
    try:
        for index in range(effective_warmup_iters):
            _, elapsed = synthetic_sample(9300 + index, cache_context_kv=args.cache_context_kv)
            print(f"[ewm-action-server] optimized warmup {index + 1}/{effective_warmup_iters}: {elapsed:.3f}s", flush=True)
    except Exception as exc:
        if not compile_active:
            raise
        print(f"[ewm-action-server] compiled core failed; falling back to eager: {type(exc).__name__}: {exc}", flush=True)
        model.predict_velocity = eager_predict_velocity
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
            print(f"[ewm-action-server] retaining compiled core: action max abs error "
                  f"{candidate_errors['action_max_abs_error']} exceeds {args.max_compile_action_error}", flush=True)
        if compile_rejected_for_error:
            print(
                "[ewm-action-server] rejecting compiled core and falling back to eager: "
                f"action max abs error {candidate_errors['action_max_abs_error']} exceeds "
                f"{args.max_compile_action_error}",
                flush=True)
            model.predict_velocity = eager_predict_velocity
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
        print(f"[ewm-action-server] startup benchmark: {json.dumps(startup_benchmark)}", flush=True)

    metadata.update({
        "compile_core_requested": args.compile_core,
        "compile_core_active": compile_active,
        "compile_mode": args.compile_mode if compile_active else None,
        "cuda_graphs_enabled": cuda_graphs_enabled if compile_active else False,
        "cache_context_kv": args.cache_context_kv,
        "allow_compile_action_error": args.allow_compile_action_error,
        "max_compile_action_error": args.max_compile_action_error,
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
                        batch, seed, steps, action_chunk_size = prepare_request(
                            request, arrays, normalizer, horizon=horizon, device="cuda")
                        step = int(request["step"])
                        if step not in checkpoints:
                            raise ValueError(f"checkpoint step {step} unavailable")
                        if step != current_step:
                            payload = normalize_policy_payload(torch.load(checkpoints[step], map_location="cpu", weights_only=True))
                            if payload.get("format") not in ("ewm_action_only_v1", "ewm_action_only_v2") or payload["model_config"].get("training_mode") != "action_only":
                                raise ValueError("checkpoint is not an action-only policy")
                            if int(payload["horizon"]) != horizon:
                                raise ValueError("checkpoint horizon differs")
                            if int(payload["step"]) != step:
                                raise ValueError("checkpoint step disagrees with filename")
                            model.load_policy_state_dict(payload["model"])
                            del payload
                            current_step = step
                        generator = torch.Generator(device="cuda").manual_seed(seed)
                        torch.cuda.synchronize()
                        inference_started = time.perf_counter()
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            result = model.sample_policy(batch, horizon=horizon,
                                                         normalizer=normalizer, num_steps=steps,
                                                         generator=generator,
                                                         decode_future=False,
                                                         cache_context_kv=args.cache_context_kv)
                        torch.cuda.synchronize()
                        inference_seconds = time.perf_counter() - inference_started
                        executed_actions = result["action"][0, :action_chunk_size].float().cpu().numpy()
                        if executed_actions.shape != (action_chunk_size, 7) or not np.isfinite(executed_actions).all():
                            raise FloatingPointError("invalid physical action chunk")
                        arrays_out = {"executed_actions": executed_actions}
                        if action_chunk_size == 1:
                            arrays_out["executed_action"] = executed_actions[0]
                        conn.send(encode_message({
                            "ok": True, "step": step, "seed": seed,
                            "sampling_steps": steps, "action_chunk_size": action_chunk_size,
                            "training_mode": "action_only", "future_decoded": False,
                            "inference_seconds": inference_seconds,
                        }, arrays_out))
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
