"""Native EWM compile setup and measured, reproducible startup validation."""
from __future__ import annotations

import time


def prepare_inference(policy, *, compile_core=False, compile_mode="default", cuda_graphs=False,
                      cache_bf16_weights=True, warmup_iters=0, benchmark_iters=None,
                      sampling_steps=4, warmup_prompt="Move the object into the container.",
                      max_compile_action_error=.02, allow_compile_action_error=True):
    import torch
    from openwam.model.architectures.ewm_inference import configure_action_compile, configure_inference_compile

    if benchmark_iters is None:
        benchmark_iters = int(compile_core)
    if min(warmup_iters, benchmark_iters) < 0 or sampling_steps <= 0 or max_compile_action_error <= 0:
        raise ValueError("invalid warmup/benchmark/sampling/error limits")
    if compile_core and policy.device.type != "cuda":
        raise ValueError("compiled policy service requires CUDA; use --no-compile-core for CPU diagnostics")
    model, d = policy.model, policy.descriptor
    batch = dict(obs_video=torch.zeros((1, 2, 3, 256, 256), device=policy.device, dtype=torch.uint8),
                 proprio=torch.zeros((1, 8), device=policy.device), prompt=[warmup_prompt])
    if d["mode"] == "joint":
        batch["obs_voxel"] = torch.zeros((1, d["event_cameras"], 2*d["voxel_bins"], 256, 256), device=policy.device)
    elif d["mode"] == "rgb_action":
        views = {"dual": slice(None), "wrist_only": slice(1, 2), "global_only": slice(0, 1)}
        batch["obs_rgb"] = batch["obs_video"][:, views[d["event_camera_mode"]]]

    def sample(seed, cache):
        if policy.device.type == "cuda":
            torch.cuda.synchronize(policy.device)
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(policy.device.type, dtype=torch.bfloat16, enabled=policy.device.type == "cuda"):
            result = model.sample_policy(batch, horizon=d["action_horizon"], normalizer=policy.normalizer,
                num_steps=sampling_steps, generator=torch.Generator(device=policy.device).manual_seed(seed),
                decode_future=False, cache_context_kv=cache)
        action = result["action_normalized"].detach().float().cpu().clone()
        if not torch.isfinite(action).all():
            raise ValueError("startup sampler returned nonfinite actions")
        if policy.device.type == "cuda":
            torch.cuda.synchronize(policy.device)
        return action, time.perf_counter() - started

    baseline, baseline_times = [], []
    eager_kv, eager_kv_times = [], []
    if benchmark_iters:
        sample(9100, False)
        for index in range(benchmark_iters):
            value, elapsed = sample(9200 + index, False)
            baseline.append(value); baseline_times.append(elapsed)
        sample(9150, True)
        for index in range(benchmark_iters):
            value, elapsed = sample(9200 + index, True)
            eager_kv.append(value); eager_kv_times.append(elapsed)
    active, graphs = False, False
    eager_velocity = model.predict_velocity
    if compile_core:
        if d["mode"] == "action_only":
            graphs = configure_action_compile(model, mode=compile_mode, cuda_graphs=cuda_graphs)
        else:
            graphs = configure_inference_compile(model, mode=compile_mode, cuda_graphs=cuda_graphs,
                                                 cache_bf16_weights=cache_bf16_weights)
        active = True
    effective = max(warmup_iters, int(compile_core))
    optimized, optimized_times = [], []
    try:
        for index in range(effective):
            sample(9300 + index, policy.cache_context_kv)
        for index in range(benchmark_iters):
            value, elapsed = sample(9200 + index, policy.cache_context_kv)
            optimized.append(value); optimized_times.append(elapsed)
        error = max((float((a-b).abs().max()) for a,b in zip(optimized, baseline)), default=0.)
        if active and error > max_compile_action_error and not allow_compile_action_error:
            raise ValueError(f"compiled action error {error} exceeds {max_compile_action_error}")
    except Exception:
        model.predict_velocity = eager_velocity
        model.disable_inference_compile()
        # A failed compiler must not publish a ready formal service running eager.
        raise
    mean = lambda values: sum(values) / len(values) if values else None
    return dict(compile_core_requested=compile_core, compile_core_active=active,
                compile_mode=compile_mode if active else None, cuda_graphs_enabled=active and graphs,
                cache_context_kv=policy.cache_context_kv,
                cache_bf16_weights=bool(getattr(model, "_inference_bf16_weights", False)),
                warmup_iters=effective, warmup_sampling_steps=sampling_steps,
                max_compile_action_error=max_compile_action_error, allow_compile_action_error=allow_compile_action_error,
                startup_benchmark=(dict(iterations=benchmark_iters, action_max_abs_error=error,
                    baseline_eager_no_kv_cache_mean_seconds=mean(baseline_times),
                    eager_kv_cache_mean_seconds=mean(eager_kv_times), optimized_mean_seconds=mean(optimized_times),
                    compile_error_exceeded=active and error > max_compile_action_error) if benchmark_iters else None))
