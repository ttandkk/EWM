"""Model-only service entry point; simulator dependencies are never imported."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metadata import load_metadata
from .resources import ResourceResolver


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("serve", "describe"))
    parser.add_argument("--family", choices=("ewm", "openwam", "fastwam"), default="ewm")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint", help="specific checkpoint filename or absolute path")
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--config", help="model-side saved, composed training config")
    parser.add_argument("--normalization", help="saved training statistics; never recomputed at evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--project-root", help="base for relative model asset paths")
    parser.add_argument("--pretrained-root", help="relocate the suffix below pretrained/ explicitly")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--compile-core", action=argparse.BooleanOptionalAction, default=None,
                        help="EWM defaults on; failed compilation never publishes readiness")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache-context-kv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-bf16-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--benchmark-iters", type=int, default=3)
    parser.add_argument("--sampling-steps", type=int, default=4, help="startup warmup sampling steps")
    parser.add_argument("--warmup-prompt", default="Move the object into the container.")
    parser.add_argument("--max-compile-action-error", type=float, default=.02)
    parser.add_argument("--allow-compile-action-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ready-file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    options = {key: getattr(args, key) for key in ("checkpoint", "checkpoint_step", "config", "normalization")}
    if args.command == "describe":
        descriptor = load_metadata(args.checkpoint_dir, args.family, **options)[3]
        print(json.dumps(descriptor, indent=2, ensure_ascii=False))
        return 0
    if not args.ready_file:
        parser.error("serve requires --ready-file")
    if Path(args.ready_file).expanduser().exists():
        raise FileExistsError(f"refusing to replace readiness file: {args.ready_file}")
    resolver = ResourceResolver(project_root=args.project_root, pretrained_root=args.pretrained_root, path_maps=args.path_map)
    if args.family == "ewm":
        from .ewm.adapter import EWMPolicy as Policy
    elif args.family == "openwam":
        from .openwam.adapter import OpenWAMPolicy as Policy
    else:
        from .fastwam.adapter import FastWAMPolicy as Policy
    if args.family == "ewm":
        inference = {key: getattr(args, key) for key in ("compile_mode", "cuda_graphs", "cache_bf16_weights",
            "warmup_iters", "benchmark_iters", "sampling_steps", "warmup_prompt", "max_compile_action_error", "allow_compile_action_error")}
        inference["compile_core"] = args.compile_core is not False
        options.update(cache_context_kv=args.cache_context_kv, inference_options=inference)
    elif args.compile_core:
        parser.error("--compile-core currently controls EWM; use each WAM model's native configuration")
    policy = Policy(args.checkpoint_dir, device=args.device, resolver=resolver, **options)
    from .server import serve_policy
    serve_policy(policy, ready_file=args.ready_file, host=args.host, port=args.port)
    return 0
