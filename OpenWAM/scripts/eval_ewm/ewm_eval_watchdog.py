"""Restart a stalled Isaac EWM client from its last completed trial.

The model server stays alive; each client attempt is a separate process group
so an interrupted ffmpeg child cannot hold the next attempt's output open.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def stop_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stall-timeout", type=float, default=180)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=10)
    parser.add_argument("client", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if min(args.stall_timeout, args.startup_timeout, args.poll_interval) <= 0 or args.max_restarts < 0:
        parser.error("timeouts must be positive and max-restarts nonnegative")
    command = args.client[1:] if args.client[:1] == ["--"] else args.client
    if not command:
        parser.error("pass a client command after --")
    output = args.output_dir.resolve()
    heartbeat = output / "progress.heartbeat"
    for attempt in range(args.max_restarts + 1):
        current = list(command)
        if (output / "manifest.json").exists() and "--resume" not in current:
            current.append("--resume")
        before = heartbeat.stat().st_mtime_ns if heartbeat.exists() else None
        print(f"[ewm-watchdog] attempt={attempt + 1} resume={'--resume' in current}", flush=True)
        process = subprocess.Popen(current, start_new_session=True)
        started = time.monotonic()
        last_progress = None
        last_mtime = before
        stalled = False
        try:
            while process.poll() is None:
                if heartbeat.exists():
                    mtime = heartbeat.stat().st_mtime_ns
                    if mtime != last_mtime:
                        last_mtime = mtime
                        last_progress = time.monotonic()
                now = time.monotonic()
                age = now - (last_progress if last_progress is not None else started)
                limit = args.stall_timeout if last_progress is not None else args.startup_timeout
                if age > limit:
                    print(f"[ewm-watchdog] no client progress for {age:.1f}s; stopping process group {process.pid}", flush=True)
                    stalled = True
                    stop_group(process)
                    break
                time.sleep(args.poll_interval)
            return_code = process.wait()
        except BaseException:
            stop_group(process)
            raise
        if return_code == 0 and (output / "complete.json").is_file():
            print("[ewm-watchdog] all trials completed", flush=True)
            return 0
        stop_group(process)
        print(f"[ewm-watchdog] client exited rc={return_code} stalled={stalled}", flush=True)
    print(f"[ewm-watchdog] exhausted {args.max_restarts + 1} attempts", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
