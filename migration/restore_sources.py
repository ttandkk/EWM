#!/usr/bin/env python3
"""Restore pinned source repositories; does not install packages or fetch LFS assets."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def checked_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe source path: {relative}")
    target = root / path
    if target.is_symlink():
        raise ValueError(f"Refusing source symlink: {target}")
    target.resolve().relative_to(root)
    return target


def git(*args, env):
    return subprocess.run(
        ["git", "--no-optional-locks", *args], check=True,
        text=True, capture_output=True, env=env,
    ).stdout.strip()


def restore(root, manifest, dry_run=False, include_optional=False):
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    entries = json.loads(manifest.read_text())["sources"]
    results = []
    targets = set()
    plan = []
    for entry in entries:
        if entry.get("optional") and not include_optional:
            continue
        target = checked_path(root, entry["path"])
        if target in targets:
            raise ValueError(f"Duplicate source destination: {target}")
        targets.add(target)
        commit = entry["commit"]
        url = entry["url"]
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError(f"Invalid pinned commit: {commit}")
        if not (url.startswith("https://github.com/") or url.startswith("file://")):
            raise ValueError(f"Unsupported source URL: {url}")
        plan.append((entry, target))
    for entry, target in plan:
        commit = entry["commit"]
        if dry_run:
            results.append(dict(entry, status="planned"))
            continue
        if target.exists():
            if not (target / ".git").exists():
                raise RuntimeError(f"Existing non-repository destination: {target}")
            head = git("-C", str(target), "rev-parse", "HEAD", env=env)
            if head != commit:
                raise RuntimeError(f"Refusing existing commit mismatch: {target}: {head}")
            status = git("-C", str(target), "status", "--porcelain", env=env)
            if status:
                raise RuntimeError(f"Refusing modified source repository: {target}")
            state = "already_present"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            git("clone", "--no-checkout", "--filter=blob:none", entry["url"], str(target), env=env)
            try:
                git("-C", str(target), "cat-file", "-e", commit + "^{commit}", env=env)
            except subprocess.CalledProcessError:
                git("-C", str(target), "fetch", "origin", commit, env=env)
            git("-C", str(target), "checkout", "--detach", commit, env=env)
            head = git("-C", str(target), "rev-parse", "HEAD", env=env)
            if head != commit:
                raise RuntimeError(f"Source verification failed: {target}")
            state = "restored"
        results.append(dict(entry, status=state))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("sources.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()
    result = restore(args.root.resolve(), args.manifest, args.dry_run, args.include_optional)
    print(json.dumps({"dry_run": args.dry_run, "sources": result}, indent=2))
    if not args.dry_run:
        print("Pinned code restored. Verify/download required LFS assets and install packages separately.")


if __name__ == "__main__":
    main()
