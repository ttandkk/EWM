"""Freeze first-party benchmark/model source without weights, data or environments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

DIRECTORIES = ('benchmark/src', 'policies/src', 'OpenWAM/openwam', 'OpenWAM/configs',
               'OpenWAM/scripts', 'scripts', 'deployment', 'launch')
FILES = ('ewm', 'benchmark/pyproject.toml', 'policies/pyproject.toml', 'OpenWAM/pyproject.toml')
IGNORED = {'__pycache__', '.git', '.pytest_cache', 'node_modules', 'build', 'dist'}


def freeze(root, destination):
    root, destination = Path(root).resolve(), Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(f'Source snapshot already exists: {destination}')
    selected = []
    for name in DIRECTORIES:
        selected.extend(path for path in (root / name).rglob('*')
                        if path.is_file() and not (set(path.relative_to(root).parts) & IGNORED)
                        and path.suffix not in ('.pyc', '.pyo'))
    selected.extend(root / name for name in FILES if (root / name).is_file())
    if not (root / 'benchmark/src/piper_bench/cli.py').is_file():
        raise ValueError('Project has no canonical benchmark package')
    hashes = {}
    destination.mkdir(parents=True)
    for source in sorted(set(selected)):
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = dict(version='piper_source_snapshot_v1', sha256=hashes,
                    exclusions=['datasets', 'pretrained', 'envs', 'results', 'logs', 'tmp', 'third_party'])
    (destination / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def verify(destination):
    destination = Path(destination)
    manifest = json.loads((destination / 'source_manifest.json').read_text())
    actual = {str(path.relative_to(destination)) for path in destination.rglob('*')
              if path.is_file() and not (set(path.relative_to(destination).parts) & IGNORED)
              and path.suffix not in ('.pyc', '.pyo') and path.name != 'source_manifest.json'}
    if actual != set(manifest['sha256']):
        raise ValueError('Source snapshot file inventory changed')
    for name, expected in manifest['sha256'].items():
        path = destination / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source snapshot changed: {name}')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    manifest = verify(args.output) if args.verify else freeze(args.root, args.output)
    print(json.dumps(dict(files=len(manifest['sha256']), directory=str(args.output))))


if __name__ == '__main__':
    main()
