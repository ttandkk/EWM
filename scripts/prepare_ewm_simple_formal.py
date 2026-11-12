#!/usr/bin/env python3
"""Freeze and verify the agreed three-mode EWM training inputs.

Run with the project's OpenWAM Python. This command never submits jobs.
"""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = 'tmp/training_data_cache/piper_dynamic_v1'
NORMALIZATION_SHA256 = '7e4291e15964624aaff460aeda60e9de697d396f67e875f111868ba2225aedd3'
SMOKE = 'tmp/ewm-simple-smoke-3muzuj7v'
TASKS = ('conveyor', 'moving_carton', 'turntable_cups', 'rotating_tube',
         'tennis_rolling', 'tennis_falling', 'tennis_parabolic', 'tennis_container')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def source_files(directory):
    return {p.relative_to(directory).as_posix(): digest(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def cache_signature(root, hash_content=False):
    # Optional full hashes protect against changes that preserve size and mtime.
    result = {}
    for task in TASKS:
        base = root / CACHE / 'current_dino_fixed32' / f'piper_{task}' / 'dino_current16'
        path = base / 'manifest.json'
        manifest = json.loads(path.read_text())
        if manifest.get('format') != 'ewm_dino_current_bf16_v1' or len(manifest['episodes']) != 50:
            raise ValueError(f'{task}: invalid current DINO cache')
        entries = {}
        for row in manifest['episodes'].values():
            file = base / row['file']
            stat = file.stat()
            entries[row['file']] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            if hash_content:
                entries[row['file']]['sha256'] = digest(file)
        result[task] = {'manifest_sha256': digest(path), 'files': entries}
        print(f'DINO_CACHE_SIGNATURE task={task} sha256={hash_content} files={len(entries)}', flush=True)
    return result


def input_signature(root, source, hash_cache=False):
    sys.path.insert(0, str(source / 'OpenWAM'))
    from openwam.dataloader.piper_training_data import source_signatures, validate_exports
    exports = validate_exports(root)
    original = json.loads((root / SMOKE / 'rgb_preflight.json').read_text())['source_exports']
    keys = ('input_signature', 'collection_sha256', 'episodes', 'frames')
    actual = {task: {key: exports[task][key] for key in keys} for task in TASKS}
    expected = {task: {key: original[task][key] for key in keys} for task in TASKS}
    if actual != expected:
        raise ValueError('dataset exports differ from the approved smoke preflight')
    norm = root / CACHE / 'shared_normalization.json'
    if digest(norm) != NORMALIZATION_SHA256:
        raise ValueError('pooled normalization differs from the approved plan')
    sources = source_signatures(root)
    if json.loads(norm.read_text())['sources'] != sources:
        raise ValueError('normalization data sources changed')
    baseline = json.loads((root / SMOKE / 'fingerprints.json').read_text())['weights']
    for name, expected_digest in baseline.items():
        if digest(root / name) != expected_digest:
            raise ValueError(f'pretrained file changed: {name}')
    runtime = {'python': sys.version, 'executable': str(Path(sys.executable).resolve()),
               'packages': sorted([dist.metadata['Name'], dist.version]
                                  for dist in importlib.metadata.distributions()),
               'deploy_env_sha256': digest(root / 'deployment/runtime_env.sh')}
    return {'exports': actual, 'normalization_sha256': NORMALIZATION_SHA256,
            'data_sources': sources, 'weights': baseline, 'dino_cache': cache_signature(root, hash_cache),
            'runtime': runtime}


def prepare(args):
    root, batch = Path(args.root).resolve(), Path(args.batch).resolve()
    if batch.exists():
        raise FileExistsError(f'refusing to overwrite existing batch: {batch}')
    batch.mkdir(parents=True)
    source = batch / 'source'
    try:
        for name in ('configs', 'openwam', 'scripts'):
            shutil.copytree(root / 'OpenWAM' / name, source / 'OpenWAM' / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (source / 'scripts').mkdir(parents=True)
        for name in ('prepare_ewm_simple_formal.py', 'train_ewm_simple_a6000.sbatch',
                     'train_ewm_simple_ada_serial.sbatch'):
            shutil.copy2(root / 'scripts' / name, source / 'scripts' / name)
        snapshot = source_files(source)
        reuse = bool(args.cache_hash_reference)
        inputs = input_signature(root, source, args.cache_content_sha256)
        reference_record = None
        if reuse:
            reference_path = Path(args.cache_hash_reference).resolve()
            reference = json.loads(reference_path.read_text())
            if not reference.get('cache_content_sha256'):
                raise ValueError('cache reference does not contain full content hashes')
            old_cache = reference['inputs']['dino_cache']
            if {k: v for k, v in reference['inputs'].items() if k != 'dino_cache'} != {
                    k: v for k, v in inputs.items() if k != 'dino_cache'}:
                raise ValueError('cache reference differs in data, weights, normalization or runtime')
            for task, current in inputs['dino_cache'].items():
                previous = old_cache[task]
                if current['manifest_sha256'] != previous['manifest_sha256'] or set(current['files']) != set(previous['files']):
                    raise ValueError(f'{task}: cache reference manifest differs')
                for name, metadata in current['files'].items():
                    saved = previous['files'][name]
                    if (metadata['size'], metadata['mtime_ns']) != (saved['size'], saved['mtime_ns']):
                        raise ValueError(f'{task}: cache reference file changed: {name}')
                    sha = saved.get('sha256')
                    if not isinstance(sha, str) or len(sha) != 64:
                        raise ValueError(f'{task}: missing cache content hash: {name}')
                    metadata['sha256'] = sha
            reference_record = {'manifest': str(reference_path), 'sha256': digest(reference_path),
                                'verified_by_size_and_mtime_only': True}
        manifest = {'protocol': 'ewm_simple_eight_v1_continuous', 'project_root': str(root),
                    'source_sha256': snapshot, 'cache_content_sha256': args.cache_content_sha256 or reuse,
                    'cache_hash_reference': reference_record, 'inputs': inputs}
        (batch / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        print(f'FROZEN_BATCH_READY {batch} source_files={len(snapshot)}')
    except BaseException:
        # Preserve incomplete snapshot for inspection; it cannot pass verify.
        print(f'INCOMPLETE_BATCH {batch}', file=sys.stderr)
        raise


def verify(args):
    batch = Path(args.batch).resolve()
    manifest = json.loads((batch / 'manifest.json').read_text())
    source = batch / 'source'
    if source_files(source) != manifest['source_sha256']:
        raise ValueError('frozen source differs from its manifest')
    root = Path(manifest['project_root'])
    if input_signature(root, source, manifest.get('cache_content_sha256', False)) != manifest['inputs']:
        raise ValueError('dataset, weight or cache fingerprint changed')
    print(f'FROZEN_BATCH_VERIFIED {batch}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--root', default=str(ROOT))
    p.add_argument('--batch', required=True)
    cache = p.add_mutually_exclusive_group()
    cache.add_argument('--cache-content-sha256', action='store_true',
                       help='hash every DINO cache array during prepare and every verify')
    cache.add_argument('--cache-hash-reference', metavar='MANIFEST',
                       help='reuse a full-hash manifest after checking all sizes and mtimes; verify still rehashes')
    p.set_defaults(func=prepare)
    p = commands.add_parser('verify')
    p.add_argument('--batch', required=True)
    p.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
