"""Freeze complete collection code for new runs; never upgrade an old snapshot."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil


def freeze_collection_source(project_root, destination):
    """Create scripts/assets and a self-contained canonical package snapshot.

    Call only for a new run. Existing destinations are rejected so resuming a
    historical run cannot silently replace its original implementation/assets.
    """
    root, destination = Path(project_root).resolve(), Path(destination).resolve()
    package = root / 'benchmark/src/piper_bench'
    scripts = root / 'Event-DataCollection/scripts'
    if not (package / '__init__.py').is_file():
        raise FileNotFoundError('collection freeze requires the benchmark package')
    if any(destination == path or path in destination.parents
           for path in (package, scripts, root / 'Event-DataCollection/assets')):
        raise ValueError('snapshot destination cannot be inside the source being copied')
    destination.mkdir(parents=True, exist_ok=False)
    for path in [*scripts.glob('*.py'), *(scripts / 'piper_tasks').rglob('*.py')]:
        target = destination / 'scripts' / path.relative_to(scripts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    if not scripts.is_dir():
        # Slim releases omit the old tree. Keep the two filenames used by the
        # collection launchers while executing the frozen canonical package.
        entrypoints = {'collect_piper_tasks_evis.py': 'piper_bench.collection.runner',
                       'validate_piper_tasks_lerobot.py': 'piper_bench.collection.validation'}
        (destination / 'scripts').mkdir()
        for name, module in entrypoints.items():
            (destination / 'scripts' / name).write_text(
                'import runpy\nimport sys\nfrom pathlib import Path\n'
                'sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark/src"))\n'
                f'if __name__ == "__main__":\n    runpy.run_module({module!r}, run_name="__main__")\n')
    shutil.copytree(package, destination / 'benchmark/src/piper_bench',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
    assets = root / 'Event-DataCollection/assets'
    if assets.is_dir():
        shutil.copytree(assets, destination / 'assets',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
    return destination


def frozen_pythonpath(source):
    """Prefer the frozen package; retain the legacy-only path for old runs."""
    source = Path(source).resolve()
    package_parent = source / 'benchmark/src'
    paths = [source / 'scripts']
    if (package_parent / 'piper_bench/__init__.py').is_file():
        paths.insert(0, package_parent)
    return os.pathsep.join(map(str, paths))


def hash_snapshot(source):
    """Hash source, resource configuration and assets with relocation-stable keys."""
    source = Path(source)
    return {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.rglob('*')) if path.is_file()}
