"""Forward first-party Python packages without mixing interpreter dependencies."""
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import tempfile


@contextmanager
def package_environment(environment=None, *, include_policies=False):
    """Keep a package-only import bridge alive until the subprocess finishes.

    A package may live inside site-packages. Forwarding that whole directory
    would also forward its Python-version-specific NumPy/Torch extensions.
    Source checkout paths remain usable, including historical frozen sources.
    """
    env = dict(os.environ if environment is None else environment)
    env.pop("PYTHONHOME", None)
    packages = {"piper_bench": Path(__file__).resolve().parents[1]}
    if include_policies:
        spec = importlib.util.find_spec("piper_policies")
        if spec is not None and spec.submodule_search_locations:
            packages["piper_policies"] = Path(next(iter(spec.submodule_search_locations))).resolve()
    installed_parents = {
        package.parent for name, package in packages.items()
        if any(package.parent.glob(name + "-*.dist-info"))
    }
    paths = []
    for entry in env.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        path = Path(entry).expanduser().resolve()
        if path.name not in ("site-packages", "dist-packages") and path not in installed_parents:
            paths.append(entry)
    with tempfile.TemporaryDirectory(prefix="piper-package-bridge-") as temporary:
        bridge = Path(temporary)
        for name, package in packages.items():
            (bridge / name).symlink_to(package, target_is_directory=True)
        env["PYTHONPATH"] = os.pathsep.join([str(bridge), *paths])
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        yield env
