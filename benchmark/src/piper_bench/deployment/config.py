"""Local deployment paths; release data never contains machine-specific roots."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def resource(name: str) -> Path:
    path = Path(__file__).resolve().parents[1] / "resources" / "deployment" / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing installed deployment resource: {name}")
    return path


def load_lock(path=None) -> dict:
    path = Path(path) if path else resource("simulation.lock.json")
    result = json.loads(path.read_text())
    if result.get("schema_version") != 1:
        raise ValueError("Unsupported simulation lock schema")
    return result


def home() -> Path:
    return Path(os.environ.get("PIPER_HOME", Path.home() / ".local/share/piper-bench")).expanduser().resolve()


def config_path(path=None) -> Path:
    return Path(path or os.environ.get("PIPER_CONFIG", home() / "deployment.json")).expanduser().resolve()


def load_config(path=None) -> dict:
    target = config_path(path)
    if not target.exists():
        return {"schema_version": 1, "prefix": str(home()), "profiles": {}, "sources": {},
                "assets_dir": str(home() / "assets")}
    data = json.loads(target.read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported local deployment config: {target}")
    return data


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def simulation_python(config=None) -> Path:
    cfg = config or load_config()
    override = os.environ.get("PIPER_SIM_PYTHON")
    value = override or cfg.get("profiles", {}).get("simulation", {}).get("python")
    if not value:
        raise ValueError("Simulation runtime is not configured. Run `ewm setup --profile simulation`, "
                         "or set PIPER_SIM_PYTHON to an existing compatible interpreter.")
    # Resolving the venv executable symlink would bypass its pyvenv.cfg.
    return Path(value).expanduser().absolute()


def runtime_environment(config=None) -> dict:
    cfg = config or load_config()
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    prefix = Path(cfg.get("prefix", home()))
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("XDG_CACHE_HOME", str(prefix / "cache"))
    if cfg.get("profiles", {}).get("simulation") or Path(cfg["assets_dir"]).is_dir():
        env.setdefault("PIPER_ASSET_ROOT", cfg["assets_dir"])
    env.setdefault("PIPER_CONFIG", str(config_path()))
    env.setdefault("OMNI_USER_CONFIG_PATH", str(prefix / "isaac-config"))
    acceptance = cfg.get("licenses", {}).get("nvidia_isaac_sim", {})
    if acceptance.get("accepted") is True and acceptance.get("version") == load_lock()["expected_simulation_versions"]["isaacsim"]:
        env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    return env
