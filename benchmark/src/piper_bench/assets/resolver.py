"""Resolve packaged assets or a verified downloaded bundle without cwd assumptions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from urllib.error import HTTPError
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from piper_bench.deployment.config import load_config, resource, sha256


def _relative(name: str) -> Path:
    value = PurePosixPath(name)
    if not name or value.is_absolute() or ".." in value.parts or "\\" in name:
        raise ValueError(f"Asset name must be a safe relative path: {name!r}")
    return Path(*value.parts)


def asset_path(name: str, *, root=None, must_exist=True) -> Path:
    relative = _relative(name)
    base = Path(root or os.environ.get("PIPER_ASSET_ROOT") or load_config()["assets_dir"])
    package = Path(__file__).resolve().parents[1] / "resources" / "assets"
    candidates = [base / relative, base / "Event-DataCollection/assets" / relative, package / relative]
    for path in candidates:
        if path.exists():
            return path.resolve()
    if not must_exist:
        return package / relative
    raise FileNotFoundError(f"Missing asset {name}; run `ewm assets download` or set PIPER_ASSET_ROOT")


def upstream_asset_path(relative: str, *, source="agilex", config=None) -> Path:
    cfg = config or load_config()
    source_dir = cfg.get("sources", {}).get(source)
    if not source_dir:
        raise FileNotFoundError(f"Source {source!r} is not installed; run `ewm setup --profile simulation`")
    path = Path(source_dir) / _relative(relative)
    if not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


def load_manifest(path=None) -> dict:
    data = json.loads((Path(path) if path else resource("assets.lock.json")).read_text())
    if data.get("schema_version") != 1 or not isinstance(data.get("files"), dict):
        raise ValueError("Unsupported asset manifest")
    revision = data.get("revision", "")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Assets require an immutable 40-character Git revision")
    for name, spec in data["files"].items():
        _relative(name)
        if len(spec.get("sha256", "")) != 64 or spec.get("size", -1) < 0:
            raise ValueError(f"Invalid asset checksum/size for {name}")
    return data


def verify(root: Path, manifest: dict) -> dict:
    root = root.resolve()
    missing, mismatched = [], []
    for name, spec in manifest["files"].items():
        path = root / _relative(name)
        if not path.is_file():
            missing.append(name)
        elif not path.resolve().is_relative_to(root):
            mismatched.append({"path": name, "reason": "symlink escapes asset root"})
        elif path.stat().st_size != spec["size"] or sha256(path) != spec["sha256"]:
            mismatched.append({"path": name, "reason": "size or SHA256 mismatch"})
    return {"passed": not missing and not mismatched, "revision": manifest["revision"],
            "root": str(root), "requested_files": len(manifest["files"]),
            "verified_files": len(manifest["files"]) - len(missing) - len(mismatched),
            "missing": missing, "mismatched": mismatched,
            "external_dependencies": manifest.get("external", {}),
            "usd_resolution_checked": False}


class _TokenSafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urlparse(req.full_url).netloc != urlparse(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def download(root: Path, manifest: dict, *, token=None, opener=None) -> dict:
    """Download immutable files atomically; never replace differing local files."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    opener = opener or build_opener(_TokenSafeRedirect())
    token = token or os.environ.get("HF_TOKEN")
    for name, spec in manifest["files"].items():
        path = root / _relative(name)
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Asset target escapes root: {name}")
        if path.exists():
            if path.is_file() and path.stat().st_size == spec["size"] and sha256(path) == spec["sha256"]:
                continue
            raise FileExistsError(f"Refusing to overwrite differing local asset: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        source = spec.get("source", {})
        if source.get("kind") == "packaged":
            packaged = Path(__file__).resolve().parents[1] / "resources/assets" / _relative(source["path"])
            response = packaged.open("rb")
        else:
            prefix = "datasets/" if manifest.get("repo_type") == "dataset" else ""
            url = source.get("url") or (f"https://huggingface.co/{prefix}{manifest['repository']}/resolve/"
                                        f"{manifest['revision']}/{quote(name, safe='/')}")
            if urlparse(url).scheme != "https":
                raise ValueError("Asset downloads require HTTPS")
            headers = {"Authorization": f"Bearer {token}"} if token and urlparse(url).hostname == "huggingface.co" else {}
            request = Request(url, headers=headers)
            try:
                response = opener.open(request, timeout=120)
            except HTTPError as exc:
                if exc.code in (401, 403):
                    raise PermissionError("Asset repository requires access; provide HF_TOKEN for an authorized account. "
                                          "No repository visibility has been changed.") from exc
                raise
        temporary = None
        try:
            with response, tempfile.NamedTemporaryFile(
                    dir=path.parent, prefix=".asset-", delete=False) as target:
                temporary = Path(target.name)
                received = 0
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    received += len(block)
                    if received > spec["size"]:
                        raise ValueError(f"Asset exceeds locked size: {name}")
                    target.write(block)
            if temporary.stat().st_size != spec["size"] or sha256(temporary) != spec["sha256"]:
                raise ValueError(f"Downloaded asset failed size/SHA256 verification: {name}")
            # Hard-link creation fails atomically if another process wrote this target.
            os.link(temporary, path)
        except HTTPError as exc:
            if exc.code in (401, 403):
                raise PermissionError("Asset repository requires access; provide HF_TOKEN for an authorized account. "
                                      "No repository visibility has been changed.") from exc
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return verify(root, manifest)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "verify", "download"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    root = args.root or Path(os.environ.get("PIPER_ASSET_ROOT", load_config()["assets_dir"]))
    if args.action == "list":
        print(json.dumps(manifest, indent=2))
        return 0
    report = download(root, manifest) if args.action == "download" else verify(root, manifest)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
