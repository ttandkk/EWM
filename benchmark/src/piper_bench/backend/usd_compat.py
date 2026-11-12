"""Auditable Piper USD derivation; original assets are always read-only.

The fixed upstream asset contains one reference to a nonexistent empty world
visual. Reference-list edits cannot cross its payload boundary from a session
layer. Copy its five-layer closure, remove only that exact source-layer arc,
and require complete native USD value equality before using the derivative.
"""
from __future__ import annotations

import hashlib
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile

from piper_bench.assets.resolver import load_manifest
from piper_bench.deployment.config import load_config, sha256, write_json

RECIPE = "piper-empty-world-reference-v1"
BAD_PRIM = "/piper/world/visuals"
BAD_TARGET = "/visuals/world"
FILES = ("piper.usd", "configuration/piper_base.usd", "configuration/piper_physics.usd",
         "configuration/piper_robot.usd", "configuration/piper_sensor.usd")


def _canonical(value):
    if isinstance(value, dict):
        return {str(k): _canonical(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _payload(stage):
    """Native values for strict equality; no numerical tolerance or omitted fields."""
    from pxr import Usd
    prims = {}
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        prims[str(prim.GetPath())] = dict(
            type=prim.GetTypeName(), schemas=list(prim.GetAppliedSchemas()),
            flags=[prim.IsActive(), prim.IsDefined(), prim.IsLoaded(), prim.IsInstance(), prim.IsInstanceProxy()],
            metadata=prim.GetAllMetadata(),
            attributes={a.GetName(): dict(value=a.Get(), metadata=a.GetAllMetadata(),
                                         connections=a.GetConnections(),
                                         samples={t: a.Get(t) for t in a.GetTimeSamples()})
                        for a in prim.GetAttributes()},
            relationships={r.GetName(): dict(targets=r.GetTargets(), metadata=r.GetAllMetadata())
                           for r in prim.GetRelationships()})
    return dict(stage=stage.GetPseudoRoot().GetAllMetadata(), prims=prims)


def _digest(value):
    return hashlib.sha256(json.dumps(_canonical(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_hashes(directory):
    manifest = load_manifest()
    root = next(name[:-len("piper.usd")] for name in manifest["files"] if name.endswith("/piper/usd/piper.usd"))
    result = {}
    for name in FILES:
        path = directory / name
        expected = manifest["files"][root + name]
        if (not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or
                path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]):
            raise ValueError(f"USD compatibility requires the exact locked Piper source: {path}")
        result[name] = expected["sha256"]
    return result


def _open_disk_stage(entry):
    """USD caches layers globally; refresh every known file, never dirty layers."""
    from pxr import Sdf, Usd
    layers = [Sdf.Layer.FindOrOpen(str(entry.parent / name)) for name in FILES]
    if any(layer is None or layer.dirty for layer in layers):
        raise ValueError("Refusing missing or in-memory edited Piper layers")
    # Check all dirty flags before touching any layer shared with an existing stage.
    for layer in layers:
        layer.Reload(True)
    stage = Usd.Stage.Open(str(entry))
    if stage is None:
        raise ValueError(f"Unable to open Piper USD: {entry}")
    return stage


def _prove(original, derived):
    from pxr import Usd
    before, after = _payload(original), _payload(derived)
    errors = [str(error) for error in derived.GetCompositionErrors()]
    # Dict equality reaches native Vt/Gf/Sdf ==. String hashes are provenance only.
    equal = before == after
    prim_count = len(before["prims"])
    mesh_count = sum(p["type"] == "Mesh" for p in before["prims"].values())
    if not equal or errors or prim_count != 761 or mesh_count != 20:
        raise ValueError(f"Derived Piper changed composed data or retained errors: equal={equal}, errors={errors}")
    return {"passed": True, "native_values_exactly_equal": True,
            "comparison": "Every composed prim including instance proxies, all flags/schemas/metadata, attributes/defaults/time samples/connections and relationships/targets/metadata; no fields excluded.",
            "source_semantic_sha256": _digest(before), "derived_semantic_sha256": _digest(after),
            "prim_count": prim_count, "mesh_count": mesh_count,
            "attribute_count": sum(len(p["attributes"]) for p in before["prims"].values()),
            "relationship_count": sum(len(p["relationships"]) for p in before["prims"].values()),
            "derived_composition_errors": errors, "usd_version": list(Usd.GetVersion())}


def _expected_base(source):
    from pxr import Sdf
    layer = Sdf.Layer.FindOrOpen(str(source.parent / "configuration/piper_base.usd"))
    derived = Sdf.Layer.CreateAnonymous("piper_base.usdc")
    derived.TransferContent(layer)
    prim = derived.GetPrimAtPath(BAD_PRIM)
    expected = Sdf.Reference("", Sdf.Path(BAD_TARGET))
    refs = prim.referenceList
    if (list(refs.prependedItems) != [expected] or list(prim.nameChildren) or
            any(list(getattr(refs, name)) for name in
                ("explicitItems", "addedItems", "appendedItems", "deletedItems", "orderedItems"))):
        raise ValueError("Pinned base layer no longer contains only the known empty reference")
    refs.prependedItems = []
    return derived


def compatible_piper(source, *, cache_root=None):
    """Return a verified derived entry layer; never edit the downloaded bundle."""
    from pxr import Sdf, Usd
    source = Path(source).resolve()
    if source.name != "piper.usd":
        raise ValueError("USD compatibility requires the exact locked Piper entry piper.usd")
    hashes = _source_hashes(source.parent)
    recipe_hash = sha256(Path(__file__))
    identity = {"recipe": RECIPE, "recipe_sha256": recipe_hash, "source_sha256": hashes,
                "usd_version": list(Usd.GetVersion())}
    key = _digest(identity)
    cache = Path(cache_root) if cache_root else Path(os.environ.get(
        "XDG_CACHE_HOME", str(Path(load_config()["prefix"]) / "cache"))) / "piper-bench/usd"
    cache = cache.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / key
    original = _open_disk_stage(source)
    empty = original.GetPrimAtPath(BAD_PRIM)
    if not empty or list(empty.GetAllChildren()) or original.GetPrimAtPath(BAD_TARGET):
        raise ValueError("Pinned dangling-reference preconditions are not satisfied")
    original_errors = [str(error) for error in original.GetCompositionErrors()]
    if len(original_errors) != 1 or BAD_TARGET not in original_errors[0] or BAD_PRIM not in original_errors[0]:
        raise ValueError(f"Unexpected original Piper USD composition state: {original_errors}")
    if target.exists():
        if target.is_symlink() or (target / "derivation.json").is_symlink():
            raise ValueError("Refusing symlinked Piper derivation cache")
        record = json.loads((target / "derivation.json").read_text())
        if record.get("identity") != identity:
            raise ValueError("Cached Piper derivation provenance differs")
        for name in FILES:
            path = target / name
            if (path.is_symlink() or not path.resolve().is_relative_to(target.resolve()) or
                    sha256(path) != record["derived_sha256"][name]):
                raise ValueError(f"Refusing modified cached Piper layer: {path}")
            if name != "configuration/piper_base.usd" and sha256(path) != hashes[name]:
                raise ValueError(f"Cached Piper changed an unpatched source layer: {path}")
        # Recheck semantics rather than trusting a locally editable manifest.
        cached = _open_disk_stage(target / "piper.usd")
        cached_base = Sdf.Layer.FindOrOpen(str(target / "configuration/piper_base.usd"))
        # Compare authored content too: hidden/unselected opinions must not slip
        # past a composed-stage check or a locally edited cache manifest.
        if cached_base.ExportToString() != _expected_base(source).ExportToString():
            raise ValueError("Cached Piper base differs from the exact single-reference derivation")
        _prove(original, cached)
        return target / "piper.usd"
    temporary = Path(tempfile.mkdtemp(prefix=".piper-derive-", dir=cache))
    try:
        for name in FILES:
            if name == "configuration/piper_base.usd":
                continue
            (temporary / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source.parent / name, temporary / name)
        derived_layer = _expected_base(source)
        if not derived_layer.Export(str(temporary / "configuration/piper_base.usd"), args={"format": "usdc"}):
            raise RuntimeError("Unable to export derived Piper base layer")
        proof = _prove(original, _open_disk_stage(temporary / "piper.usd"))
        derived_hashes = {name: sha256(temporary / name) for name in FILES}
        if any(derived_hashes[name] != hashes[name] for name in FILES if name != "configuration/piper_base.usd"):
            raise ValueError("Unexpected changes outside the single patched USD layer")
        if _source_hashes(source.parent) != hashes:
            raise ValueError("Original Piper source changed during derivation")
        write_json(temporary / "derivation.json", {"schema_version": 1, "identity": identity,
                   "source_entry": str(source), "original_composition_errors": original_errors,
                   "operation": {"prim": BAD_PRIM, "remove_only_prepend_reference": BAD_TARGET},
                   "source_files_unchanged": True, "derived_sha256": derived_hashes, "proof": proof})
        try:
            temporary.rename(target)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            return compatible_piper(source, cache_root=cache)
        return target / "piper.usd"
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
