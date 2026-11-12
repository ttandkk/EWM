"""Real CPU USD checks; opt in with the exact downloaded source bundle."""
import errno
import json
import os
from pathlib import Path
import shutil

import pytest

from piper_bench.backend import usd_compat


@pytest.fixture
def source():
    value = os.environ.get("PIPER_USD_TEST_SOURCE")
    if not value:
        pytest.skip("Set PIPER_USD_TEST_SOURCE to run real CPU USD compatibility checks")
    pytest.importorskip("pxr.Usd")
    return Path(value)


def test_derived_usd_exact_native_values_and_source_bytes(source, tmp_path):
    before = usd_compat._source_hashes(source.parent)
    derived = usd_compat.compatible_piper(source, cache_root=tmp_path)
    assert usd_compat.compatible_piper(source, cache_root=tmp_path) == derived
    record = json.loads((derived.parent / "derivation.json").read_text())
    assert record["proof"]["native_values_exactly_equal"]
    assert record["proof"]["derived_composition_errors"] == []
    assert record["proof"]["prim_count"] == 761
    assert record["proof"]["attribute_count"] == 6921
    assert usd_compat._source_hashes(source.parent) == before


def test_unknown_source_is_rejected_without_changes(source, tmp_path):
    copied = tmp_path / "source"
    shutil.copytree(source.parent, copied)
    base = copied / "configuration/piper_base.usd"
    base.write_bytes(base.read_bytes() + b"modified")
    before = base.read_bytes()
    with pytest.raises(ValueError, match="exact locked"):
        usd_compat.compatible_piper(copied / "piper.usd", cache_root=tmp_path / "cache")
    assert base.read_bytes() == before
    assert not (tmp_path / "cache").exists()


def test_cached_layer_corruption_is_rejected(source, tmp_path):
    derived = usd_compat.compatible_piper(source, cache_root=tmp_path)
    base = derived.parent / "configuration/piper_base.usd"
    base.write_bytes(base.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="modified cached"):
        usd_compat.compatible_piper(source, cache_root=tmp_path)


def test_forged_cache_manifest_cannot_hide_disk_edit_behind_usd_memory_cache(source, tmp_path):
    from pxr import Sdf, Usd
    derived = usd_compat.compatible_piper(source, cache_root=tmp_path)
    live_stage = Usd.Stage.Open(str(derived))
    base = derived.parent / "configuration/piper_base.usd"
    edited = Sdf.Layer.CreateAnonymous("edited.usdc")
    edited.TransferContent(Sdf.Layer.FindOrOpen(str(base)))
    edited.GetPrimAtPath("/piper").customData = {"unexpected_change": 1}
    assert edited.Export(str(base), args={"format": "usdc"})
    manifest = derived.parent / "derivation.json"
    record = json.loads(manifest.read_text())
    record["derived_sha256"]["configuration/piper_base.usd"] = usd_compat.sha256(base)
    manifest.write_text(json.dumps(record))
    assert "unexpected_change" not in live_stage.GetPrimAtPath("/piper").GetCustomData()
    with pytest.raises(ValueError, match="exact single-reference derivation"):
        usd_compat.compatible_piper(source, cache_root=tmp_path)


@pytest.mark.parametrize("error_number", [errno.EEXIST, errno.ENOTEMPTY])
def test_racing_first_creation_rechecks_winning_cache(source, tmp_path, monkeypatch, error_number):
    original = Path.rename
    collided = []

    def race(path, target):
        if path.name.startswith(".piper-derive-") and not collided:
            shutil.copytree(path, target)
            collided.append(True)
            raise OSError(error_number, "Another process published this valid cache")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", race)
    result = usd_compat.compatible_piper(source, cache_root=tmp_path)
    assert result.is_file() and collided
    assert not list(tmp_path.glob(".piper-derive-*"))
