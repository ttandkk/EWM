"""Metadata identities that survive moving a dataset to another machine.

These fingerprints cover the named protocol/calibration files. They do not
claim to hash every image, video or trajectory in an export.
"""
import hashlib
import json
from pathlib import Path

IDENTITY_FILES = (
    "piper_collection_config.json",
    "piper_camera_calibration.json",
    "piper_temporal_contract.json",
)


def dataset_identity(directory):
    directory = Path(directory)
    values = {}
    for name in IDENTITY_FILES:
        value = json.loads((directory / "meta" / name).read_text())
        payload = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        values[name] = hashlib.sha256(payload).hexdigest()
    return {"version": "piper_dataset_metadata_v1", "sha256": values}


def verify_identity(actual, expected):
    if not expected:
        return
    if actual != expected:
        raise ValueError("Dataset content identity differs from the policy's training reference")
