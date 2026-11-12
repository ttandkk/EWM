"""Portable, content-addressed task definitions shipped with the benchmark."""
import hashlib
import json
from importlib.resources import files

DEFAULT_SUITE = "piper_eight_v1"


def load_suite(name=DEFAULT_SUITE):
    if name != DEFAULT_SUITE:
        raise ValueError(f"Unknown suite {name!r}; available: {DEFAULT_SUITE}")
    return json.loads(files("piper_bench").joinpath("resources", "suites", name + ".json").read_text())


def load_collection(task, suite=DEFAULT_SUITE):
    definition = load_suite(suite)
    try:
        entry = definition["tasks"][task]
    except KeyError as exc:
        raise ValueError(f"Task {task!r} is not part of suite {suite}") from exc
    resource = files("piper_bench").joinpath("resources", "suites", "collections", task + ".json")
    payload = resource.read_bytes()
    if hashlib.sha256(payload).hexdigest() != entry["collection_file_sha256"]:
        raise ValueError(f"Packaged collection metadata changed for {task}")
    collection = json.loads(payload)
    if collection["version"] != entry["collection_protocol"] or collection["task"] != task:
        raise ValueError(f"Suite/collection version mismatch for {task}")
    return collection
