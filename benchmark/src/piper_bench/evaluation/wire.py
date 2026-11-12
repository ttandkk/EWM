"""Pickle-free binary WebSocket messages for EWM evaluation."""
from __future__ import annotations

import json
import struct

import numpy as np

MAX_HEADER = 64 * 1024
MAX_ARRAY = 128 * 1024 * 1024
MAX_MESSAGE = 160 * 1024 * 1024
DTYPES = {"uint8", "float32", "bool", "int64"}


def encode_message(metadata: dict, arrays: dict[str, np.ndarray] | None = None) -> bytes:
    arrays = arrays or {}
    specs = []
    payloads = []
    for name, value in arrays.items():
        value = np.asarray(value)
        if not value.flags.c_contiguous:
            value = np.ascontiguousarray(value)
        if value.dtype.name not in DTYPES or value.nbytes > MAX_ARRAY:
            raise ValueError(f"unsupported evaluation array {name}: {value.dtype}, {value.nbytes} bytes")
        specs.append({"name": name, "dtype": value.dtype.name,
                      "shape": list(value.shape), "nbytes": value.nbytes})
        payloads.append(value.tobytes(order="C"))
    header = json.dumps({**metadata, "arrays": specs}, separators=(",", ":")).encode("utf-8")
    if not 0 < len(header) <= MAX_HEADER:
        raise ValueError("evaluation header too large")
    message = struct.pack("!I", len(header)) + header + b"".join(payloads)
    if len(message) > MAX_MESSAGE:
        raise ValueError("evaluation message too large")
    return message


def decode_message(message: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    if not isinstance(message, bytes) or not 4 <= len(message) <= MAX_MESSAGE:
        raise ValueError("invalid evaluation binary message")
    header_size = struct.unpack("!I", message[:4])[0]
    if not 0 < header_size <= MAX_HEADER or len(message) < 4 + header_size:
        raise ValueError("invalid evaluation header size")
    metadata = json.loads(message[4:4 + header_size])
    specs = metadata.pop("arrays", [])
    arrays = {}
    offset = 4 + header_size
    for spec in specs:
        name, dtype, shape, nbytes = spec["name"], spec["dtype"], spec["shape"], spec["nbytes"]
        if name in arrays or dtype not in DTYPES or not isinstance(shape, list):
            raise ValueError("invalid evaluation array descriptor")
        if not isinstance(nbytes, int) or not 0 <= nbytes <= MAX_ARRAY:
            raise ValueError("invalid evaluation array length")
        if any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError("invalid evaluation array shape")
        data_type = np.dtype(dtype)
        expected = data_type.itemsize
        for n in shape:
            expected *= n
            if expected > MAX_ARRAY:
                raise ValueError("evaluation array too large")
        if expected != nbytes or offset + nbytes > len(message):
            raise ValueError("evaluation array shape/length mismatch")
        arrays[name] = np.frombuffer(message, dtype=data_type, count=nbytes // data_type.itemsize,
                                     offset=offset).reshape(shape).copy()
        offset += nbytes
    if offset != len(message):
        raise ValueError("trailing evaluation message bytes")
    return metadata, arrays
