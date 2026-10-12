"""Losslessly predecode aligned event PNGs into row-indexed uint8 .npy files.

The cache is optional. Its manifest is written last, so an interrupted build is
never mistaken for a complete cache by EWMDataset.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq


CAMERAS = ("cam0", "cam_wrist")


def build(dataset_dir: Path, cache_dir: Path):
    if not (dataset_dir / "meta" / "piper_temporal_contract.json").exists():
        raise ValueError("dataset is missing the Piper temporal contract")
    sources = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not sources:
        raise ValueError("dataset has no Parquet data files")
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "ewm_raw_event_frames_v1", "cameras": list(CAMERAS), "files": {}}
    for source in sources:
        relative = source.relative_to(dataset_dir)
        cache_relative = relative.with_suffix(".npy")
        target = cache_dir / cache_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.stem + ".tmp.npy")
        columns = [f"observation.event_frame.{camera}" for camera in CAMERAS]
        table = pq.read_table(source, columns=columns)
        array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8,
                                          shape=(table.num_rows, 2, 2, 256, 256))
        for camera_index, column_name in enumerate(columns):
            column = table[column_name]
            for row_index in range(table.num_rows):
                encoded = column[row_index].as_py()
                if encoded["bytes"] is None:
                    raise ValueError(f"missing event PNG in {relative} row {row_index}")
                with Image.open(BytesIO(encoded["bytes"])) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                if rgb.shape != (256, 256, 3) or np.any(rgb[..., 2]):
                    raise ValueError(f"invalid event PNG in {relative} row {row_index}")
                array[row_index, camera_index, 0] = rgb[..., 0]
                array[row_index, camera_index, 1] = rgb[..., 1]
            print(f"decoded {relative} {column_name}: {table.num_rows} rows", flush=True)
        array.flush()
        del array, table
        os.replace(temporary, target)
        source_stat = source.stat()
        manifest["files"][relative.as_posix()] = {
            "cache_file": cache_relative.as_posix(), "rows": pq.ParquetFile(source).metadata.num_rows,
            "source_size": source_stat.st_size, "source_mtime_ns": source_stat.st_mtime_ns,
        }
    temporary_manifest = cache_dir / "manifest.tmp.json"
    temporary_manifest.write_text(json.dumps(manifest, indent=2))
    os.replace(temporary_manifest, cache_dir / "manifest.json")
    print(f"ready: {cache_dir / 'manifest.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.dataset_dir, args.cache_dir)


if __name__ == "__main__":
    main()
