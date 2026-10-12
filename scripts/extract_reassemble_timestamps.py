#!/usr/bin/env python3
"""Fetch one original REASSEMBLE HDF5 member via HTTP ranges and export camera clocks."""
from __future__ import annotations

import argparse
import io
import json
import struct
import tempfile
import zipfile
import zlib
from pathlib import Path

import h5py
import numpy as np
import requests

ROOT = Path('/projects/hdd/myproject/CVPR2027')
DATASET = ROOT / 'datasets/reassemble'
URL = 'https://researchdata.tuwien.ac.at/api/records/0ewrv-8cb44/files/data.zip/content'
CAMERAS = ('hand', 'hama1', 'hama2', 'capture_node-camera-image')


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str):
        self.url = url
        self.position = 0
        response = requests.head(url, timeout=45)
        response.raise_for_status()
        self.size = int(response.headers['Content-Length'])
        if response.headers.get('Accept-Ranges') != 'bytes':
            raise RuntimeError('remote archive does not advertise byte-range access')

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError('negative seek')
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size == 0:
            return b''
        start, end = self.position, self.position + size - 1
        response = requests.get(self.url, headers={'Range': f'bytes={start}-{end}'}, timeout=90)
        if response.status_code != 206 or len(response.content) != size:
            raise IOError(f'incorrect HTTP range response: {response.status_code}, {len(response.content)} bytes')
        self.position += size
        return response.content


def extract_member(remote: HTTPRangeFile, member: zipfile.ZipInfo, target: Path) -> None:
    remote.seek(member.header_offset)
    header = remote.read(30)
    fields = struct.unpack('<IHHHHHIIIHH', header)
    if fields[0] != 0x04034B50 or fields[3] != zipfile.ZIP_DEFLATED:
        raise ValueError('unexpected ZIP member header or compression')
    start = member.header_offset + 30 + fields[9] + fields[10]
    end = start + member.compress_size - 1
    with requests.get(remote.url, headers={'Range': f'bytes={start}-{end}'}, timeout=120, stream=True) as response:
        if response.status_code != 206:
            raise IOError(f'HTTP range request failed: {response.status_code}')
        decoder = zlib.decompressobj(-15)
        crc = 0
        total = 0
        with target.open('wb') as output:
            for block in response.iter_content(chunk_size=1 << 20):
                if not block:
                    continue
                plain = decoder.decompress(block)
                if plain:
                    output.write(plain)
                    crc = zlib.crc32(plain, crc)
                    total += len(plain)
            plain = decoder.flush()
            output.write(plain)
            crc = zlib.crc32(plain, crc)
            total += len(plain)
    if total != member.file_size or crc != member.CRC:
        raise IOError(f'ZIP verification failed: size {total}/{member.file_size}, CRC {crc}/{member.CRC}')


def nearest(src: np.ndarray, queries: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(src, queries), 1, len(src) - 1)
    left, right = src[idx - 1], src[idx]
    idx = idx - ((queries - left) < (right - queries))
    return np.clip(idx, 0, len(src) - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'tmp/reassemble_alignment')
    parser.add_argument('--max-compressed-mib', type=int, default=128)
    args = parser.parse_args()
    mapping = json.loads((DATASET / 'meta/splits.json').read_text())
    recording = mapping[str(args.episode)]['recording']
    remote = HTTPRangeFile(URL)
    with zipfile.ZipFile(remote) as archive:
        matches = [entry for entry in archive.infolist() if entry.filename.endswith('/' + recording + '.h5')]
        if len(matches) != 1:
            raise ValueError(f'expected one original HDF5 for episode {args.episode}, got {len(matches)}')
        member = matches[0]
    if member.compress_size > args.max_compressed_mib * 2**20:
        raise RuntimeError(f'{recording} needs {member.compress_size / 2**20:.1f} MiB; increase --max-compressed-mib to allow')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix='.h5', dir=args.output_dir, delete=True) as tmp:
        print(f'fetching {recording}: {member.compress_size / 2**20:.1f} MiB compressed', flush=True)
        extract_member(remote, member, Path(tmp.name))
        with h5py.File(tmp.name) as h5:
            group = h5['timestamps']
            master = next((name for name in CAMERAS if name in h5 and name in group), None)
            if master is None:
                raise RuntimeError('original recording has no master-camera timestamps')
            t0 = float(group[master][0])
            t1 = float(group[master][-1])
            n = max(int((t1 - t0) * 30) + 1, 2)
            grid = t0 + np.arange(n, dtype=np.float64) / 30
            arrays = {}
            cameras = {}
            for name in CAMERAS:
                if name not in group:
                    continue
                timestamps = np.asarray(group[name], dtype=np.float64).ravel()
                if not len(timestamps):
                    continue
                arrays[name] = timestamps
                indices = nearest(timestamps, grid) if len(timestamps) > 1 else np.zeros(n, dtype=np.int64)
                err_ms = 1000 * (timestamps[indices] - grid)
                cameras[name] = {
                    'frames': len(timestamps), 'first': float(timestamps[0]), 'last': float(timestamps[-1]),
                    'nearest_error_ms_p50': float(np.median(np.abs(err_ms))),
                    'nearest_error_ms_p95': float(np.percentile(np.abs(err_ms), 95)),
                    'nearest_error_ms_max': float(np.max(np.abs(err_ms))),
                }
            if 'events' in group:
                event_ts = group['events']
                event_clock = {'count': len(event_ts), 'first': float(event_ts[0]), 'last': float(event_ts[-1])}
            else:
                event_clock = None
    output = args.output_dir / f'episode_{args.episode:06d}'
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'camera_timestamps.npz', **arrays)
    report = {
        'episode': args.episode, 'recording': recording, 'source': URL, 'source_member': member.filename,
        'source_member_compressed_bytes': member.compress_size,
        'master_camera': master, 'absolute_t0': t0, 'absolute_t1': t1,
        'grid_fps': 30, 'grid_frames': n, 'cameras': cameras, 'events': event_clock,
    }
    (output / 'timing.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
