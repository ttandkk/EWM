#!/usr/bin/env python3
"""Held-out PCA reconstruction decoded by frozen LeanVAE; raw-input metrics.

The fitted basis and split are read from another analysis; no refitting occurs here.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]


def setup():
    for name, path in [('openwam', ROOT/'OpenWAM/openwam'),
                       ('openwam.model', ROOT/'OpenWAM/openwam/model'),
                       ('openwam.model.architectures', ROOT/'OpenWAM/openwam/model/architectures'),
                       ('openwam.model.video_backbone', ROOT/'OpenWAM/openwam/model/video_backbone'),
                       ('openwam.model.video_backbone.encoder', ROOT/'OpenWAM/openwam/model/video_backbone/encoder')]:
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod
    from openwam.model.architectures.lean_event_codec import LeanEventCodec
    from openwam.model.architectures.lean_rgb_codec import LeanRGBCodec
    spec = importlib.util.spec_from_file_location('direct_video_io', ROOT/'OpenWAM/openwam/dataloader/utils/video_io.py')
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)
    codec = LeanEventCodec(ROOT/'pretrained/leanvae', batch_size=1).eval()
    return codec, LeanRGBCodec(vae=codec.vae.model, batch_size=1).eval(), io


def metrics(pred, orig, baseline, is_event):
    # Only future input frames: index 0 is the original, uncompressed observation.
    pred = pred[1:].astype(np.float64)
    orig = orig[1:].astype(np.float64)
    if baseline is None:
        baseline = pred
    else:
        baseline = baseline[1:].astype(np.float64)
    diff = pred-orig
    m = {'input_mse': float(np.mean(diff**2)),
         'input_mae': float(np.mean(np.abs(diff))),
         'decoder_delta_mse': float(np.mean((pred-baseline)**2))}
    if is_event:
        mask = orig > 0
        m['active_fraction'] = float(np.mean(mask))
        m['active_mse'] = float(np.mean(diff[mask]**2)) if mask.any() else None
        # Fixed threshold on decoded normalized count >= 1/15. This is a proxy,
        # not a physical event-count metric or a matched sensor detection test.
        detected = pred >= 1/15
        tp = np.count_nonzero(detected & mask)
        fp = np.count_nonzero(detected & ~mask)
        fn = np.count_nonzero(~detected & mask)
        m['tp'], m['fp'], m['fn'] = int(tp), int(fp), int(fn)
        m['precision'] = float(tp/(tp+fp)) if tp+fp else None
        m['recall'] = float(tp/(tp+fn)) if tp+fn else None
    return m


def image(arr, event):
    if event:
        n, p = arr
        arr = np.stack([p, (n+p)*.5, n], -1)
    else:
        arr = arr.transpose(1, 2, 0)
    return Image.fromarray(np.uint8(np.clip(arr, 0, 1)*255+.5), 'RGB')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--latent', type=Path, required=True)
    p.add_argument('--pca', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=2)
    p.add_argument('--max-windows', type=int, default=0, help='0 means all windows of selected episodes')
    p.add_argument('--ks', nargs='+', type=int, default=[0,8,16,32,64])
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    zdata = np.load(args.latent)
    pca = np.load(args.pca)
    for key in ('episode', 'anchor'):
        if not np.array_equal(zdata[key], pca[key]):
            raise ValueError(f'{key}: latent and PCA inputs differ')
    test_idx = pca['test_indices']
    train_idx = pca['train_indices']
    if set(zdata['episode'][test_idx]) & set(zdata['episode'][train_idx]):
        raise ValueError('train/test trajectory leakage')
    test_eps = np.unique(zdata['episode'][test_idx])[:args.episodes]
    chosen = test_idx[np.isin(zdata['episode'][test_idx], test_eps)]
    if args.max_windows:
        chosen = chosen[:args.max_windows]
    codec, rgb_codec, io = setup()
    task = 'tennis_falling'
    base = ROOT/'datasets'/('piper_'+task)
    rows = pq.read_table(sorted((base/'meta/episodes').rglob('*.parquet'))).to_pandas().sort_values('episode_index')
    rows['data_offset'] = rows['dataset_from_index'] - rows.groupby(['data/chunk_index','data/file_index'])['dataset_from_index'].transform('min')
    rows = {int(row['episode_index']): row for row in rows.to_dict('records')}
    clip = json.loads((ROOT/'tmp/training_data_cache/piper_dynamic_v1/shared_normalization.json').read_text())['voxel']['count_clip']
    results = []
    start = time.monotonic()
    for idx in chosen:
        ep, anchor = int(zdata['episode'][idx]), int(zdata['anchor'][idx])
        row = rows[ep]
        event_path = ROOT/'tmp/training_data_cache/piper_dynamic_v1'/('piper_'+task)/'event_frames/data'/f"chunk-{int(row['data/chunk_index']):03d}"/f"file-{int(row['data/file_index']):03d}.npy"
        video = base/'videos/observation.images.cam0'/f"chunk-{int(row['videos/observation.images.cam0/chunk_index']):03d}"/f"file-{int(row['videos/observation.images.cam0/file_index']):03d}.mp4"
        ids = np.arange(anchor, anchor+25, 2, dtype=np.int64)
        offset = int(round(float(row['videos/observation.images.cam0/from_timestamp'])*30))
        ev = np.load(event_path, mmap_mode='r')[int(row['data_offset'])+ids,0].astype(np.float32)
        ev = np.clip(ev, 0, clip)/clip
        rgb = np.stack([np.asarray(im, dtype=np.uint8) for im in io.decode_video_frames(str(video), (offset+ids).tolist(), 256, 256)]).transpose(0,3,1,2).astype(np.float32)/255
        frame = int(np.argmax((ev[1:]>0).sum(axis=(1,2,3))))+1
        print(f'episode={ep} anchor={anchor} frame={frame}', flush=True)
        for modality, original, c in [('event', ev, codec), ('rgb', rgb, rgb_codec)]:
            z = zdata[modality][idx].astype(np.float64)
            future = z[1:].reshape(-1)
            mu, basis = pca[modality+'_mean'], pca[modality+'_basis']
            if basis.shape[1] != future.size or mu.shape != future.shape:
                raise ValueError(f'{modality} PCA shape mismatch')
            if max(args.ks) > len(basis):
                raise ValueError('K exceeds fitted PCA rank')
            coeff = basis @ (future-mu)
            baseline = None
            tiles = [('input', image(original[frame], modality == 'event'))]
            for k in [None]+args.ks:
                work = z.copy()
                if k is not None:
                    work[1:] = (mu+coeff[:k] @ basis[:k]).reshape(work[1:].shape)
                latent = torch.from_numpy(work.astype(np.float32))[None,:,None]
                pred = c.decode(latent)[0,:,0].cpu().numpy()
                m = metrics(pred, original, baseline, modality == 'event')
                if k is None:
                    baseline = pred
                results.append({'episode':ep,'anchor':anchor,'modality':modality,
                                'k':k,'metrics':m})
                if idx == chosen[0]:
                    tiles.append(('full' if k is None else f'K={k}', image(pred[frame], modality == 'event')))
            if idx == chosen[0]:
                canvas = Image.new('RGB', (256*len(tiles), 280), 'white')
                draw = ImageDraw.Draw(canvas)
                for i, (label, tile) in enumerate(tiles):
                    draw.text((256*i+5, 3), label, fill='black')
                    canvas.paste(tile, (256*i,24))
                canvas.save(args.output/f'{modality}_example_ep{ep}_anchor{anchor}_frame{frame}.png')
        print(f'done {time.monotonic()-start:.1f}s', flush=True)
    result = {'task':task,'protocol':'frozen LeanVAE, held-out episodes, 13 temporally sampled frames, metrics on 12 future frames, current latent block retained',
              'latent_file':str(args.latent),'pca_file':str(args.pca),'event_clip':clip,
              'train_episode_count':len(np.unique(zdata['episode'][train_idx])), 'test_episode_count':len(np.unique(zdata['episode'][test_idx])),
              'evaluated_test_episodes':np.unique(zdata['episode'][chosen]).tolist(),'evaluated_window_count':len(chosen),
              'ks':args.ks,'metrics':results,'runtime_seconds':time.monotonic()-start}
    path = args.output/'decoder_results.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2))
    print(f'wrote {path}', flush=True)

if __name__ == '__main__':
    main()
