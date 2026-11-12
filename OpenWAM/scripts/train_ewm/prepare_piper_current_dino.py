"""Resume full-grid current DINO caching; torchrun may split tasks across GPUs.

Source videos and the existing pooled history cache are never modified.
Each completed episode has a receipt; a manifest is published only after
independent re-encoding checks pass for every episode.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'OpenWAM'))


def write_json(path, value):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def encode_fixed_batch(dataset, row, indices, encoder, device, batch_size):
    """Always encode B frame pairs; padding is repeated input, never stored.

    Use the identical shape for cache generation and independent verification.
    In particular, a one-frame remainder must not select a different GEMM path.
    The caller controls inference mode and autocast, as in online training.
    """
    import torch
    from openwam.model.architectures.ewm import EWMContextEncoder
    if batch_size < 1 or not len(indices):
        raise ValueError('positive fixed batch size and nonempty frame indices required')
    dtype = next(encoder.parameters()).dtype
    outputs = []
    for start in range(0,len(indices),batch_size):
        chunk = indices[start:start+batch_size]
        rgb = dataset._rgb(row,chunk)
        count = len(chunk)
        if count < batch_size:
            rgb = torch.cat((rgb,rgb[-1:].expand(batch_size-count,*rgb.shape[1:])),0)
        pixels = EWMContextEncoder.preprocess_rgb(rgb,device=device,dtype=dtype)
        grid = encoder.batch_encode(pixels)[:, :, 0]
        if (grid.dtype not in (torch.float32,torch.bfloat16)
                or grid.shape != (batch_size*2,1024,16,16)):
            raise ValueError(f'unexpected online DINO representation: {grid.dtype}, {grid.shape}')
        outputs.append(grid.reshape(batch_size,2,1024,16,16)[:count].cpu().contiguous())
    return torch.cat(outputs,0)


def prepare_task(root, cache, task, encoder, device, batch_size):
    import numpy as np
    import torch
    from openwam.dataloader.ewm_dataset import EWMDataset
    from openwam.dataloader.current_dino_cache import dino_fingerprint, source_signature

    target = cache/f'piper_{task}/dino_current16'
    target.mkdir(parents=True, exist_ok=True)
    with (target/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        dataset = EWMDataset(root/f'datasets/piper_{task}', chunk_size=4, voxel_bins=1,
                             event_camera_mode='dual', gpu_voxelization=False)
        spec = dict(format='ewm_dino_current_bf16_v1', grid=16, cameras=2, dim=1024,
                    dtype='bfloat16_uint16_bits',
                    encoding_batch_size=batch_size,
                    padding='repeat_last_frame_pair_discard_padding',
                    dino_fingerprint=dino_fingerprint(root/'pretrained/dinov3-vitl16'),
                    preprocessing='EWM ImageNet normalization; independent T1; full grid')
        manifest = dict(spec, episodes={})
        def encode(row, indices):
            return encode_fixed_batch(dataset,row,indices,encoder,device,batch_size)

        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for _,row in dataset.episodes.iterrows():
                ep,length = int(row['episode_index']),int(row['length'])
                name = f'episode_{ep:06d}.npy'
                entry = dict(file=name, length=length, source=source_signature(dataset,row))
                receipt = target/f'episode_{ep:06d}.json'
                path = target/name
                if receipt.exists():
                    saved = json.loads(receipt.read_text())
                    if saved.get('spec') != spec or saved.get('entry') != entry or not saved.get('verified'):
                        raise ValueError(f'stale current DINO receipt: {receipt}')
                    array = np.load(path, mmap_mode='r', allow_pickle=False)
                    if array.shape != (length,2,1024,16,16) or array.dtype != np.uint16:
                        raise ValueError(f'invalid completed cache: {path}')
                    manifest['episodes'][str(ep)] = entry
                    continue
                partial = path.with_suffix('.partial.npy')
                array = np.lib.format.open_memmap(partial, mode='w+', dtype=np.uint16,
                                                 shape=(length,2,1024,16,16))
                for start in range(0,length,batch_size):
                    stop = min(length,start+batch_size)
                    array[start:stop] = encode(row,np.arange(start,stop)).to(torch.bfloat16).view(torch.uint16).numpy()
                array.flush()
                indices = np.array([0,length//2,length-1])
                fresh = encode(row,indices)
                stored = torch.from_numpy(array[indices].copy()).view(torch.bfloat16)
                # Keep the original tolerances; fixed shape removes the ragged-batch change.
                torch.testing.assert_close(stored.float(),fresh.float(),rtol=.01,atol=.02)
                error = float((stored.float()-fresh.float()).abs().max())
                if entry['source'] != source_signature(dataset,row):
                    raise ValueError('source changed while caching')
                del array
                partial.replace(path)
                write_json(receipt,dict(spec=spec,entry=entry,verified=True,
                                       checked_frames=indices.tolist(),max_abs_error=error))
                manifest['episodes'][str(ep)] = entry
                print('CURRENT_DINO_VERIFIED',task,ep,length,error,flush=True)
        write_json(target/'manifest.json',manifest)
        print('CURRENT_DINO_COMPLETE',task,len(manifest['episodes']),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--cache-root',type=Path)
    p.add_argument('--batch-size',type=int,default=32)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('batch size must be positive')
    import torch
    from openwam.dataloader.piper_training_data import DYNAMIC_TASKS
    from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder
    root = args.root.resolve()
    cache = (args.cache_root or root/'tmp/training_data_cache/piper_dynamic_v1').resolve()
    if cache == root/'datasets' or root/'datasets' in cache.parents:
        raise ValueError('derived caches must be outside source datasets')
    rank,world = int(os.environ.get('RANK',0)),int(os.environ.get('WORLD_SIZE',1))
    local = int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    device = torch.device('cuda',local)
    encoder = DinoV3VideoEncoder.from_pretrained(str(root/'pretrained/dinov3-vitl16'))
    encoder = encoder.to(device).eval().requires_grad_(False)
    for task in DYNAMIC_TASKS[rank::world]:
        prepare_task(root,cache,task,encoder,device,args.batch_size)


if __name__ == '__main__':
    main()
