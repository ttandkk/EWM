"""Cache frozen per-frame DINO semantic grids for RGB4/event10 training."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if (ROOT/'openwam').is_dir():sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--dino-path',required=True,type=Path)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-size',type=int,default=16,help='Frame pairs per encoder batch')
    args=parser.parse_args()
    if args.batch_size<=0:parser.error('--batch-size must be positive')
    import numpy as np
    import torch
    from openwam.dataloader.ewm_dataset import EWMDataset
    from openwam.dataloader.hierarchical_dataset import dino_fingerprint,source_signature
    from openwam.model.architectures.ewm import EWMContextEncoder
    from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder
    from torch.nn import functional as F
    if (args.output/'manifest.json').exists():
        raise FileExistsError('refusing to replace a completed DINO cache')
    args.output.mkdir(parents=True,exist_ok=True)
    dataset=EWMDataset(args.dataset,chunk_size=4,voxel_bins=1,event_camera_mode='dual',gpu_voxelization=False)
    encoder=DinoV3VideoEncoder.from_pretrained(str(args.dino_path)).to(args.device).requires_grad_(False).eval()
    device=torch.device(args.device)
    input_dtype=next(encoder.parameters()).dtype
    width=encoder.properties.z_dim
    manifest=dict(format='ewm_dino_history_bf16_v1',grid=8,cameras=2,dim=width,
        dtype='bfloat16_uint16_bits',dino_fingerprint=dino_fingerprint(args.dino_path),
        preprocessing='EWM ImageNet normalization; independent T1; area pool8',episodes={})
    with torch.inference_mode(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        for _,row in dataset.episodes.iterrows():
            ep,length=int(row['episode_index']),int(row['length'])
            name=f'episode_{ep:06d}.npy'
            target=args.output/name
            if target.exists():raise FileExistsError(f'refusing to replace {target}; use a new cache directory')
            array=np.lib.format.open_memmap(target,mode='w+',dtype=np.uint16,shape=(length,2,width,8,8))
            for start in range(0,length,args.batch_size):
                end=min(length,start+args.batch_size)
                rgb=dataset._rgb(row,np.arange(start,end))
                pixels=EWMContextEncoder.preprocess_rgb(rgb,device=device,dtype=input_dtype)
                grid=encoder.batch_encode(pixels)[:,:,0]
                pooled=F.adaptive_avg_pool2d(grid,(8,8)).reshape(end-start,2,width,8,8)
                array[start:end]=pooled.to(dtype=torch.bfloat16,device='cpu').contiguous().view(torch.uint16).numpy()
            array.flush();del array
            manifest['episodes'][str(ep)]=dict(file=name,length=length,source=source_signature(dataset,row))
            print(f'cached episode {ep}: {length} frame pairs',flush=True)
    temporary=args.output/'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest,indent=2)+'\n')
    temporary.replace(args.output/'manifest.json')
    print(f'completed {len(manifest["episodes"])} episodes: {args.output}',flush=True)


if __name__=='__main__':main()
