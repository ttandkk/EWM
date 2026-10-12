"""Render aligned LeRobot RGB and measured OFF/ON event frames as four horizontal panels."""
from __future__ import annotations
import argparse,json,subprocess
from io import BytesIO
from pathlib import Path
import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image,ImageDraw,ImageFont


def white_redblue(event):
    """Show measured OFF in blue, ON in red, and overlapping polarities in purple."""
    off,on=event[:,:,0]>0,event[:,:,1]>0
    visual=np.full_like(event,255)
    visual[off]=(45,65,220)
    visual[on]=(225,45,45)
    visual[off&on]=(160,45,180)
    return Image.fromarray(visual)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--episode',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--panel-size',type=int,default=256)
    parser.add_argument('--allow-probe',action='store_true')
    args=parser.parse_args();root=args.dataset.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    if args.panel_size<64 or args.panel_size%2:raise ValueError('panel size must be even and at least 64')
    validation_path=root/'meta/piper_validation.json'
    validation=json.loads(validation_path.read_text())
    if not validation['passed']:raise ValueError('dataset validation did not pass')
    if validation['probe_only'] and not args.allow_probe:raise ValueError('a probe is excluded from successful trajectory previews')
    rows=[]
    for p in sorted((root/'data').rglob('*.parquet')):
        for row in pq.read_table(p).to_pylist():
            if row['episode_index']==args.episode:rows.append(row)
    rows.sort(key=lambda row:row['frame_index'])
    if not rows or [r['frame_index'] for r in rows]!=list(range(len(rows))):raise ValueError('missing or noncontiguous episode')
    metadata=[]
    for p in sorted((root/'meta/episodes').rglob('*.parquet')):metadata.extend(pq.read_table(p).to_pylist())
    episode=next(r for r in metadata if r['episode_index']==args.episode)
    info=json.loads((root/'meta/info.json').read_text())
    cameras=('cam0','cam_wrist');containers=[];readers=[];offsets=[]
    for camera in cameras:
        key='observation.images.'+camera
        path=root/info['video_path'].format(video_key=key,chunk_index=episode[f'videos/{key}/chunk_index'],file_index=episode[f'videos/{key}/file_index'])
        container=av.open(str(path));containers.append(container);reader=container.decode(video=0)
        offset=int(round(float(episode.get(f'videos/{key}/from_timestamp',0))*info['fps']))
        for _ in range(offset):next(reader)
        readers.append(reader);offsets.append(offset)
    output.parent.mkdir(parents=True,exist_ok=True)
    project=Path(__file__).resolve().parents[2]
    ffmpeg=project/'envs/torchcodec_ffmpeg7/bin/ffmpeg';ffprobe=project/'envs/torchcodec_ffmpeg7/bin/ffprobe'
    size=args.panel_size
    video=subprocess.Popen([str(ffmpeg),'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{4*size}x{size}','-r',str(info['fps']),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',max(10,round(size/24)))
    except OSError:font=ImageFont.load_default()
    indices=sorted(set(min(round(t*info['fps']),len(rows)-1) for t in (0,2,4,6,8,12,14,16,18,22,(len(rows)-1)/info['fps'])))
    keyframes=[]
    try:
        for k,row in enumerate(rows):
            panels=[]
            for camera,reader in zip(cameras,readers):
                rgb=next(reader).to_image().convert('RGB')
                event=np.asarray(Image.open(BytesIO(row[f'observation.event_frame.{camera}']['bytes'])).convert('RGB'))
                if event.shape!=(256,256,3) or event[:,:,:2].max()>15 or np.any(event[:,:,2]):raise ValueError('invalid event encoding')
                panels.extend((rgb.resize((size,size),Image.Resampling.BICUBIC),white_redblue(event).resize((size,size),Image.Resampling.NEAREST)))
            canvas=Image.new('RGB',(4*size,size),'white')
            labels=('cam0 RGB','cam0 Events','cam_wrist RGB','cam_wrist Events')
            for i,panel in enumerate(panels):
                canvas.paste(panel,(i*size,0))
                ImageDraw.Draw(canvas).text((i*size+8,5),labels[i],font=font,fill='white' if i%2==0 else 'black',stroke_width=1,stroke_fill='black' if i%2==0 else 'white')
            draw=ImageDraw.Draw(canvas);draw.text((8,size-max(18,round(size/17))),f"{row['timestamp']:.2f}s",font=font,fill='white',stroke_width=1,stroke_fill='black')
            for x in (size,2*size,3*size):draw.line((x,0,x,size),fill=(220,220,220),width=1)
            video.stdin.write(canvas.tobytes())
            if k in indices:
                keyframes.append(canvas.copy());canvas.save(output.with_name(output.stem+f'_frame_{k:04d}.png'))
    finally:
        video.stdin.close()
        if video.wait(timeout=60):raise RuntimeError('video encoding failed')
        for container in containers:container.close()
    subprocess.run([str(ffmpeg),'-v','error','-i',str(output),'-f','null','-'],check=True)
    probe=json.loads(subprocess.check_output([str(ffprobe),'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,avg_frame_rate','-of','json',str(output)]))['streams'][0]
    assert int(probe['nb_read_frames'])==len(rows)
    sheet=Image.new('RGB',(4*size,size*len(keyframes)))
    for i,frame in enumerate(keyframes):sheet.paste(frame,(0,i*size))
    sheet.save(output.with_name(output.stem+'_contact_sheet.png'))
    report=dict(dataset=str(root),episode=args.episode,frames=len(rows),fps=info['fps'],duration_seconds=len(rows)/info['fps'],
                panels=list(labels),event_source='measured_EVIS_OFF_ON_lossless_frames',event_display='white background, OFF blue, ON red, overlap purple',
                full_decode_passed=True,video=probe,probe_only=validation['probe_only'])
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
