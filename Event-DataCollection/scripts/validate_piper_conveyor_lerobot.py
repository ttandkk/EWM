"""Validate real conveyor LeRobot data, event encoding, timing and physical terminal success."""
import argparse,json,subprocess
from io import BytesIO
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import piper_conveyor
from piper_conveyor_speed import sample_episode_speed
from piper_conveyor import CAMERAS,TASK,PROMPT,ConveyorConfig,camera_metadata,success_now


def require(condition,message):
    if not condition:raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path)
    p.add_argument('--episodes',type=int,required=True);p.add_argument('--allow-probe',action='store_true')
    args=p.parse_args();root=args.dataset.resolve();meta=root/'meta'
    info=json.loads((meta/'info.json').read_text());contract=json.loads((meta/'piper_temporal_contract.json').read_text())
    cal=json.loads((meta/'piper_camera_calibration.json').read_text());collection=json.loads((meta/'piper_collection_config.json').read_text())
    require(info['total_episodes']==args.episodes,'wrong episode count')
    require(info['fps']==30 and contract['control_hz']==30,'wrong clock')
    require(contract['task_id']==TASK and contract['object_count']==2,'wrong task')
    require(contract['temporal_contract']=='aligned_event_frame_pre_action_v1','wrong temporal alignment')
    require(cal==camera_metadata(),'calibration differs from approved scene')
    require(args.allow_probe or not collection['probe_only'],'interface probe is not demonstration data')
    require((root/'.collection_complete').is_file(),'collection did not finish')
    dims={'state':8,'joint_vel':8,'ee_pose':7,'object_pose':14,'object_velocity':12,'object_active':2,'sim_time':1}
    for key,size in dims.items():require(info['features']['observation.'+key]['shape']==[size],f'wrong {key} shape')
    require(info['features']['action']['shape']==[7],'wrong action shape')
    for name in CAMERAS:
        require(info['features']['observation.images.'+name]['dtype']=='video','RGB must be video')
        require(info['features']['observation.event_frame.'+name]['dtype']=='image','events must be lossless PNG')
    counts={i:0 for i in range(args.episodes)};event_mass={name:0 for name in CAMERAS};total=0
    for path in sorted((root/'data').rglob('*.parquet')):
        table=pq.read_table(path)
        for batch in table.to_batches(max_chunksize=256):
            for row in batch.to_pylist():
                episode=int(row['episode_index']);require(episode in counts,'unknown episode')
                k=counts[episode];require(row['frame_index']==k,'noncontiguous frame index')
                require(abs(row['timestamp']-k/30)<1e-5,'timestamp mismatch')
                require(abs(float(np.asarray(row['observation.sim_time']).reshape(-1)[0])-k/30)<1e-5,'simulation clock mismatch')
                for name,size in dims.items():
                    a=np.asarray(row['observation.'+name]);require(a.size==size and np.isfinite(a).all(),f'invalid {name}')
                require(np.isfinite(row['action']).all(),'nonfinite action')
                active=np.asarray(row['observation.object_active']);require(np.array_equal(active,[1.,float(k/30+1e-8>=collection['scene_config']['spawn_interval'])]),'supply schedule differs')
                for name in CAMERAS:
                    payload=row['observation.event_frame.'+name]
                    event=np.asarray(Image.open(BytesIO(payload['bytes'])).convert('RGB'))
                    require(event.shape==(256,256,3) and event.dtype==np.uint8,'wrong event shape/dtype')
                    require(event[:,:,:2].max()<=15 and not np.any(event[:,:,2]),'invalid OFF/ON count encoding')
                    if k==0:require(not np.any(event),'first event frame must be zero')
                    event_mass[name]+=int(event[:,:,:2].sum())
                counts[episode]+=1;total+=1
    require(all(n>=25 for n in counts.values()),'episodes must support current1+future24')
    require(total==info['total_frames'],'frame count mismatch')
    attempts=json.loads((meta/'piper_conveyor_attempts.json').read_text())
    saved={a['saved_episode']:a for a in attempts if 'saved_episode' in a}
    require(set(saved)==set(counts),'missing episode attempt metadata')
    speed_spec=collection.get('speed_sampling')
    if speed_spec is not None:
        for episode,a in saved.items():
            expected=sample_episode_speed(speed_spec,collection['seed'],episode)
            require(a.get('speed_episode_index')==episode,'wrong speed episode index')
            require(a.get('conveyor_speed_m_s')==expected and a['scene']['config']['speed']==expected,
                    f'episode {episode}: speed sampling metadata mismatch')
        for a in attempts:
            expected=sample_episode_speed(speed_spec,collection['seed'],a['speed_episode_index'])
            require(a['conveyor_speed_m_s']==expected and a['scene']['config']['speed']==expected,
                    'retry changed the sampled episode speed')
    if not collection['probe_only']:
        for episode,a in saved.items():
            cfg=ConveyorConfig(**a['scene']['config'])
            require(a['success'] and all(a['grasp_lifted']),f'episode {episode}: incomplete physical grasp')
            require(a['stability_steps']>=30,f'episode {episode}: insufficient stability')
            require(success_now(a['terminal_poses'],a['terminal_velocities'],np.asarray(a['terminal_tcp']),[True,True],cfg),
                    f'episode {episode}: physical terminal predicate failed')
    require(all(v>0 for v in event_mass.values()),'empty event stream')
    normalization=json.loads((meta/'ewm_normalization_stats.json').read_text())
    require(normalization['voxel']['count_clip']==15 and normalization['voxel']['num_bins']==1,'wrong event normalization')
    for group in ('proprio','action'):
        for key,values in normalization[group].items():
            if isinstance(values,list):require(np.isfinite(values).all(),'nonfinite normalization')
    ffmpeg=Path(piper_conveyor.__file__).resolve().parents[2]/'envs/torchcodec_ffmpeg7/bin/ffmpeg'
    videos=sorted(root.rglob('*.mp4'));require(bool(videos),'no RGB videos')
    for video in videos:subprocess.run([str(ffmpeg),'-v','error','-i',str(video),'-f','null','-'],check=True)
    report=dict(passed=True,episodes=args.episodes,frames=total,event_count_mass=event_mass,
                videos_decoded=len(videos),probe_only=collection['probe_only'],failed_attempts=sum(not a['success'] for a in attempts))
    if speed_spec is not None:
        report['speed_sampling']=speed_spec
        report['saved_episode_speeds_m_s']=[saved[i]['conveyor_speed_m_s'] for i in range(args.episodes)]
    (meta/'piper_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    if not collection['probe_only']:(root/'.export_complete').write_text(json.dumps(report)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
