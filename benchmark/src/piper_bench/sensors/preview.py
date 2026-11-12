"""Synchronized HDR EVIS preview using the existing conveyor event contract."""
import json
import math
import subprocess

import numpy as np
from PIL import Image
from PIL import ImageDraw

from piper_bench.tasks.conveyor.core import CAMERAS
from piper_bench.tasks.conveyor.core import CONTROL_HZ
from piper_bench.sensors.capture import snapshot
from piper_bench.sensors.event_frames import COUNT_CLIP
from piper_bench.sensors.event_frames import events_to_frame

WARP = 8


def create_event_camera(env, output_dir):
    from dvs_gen.sensors import DVSCamera
    return DVSCamera.from_scene(env.scene, list(CAMERAS), out_dir=str(output_dir),
                               compression='lzf', event_source='hdr', antialiasing=None)


def event_metadata(dvs):
    return dict(source='EVIS linear HDR', cameras=list(CAMERAS), warp=WARP,
                interpolation_hz=CONTROL_HZ*WARP, output_hz=CONTROL_HZ,
                threshold={name: proc.threshold for name, proc in zip(CAMERAS, dvs.procs)},
                window='(t - 1/30, t]', count_channels=['OFF', 'ON'], count_clip=COUNT_CLIP,
                display='white background; OFF blue, ON red, overlap purple',
                video_layout=['front RGB', 'wrist RGB', 'front events', 'wrist events'],
                video_clock='same post-step RGB/event time; initial t=0 saved separately')


class EventPreview:
    def __init__(self, dvs, runtime, out, ffmpeg, *, labels=True):
        import torch
        self.dvs, self.runtime, self.out, self.ffmpeg = dvs, runtime, out, ffmpeg
        self.labels = labels
        self.frames = {name: [] for name in CAMERAS}
        self.times = []
        dvs.reset(torch.tensor([0], dtype=torch.long, device='cuda:0'))
        dvs.recorder._events.pop(0, None)
        self.previous = snapshot(dvs, runtime.env)
        self.previous_time = runtime.time
        self.counts = self._observe(runtime.time)
        if any(np.any(counts) for counts in self.counts.values()):
            raise RuntimeError('initial EVIS frame must be empty after reset')
        self.initial_counts = {name: frame[:, :, :2].copy() for name, frame in self.counts.items()}
        self.path = out/'preview_events.mp4'
        self.video = subprocess.Popen([ffmpeg, '-v', 'error', '-y', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', '512x512', '-r', str(CONTROL_HZ), '-i', 'pipe:0',
            '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.path)], stdin=subprocess.PIPE)
        self.last_panel = None

    def _observe(self, time):
        # Include the interval's final keyframe exactly as the conveyor evaluator does.
        self.dvs.warp_and_process(self.previous, self.previous, 1, time, 1/(CONTROL_HZ*WARP))
        chunks = self.dvs.recorder._events.pop(0, {})
        return {name: events_to_frame(chunks.get('DVS/'+name, []), time-1/CONTROL_HZ, time)
                for name in CAMERAS}

    def panel(self, rgb):
        canvas = Image.new('RGB', (512, 512), 'white')
        canvas.paste(rgb, (0, 0))
        for i, name in enumerate(CAMERAS):
            counts = self.counts[name]
            off, on = counts[:, :, 0] > 0, counts[:, :, 1] > 0
            visual = np.full((256, 256, 3), 255, dtype=np.uint8)
            visual[off] = (45, 65, 220)
            visual[on] = (225, 45, 45)
            visual[off & on] = (160, 45, 180)
            tile = Image.fromarray(visual)
            if self.labels:
                draw = ImageDraw.Draw(tile)
                draw.text((5, 5), f'{name} EVIS t={self.previous_time:.2f}s', fill='black',
                          stroke_width=1, stroke_fill='white')
                draw.text((5, 241), 'OFF blue / ON red / both purple', fill='black',
                          stroke_width=1, stroke_fill='white')
            canvas.paste(tile, (256*i, 256))
        return canvas

    def step(self, rgb):
        now = self.runtime.time
        if not math.isclose(now-self.previous_time, 1/CONTROL_HZ, abs_tol=1e-9):
            raise RuntimeError('event preview requires contiguous 30 Hz ticks')
        current = snapshot(self.dvs, self.runtime.env)
        self.dvs.warp_and_process(self.previous, current, WARP, self.previous_time, 1/(CONTROL_HZ*WARP))
        self.previous, self.previous_time = current, now
        self.counts = self._observe(now)
        self.times.append(now)
        stats = {}
        for name, counts in self.counts.items():
            data = counts[:, :, :2]
            self.frames[name].append(data.copy())
            stats[name] = dict(off=int(data[:, :, 0].sum()), on=int(data[:, :, 1].sum()),
                               active_pixels=int(np.any(data, axis=-1).sum()))
        self.last_panel = self.panel(rgb)
        self.video.stdin.write(self.last_panel.tobytes())
        return stats

    def close(self):
        self.video.stdin.close()
        if self.video.wait(timeout=60):
            raise RuntimeError('event preview encoding failed')
        if not self.times:
            return
        frames = {name: np.stack(values) for name, values in self.frames.items()}
        np.savez_compressed(self.out/'event_frames.npz', time_s=np.asarray(self.times),
                            **frames, **{'initial_'+name: value for name, value in self.initial_counts.items()})
        self.last_panel.save(self.out/'terminal_events.png')
        subprocess.run([self.ffmpeg, '-v', 'error', '-i', str(self.path), '-f', 'null', '-'], check=True)
        report = event_metadata(self.dvs)
        report.update(labels=self.labels, frames=len(self.times), duration_s=len(self.times)/CONTROL_HZ,
                      initial_frame_empty=True, video_decoded=True,
                      cameras_stats={name: dict(off=int(data[:, :, :, 0].sum()),
                          on=int(data[:, :, :, 1].sum()), max_count=int(data.max()),
                          nonempty_frames=int(np.any(data, axis=(1, 2, 3)).sum()))
                          for name, data in frames.items()})
        report['passed'] = all(r['off'] > 0 and r['on'] > 0 for r in report['cameras_stats'].values())
        (self.out/'events.json').write_text(json.dumps(report, indent=2)+'\n')
        self.report = report
