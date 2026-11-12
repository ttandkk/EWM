"""Per-trial physical imaging; transformed RGB reaches policy and memory together."""
import numpy as np
from piper_bench.sensors.motion_blur import RGBExposureObserver


class TrialImaging:
    def __init__(self, runtime, dvs, spec, lights, background=None):
        self.runtime, self.dvs = runtime, dvs
        self.spec = dict(spec)
        self.lights, self.background = lights, background
        self.blur = RGBExposureObserver(dvs) if spec['condition'] == 'blur_ood' else None

    def apply_rgb(self, rgb):
        if self.blur is None:
            return rgb
        frame = {'observation.sim_time': np.array([self.runtime.time])}
        for i, name in enumerate(('cam0', 'cam_wrist')):
            frame['observation.images.'+name] = rgb[i].transpose(1, 2, 0)
        self.blur(frame, self.dvs._conveyor_snapshot)
        images = frame['_blurred_rgb'][self.spec['intervals']]
        return np.stack([images[name].transpose(2, 0, 1) for name in ('cam0', 'cam_wrist')])

    def metadata(self):
        return dict(condition=self.spec['condition'], parameters=self.spec,
                    physical_lighting=self.lights, background=self.background,
                    rgb_exposure_method='piper_causal_rgb_flow_exposure_v1' if self.blur else None,
                    event_input='sharp HDR from the condition scene',
                    output_hz=30, event_window='(t_(k-1), t_k]')


class EvaluationEvents:
    """Both sensor streams, including t=0 and the final post-action observation.

    Evaluation has N actions and N+1 observations, so unlike training there is
    no unobserved terminal action tail. Raw xypt is never clipped; PNG-equivalent
    count arrays use the same OFF/ON clip=15 convention as training.
    """
    def __init__(self):
        self.times = []
        self.raw = {c: [] for c in ('cam0', 'cam_wrist')}
        self.offsets = {c: [0] for c in self.raw}
        self.frames = {c: [] for c in self.raw}

    def observe(self, chunks, tick, policy_cameras, policy_counts):
        from piper_bench.sensors.event_archive import checked_events
        from piper_bench.sensors.event_frames import events_to_frame
        if tick != len(self.times):
            raise ValueError('evaluation event observations must be contiguous from t=0')
        self.times.append(tick/30)
        result = {}
        for cam in self.raw:
            raw = checked_events(chunks.get('DVS/'+cam, []), (tick-1)/30, tick/30)
            counts = events_to_frame([raw], (tick-1)/30, tick/30)[:, :, :2].transpose(2, 0, 1)
            if tick == 0 and len(raw[2]):
                raise ValueError('initial evaluation event frame must be empty')
            if cam in policy_cameras and not np.array_equal(counts, policy_counts[policy_cameras.index(cam)]):
                raise ValueError('policy events differ from archived raw events')
            self.raw[cam].append(raw)
            self.offsets[cam].append(self.offsets[cam][-1]+len(raw[2]))
            self.frames[cam].append(counts)
            result[cam] = counts
        return result

    def save(self, directory):
        from pathlib import Path
        from piper_bench.collection.protocol import write_json
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        times = np.asarray(self.times, dtype=np.float64)
        if not len(times):
            raise ValueError('cannot save an empty evaluation event archive')
        counts = {}
        for cam, chunks in self.raw.items():
            arrays = {key:np.concatenate([chunk[i] for chunk in chunks]) for i,key in enumerate(('x','y','t','p'))}
            np.savez_compressed(directory/(cam+'_raw.npz'), **arrays,
                                frame_offsets=np.asarray(self.offsets[cam],dtype=np.int64), frame_times_s=times)
            np.savez_compressed(directory/(cam+'_frames.npz'), frames=np.stack(self.frames[cam]), time_s=times)
            counts[cam] = len(arrays['t'])
        metadata = dict(version='piper_evaluation_raw_and_count_frames_v1', frames=len(times),
                        actions=len(times)-1, raw_events=counts, frame_window='(t_(k-1), t_k]',
                        first_frame_empty=True, terminal_observation_included=True,
                        unobserved_terminal_tail=False, count_channels=['OFF','ON'],count_clip=15,
                        directory=str(directory))
        write_json(directory/'metadata.json',metadata)
        return metadata
