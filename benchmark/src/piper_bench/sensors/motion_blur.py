"""Causal 30 Hz RGB exposure approximation; sharp HDR EVIS is untouched.

RGB keyframes are inverse-sRGB transformed, flow-warped at 240 Hz, integrated,
and sRGB encoded. This approximates exposure in display-linear RGB; it is not
an HDR sensor/radiometric model and cannot recover clipped keyframe values.
"""
import numpy as np

VERSION = "piper_causal_rgb_flow_exposure_v1"
INTERVALS = (4, 5, 6, 7, 8)


def exposure_weights(intervals):
    """Trapezoidal quadrature on nine points t-8/240,...,t (no future input)."""
    if intervals not in INTERVALS:
        raise ValueError("exposure must be 4, 5, 6, 7 or 8 intervals at 240 Hz")
    weights = np.zeros(9, dtype=np.float32)
    weights[8-intervals:] = 1./intervals
    weights[8-intervals] *= .5
    weights[-1] *= .5
    return weights


def srgb_to_linear(rgb):
    value = np.asarray(rgb, dtype=np.float32)/255.
    return np.where(value <= .04045, value/12.92, ((value+.055)/1.055)**2.4)


def sample_plan(seed):
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(np.random.SeedSequence([20261021, seed, 902]))
    rows = np.repeat(INTERVALS, 10)
    return dict(version=VERSION, seed=seed, output_hz=30, event_input="sharp HDR",
                episodes=[dict(trial=i, intervals=int(n), exposure_s=int(n)/240)
                          for i, n in enumerate(rng.permutation(rows))])


class RGBExposureObserver:
    """Observe already-captured pre-action rows; never change actions or events."""
    def __init__(self, dvs):
        if any(dvs.margin):
            raise ValueError("RGB exposure preview currently requires zero overscan margin")
        if dvs.blur_cfgs:
            raise ValueError("EVIS native blur must be disabled to retain sharp HDR events")
        self.dvs = dvs
        self.previous = None
        self.previous_snapshot = None
        self.step = 0

    def __call__(self, frame, snapshot):
        import torch
        from dvs_gen.warp import bidir_warp_gap
        time = float(frame["observation.sim_time"][0])
        if abs(time-self.step/30) > 1e-5:
            raise ValueError("RGB exposure requires contiguous pre-action 30 Hz rows")
        names = self.dvs.names
        current = torch.stack([torch.from_numpy(srgb_to_linear(frame["observation.images."+c]))
                               for c in names]).to("cuda:0")
        if self.previous is None:
            # The scene is stationary before t=0: extend the initial image into
            # the pre-roll exposure window, never borrow future movement.
            fine = current[None].expand(9, -1, -1, -1, -1)
        else:
            prev = self.previous_snapshot
            mv_a = torch.cat([prev[c][1] for c in names], 0)
            mv_b = torch.cat([snapshot[c][1] for c in names], 0)
            depth_a = torch.cat([prev[c][2] for c in names], 0) if self.dvs.enable_warp else None
            depth_b = torch.cat([snapshot[c][2] for c in names], 0) if self.dvs.enable_warp else None
            hf = self.dvs.hole_fill
            if hf == "bg":
                hf = float(torch.maximum(self.previous.max(), current.max()))
            mids = bidir_warp_gap(self.previous, current, mv_a, mv_b, 8, self.dvs.composite,
                                  depthA=depth_a, depthB=depth_b, hole_fill=hf,
                                  mv_dilate=self.dvs.mv_dilate)
            fine = torch.stack([self.previous, *mids, current])
        weights = torch.as_tensor(np.stack([exposure_weights(n) for n in INTERVALS]), device="cuda:0")
        linear = torch.einsum("ek,kchwd->echwd", weights, fine).clamp(0,1)
        display = torch.where(linear <= .0031308, linear*12.92,
                              1.055*linear.clamp_min(0).pow(1/2.4)-.055)
        images = (display*255).round().clamp(0,255).byte().cpu().numpy()
        frame["_blurred_rgb"] = {n: {c: images[i,j] for j,c in enumerate(names)}
                                 for i,n in enumerate(INTERVALS)}
        self.previous, self.previous_snapshot = current, snapshot
        self.step += 1
