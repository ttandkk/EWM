from dataclasses import dataclass
from dataclasses import asdict
import math


@dataclass(frozen=True)
class PushTConfig:
    source_x: float = .36
    source_y: float = -.105
    target_x: float = .36
    target_y: float = .105
    target_yaw: float = 0.
    width: float = .10
    length: float = .10
    bar: float = .025
    stem: float = .028
    height: float = .025
    position_jitter: float = .004
    yaw_jitter: float = math.radians(8)
    position_tolerance: float = .008
    yaw_tolerance: float = math.radians(5)
    min_coverage: float = .90
    stable_seconds: float = .7
    max_seconds: float = 45.

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('configuration must be finite')
        if not (0 < self.bar < self.length and 0 < self.stem < self.width and self.height > 0):
            raise ValueError('invalid T geometry')
        if min(self.position_jitter, self.yaw_jitter) < 0:
            raise ValueError('jitter must be nonnegative')
        if not (0 < self.min_coverage <= 1 and min(self.position_tolerance, self.yaw_tolerance,
                                                 self.stable_seconds, self.max_seconds) > 0):
            raise ValueError('invalid success thresholds')

    def metadata(self):
        return asdict(self)
