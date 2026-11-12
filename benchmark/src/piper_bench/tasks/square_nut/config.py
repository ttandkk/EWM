"""Dimensions in metres; the nut and its side handle form one passive body."""
from dataclasses import asdict
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SquareNutConfig:
    source_x: float = .38
    source_y: float = -.12
    source_yaw: float = math.atan2(-.12, .38)
    peg_x: float = .39
    peg_y: float = .10
    peg_yaw: float = math.radians(15)
    outer_width: float = .066
    hole_width: float = .030
    thickness: float = .018
    handle_length: float = .040
    handle_width: float = .022
    peg_width: float = .024
    peg_height: float = .045
    base_width: float = .10
    base_height: float = .013
    table_z: float = -.003
    position_jitter: float = .004
    yaw_jitter: float = math.radians(10)
    support_tolerance: float = .0015
    stable_seconds: float = .7
    max_seconds: float = 30.

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('configuration must be finite')
        if not (0 < self.peg_width < self.hole_width < self.outer_width < self.base_width):
            raise ValueError('require peg < hole < nut < base width')
        if not (0 < self.thickness < self.peg_height and self.base_height > 0):
            raise ValueError('invalid vertical geometry')
        if not (self.handle_length > 0 and 0 < self.handle_width < self.outer_width):
            raise ValueError('invalid handle geometry')
        if min(self.position_jitter, self.yaw_jitter) < 0:
            raise ValueError('jitter must be nonnegative')
        if min(self.support_tolerance, self.stable_seconds, self.max_seconds) <= 0:
            raise ValueError('success thresholds must be positive')

    @property
    def base_top(self):
        return self.table_z + self.base_height

    @property
    def handle_x(self):
        return -(self.outer_width + self.handle_length) / 2

    def metadata(self):
        return asdict(self)
