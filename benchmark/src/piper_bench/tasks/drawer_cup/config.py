from dataclasses import dataclass
from dataclasses import asdict
import math


@dataclass(frozen=True)
class DrawerCupConfig:
    cabinet_x: float = .34
    cabinet_y: float = -.18
    cabinet_yaw: float = math.pi/2
    scale_x: float = .36
    scale_y: float = .28
    scale_z: float = .65
    drawer_z: float = .050
    handle_extension: float = .018
    source_x: float = .38
    source_y: float = .18
    position_jitter: float = .004
    yaw_jitter: float = math.radians(10)
    travel: float = .13
    open_threshold: float = .115
    closed_threshold: float = .008
    # These names match the reused hollow-container generator: a shallow bowl.
    cup_height: float = .032
    cup_radius: float = .038
    bowl_bottom_radius: float = .022
    bowl_wall: float = .0025
    bowl_floor: float = .0025
    # A single 2.5 mm rim does not need the full 80 mm opening, which can
    # bring the outer finger into the open drawer at randomized bowl poses.
    gripper_open_half_width: float = .025
    stable_seconds: float = 1.
    max_seconds: float = 55.

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('configuration must be finite')
        if not (0 < self.closed_threshold < self.open_threshold < self.travel <= .18):
            raise ValueError('invalid drawer travel thresholds')
        if not (0 < self.bowl_bottom_radius < self.cup_radius and
                0 < self.bowl_wall < self.bowl_bottom_radius and 0 < self.bowl_floor < self.cup_height):
            raise ValueError('bowl must have an open cavity and a solid floor')
        if min(self.scale_x, self.scale_y, self.scale_z) <= 0 or self.handle_extension < 0:
            raise ValueError('invalid cabinet scale or handle clearance')
        if not .025 <= self.gripper_open_half_width <= .040:
            raise ValueError('rim approach requires a bounded gripper opening')
        if .110356*self.scale_z <= self.cup_height+.01:
            raise ValueError('drawer must retain overhead clearance for the bowl')
        if min(.245*self.scale_x, .338*self.scale_y) <= self.cup_radius+.01:
            raise ValueError('drawer is too narrow for the bowl')
        if min(self.position_jitter, self.yaw_jitter) < 0 or min(self.stable_seconds, self.max_seconds) <= 0:
            raise ValueError('invalid jitter or timing')

    @property
    def floor_z(self):
        # Top face of the original Sektion drawer bottom, from its collision USD.
        return self.drawer_z-.040356*self.scale_z

    @property
    def pull_direction(self):
        return (math.cos(self.cabinet_yaw), math.sin(self.cabinet_yaw), 0.)

    def metadata(self):
        return asdict(self)
