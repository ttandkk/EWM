"""Validated dimensions and motion defaults for this task."""
from dataclasses import dataclass
from dataclasses import asdict
import math


@dataclass(frozen=True)
class TubeConfig:
    rack_x: float=.40
    rack_y: float=0.
    rack_radius: float=.078
    plate_inner_radius: float=.028
    hole_orbit: float=.051
    hole_radius: float=.009
    holes: int=8
    base_thickness: float=.009
    plate_top: float=.065
    plate_thickness: float=.008
    omega: float=.21
    tube_length: float=.080
    tube_radius: float=.007
    cap_height: float=.009
    cap_radius: float=.00825
    grasp_offset: float=.020
    tube_x: float=.18
    tube_y: float=.19
    max_seconds: float=30.
    stable_seconds: float=1.

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):raise ValueError('nonfinite config')
        if self.holes<3 or not isinstance(self.holes,int):raise ValueError('invalid hole count')
        if not 0<self.tube_radius<self.cap_radius<self.hole_radius:raise ValueError('tube must clear hole')
        if not self.plate_inner_radius<self.hole_orbit-self.hole_radius:raise ValueError('inner plate overlap')
        if self.hole_orbit+self.hole_radius>=self.rack_radius:raise ValueError('hole outside plate')
        if self.hole_radius>=self.hole_orbit*math.sin(math.pi/self.holes):raise ValueError('holes overlap')
        if not 0<self.base_thickness<self.plate_top-self.plate_thickness:raise ValueError('invalid rack height')
        if not self.base_thickness+self.tube_length>self.plate_top+self.cap_height+.01:raise ValueError('tube too short')
        if not 0<abs(self.omega)<=2:raise ValueError('rack must continuously rotate')
        if not 0<self.grasp_offset<self.tube_length/2:raise ValueError('invalid grasp height')
        if self.max_seconds<=0 or self.stable_seconds<=0:raise ValueError('invalid episode time')

    def metadata(self):return asdict(self)

