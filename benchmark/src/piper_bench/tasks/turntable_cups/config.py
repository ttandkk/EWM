"""Validated dimensions and motion defaults for this task."""
from dataclasses import dataclass
import math
from piper_bench.tasks.turntable_cups.turntable import TurntableConfig


@dataclass(frozen=True)
class CupConfig(TurntableConfig):
    speed: float=.315 * .095  # m/s at the nominal orbit; omega = 0.315 rad/s
    object_size: float=.06  # inherited spawn uses height/2; shape is explicitly a hollow cup
    cup_height: float=.06
    cup_bottom_radius: float=.017
    cup_top_radius: float=.026
    cup_wall: float=.0015
    cup_floor: float=.002
    max_seconds: float=40.

    def __post_init__(self):
        super().__post_init__()
        vals=(self.cup_height,self.cup_bottom_radius,self.cup_top_radius,self.cup_wall,self.cup_floor)
        if not all(math.isfinite(x) and x>0 for x in vals):raise ValueError('invalid cup dimensions')
        if self.cup_top_radius<=self.cup_bottom_radius or self.cup_wall>=self.cup_bottom_radius or self.cup_floor>=self.cup_height:
            raise ValueError('cup must be open, tapered and hollow')
        if self.object_size!=self.cup_height:raise ValueError('spawn height must equal cup height')
        if self.cup_wall/self.taper>=self.cup_height/2:raise ValueError('cups must nest deeply')

    @property
    def taper(self):return (self.cup_top_radius-self.cup_bottom_radius)/self.cup_height


@dataclass(frozen=True)
class BowlConfig(CupConfig):
    """Same graspable 52 mm opening, half-height shallow nesting bowls."""
    object_size: float = .03
    cup_height: float = .03

