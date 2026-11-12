"""Dimensions in metres and sampled launch schedules for the four tennis tasks."""
from dataclasses import asdict
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TennisConfig:
    mode: str = 'fall'
    radius: float = .0335
    mass: float = .057
    restitution: float = .55
    gravity: float = 9.81
    launch_delay: float = 4.
    launch_delay_min: float = 0.
    launch_delay_max: float = 2.
    source_x: float = .40
    source_y: float = 0.
    source_z: float = .60
    velocity_x: float = 0.
    velocity_y: float = 0.
    velocity_z: float = 0.
    catch_z: float = .22
    rolling_speed: float = .12
    position_jitter: float = .008
    close_lead: float = .067
    lift_height: float = .07
    stable_seconds: float = .5
    max_seconds: float = 16.
    container_inner: float = .14
    container_height: float = .09
    wall: float = .006
    handle_length: float = .045
    handle_width: float = .025
    container_x: float = .37
    container_y: float = -.17
    container_hold_z: float = .12

    def __post_init__(self):
        if self.mode not in ('roll', 'fall', 'lob', 'container'):
            raise ValueError('unknown tennis mode')
        numeric = [v for k, v in asdict(self).items() if k not in ('mode', 'receiving_cell')]
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError('tennis parameters must be finite')
        if not (.02 <= self.radius <= .04 and self.mass > 0 and 0 <= self.restitution <= 1):
            raise ValueError('invalid ball geometry or material')
        if not (self.gravity > 0 and 0 < self.launch_delay < self.max_seconds-self.stable_seconds-1):
            raise ValueError('invalid gravity or launch schedule')
        if not (0 <= self.launch_delay_min <= self.launch_delay_max <= 2.
                and all(abs(t*30-round(t*30)) < 1e-8
                        for t in (self.launch_delay_min, self.launch_delay_max))):
            raise ValueError('launch range must use inclusive 30 Hz ticks in [0, 2]')
        if not (0 < self.stable_seconds and 0 < self.close_lead < .15 and self.lift_height >= .05):
            raise ValueError('invalid catch timing or lift')
        if not (0 <= self.position_jitter <= (.08 if self.mode == 'lob' else .02) and .03 <= self.rolling_speed <= (.6 if self.mode == 'roll' else .25)):
            raise ValueError('invalid launch randomization or rolling speed')
        if self.catch_z <= self.radius+.06 or self.source_z <= self.radius:
            raise ValueError('airborne catch must clear the table')
        if not (.15 < self.source_x < (1.65 if self.mode == 'lob' else (1.5 if self.mode == 'container' else (1.0 if self.mode == 'roll' else .65))) and abs(self.source_y)+self.position_jitter < .42):
            raise ValueError('release position must clear the base and table edge')
        if self.mode == 'fall' and (self.source_z <= self.catch_z or
                                   self.velocity_x != 0 or self.velocity_y != 0 or self.velocity_z > 0):
            raise ValueError('free fall requires downward vertical motion above the catch')
        if self.mode in ('lob', 'container') and (self.velocity_z <= 0 or
                math.hypot(self.velocity_x, self.velocity_y) == 0):
            raise ValueError('parabolic launch requires upward and horizontal velocity')
        if not (self.wall > 0 and self.container_inner > 2*self.radius+.02 and
                self.container_height > 2*self.radius+self.wall and
                self.handle_length > .02 and .01 < self.handle_width < .04 and self.container_hold_z > (.02 if self.mode == 'container' else .08)):
            raise ValueError('container must contain the whole ball and have a graspable handle')
        if self.mode in ('lob', 'container'):
            z = self.catch_z if self.mode == 'lob' else self.container_hold_z+self.wall+self.container_height+self.radius
            if self.velocity_z**2+2*self.gravity*(self.source_z-z) <= 0:
                raise ValueError('launch cannot reach the receiving height')

    @property
    def handle_offset(self):
        return (-(self.container_inner/2+self.wall+self.handle_length/2), 0., self.container_height*.6)

    def metadata(self):
        return asdict(self)


@dataclass(frozen=True)
class RollingConfig(TennisConfig):
    mode: str = 'roll'
    launch_delay: float = 1.
    source_x: float = .74
    source_y: float = 0.
    source_z: float = .034
    position_jitter: float = 0.
    release_x_jitter: float = .05
    release_y_jitter: float = .10
    rolling_speed: float = .38
    rolling_speed_jitter: float = .08
    rolling_yaw_jitter_deg: float = 25.
    intercept_x: float = .34
    intercept_y_min: float = -.18
    intercept_y_max: float = .18
    reaction_delay: float = .067
    intercept_seconds: float = .45
    translation_limit: float = .04
    stable_position_diameter_m: float = .004

    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.release_x_jitter <= .15 and 0 <= self.release_y_jitter <= .2):
            raise ValueError('invalid rolling release region')
        if not (.20 <= self.intercept_x <= .55 and
                self.intercept_x+.2 < self.source_x-self.release_x_jitter and
                self.source_x+self.release_x_jitter < 1.0 and
                abs(self.source_y)+self.release_y_jitter < .4):
            raise ValueError('rolling release must be in front of the receiving plane on the table')
        if not (0 <= self.rolling_speed_jitter and
                .03 <= self.rolling_speed-self.rolling_speed_jitter and
                self.rolling_speed+self.rolling_speed_jitter <= .6):
            raise ValueError('invalid rolling speed range')
        if not (0 <= self.rolling_yaw_jitter_deg <= 30. and
                -.25 <= self.intercept_y_min < self.intercept_y_max <= .25):
            raise ValueError('invalid rolling yaw or receiving workspace')
        if not (0 < self.stable_position_diameter_m <= .01 and
                0 <= self.reaction_delay <= .2 and .1 <= self.intercept_seconds <= 2. and
                0 < self.translation_limit <= .04):
            raise ValueError('invalid rolling reaction or hold parameters')


@dataclass(frozen=True)
class FallingConfig(TennisConfig):
    source_z: float = .80
    catch_z: float = .18
    launch_delay: float = 1.
    stable_position_diameter_m: float = .002
    close_lead: float = .040
    lift_height: float = .10
    grasp_depth: float = .025
    reaction_delay: float = 2/30
    translation_limit: float = .16
    angular_limit: float = .20
    approach_gain: float = 1.5
    downward_speed_min: float = 0.
    downward_speed_max: float = .6

    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.downward_speed_min <= self.downward_speed_max <= 1.5):
            raise ValueError('invalid downward launch speed range')
        if self.velocity_z != 0:
            raise ValueError('configure downward_speed_min/max instead of velocity_z')
        if not .01 <= self.grasp_depth <= .035:
            raise ValueError('invalid falling grasp depth')
        if not 0 < self.stable_position_diameter_m <= .01:
            raise ValueError('invalid falling hold tolerance')
        if not (0 <= self.reaction_delay <= .1 and
                .04 <= self.translation_limit <= .2 and
                .08 <= self.angular_limit <= .3 and
                1. <= self.approach_gain <= 2.):
            raise ValueError('invalid falling observation or approach control')


@dataclass(frozen=True)
class LobConfig(TennisConfig):
    mode: str = 'lob'
    launch_delay: float = 1.
    source_x: float = 1.20
    source_y: float = 0.
    source_z: float = .35
    velocity_x: float = -1.60
    velocity_y: float = 0.
    # 0.45 s nominal flight from z=.35 to z=.55 under Earth gravity.
    velocity_z: float = 2.651694444444445
    catch_z: float = .55
    lift_height: float = .06
    position_jitter: float = .035
    close_lead: float = .040
    reaction_delay: float = 2/30
    grasp_depth: float = .025
    intercept_seconds: float = .10
    translation_limit: float = .16
    velocity_jitter: float = .035
    launch_yaw_jitter_deg: float = 15.
    stable_position_diameter_m: float = .002

    def __post_init__(self):
        super().__post_init__()
        if not (0.03 <= self.reaction_delay <= .2 and 0 <= self.velocity_jitter <= .1):
            raise ValueError('invalid observation delay or launch variation')
        if not (.01 <= self.grasp_depth <= .035 and .10 <= self.intercept_seconds <= .3):
            raise ValueError('invalid finger insertion depth or approach duration')
        if not (.09 <= self.translation_limit <= .20):
            raise ValueError('invalid reactive translation limit')
        if not (0 < self.stable_position_diameter_m <= .01):
            raise ValueError('stable position diameter must be positive and at most 10 mm')
        if not (0 <= self.launch_yaw_jitter_deg <= 15.):
            raise ValueError('launch yaw variation must be between zero and fifteen degrees')
        if -self.velocity_x <= self.velocity_jitter:
            raise ValueError('horizontal speed variation must preserve forward flight')
        if self.source_y != 0 or self.velocity_y != 0 or self.velocity_x >= 0:
            raise ValueError('lob must approach head-on along the robot positive-X center plane')


@dataclass(frozen=True)
class ContainerConfig(TennisConfig):
    paired_launch: bool = True
    # Optional per-slot spatial stratum; absent in v1/v2 serialized configs.
    receiving_cell: tuple | None = None

    def metadata(self):
        result = super().metadata()
        if self.paired_launch:
            result.pop('paired_launch')  # Preserve v1 serialized configurations.
        if self.receiving_cell is None:
            result.pop('receiving_cell')
        else:
            result['receiving_cell'] = list(self.receiving_cell)
        return result

    mode: str = 'container'
    # launch_delay is legacy; container episodes use the sampled range below.
    launch_delay: float = 1.
    launch_delay_min: float = 0.
    launch_delay_max: float = 2.
    source_x: float = 1.40
    source_y: float = 0.
    source_z: float = .475
    velocity_x: float = -1.60
    velocity_y: float = 0.
    velocity_z: float = 2.651694444444445
    position_jitter: float = 0.
    release_x_jitter: float = .20
    release_y_jitter: float = 0.
    release_z_jitter: float = .125
    flight_time_min: float = .65
    flight_time_max: float = .70
    minimum_receiver_displacement: float = .08
    # Legacy nominal velocity fields are retained for config compatibility.
    # Actual container velocity is solved from sampled flight time and region.
    velocity_jitter: float = .15
    launch_yaw_jitter_deg: float = 30.
    # Conservative footprint shared by the visible top and SeattleLabTable.
    landing_x_min: float = -.10
    landing_x_max: float = .98
    landing_y_limit: float = .45
    landing_table_z: float = .0002
    landing_edge_margin: float = .025
    receiving_x_min: float = .25
    receiving_x_max: float = .52
    receiving_radius_min: float = .26
    receiving_radius_max: float = .56
    receiving_y_limit: float = .24
    stable_position_diameter_m: float = .005
    reaction_delay: float = 1/30
    translation_limit: float = .08
    container_inner: float = .16
    container_height: float = .10
    container_x: float = .36
    container_y: float = 0.
    container_hold_z: float = .24
    grip_pitch: float = math.pi/2
    handle_length: float = .10
    handle_width: float = .028
    handle_thickness: float = .018
    container_min_clearance: float = .01

    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.launch_delay_min <= self.launch_delay_max <= 2.
                and all(abs(t*30-round(t*30)) < 1e-8
                        for t in (self.launch_delay_min, self.launch_delay_max))):
            raise ValueError('container launch times must be control ticks in [0, 2] seconds')
        if not (self.velocity_x < 0 and self.velocity_y == 0 and self.source_y == 0):
            raise ValueError('container launch must approach along negative world X')
        if not (0 <= self.release_x_jitter <= .3 and self.release_y_jitter == 0
                and 1.0 <= self.source_x-self.release_x_jitter
                and self.source_x+self.release_x_jitter <= 1.6
                and 0 <= self.release_z_jitter <= .25
                and .2 <= self.source_z-self.release_z_jitter
                and self.source_z+self.release_z_jitter <= .8
                # Allow the historical ID envelope [.65, .70] scaled by the
                # evaluation speed multipliers [.8, 1.2]; sampling defaults stay ID.
                and .65/1.2 <= self.flight_time_min <= self.flight_time_max <= .70/.8
                and 0 <= self.minimum_receiver_displacement <= .2
                and 0 <= self.launch_yaw_jitter_deg <= 60.
                and 0 <= self.velocity_jitter < -self.velocity_x):
            raise ValueError('invalid container launch sampling')
        if not (0 < self.stable_position_diameter_m <= .01):
            raise ValueError('invalid container stability tolerance')
        clearance = self.radius+self.landing_edge_margin
        if not (-.10 <= self.landing_x_min < self.landing_x_max <= .98
                and 0 < self.landing_y_limit <= .45
                and .01 <= self.landing_edge_margin <= .10
                and self.landing_x_max-self.landing_x_min > 2*clearance
                and self.landing_y_limit > clearance
                and self.landing_table_z == .0002):
            raise ValueError('invalid container tabletop landing region')
        if not (.1 <= self.receiving_x_min < self.receiving_x_max <= .65
                and .2 <= self.receiving_radius_min < self.receiving_radius_max <= .65
                and 0 < self.receiving_y_limit <= .5 and 0 <= self.reaction_delay <= .2
                and 0 < self.translation_limit <= .16):
            raise ValueError('invalid container receiving workspace or reaction')
        if self.receiving_cell is not None:
            if self.paired_launch or len(self.receiving_cell) != 4:
                raise ValueError('receiving cell requires independent launch and four bounds')
            x0, x1, y0, y1 = self.receiving_cell
            if not (self.receiving_x_min <= x0 < x1 <= self.receiving_x_max
                    and -self.receiving_y_limit <= y0 < y1 <= self.receiving_y_limit):
                raise ValueError('receiving cell is outside the configured region')
        if not (1.2 <= self.grip_pitch <= 1.8 and
                0 < self.container_min_clearance < self.container_hold_z and
                math.hypot(self.container_x, self.container_y) > abs(self.grip_offset[0])):
            raise ValueError('invalid front-handle grasp pose')
        if not (.006 <= self.handle_thickness <= .02 and
                self.handle_length > self.handle_thickness+.01 and self.handle_width > .016):
            raise ValueError('invalid pan-handle dimensions')

    @property
    def handle_offset(self):
        return (-(self.container_inner/2+self.wall+self.handle_length-.025),
                0., self.container_height-.02)

    @property
    def grip_offset(self):
        return self.handle_offset

    @property
    def initial_container_yaw(self):
        return math.atan2(self.container_y, self.container_x)

    @property
    def initial_tool_yaw(self):
        return self.initial_container_yaw

    @property
    def initial_tool_quaternion(self):
        # Rz(yaw) Ry(pitch): hold the horizontal handle from its two sides.
        a, b, c = self.initial_tool_yaw/2, self.grip_pitch/2, 0.
        return (math.sin(b)*math.sin(c-a), math.sin(b)*math.cos(a-c),
                math.cos(b)*math.sin(a+c), math.cos(b)*math.cos(a+c))

    @property
    def initial_tcp_position(self):
        yaw = self.initial_container_yaw
        x, _, z = self.grip_offset
        radial = x+.0192*math.sin(self.grip_pitch)
        return (self.container_x+radial*math.cos(yaw),
                self.container_y+radial*math.sin(yaw),
                self.container_hold_z+z+.0192*math.cos(self.grip_pitch))
