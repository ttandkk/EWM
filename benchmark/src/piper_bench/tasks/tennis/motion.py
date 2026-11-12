"""Reset sampling and ideal flight prediction; never used to drive ball poses."""
import math
import numpy as np


def launch_state(cfg, seed):
    if cfg.mode == 'container':
        paired = cfg.paired_launch
        position, velocities = container_launch_pair(cfg, seed//2 if paired else seed)
        return position, velocities[seed%2 if paired else 0].copy()
    rng = np.random.default_rng(seed)
    p = np.array([cfg.source_x, cfg.source_y, cfg.source_z], dtype=float)
    p[:2] += rng.uniform(-cfg.position_jitter, cfg.position_jitter, 2)
    v = np.array([cfg.velocity_x, cfg.velocity_y, cfg.velocity_z], dtype=float)
    angular = np.zeros(3)
    if cfg.mode == 'fall':
        v[2] = -rng.uniform(cfg.downward_speed_min, cfg.downward_speed_max)
    if cfg.mode == 'lob':
        # Keep the old seeded fore-aft samples for exact zero-yaw replay.
        speed_delta, _ = rng.uniform(-cfg.velocity_jitter, cfg.velocity_jitter, 2)
        speed = -cfg.velocity_x-speed_delta
        yaw = math.radians(rng.uniform(-cfg.launch_yaw_jitter_deg, cfg.launch_yaw_jitter_deg))
        # Rotate a sampled horizontal speed; yaw must not increase its norm.
        v[0], v[1] = -speed*math.cos(yaw), speed*math.sin(yaw)
        p[1] = cfg.source_y
        angular = np.array([0., 1.2, .4])
    if cfg.mode == 'roll':
        # Independent release/speed/yaw draws, conditioned only on a fixed
        # world-space receiving region. Never query the robot or its target.
        for _ in range(256):
            p = np.array([rng.uniform(cfg.source_x-cfg.release_x_jitter, cfg.source_x+cfg.release_x_jitter),
                          rng.uniform(cfg.source_y-cfg.release_y_jitter, cfg.source_y+cfg.release_y_jitter),
                          cfg.radius+.0005])
            speed = rng.uniform(cfg.rolling_speed-cfg.rolling_speed_jitter,
                                cfg.rolling_speed+cfg.rolling_speed_jitter)
            yaw = math.radians(rng.uniform(-cfg.rolling_yaw_jitter_deg, cfg.rolling_yaw_jitter_deg))
            v = np.array([-speed*math.cos(yaw), speed*math.sin(yaw), 0.])
            crossing_y = p[1]+(cfg.intercept_x-p[0])*v[1]/v[0]
            if cfg.intercept_y_min <= crossing_y <= cfg.intercept_y_max:
                break
        else:
            raise ValueError('rolling release ranges cannot reach the receiving region')
        # No-slip rolling in either planar direction.
        angular = np.array([-v[1]/cfg.radius, v[0]/cfg.radius, 0.])
    return p, np.r_[v, angular]


def launch_time(cfg, seed):
    """Independent schedule stream; container pair members share the same tick."""
    schedule_seed = seed//2 if cfg.mode == 'container' and cfg.paired_launch else seed
    rng = np.random.default_rng(np.random.SeedSequence([schedule_seed, 731]))
    low, high = round(cfg.launch_delay_min*30), round(cfg.launch_delay_max*30)
    return int(rng.integers(low, high+1))/30


def container_launch_pair(cfg, pair_id):
    """Adjacent seeds share reset-visible state, but require different catches.

    Both velocities are generated together so selecting a pair member cannot
    alter its source, flight time or launch schedule. No live robot state enters.
    """
    rng = np.random.default_rng(np.random.SeedSequence([pair_id, 739]))
    duration = rng.uniform(cfg.flight_time_min, cfg.flight_time_max)
    p = np.array([rng.uniform(cfg.source_x-cfg.release_x_jitter, cfg.source_x+cfg.release_x_jitter),
                  0., rng.uniform(cfg.source_z-cfg.release_z_jitter, cfg.source_z+cfg.release_z_jitter)])
    rim_z = cfg.container_hold_z+cfg.wall+cfg.container_height+cfg.radius
    vz = (rim_z-p[2]+.5*cfg.gravity*duration**2)/duration
    floor_time = descending_time(p, [0., 0., vz],
                                cfg.container_hold_z+cfg.wall+cfg.radius, cfg.gravity)
    middle_time = .5*(duration+floor_time)
    clearance = cfg.radius+cfg.landing_edge_margin
    velocities, centers = [], []
    for member in range(2 if cfg.paired_launch else 1):
        for _ in range(256):
            x0, x1, y0, y1 = (cfg.receiving_cell if cfg.receiving_cell is not None else
                               (cfg.receiving_x_min, cfg.receiving_x_max, -cfg.receiving_y_limit, cfg.receiving_y_limit))
            center = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
            if not cfg.receiving_radius_min <= np.linalg.norm(center) <= cfg.receiving_radius_max:
                continue
            if np.linalg.norm(center-[cfg.container_x, cfg.container_y]) < cfg.minimum_receiver_displacement:
                continue
            v = np.r_[(center-p[:2])/middle_time, vz]
            yaw = math.degrees(math.atan2(v[1], -v[0]))
            if abs(yaw) > cfg.launch_yaw_jitter_deg:
                continue
            landing = container_table_landing(p, v, cfg)
            if (landing is None or not cfg.landing_x_min+clearance <= landing[0] <= cfg.landing_x_max-clearance
                    or abs(landing[1])+clearance > cfg.landing_y_limit):
                continue
            if member:
                other = velocities[0]
                other_yaw = math.degrees(math.atan2(other[1], -other[0]))
                if (np.linalg.norm(center-centers[0]) < .16 or abs(yaw-other_yaw) < 8.
                        or abs(np.linalg.norm(v[:2])-np.linalg.norm(other[:2])) < .12):
                    continue
            velocities.append(np.r_[v, 0., 1.2, .4])
            centers.append(center)
            break
        else:
            raise ValueError('container launch pair cannot land inside the tabletop margin with distinct trajectories')
    return p, np.asarray(velocities)


def predict(position, velocity, seconds, cfg):
    p, v = np.asarray(position, dtype=float).copy(), np.asarray(velocity, dtype=float)[:3].copy()
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('prediction time must be finite and nonnegative')
    if cfg.mode == 'roll':
        # Table contacts can report vertical jitter; predict planar rolling only.
        p[2], v[2] = cfg.radius+.0005, 0.
    p += seconds*v
    if cfg.mode != 'roll':
        p[2] -= .5*cfg.gravity*seconds**2
        v[2] -= cfg.gravity*seconds
    return p, v


def descending_time(position, velocity, height, gravity):
    """Time until the future descending crossing; None for an unreachable plane."""
    values = (*np.asarray(position), *np.asarray(velocity)[:3], height, gravity)
    if not all(math.isfinite(v) for v in values) or gravity <= 0:
        raise ValueError('invalid flight state')
    vz = velocity[2]
    discriminant = vz*vz+2*gravity*(position[2]-height)
    if discriminant < 0:
        return None
    seconds = (vz+math.sqrt(discriminant))/gravity
    return seconds if seconds >= 0 else None


def container_table_landing(position, velocity, cfg):
    """Free-flight ball centre at first tabletop contact height, before a bounce.

    Used only to choose/reset a launch and report its intended landing point.
    Contacts with the container can subsequently change the actual trajectory.
    """
    seconds = descending_time(position, velocity, cfg.landing_table_z+cfg.radius, cfg.gravity)
    return None if seconds is None else predict(position, velocity, seconds, cfg)[0]


def interception(position, velocity, cfg):
    if cfg.mode == 'roll':
        if velocity[0] >= -1e-6:
            return None
        seconds = (cfg.intercept_x-position[0])/velocity[0]
        if seconds < 0:
            return None
    else:
        height = cfg.catch_z
        if cfg.mode == 'container':
            height = cfg.container_hold_z+cfg.wall+cfg.container_height+cfg.radius
        seconds = descending_time(position, velocity, height, cfg.gravity)
        if seconds is None:
            return None
    return predict(position, velocity, seconds, cfg)[0], seconds
