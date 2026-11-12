"""Shared imaging conditions for the eight dynamic tasks; speed rules are separate.

This manifest describes perturbations, not a policy-evaluation runner. Pair
trial i with slot i of dynamic_collection.make_plan(..., 'evaluation'), retry=0.
"""
from piper_bench.collection.sampling import TASKS
from piper_bench.sensors.lighting import sample_plan as lighting_plan
from piper_bench.sensors.motion_blur import sample_plan as blur_plan

VERSION = "piper_eight_dynamic_imaging_ood_v3"
BACKGROUND_VERSION = "piper_eight_dynamic_background_ood_v5_grey_green"
LEGACY_BACKGROUND_VERSION = "piper_eight_dynamic_background_ood_v4"
BACKGROUND_PROFILES = (BACKGROUND_VERSION, LEGACY_BACKGROUND_VERSION)
CONDITIONS = ("id", "lighting_ood", "blur_ood", "background_ood")


def background_metadata(version=BACKGROUND_VERSION):
    if version not in BACKGROUND_PROFILES:
        raise ValueError(f"Unknown background profile: {version}")
    from piper_bench.scenes.background import BACKGROUND_ID
    from piper_bench.scenes.background_ood import metadata as room_metadata
    from piper_bench.scenes.tabletop_marble import metadata as tabletop_metadata
    return dict(room=room_metadata(), tabletop=tabletop_metadata(low_contrast=version == LEGACY_BACKGROUND_VERSION),
                reference_background_id=BACKGROUND_ID, changed_surfaces=["walls", "floor", "tabletop"],
                fixed_profile=True, changes_cameras=False, changes_lights=False,
                changes_task_objects=False, changes_physics=False)


def configure_background(scene, version=BACKGROUND_VERSION):
    """Use the formal grey-green v5 profile; v4 is an explicit historical option."""
    metadata = background_metadata(version)
    from piper_bench.scenes.background import configure_room_background
    from piper_bench.scenes.background_ood import BACKGROUND_ID
    from piper_bench.scenes.tabletop_marble import configure_marble_tabletop
    configure_room_background(scene, profile=BACKGROUND_ID)
    configure_marble_tabletop(scene, low_contrast=version == LEGACY_BACKGROUND_VERSION)
    return metadata


def make_plan(task, condition, seed, *, background_version=BACKGROUND_VERSION):
    if task not in TASKS or condition not in CONDITIONS or seed < 0:
        raise ValueError("invalid dynamic task, imaging condition or seed")
    rows = [dict(trial=i) for i in range(50)]
    implementation = None
    if condition == "lighting_ood":
        sampled = lighting_plan(seed)
        rows, implementation = sampled["episodes"], sampled["version"]
    elif condition == "blur_ood":
        sampled = blur_plan(seed)
        rows, implementation = sampled["episodes"], sampled["version"]
    return dict(version=background_version if condition == 'background_ood' else VERSION,
                task=task, condition=condition, seed=seed,
                trials=50, episodes=rows, implementation=implementation,
                background=background_metadata(background_version) if condition == "background_ood" else None,
                motion_distribution="ID", combined_perturbations=False,
                scene_pairing="same evaluation seed and slot; retry=0; no success filtering",
                cameras=["cam0", "cam_wrist"], output_hz=30, event_hz=240,
                event_threshold=.15, event_window="(t_(k-1), t_k]",
                initial_event_frame="empty", raw_events=True,
                event_input="sharp HDR from the condition's scene")
