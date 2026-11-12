"""Public dataset API, loaded lazily so inference can import normalization alone.

Requesting the registry still performs the existing full built-in registration;
importing a single normalizer does not require every dataset's I/O dependencies.
"""
from importlib import import_module

_EXPORTS = {
    "BaseDataset": "bases",
    "EBenchDataset": "ebench", "MultiEBenchDataset": "ebench",
    "DATASET_REGISTRY": "registry", "build_dataset": "registry",
    "list_registered_datasets": "registry", "register_dataset": "registry",
    "RoboDojoDataset": "robodojo", "MultiTaskRoboDojoDataset": "robodojo",
    "RoboTwinDataset": "robotwin", "MultiTaskRoboTwinDataset": "robotwin",
    "ROBOTWIN_ALL_TASKS": "robotwin", "EEF_ACTION_DIM": "robotwin",
    "EEF_GRIPPER_INDICES": "robotwin", "JOINT_GRIPPER_INDICES": "robotwin",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
