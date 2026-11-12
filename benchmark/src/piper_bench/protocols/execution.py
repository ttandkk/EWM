"""Versioned execution settings, independent of GPU resources and model options."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExecutionProtocol:
    version: str
    control_hz: int = 30
    latency_aware: bool = True
    latency_controller: str = "continuous"
    min_execution_steps: int = 8
    execution_horizon: int = 10
    trials: int = 50
    scene_seed: int = 20280002
    policy_seed: int = 10000
    prediction_decode_phase: str = "post_trial"
    prediction_decode_in_equivalent_latency: bool = False

    def metadata(self):
        return asdict(self)


PROFILES = {
    "continuous_min8_v1": ExecutionProtocol("piper_eight_continuous_min8_v1"),
    "legacy_buffered_min4_v1": ExecutionProtocol(
        "piper_legacy_buffered_min4_v1", latency_controller="buffered", min_execution_steps=4
    ),
}
DEFAULT_PROFILE = "continuous_min8_v1"


def load_execution(name=DEFAULT_PROFILE):
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown execution profile {name!r}; choose from {', '.join(PROFILES)}") from exc
