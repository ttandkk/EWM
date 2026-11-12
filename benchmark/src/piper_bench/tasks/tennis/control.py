"""Observe low ball crossings at physics frequency without driving ball motion."""
from piper_bench.robots.piper.control import ConveyorIKAction


class TennisIKAction(ConveyorIKAction):
    def apply_actions(self):
        runtime = getattr(self._env, '_tennis_runtime', None)
        if runtime is not None:
            runtime.observe_ball()
        super().apply_actions()
