"""Advance the physical carousel at every PhysX substep."""
from piper_bench.robots.piper.control import ConveyorIKAction


class RotatingRackIKAction(ConveyorIKAction):
    def apply_actions(self):
        runtime=getattr(self._env,'_rotating_tube_runtime',None)
        if runtime is not None:runtime.advance_rack()
        super().apply_actions()
