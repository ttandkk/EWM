"""Apply the basket's prescribed motion before each PhysX substep."""
from piper_bench.robots.piper.control import ConveyorIKAction


class MovingBasketIKAction(ConveyorIKAction):
    def apply_actions(self):
        runtime = getattr(self._env,'_moving_basket_runtime',None)
        if runtime is not None:
            runtime.advance_basket()
        super().apply_actions()
