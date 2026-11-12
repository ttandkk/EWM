"""Conveyor contacts and success predicate, with reduced end-effector shadows."""
from piper_bench.tasks.conveyor.core import ConveyorConfig as Config
from piper_bench.tasks.conveyor.core import ConveyorRuntime
from piper_bench.tasks.conveyor.core import TASK
from piper_bench.tasks.conveyor.core import make_env_cfg as base_env_cfg
from piper_bench.sensors.lighting import spawn_light_without_wrist_shadows as spawn_conveyor_light

PROMPT = "Pick up the Rubik's cube and the red die from the conveyor belt and place both in the basket."


def make_env_cfg(config=Config(), *, device='cpu', seed=42):
    cfg = base_env_cfg(config, device=device, seed=seed)
    cfg.scene.light.spawn.func = spawn_conveyor_light
    from piper_bench.scenes.task_assets import configure_task_assets
    configure_task_assets(cfg.scene, 'conveyor', seed)
    return cfg


class Runtime(ConveyorRuntime):
    settle_steps = 0

    def reset(self, seed):
        super().reset(seed)
        from piper_bench.scenes.task_assets import reset_task_assets
        self.tabletop_appearance = reset_task_assets(self.env, 'conveyor', seed)

    def metadata(self):
        return dict(super().metadata(), prompt=PROMPT, appearance=self.tabletop_appearance)

    def before_step(self, action):
        self.supply()

    def after_step(self, gripper_open=None):
        return super().after_step()

    def metrics(self, gripper_open=None):
        return dict(active=list(self.active), stable_steps=self.tracker.steps)
