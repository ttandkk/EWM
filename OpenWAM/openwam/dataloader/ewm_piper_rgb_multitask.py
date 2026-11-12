"""Eight-task RGB/action training using the shared balanced sampler."""
from .ewm_piper_multitask import PiperMultiTaskDataset
from .ewm_rgb_action import PiperRGBActionDataset


class PiperMultiTaskRGBActionDataset(PiperMultiTaskDataset):
    dataset_class = PiperRGBActionDataset
