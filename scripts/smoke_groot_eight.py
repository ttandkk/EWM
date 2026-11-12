"""Five real single-GPU batch64 steps before the formal run; never save weights."""
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from gr00t.experiment import experiment

ROOT = Path(os.environ["TRAINING_PROJECT_ROOT"])
OUT = Path(os.environ["GROOT_EIGHT_OUTPUT"]) / "gpu_preflight"
NativeTrainer = experiment.Gr00tTrainer
NativeArguments = experiment.TrainingArguments
native_run = experiment.run


class FiniteMetrics(TrainerCallback):
    def __init__(self):
        self.metrics = []
        self.step_seconds = []

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.started = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.step_seconds.append(time.perf_counter() - self.started)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            assert math.isfinite(float(logs["loss"])), logs
            assert "grad_norm" in logs and math.isfinite(float(logs["grad_norm"])), logs
            self.metrics.append({"step": state.global_step, **logs})


class SmokeTrainer(NativeTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finite_metrics = FiniteMetrics()
        self.add_callback(self.finite_metrics)

    def save_model(self, *args, **kwargs):
        # experiment.run also saves at train end; diagnostics must not save a model.
        pass

    def train(self, *args, **kwargs):
        assert self.accelerator.num_processes == 1
        assert self.args.per_device_train_batch_size == 64
        assert self.args.gradient_accumulation_steps == 1
        assert not self.is_deepspeed_enabled
        assert (self.args.adam_beta1, self.args.adam_beta2) == (.9, .95)
        assert self.args.weight_decay == .01
        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        assert trainable and all(n.startswith('action_head.') for n in trainable), trainable
        assert any('action_encoder' in n for n in trainable)
        assert any('action_decoder' in n for n in trainable)
        result = super().train(*args, **kwargs)
        assert self.state.global_step == 5
        assert hasattr(self.model_wrapped, '_orig_mod') == bool(self.args.torch_compile), 'Unexpected model compilation state'
        assert len(self.finite_metrics.metrics) == 5
        assert any(float(m["grad_norm"]) > 0 for m in self.finite_metrics.metrics)
        report = {"rank": self.accelerator.process_index, "world_size": self.accelerator.num_processes,
                  "gpu": torch.cuda.get_device_name(), "steps": self.state.global_step,
                  "torch_compile": self.args.torch_compile,
                  "compile_backend": self.args.torch_compile_backend,
                  "deepspeed": self.is_deepspeed_enabled,
                  "engine_is_compiled": hasattr(self.model_wrapped, "_orig_mod"),
                  "batch_per_gpu": 64, "gradient_accumulation_steps": 1,
                  "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                  "trainable_names": trainable,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "step_compute_seconds": self.finite_metrics.step_seconds,
                  "metrics": self.finite_metrics.metrics, "train_metrics": result.metrics}
        (OUT / f"rank_{self.accelerator.process_index}_passed.json").write_text(json.dumps(report, indent=2))
        return result


def smoke_arguments(*args, **kwargs):
    kwargs.update(save_strategy="no", logging_steps=1, logging_first_step=True,
                  logging_nan_inf_filter=False)
    return NativeArguments(*args, **kwargs)


def smoke_run(config):
    config.training.max_steps = 5
    config.training.output_dir = str(OUT)
    config.training.experiment_name = None
    config.training.use_wandb = False
    config.training.enable_profiling = False
    native_run(config)
    assert not list(OUT.glob("checkpoint-*")), "Smoke must not write model checkpoints"
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


experiment.Gr00tTrainer = SmokeTrainer
experiment.TrainingArguments = smoke_arguments
experiment.run = smoke_run
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_LOG_MODEL"] = "false"
sys.argv[0] = str(ROOT / "Isaac-GR00T/gr00t/experiment/launch_finetune.py")
runpy.run_path(sys.argv[0], run_name="__main__")
