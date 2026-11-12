"""CPU adapter boundaries with the native processor/engine and a sampler double."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf


def descriptor():
    result=json.loads((Path(__file__).parent / "fixtures/hello_joint.json").read_text())["policy_descriptor"]
    result.update(mode="rgb_action",action_horizon=4,event_horizon=4,checkpoint_step=42,
                  observation_contract="rgb_state_pre_action_rgb_latent_v1",post_trial_visualization_supported=False)
    return result


def request():
    return dict(op="predict",prompt="move",raw=True,sampling_steps=4,action_chunk_size=4,
                observation_contract="rgb_state_pre_action_rgb_latent_v1",timing=dict(obs_video_time_s=0.,proprio_time_s=0.))


def arrays():
    video=np.zeros((2,3,256,256),np.uint8);video[1]=255
    return dict(obs_video=video,proprio=np.full(8,4.,np.float32))


def test_openwam_native_engine_preserves_physical_actions_and_layout():
    torch=pytest.importorskip("torch")
    pytest.importorskip("openwam")
    from openwam.deploy.engine import JointInferenceEngine
    from piper_policies.openwam.adapter import OpenWAMPolicy
    from openwam.model.action_backbone.scheduler import ActionScheduler
    from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler
    class Architecture:
        video_scheduler = FlowMatchScheduler("Wan")
        action_scheduler = ActionScheduler()
        def normalize_deploy_proprio(self,state):
            return np.asarray(state)/2
        def generate(self,**kwargs):
            frame=np.asarray(kwargs["first_frame_image"][0])
            assert frame.shape==(256,512,3) and not frame[:,:256].any() and (frame[:,256:]==255).all()
            assert (kwargs["proprio"]==2).all()
            assert kwargs["num_frames"]==5 and kwargs["action_num_frames"]==5
            assert kwargs["decode_video"] is False
            return dict(actions=np.full((4,7),.05,np.float32))
    policy=OpenWAMPolicy.__new__(OpenWAMPolicy)
    policy.model=Architecture();policy.device=torch.device("cpu");policy.descriptor=descriptor();policy.video_frames=5
    cfg=OmegaConf.create(dict(inference=dict(denoise_steps=4,denoise_mode="sync"),optimization=dict(decode_video=False)))
    policy.engine=JointInferenceEngine(cfg,architecture=policy.model)
    reply,output=policy.handle(request(),arrays())
    np.testing.assert_array_equal(output["executed_actions"],np.full((4,7),.05,np.float32))
    assert not reply["prediction_recorded"]


def test_fastwam_native_processor_roundtrip_and_layout():
    torch=pytest.importorskip("torch")
    pytest.importorskip("fastwam")
    pytest.importorskip("git", reason="FastWAM native normalizer requires its own GitPython dependency")
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
    from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
    from piper_policies.fastwam.adapter import FastWAMPolicy
    shape={"action":[dict(key="default",raw_shape=7,shape=7)],"state":[dict(key="default",raw_shape=8,shape=8)],"images":[]}
    processor=FastWAMProcessor(shape_meta=shape,num_obs_steps=5,num_output_cameras=2,action_output_dim=7,
        proprio_output_dim=8,action_state_transforms=None,use_stepwise_action_norm=False,norm_default_mode="min/max",
        norm_exception_mode=None,action_state_merger=ConcatLeftAlign(),train_transforms=None,val_transforms=None).eval()
    processor.set_normalizer_from_stats({"action":{"default":{"global_min":torch.full((7,),-2.),"global_max":torch.full((7,),4.)}},
                                         "state":{"default":{"global_min":torch.zeros(8),"global_max":torch.full((8,),8.)}}})
    class Model:
        torch_dtype=torch.float32
        def infer_action(self,**kwargs):
            image=kwargs["input_image"]
            assert image.shape==(1,3,256,512) and (image[:,:,:,:256]==-1).all() and (image[:,:,:,256:]==1).all()
            assert torch.equal(kwargs["proprio"],torch.zeros((1,8)))
            assert kwargs["prompt"].endswith("instruction: move")
            return {"action":torch.zeros((4,7))}
    policy=FastWAMPolicy.__new__(FastWAMPolicy)
    policy.processor=processor;policy.model=Model();policy.device=torch.device("cpu")
    policy.descriptor=descriptor();policy.video_frames=5;policy.action_key=policy.state_key="default"
    reply,output=policy.handle(request(),arrays())
    np.testing.assert_allclose(output["executed_actions"],np.ones((4,7)),rtol=0,atol=1e-6)
    assert not reply["prediction_recorded"]
