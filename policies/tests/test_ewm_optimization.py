"""Exercise native compiler configuration and readiness gates without a GPU."""
import contextlib
import copy
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("allow,error,raises", [(True,.1,False),(False,.1,True),(False,.001,False)])
def test_compile_deviations_recorded_unless_strict(monkeypatch, allow, error, raises):
    torch = pytest.importorskip("torch")
    pytest.importorskip("openwam")
    from piper_policies.ewm.optimization import prepare_inference
    zeros, generator = torch.zeros, torch.Generator
    monkeypatch.setattr(torch, "zeros", lambda *a, **kw: zeros(*a, **{**kw, "device":"cpu"}))
    monkeypatch.setattr(torch, "Generator", lambda **kw: generator(device="cpu"))
    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    class Model:
        compiled = False
        _inference_bf16_weights = False
        def predict_velocity(self):
            pass
        def compile_inference(self, **kwargs):
            assert kwargs["options"]["triton.cudagraphs"] is False
            self.compiled = True
            self._inference_bf16_weights = kwargs["cache_bf16_weights"]
        def disable_inference_compile(self):
            self.compiled = False
        def sample_policy(self,*args,**kwargs):
            assert kwargs["num_steps"] == 4 and kwargs["decode_future"] is False
            return {"action_normalized":torch.tensor([error if self.compiled else 0.])}
    descriptor = dict(mode="joint",event_cameras=2,voxel_bins=1,action_horizon=24)
    unchanged = copy.deepcopy(descriptor)
    policy = SimpleNamespace(model=Model(),descriptor=descriptor,device=torch.device("cuda"),normalizer=None,cache_context_kv=True)
    if raises:
        with pytest.raises(ValueError,match="compiled action error"):
            prepare_inference(policy,compile_core=True,allow_compile_action_error=allow)
        assert not policy.model.compiled
    else:
        result = prepare_inference(policy,compile_core=True,allow_compile_action_error=allow)
        assert result["compile_core_active"] and not result["cuda_graphs_enabled"]
        assert result["startup_benchmark"]["action_max_abs_error"] == pytest.approx(error)
        assert result["startup_benchmark"]["compile_error_exceeded"] == (error>.02)
    assert descriptor == unchanged
