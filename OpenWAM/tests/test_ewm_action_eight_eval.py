"""Action-only eight-task compatibility and real RGB-only transport probe."""
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed

from scripts.eval_ewm.ewm_action_only_protocol import OBSERVATION_CONTRACT, prepare_request
from scripts.eval_ewm.ewm_eval_wire import MAX_MESSAGE, encode_message, decode_message
from scripts.eval_ewm.check_ewm_policy_server import check_server
from scripts.eval_ewm import eval_ewm_piper_tasks_sim as evaluator
from tests.test_ewm_piper_tasks_eval import export


@pytest.mark.parametrize('mode', ['default', 'reduce-overhead', 'max-autotune'])
def test_action_compile_explicitly_disables_graphs(monkeypatch, mode):
    import torch
    from torch._inductor import config
    from scripts.eval_ewm.ewm_action_only_eval_server import configure_action_compile
    calls = []
    def compile(function, **kwargs):
        calls.append(kwargs)
        return function
    monkeypatch.setattr(torch, 'compile', compile)
    model = SimpleNamespace(predict_velocity=lambda:None)
    with config.patch({'triton.cudagraphs':True}):
        assert configure_action_compile(model,mode=mode,cuda_graphs=False) is False
    assert calls[0]['options']['triton.cudagraphs'] is False
    assert calls[0]['dynamic'] is False and 'mode' not in calls[0]


def test_multitask_action_settings_keep_dataset_and_mode_guards(tmp_path):
    dataset = export(tmp_path/'data', 'drawer_cup')
    dataset.rename(dataset.with_name('piper_drawer_cup'))
    dataset = dataset.with_name('piper_drawer_cup')
    ckpt = tmp_path/'run';ckpt.mkdir()
    config = dict(model=dict(training_mode='action_only'), dataloader=dict(
        type='ewm_piper_multitask_action_only', dataset_dir=str(dataset.parent), tasks=['drawer_cup']))
    file = ckpt/'config.yaml';file.write_text(json.dumps(config))
    args = evaluator.build_parser().parse_args(['--task','drawer_cup','--policy-mode','action_only',
        '--checkpoint-dir',str(ckpt),'--dataset-dir',str(dataset),'--plan-only'])
    assert evaluator.load_settings(args)[0] == dataset
    # Dataset/type guards now belong to the independent model service. Planning
    # reads only the benchmark export and never opens this training config.
    from piper_policies.metadata import validate_training_contract
    validate_training_contract(config, task='drawer_cup', policy_mode='action_only')
    config['dataloader']['tasks'] = ['conveyor'];file.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='task is not present'):
        validate_training_contract(config, task='drawer_cup', policy_mode='action_only')
    assert evaluator.load_settings(args)[0] == dataset
    config['dataloader']['tasks'] = ['drawer_cup']
    config['dataloader']['type'] = 'ewm_piper_multitask_memory';file.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='RGB/state training contract'):
        validate_training_contract(config, task='drawer_cup', policy_mode='action_only')


def test_action_probe_uses_strict_server_contract_on_real_websocket():
    requests = []
    normalizer = SimpleNamespace(normalize_proprio=lambda x:x)
    def handler(socket):
        try:
            while True:
                request,arrays = decode_message(socket.recv())
                if request['op']=='hello':
                    socket.send(encode_message(dict(ok=True,training_mode='action_only',
                        observation_contract=OBSERVATION_CONTRACT,compile_core_active=True,
                        cuda_graphs_enabled=False,checkpoints=[80000],horizon=24)))
                else:
                    batch,seed,steps,chunk = prepare_request(request,arrays,normalizer,horizon=24,device='cpu')
                    assert set(batch)=={'obs_video','proprio','prompt'}
                    requests.append(request)
                    socket.send(encode_message(dict(ok=True,step=80000,seed=seed,inference_seconds=.01),
                        dict(executed_actions=np.zeros((chunk,7),np.float32))))
        except ConnectionClosed:
            pass
    with serve(handler,'127.0.0.1',0,compression=None,max_size=MAX_MESSAGE) as server:
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            report=check_server(dict(url=f'ws://127.0.0.1:{server.socket.getsockname()[1]}'),
                'catch',require_compiled=True,require_no_cudagraphs=True)
            assert report['passed'] and report['requests']==4 and report['connections']==2
            assert len(requests)==4
        finally:
            server.shutdown();thread.join(timeout=5)
        assert not thread.is_alive()
