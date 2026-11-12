"""WebSocket transport, compile options, and ownership of inference state."""
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from websockets.sync.server import serve

from scripts.eval_ewm.check_ewm_policy_server import check_server
from scripts.eval_ewm.ewm_eval_time_audit import EVENT_TIME_CONTRACT
from scripts.eval_ewm.ewm_eval_wire import MAX_MESSAGE, decode_message, encode_message


@pytest.mark.parametrize('mode', ['default', 'reduce-overhead', 'max-autotune'])
def test_no_cudagraphs_overrides_mode_and_global_default(mode):
    from torch._inductor import config
    from scripts.eval_ewm.ewm_eval_server import configure_inference_compile
    calls = []
    model = SimpleNamespace(compile_inference=lambda **kwargs:calls.append(kwargs))
    with config.patch({'triton.cudagraphs': True}):
        enabled = configure_inference_compile(model,mode=mode,cache_bf16_weights=True,cuda_graphs=False)
    assert enabled is False
    assert calls[0]['mode'] is None and calls[0]['cache_bf16_weights']
    assert calls[0]['options']['triton.cudagraphs'] is False


@pytest.mark.parametrize('active,graphs,error', [(True,False,None),(False,False,'not running compiled'),
                                               (True,True,'disabled CUDA Graphs')])
def test_real_websocket_probe_across_handler_threads(active,graphs,error):
    from websockets.exceptions import ConnectionClosed
    calls, threads = [], []
    main_thread = threading.get_ident()

    def handler(socket):
        threads.append(threading.get_ident())
        try:
            while True:
                request, arrays = decode_message(socket.recv())
                if request['op'] == 'hello':
                    socket.send(encode_message(dict(ok=True,event_time_contract=EVENT_TIME_CONTRACT,
                        compile_core_active=active,cuda_graphs_enabled=graphs,compile_mode='default',
                        checkpoints=[40000],horizon=24,event_cameras=2,voxel_bins=1)))
                else:
                    calls.append(request)
                    anchor = round(request['timing']['obs_video_time_s'] * 30)
                    assert set(arrays) == {'obs_video', 'obs_voxel', 'proprio'}
                    assert request['timing']['obs_video_time_s'] == anchor/30
                    assert request['timing']['obs_event_end_s']-request['timing']['obs_event_start_s'] == pytest.approx(1/30)
                    assert not request['decode_future_events']
                    actions = np.full((16,7),anchor/100,np.float32)
                    socket.send(encode_message(dict(ok=True,step=40000,seed=request['seed'],inference_seconds=.05),
                                               dict(executed_actions=actions)))
        except ConnectionClosed:
            pass

    with serve(handler,'127.0.0.1',0,compression=None,max_size=MAX_MESSAGE) as server:
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            ready = dict(url=f'ws://127.0.0.1:{server.socket.getsockname()[1]}')
            if error:
                with pytest.raises(ValueError,match=error):
                    check_server(ready,'catch the ball',require_compiled=True,require_no_cudagraphs=True)
                assert not calls
            else:
                result = check_server(ready,'catch the ball',require_compiled=True,require_no_cudagraphs=True)
                assert result['passed'] and result['connections'] == 2 and result['requests'] == 4
                assert result['max_reconnect_action_error'] == 0 and len(threads) == 2
                assert all(t != main_thread for t in threads)
                assert len(calls) == 4
        finally:
            server.shutdown()
            worker.join(timeout=5)
        assert not worker.is_alive()


@pytest.fixture
def serialized_policy_server(tmp_path):
    from websockets.sync.client import connect
    from scripts.eval_ewm.ewm_eval_server import serve_on_inference_thread

    ready_file = tmp_path/'ready.json'
    local = threading.local()
    state = dict(calls=[], failures=[], release_visualization=threading.Event())
    metadata = dict(event_time_contract=EVENT_TIME_CONTRACT, compile_core_active=True,
        cuda_graphs_enabled=True, compile_mode='reduce-overhead', checkpoints=[40000],
        horizon=24, event_cameras=2, voxel_bins=1)

    def run():
        # Like CUDA Graph warmup, initialization belongs to exactly one thread.
        local.initialized = True
        state['owner'] = threading.get_ident()

        def process(request, arrays):
            assert getattr(local, 'initialized', False), 'inference moved to an uninitialized thread'
            state['calls'].append((request['op'], threading.get_ident()))
            if request['op'] == 'hello':
                yield encode_message(dict(ok=True, **metadata))
            elif request['op'] == 'fail':
                raise ValueError('invalid test request')
            elif request['op'] == 'deferred':
                yield encode_message(dict(ok=True, op='actions'))
                assert state['release_visualization'].wait(5), 'action response was held behind decoding'
                yield encode_message(dict(ok=True, op='predict_visualization'))
            else:
                anchor = round(request['timing']['obs_video_time_s']*30)
                yield encode_message(dict(ok=True, step=40000, seed=request['seed'], inference_seconds=.05),
                    dict(executed_actions=np.full((16,7),anchor/100,np.float32)))

        try:
            serve_on_inference_thread(process, metadata, ready_file)
        except BaseException as exc:
            state['failures'].append(exc)

    owner = threading.Thread(target=run, daemon=True)
    owner.start()
    try:
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            if state['failures']:
                raise state['failures'][0]
            try:
                state['ready'] = json.loads(ready_file.read_text())
                break
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(.01)
        else:
            pytest.fail('inference server did not become ready')
        yield state
    finally:
        state['release_visualization'].set()
        if owner.is_alive() and 'ready' in state:
            with connect(state['ready']['url'], open_timeout=5) as socket:
                socket.send(encode_message(dict(op='shutdown')))
                reply, _ = decode_message(socket.recv(timeout=5))
                assert reply['ok']
        owner.join(timeout=5)
        assert not owner.is_alive()
        assert not state['failures']
        assert not ready_file.exists()


def test_inference_owner_survives_request_error_and_reconnections(serialized_policy_server):
    from websockets.sync.client import connect
    state = serialized_policy_server
    with connect(state['ready']['url']) as socket:
        socket.send(encode_message(dict(op='fail')))
        reply, _ = decode_message(socket.recv(timeout=5))
        assert not reply['ok'] and 'invalid test request' in reply['error']

    result = check_server(state['ready'], 'catch the ball', require_compiled=True)
    assert result['passed'] and result['cuda_graphs_enabled']
    assert result['connections'] == 2 and result['requests'] == 4
    assert result['max_reconnect_action_error'] == 0
    assert {thread for _, thread in state['calls']} == {state['owner']}
    assert state['owner'] != threading.get_ident()


def test_action_reply_precedes_deferred_decode_and_other_requests(serialized_policy_server):
    from websockets.sync.client import connect
    state = serialized_policy_server
    with connect(state['ready']['url']) as first, connect(state['ready']['url']) as second:
        first.send(encode_message(dict(op='deferred')))
        reply, _ = decode_message(first.recv(timeout=5))
        assert reply['op'] == 'actions'
        second.send(encode_message(dict(op='hello')))
        with pytest.raises(TimeoutError):
            second.recv(timeout=.1)
        state['release_visualization'].set()
        reply, _ = decode_message(first.recv(timeout=5))
        assert reply['op'] == 'predict_visualization'
        reply, _ = decode_message(second.recv(timeout=5))
        assert reply['ok'] and reply['cuda_graphs_enabled']
    assert {thread for _, thread in state['calls']} == {state['owner']}
