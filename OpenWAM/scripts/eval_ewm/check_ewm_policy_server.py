"""Validate real threaded WebSocket inference before starting physical rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from websockets.sync.client import connect

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval_ewm.ewm_eval_time_audit import EVENT_TIME_CONTRACT
from scripts.eval_ewm.ewm_action_only_protocol import OBSERVATION_CONTRACT
from scripts.eval_ewm.ewm_rgb_action_protocol import OBSERVATION_CONTRACT as RGB_ACTION_CONTRACT
from scripts.eval_ewm.ewm_eval_wire import MAX_MESSAGE, decode_message, encode_message


def check_server(ready, prompt, *, require_compiled=False, require_no_cudagraphs=False):
    calls, references, metadata = [], {}, None
    max_reconnect_error = 0.
    # Separate connections exercise newly created handler threads, unlike main-thread warmup.
    for connection in range(2):
        with connect(ready['url'], open_timeout=30, max_size=MAX_MESSAGE, compression=None) as socket:
            socket.send(encode_message(dict(op='hello')))
            metadata, _ = decode_message(socket.recv(timeout=600))
            rgb_action = metadata.get('training_mode') == 'rgb_action'
            action_only = metadata.get('training_mode') == 'action_only'
            rgb_state_only = action_only or rgb_action
            contract = RGB_ACTION_CONTRACT if rgb_action else OBSERVATION_CONTRACT
            contract_ok = (metadata.get('observation_contract') == contract if rgb_state_only
                           else metadata.get('event_time_contract') == EVENT_TIME_CONTRACT)
            if not metadata.get('ok') or not contract_ok:
                raise ValueError('server hello has an invalid policy/time contract')
            if require_compiled and metadata.get('compile_core_active') is not True:
                raise ValueError('server is not running compiled inference')
            if require_no_cudagraphs and metadata.get('cuda_graphs_enabled') is not False:
                raise ValueError('server has not disabled CUDA Graphs')
            step, horizon = max(metadata['checkpoints']), min(16, metadata['horizon'])
            cameras = 2 if rgb_state_only else metadata['event_cameras']
            bins = 1 if rgb_state_only else metadata['voxel_bins']
            for anchor in (0, 9):
                rng = np.random.default_rng(anchor)
                arrays = dict(obs_video=rng.integers(0,256,(2,3,256,256),dtype=np.uint8),
                    obs_voxel=np.zeros((cameras,2*bins,256,256),np.float32),
                    proprio=np.zeros(8,np.float32))
                now = anchor/30
                header = dict(op='predict', step=step, seed=812000+anchor, prompt=prompt,
                    sampling_steps=4, action_chunk_size=horizon, raw_observation=True,
                    decode_future_events=False, event_time_contract=EVENT_TIME_CONTRACT,
                    observation_row_offset=0, timing=dict(obs_video_time_s=now, proprio_time_s=now,
                        obs_event_start_s=(anchor-1)/30, obs_event_end_s=now))
                if rgb_state_only:
                    arrays = {k: arrays[k] for k in ('obs_video', 'proprio')}
                    header.pop('event_time_contract')
                    header['observation_contract'] = contract
                    if rgb_action:
                        header['decode_future_rgb'] = False
                    header['timing'] = {k: header['timing'][k] for k in ('obs_video_time_s', 'proprio_time_s')}
                started = time.perf_counter()
                socket.send(encode_message(header,arrays))
                reply, outputs = decode_message(socket.recv(timeout=600))
                elapsed = time.perf_counter()-started
                if not reply.get('ok'):
                    raise RuntimeError(f"WebSocket policy check failed: {reply.get('error', reply)}")
                actions = outputs.get('executed_actions')
                if (reply.get('step') != step or reply.get('seed') != header['seed'] or actions is None
                        or actions.shape != (horizon,7) or actions.dtype != np.float32
                        or not np.isfinite(actions).all()):
                    raise ValueError('invalid action reply in WebSocket policy check')
                inference = float(reply['inference_seconds'])
                if not np.isfinite(inference) or inference < 0:
                    raise ValueError('invalid inference timing in WebSocket policy check')
                if anchor in references:
                    error = float(np.max(np.abs(actions-references[anchor])))
                    max_reconnect_error = max(max_reconnect_error,error)
                    if error > .02:
                        raise ValueError(f'action outputs changed across connections: {error}')
                else:
                    references[anchor] = actions.copy()
                calls.append(dict(connection=connection, anchor=anchor, inference_seconds=inference,
                                  round_trip_seconds=elapsed))
    return dict(passed=True, synthetic_diagnostic=True, connections=2, requests=len(calls),
                checkpoint_step=step, compile_core_active=metadata.get('compile_core_active'),
                compile_mode=metadata.get('compile_mode'), cuda_graphs_enabled=metadata.get('cuda_graphs_enabled'),
                max_reconnect_action_error=max_reconnect_error, calls=calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prompt',required=True)
    parser.add_argument('--require-compiled',action='store_true')
    parser.add_argument('--require-no-cudagraphs',action='store_true')
    args = parser.parse_args()
    report = check_server(json.loads(args.ready_file.read_text()),args.prompt,
        require_compiled=args.require_compiled,require_no_cudagraphs=args.require_no_cudagraphs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('WEBSOCKET_POLICY_CHECK_PASSED',json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
