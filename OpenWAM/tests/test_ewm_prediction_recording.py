"""Post-trial visualization must preserve actions, clocks, and observation alignment."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.eval_ewm import ewm_prediction_recording as recording
from scripts.eval_ewm import eval_ewm_piper_conveyor_sim as client
from scripts.eval_ewm import eval_ewm_piper_tasks_sim as tasks


class Decoder(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.training_mode = kind
        self.calls = 0

    def _decode_future(self, observed, future):
        self.calls += 1
        return observed[:, None] + future


def sample_record(archive, anchor=0):
    request = dict(record_prediction=dict(session="trial", observation_step=anchor),
                   seed=123, step=80000, sampling_steps=4, timing=dict(obs_video_time_s=anchor/30))
    result = dict(observed_latent=torch.full((1, 2, 3, 4, 4), .1),
                  future_latent=torch.full((1, 3, 2, 3, 4, 4), .3))
    archive.record(request, result)
    return request, result


@pytest.mark.parametrize("kind,key", [("joint", "future_voxel"), ("rgb_action", "future_rgb")])
def test_archive_preserves_exact_condition_without_decoding_or_gpu_aliases(kind, key):
    model = Decoder(kind)
    archive = recording.PredictionArchive()
    archive.handle(dict(op="visualization_begin", session="trial"), model)
    request, result = sample_record(archive)
    result["observed_latent"].zero_()
    result["future_latent"].zero_()  # Simulate storage reuse by the next request/graph.
    assert model.calls == 0
    with pytest.raises(ValueError, match="duplicate"):
        archive.record(request, result)
    reply, arrays = archive.handle(dict(op="visualization_decode", session="trial", observation_step=0), model)
    assert model.calls == 1
    assert reply["source_observation_step"] == 0 and reply["sampling_steps"] == 4
    np.testing.assert_allclose(arrays[key], 102 if kind == "rgb_action" else .4, atol=1e-6)
    assert not archive.records and archive.bytes == 0
    reply, _ = archive.handle(dict(op="visualization_end", session="trial"), model)
    assert reply["recorded_predictions"] == 1 and reply["peak_archive_bytes"] > 0
    assert archive.session is None


def test_archive_limits_session_validation_and_interrupted_trial_reset():
    archive = recording.PredictionArchive(max_bytes=1)
    model = Decoder("joint")
    archive.handle(dict(op="visualization_begin", session="trial"), model)
    with pytest.raises(MemoryError, match="archive"):
        sample_record(archive)
    assert archive.bytes == archive.count == 0
    with pytest.raises(ValueError, match="session"):
        archive.handle(dict(op="visualization_decode", session="other", observation_step=0), model)
    archive.max_bytes = 10000
    sample_record(archive)
    archive.handle(dict(op="visualization_begin", session="new_trial"), model)
    assert archive.bytes == archive.count == 0 and not archive.records
    with pytest.raises(ValueError, match="not recorded"):
        archive.handle(dict(op="visualization_decode", session="new_trial", observation_step=0), model)


@pytest.mark.parametrize("mode", ["joint", "rgb_action", "action_only"])
def test_default_cli_recording_and_explicit_disable(mode):
    args = tasks.build_parser().parse_args(["--task", "tennis_rolling", "--policy-mode", mode])
    assert args.record_predictions is None  # capability negotiation occurs at hello
    assert not recording.recording_enabled(args)
    args = tasks.build_parser().parse_args(["--task", "tennis_rolling", "--no-record-predictions"])
    assert not recording.recording_enabled(args)
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("recording_suite", root/"scripts/eval_piper_eight_dynamic.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    assert suite.parser().parse_args(["--output-dir", "/tmp/unused"]).record_predictions is None
    assert not suite.parser().parse_args(["--output-dir", "/tmp/unused", "--no-record-predictions"]).record_predictions


@pytest.mark.parametrize("mode", ["joint", "rgb_action"])
def test_recording_requests_never_decode_inline(monkeypatch, mode):
    seen = {}
    def request(conn, header, arrays=None):
        seen.update(header=header, arrays=arrays)
        return dict(ok=True, prediction_recorded=True), {}
    from piper_bench.evaluation import policy_client
    monkeypatch.setattr(policy_client, "request", request)
    args = SimpleNamespace(policy_mode=mode, _prediction_session="trial")
    client.policy_request(None, args,
        dict(op="predict", timing=dict(obs_video_time_s=7/30, proprio_time_s=7/30),
             decode_future_events=True, defer_visualization=True),
        dict(obs_video="rgb", proprio="state", obs_voxel="events"))
    header = seen["header"]
    assert header["record_prediction"] == dict(session="trial", observation_step=7)
    assert header["decode_future_events"] is False
    assert not header.get("defer_visualization", False)
    if mode == "rgb_action":
        assert header["decode_future_rgb"] is False and "obs_voxel" not in seen["arrays"]


def test_prediction_indices_include_expired_prefix_and_paused_actions():
    assert recording.prediction_target(dict(step=8, source_observation_step=4, chunk_index=3)) == (4, 3)
    assert recording.prediction_target(dict(step=8, chunk_index=3)) == (4, 3)
    assert recording.prediction_target(dict(step=8, source_observation_step=4, chunk_index=0)) == (4, 3)
    assert recording.prediction_target(dict(step=1, action_source="bootstrap_wait_hold", chunk_index=None)) is None
    with pytest.raises(ValueError, match="alignment"):
        recording.prediction_target(dict(step=8, source_observation_step=8, chunk_index=3))


def test_failed_postprocessing_clears_archive_session(monkeypatch, tmp_path):
    args = SimpleNamespace(policy_mode="rgb_action", _prediction_session="trial", record_predictions=True)
    calls = []
    def request(conn, header):
        calls.append(header)
        return dict(recorded_predictions=3, peak_archive_bytes=100), {}
    def fail(*args, **kwargs):
        raise RuntimeError("decoder failed")
    monkeypatch.setattr(recording, "append_prediction_video", fail)
    with pytest.raises(RuntimeError, match="decoder failed"):
        recording.finish_recording(args, None, {"rgb_cameras":2}, request,
                                   SimpleNamespace(path=tmp_path/"video.mp4", ffmpeg="unused"), [], tmp_path/"heartbeat")
    assert calls == [dict(op="visualization_end", session="trial")]
    assert not hasattr(args, "_prediction_session")


@pytest.mark.parametrize("kind,cameras", [("rgb", ("cam0", "cam_wrist")), ("events", ("cam_wrist",))])
def test_real_video_append_has_exact_frame_alignment_and_no_repeated_tail(tmp_path, kind, cameras):
    import cv2
    import subprocess

    root = Path(__file__).resolve().parents[2]
    ffmpeg = sorted((root/"envs/openwam/lib").glob("python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*"))[0]
    path = tmp_path/"trial_000_recording.mp4"
    source = np.full((4, 512, 512, 3), 30, np.uint8)
    subprocess.run([str(ffmpeg), "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "512x512",
        "-r", "30", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        input=source.tobytes(), check=True)
    original = cv2.VideoCapture(str(path))
    try:
        ok, decoded_source = original.read()
        assert ok
    finally:
        original.release()
    calls = []
    def decode(anchor):
        calls.append(anchor)
        if kind == "rgb":
            forecast = np.zeros((2, 2, 3, 256, 256), np.uint8)
            forecast[1, :, 0] = 220
        else:
            forecast = np.zeros((2, 1, 2, 256, 256), np.uint8)
            forecast[1, :, 1] = 1
        return forecast, dict(source_observation_step=anchor)
    rows = [dict(step=1, action_source="bootstrap_wait_hold", chunk_index=None),
            dict(step=2, source_observation_step=0, chunk_index=1),
            dict(step=3, source_observation_step=0, chunk_index=2)]
    result = recording.append_prediction_video(path, rows, ffmpeg=str(ffmpeg), decode=decode,
        kind=kind, cameras=cameras, heartbeat=tmp_path/"heartbeat")
    assert calls == [0] and result["decoded_predictions"] == 1
    video = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        video.release()
    assert len(frames) == 4 and all(f.shape == (768, 512, 3) for f in frames)
    x = 100 if kind == "rgb" else 356
    assert frames[2][700, x, 2] > 200 and frames[2][700, x, 0] < 20
    assert frames[3][700, x].min() > 240  # Exhausted forecast is blank, not repeated.
    # Compare against the actual H.264 input, not the pre-encoding synthetic pixels.
    assert np.abs(frames[2][:500].astype(float)-decoded_source[:500]).mean() < 5


def test_post_trial_decode_uses_exact_latents_and_does_not_change_sampled_actions():
    from tests.test_ewm_asymmetric_horizon import policy, inputs

    torch.set_num_threads(2)
    model, batch = policy(), inputs()
    normalizer = SimpleNamespace(denormalize_action=lambda x: x)
    decoded = model.sample_policy(batch, horizon=40, normalizer=normalizer, num_steps=2,
        generator=torch.Generator().manual_seed(18))
    recorded = model.sample_policy(batch, horizon=40, normalizer=normalizer, num_steps=2,
        generator=torch.Generator().manual_seed(18), decode_future=False, return_observed_latent=True)
    assert "future_voxel" not in recorded
    torch.testing.assert_close(decoded["action"], recorded["action"], rtol=0, atol=0)
    torch.testing.assert_close(decoded["future_voxel"],
        model._decode_future(recorded["observed_latent"], recorded["future_latent"]), rtol=0, atol=0)


def test_continuous_recording_and_slow_video_do_not_change_equivalent_latency(monkeypatch, tmp_path):
    from tests.test_ewm_continuous_min4 import observation

    clock = SimpleNamespace(value=0.)
    monkeypatch.setattr(client, "time", SimpleNamespace(perf_counter=lambda: clock.value))
    def request(conn, args, header, arrays):
        clock.value += .04
        return dict(inference_seconds=.03), {}
    def validate(*args):
        actions = np.zeros((24, 7), np.float32)
        actions[:, 0] = np.arange(24)
        actions[:, 6] = 1.
        return actions, None, .03
    monkeypatch.setattr(client, "policy_request", request)
    monkeypatch.setattr(client, "validate_reply", validate)
    class Connection:
        def recv(self, **kwargs):
            raise AssertionError("control loop must not wait for visualization replies")
    results = []
    for enabled in (False, True):
        clock.value = 0.
        runtime = SimpleNamespace(env=SimpleNamespace(device=torch.device("cpu")), time=0.)
        def capture(runtime, *args):
            step = round(runtime.time*30)+1
            runtime.time = step/30
            obs = observation(step)
            obs["event_counts"] = {}
            return obs, step == 15, None
        class Video:
            def write(self, *args, **kwargs):
                clock.value += 20. if enabled else 0.
        options = SimpleNamespace(seed=42, max_steps=15, sampling_steps=4, policy_mode="joint",
            decode_predicted_events=enabled, latency_controller="continuous", min_execution_steps=4,
            execution_horizon=10, profile_timing=False, trials=1,
            _prediction_session="trial" if enabled else None)
        rows, chunks = [], []
        outcome = client.run_latency_loop(options, runtime, None, Connection(), {"horizon":24}, 80000,
            ("cam0", "cam_wrist"), observation(0), Video(), rows, chunks, [], tmp_path/"heartbeat", tmp_path, 0,
            capture=capture, stable_steps=lambda _: 0)
        results.append((rows, chunks, outcome))
    assert results[0][1] == results[1][1]
    for left, right in zip(results[0][0], results[1][0], strict=True):
        for key in ("action", "source_observation_step", "chunk_index", "chunk_end_reason"):
            assert left[key] == right[key]
    np.testing.assert_allclose(results[0][2][6], results[1][2][6], atol=1e-12)
    assert all(abs(v-.04) < 1e-12 for v in results[1][2][6])


def test_rgb_post_trial_decode_matches_inline_without_changing_actions():
    from tests.test_ewm_rgb_action import rgb_model, TinyRGBDataset

    torch.set_num_threads(2)
    model = rgb_model()
    item = TinyRGBDataset()[0]
    batch = {k: [v] if isinstance(v, str) else v[None] for k, v in item.items()}
    normalizer = SimpleNamespace(denormalize_action=lambda x: x)
    decoded = model.sample_policy(batch, horizon=4, normalizer=normalizer, num_steps=2,
        generator=torch.Generator().manual_seed(18))
    recorded = model.sample_policy(batch, horizon=4, normalizer=normalizer, num_steps=2,
        generator=torch.Generator().manual_seed(18), decode_future=False, return_observed_latent=True)
    assert "future_rgb" not in recorded
    torch.testing.assert_close(decoded["action"], recorded["action"], rtol=0, atol=0)
    torch.testing.assert_close(decoded["future_rgb"],
        model._decode_future(recorded["observed_latent"], recorded["future_latent"]), rtol=0, atol=0)
