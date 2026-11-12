"""Versioned sensor/action clock shared by policy clients and controllers."""
CONTROL_HZ = 30
CAMERAS = ("cam0", "cam_wrist")
WARP = 8
RESET_WARMUP_STEPS = 60
EVENT_TIME_CONTRACT = "aligned_event_frame_pre_action_v1"
OBSERVATION_CONTRACT = "rgb_state_pre_action_v1"
RGB_ACTION_CONTRACT = "rgb_state_pre_action_rgb_latent_v1"
PAUSED_CLOCK = "fixed_simulation_time_inference_paused"
LATENCY_CLOCK = "offline_30hz_simulation_with_measured_policy_latency"
