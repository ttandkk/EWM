"""Shared Piper stack camera calibration for new event-frame datasets.

TCP-local offsets are in metres.  Isaac Lab PinholeCameraCfg stores focal
length and horizontal aperture in centimetres; their ratio sets the FOV.
"""

CALIBRATION_ID = "piper_stack_front_global_tight_wrist_low_v3"
CAM0_EYE_WORLD = (1.25, 0.0, 0.62)
CAM0_TARGET_WORLD = (0.24, 0.0, 0.16)
CAM0_FOCAL_LENGTH_CM = 30.0
CAM0_HORIZONTAL_APERTURE_CM = 20.955
WRIST_EYE_LOCAL = (-0.12, 0.0, -0.07)
WRIST_TARGET_LOCAL = (0.0, 0.0, 0.17)
WRIST_FOCAL_LENGTH_CM = 8.5
WRIST_HORIZONTAL_APERTURE_CM = 20.955
