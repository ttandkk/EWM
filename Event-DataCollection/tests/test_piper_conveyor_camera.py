"""Rigid wrist-camera mounting remains continuous through a downward view."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from piper_conveyor import rotation_xyzw, wrist_mount_rotation


def world_up_view(forward):
    backward = -forward/np.linalg.norm(forward)
    right = np.cross([0., 0., 1.], backward)
    right /= np.linalg.norm(right)
    return np.stack((right, np.cross(backward, right), backward), axis=1)


def angle_between(a,b):
    return np.rad2deg(np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1)))


class WristCameraTests(unittest.TestCase):
    def test_optical_axis_matches_approved_local_eye_target(self):
        mount = wrist_mount_rotation()
        forward = np.array([.12, 0., .24]); forward /= np.linalg.norm(forward)
        np.testing.assert_allclose(-mount[:,2],forward,atol=1e-12)
        np.testing.assert_allclose(mount.T@mount,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(mount),1.)

    def test_initial_view_matches_world_up_reference(self):
        # The approved zero-joint scene has TCP pitch about 85 degrees.
        pitch=np.deg2rad(85.)
        robot=rotation_xyzw([0.,np.sin(pitch/2),0.,np.cos(pitch/2)])
        mount=wrist_mount_rotation()
        np.testing.assert_allclose(robot@mount,world_up_view(-robot@mount[:,2]),atol=1e-12)

    def test_downward_crossing_has_no_camera_flip(self):
        mount=wrist_mount_rotation()
        pitches=np.deg2rad([153.,154.])
        robots=[rotation_xyzw([0.,np.sin(t/2),0.,np.cos(t/2)]) for t in pitches]
        cameras=[robot@mount for robot in robots]
        old=[world_up_view(-camera[:,2]) for camera in cameras]
        self.assertGreater(angle_between(old[0],old[1]),179.)
        self.assertAlmostEqual(angle_between(cameras[0],cameras[1]),1.,places=9)


if __name__ == '__main__':unittest.main()
