"""Physical success edge cases, tested without launching Isaac Sim."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from piper_conveyor import ConveyorConfig, SuccessTracker, object_in_basket, success_now


class ConveyorGeometryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ConveyorConfig()
        self.poses = np.array([[0.155, 0.18, 0.022, 0, 0, 0, 1],
                               [0.205, 0.18, 0.022, 0, 0, 0, 1]], dtype=float)
        self.vel = np.zeros((2, 6))
        self.tcp = np.array([0.30, 0.0, 0.30])

    def test_simultaneous_supply_has_clearance_and_stays_on_belt(self):
        self.assertEqual(self.cfg.spawn_interval, 0.)
        for seed in range(100):
            x,y,yaw=self.cfg.supply_positions(seed)
            self.assertGreater(y[0]-y[1],self.cfg.object_size*np.sqrt(2)+.01)
            self.assertTrue(np.all(np.abs(x-self.cfg.belt_x)+self.cfg.object_size*np.sqrt(2)/2<self.cfg.belt_width/2))
            self.assertTrue(np.all(np.abs(y)+self.cfg.object_size*np.sqrt(2)/2<self.cfg.belt_length/2))
            self.assertTrue(np.all(np.abs(yaw)<=self.cfg.yaw_jitter))
        with self.assertRaises(ValueError):ConveyorConfig(object_spacing=.04)
        with self.assertRaises(ValueError):ConveyorConfig(spawn_interval=-1.)

    def test_random_spacing_and_yaw_are_reproducible_and_bounded(self):
        cfg=ConveyorConfig(object_spacing_jitter=.01,yaw_jitter=np.deg2rad(20.))
        gaps=[];angles=[]
        for seed in range(100):
            x,y,yaw=cfg.supply_positions(seed)
            repeated=cfg.supply_positions(seed)
            for first,second in zip((x,y,yaw),repeated):np.testing.assert_array_equal(first,second)
            gaps.append(y[0]-y[1]);angles.extend(yaw)
            self.assertTrue(.13<=gaps[-1]<=.15)
            self.assertTrue(np.all(np.abs(yaw)<=np.deg2rad(20.)))
        self.assertGreater(np.ptp(gaps),.018)
        self.assertGreater(np.ptp(angles),np.deg2rad(35.))
        with self.assertRaises(ValueError):ConveyorConfig(object_spacing_jitter=-.01)
        with self.assertRaises(ValueError):ConveyorConfig(object_spacing_jitter=.1)

    def test_both_objects_spawn_once_at_time_zero(self):
        from types import SimpleNamespace
        from piper_conveyor import ConveyorRuntime
        class Object:
            def __init__(self):self.poses=[];self.velocities=[]
            def write_root_pose_to_sim(self,pose):self.poses.append(pose.clone())
            def write_root_velocity_to_sim(self,velocity):self.velocities.append(velocity.clone())
        cubes=[Object(),Object()]
        runtime=ConveyorRuntime(SimpleNamespace(scene={'cube_1':cubes[0],'cube_2':cubes[1]},device='cpu'),self.cfg)
        self.assertEqual(runtime.supply(),[0,1])
        self.assertEqual(runtime.active,[True,True])
        runtime.control_step=90
        self.assertEqual(runtime.supply(),[])
        self.assertEqual([len(c.poses) for c in cubes],[1,1])
        self.assertEqual(runtime.metadata()['spawn_times'],[0.,0.])

    def test_compact_layout_keeps_basket_clear_of_belt(self):
        outer_right = self.cfg.basket_x+self.cfg.basket_length/2+self.cfg.basket_taper+self.cfg.wall_thickness+0.003
        belt_left = self.cfg.belt_x-self.cfg.belt_width/2
        self.assertGreater(belt_left-outer_right, 0.01)
        self.assertLess(np.linalg.norm([self.cfg.belt_x, self.cfg.pick_y]), 0.4)
        self.assertLess(np.linalg.norm([self.cfg.basket_x, self.cfg.basket_y]), 0.3)

    def test_red_lattice_has_open_walls_and_finite_small_geometry(self):
        from piper_conveyor_basket import basket_mesh
        vertices, faces = basket_mesh(self.cfg)
        points = np.asarray(vertices)
        self.assertTrue(np.isfinite(points).all())
        self.assertLess(np.ptp(points[:,0]), 0.15)
        self.assertLess(np.ptp(points[:,1]), 0.13)
        self.assertLess(np.ptp(points[:,2]), 0.06)
        self.assertLess(max(max(f) for f in faces), len(points))

    def test_two_released_objects_require_one_second(self):
        tracker = SuccessTracker(self.cfg)
        for _ in range(29):
            self.assertFalse(tracker.update(self.poses, self.vel, self.tcp, [True, True]))
        self.assertTrue(tracker.update(self.poses, self.vel, self.tcp, [True, True]))

    def test_center_inside_but_crossing_wall_is_not_success(self):
        pose = self.poses[1].copy()
        pose[0] = self.cfg.basket_x+self.cfg.basket_length/2-0.005
        self.assertFalse(object_in_basket(pose, self.cfg))

    def test_rotated_corners_are_checked(self):
        pose = self.poses[1].copy()
        pose[0] = self.cfg.basket_x+self.cfg.basket_length/2-0.024
        self.assertTrue(object_in_basket(pose, self.cfg))
        pose[5:7] = [np.sin(np.pi/8), np.cos(np.pi/8)]
        self.assertFalse(object_in_basket(pose, self.cfg))

    def test_only_one_spawned_object_is_not_success(self):
        self.assertFalse(success_now(self.poses, self.vel, self.tcp, [True, False], self.cfg))

    def test_held_or_high_object_is_not_success(self):
        self.assertFalse(success_now(self.poses, self.vel, self.poses[0, :3], [True, True], self.cfg))
        self.poses[0, 2] = 0.20
        self.assertFalse(success_now(self.poses, self.vel, self.tcp, [True, True], self.cfg))

    def test_bounce_resets_stability(self):
        tracker = SuccessTracker(self.cfg)
        for _ in range(20):
            tracker.update(self.poses, self.vel, self.tcp, [True, True])
        self.vel[0, 2] = 0.1
        self.assertFalse(tracker.update(self.poses, self.vel, self.tcp, [True, True]))
        self.assertEqual(tracker.steps, 0)
        self.vel[:] = 0
        for _ in range(29):
            self.assertFalse(tracker.update(self.poses, self.vel, self.tcp, [True, True]))
        self.assertTrue(tracker.update(self.poses, self.vel, self.tcp, [True, True]))


if __name__ == '__main__':
    unittest.main()
