"""PhysX-backed rigid wrist poses for Isaac Lab's Fabric camera backend.

FabricFrameView creates an authoritative world matrix when a camera is first
read. Thus its USD parent alone does not keep that matrix attached to the hand.
Resolve the actual physical hand and the fixed local installation at capture
rather than from a controller's possibly stale cached end-effector frame.
"""
import torch
import omni.physics.tensors as physics_tensors
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera
from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_mul


class RigidPiperWristCamera(Camera):
    def _initialize_impl(self):
        self._hand_sim_view = physics_tensors.create_simulation_view(
            'torch', stage_id=sim_utils.get_current_stage_id())
        self._hand_sim_view.set_subspace_roots('/')
        hand_path = self.cfg.prim_path.rsplit('/', 1)[0].replace('.*', '*')
        self._hand_view = self._hand_sim_view.create_rigid_body_view(hand_path)
        if self._hand_view.count < 1:
            raise RuntimeError(f'No physical wrist link matched {hand_path}')
        if self.cfg.offset.convention != 'opengl':
            raise ValueError('Rigid wrist offsets must use the OpenGL convention')
        super()._initialize_impl()
        if self._hand_view.count != self._view.count:
            raise RuntimeError('Camera/physical-wrist instance counts differ')

    def _update_poses(self, env_ids=None, env_mask=None, frame_op=0):
        hand = self._hand_view.get_transforms().to(device=self.device)
        count = hand.shape[0]
        local_position = torch.tensor(self.cfg.offset.pos, dtype=torch.float32, device=self.device)[None].expand(count, -1)
        local_rotation = torch.tensor(self.cfg.offset.rot, dtype=torch.float32, device=self.device)[None].expand(count, -1)
        self.set_world_poses(
            positions=hand[:, :3]+quat_apply(hand[:, 3:7], local_position),
            orientations=quat_mul(hand[:, 3:7], local_rotation), convention='opengl')
        super()._update_poses(env_ids=env_ids, env_mask=env_mask, frame_op=frame_op)
