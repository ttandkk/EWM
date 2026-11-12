"""Piper IK with the TCP lever arm expressed in the same frame as its Jacobian."""
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import math as math_utils
import torch


class ConveyorIKAction(DifferentialInverseKinematicsAction):
    def _compute_frame_jacobian(self):
        # Clone the cached body Jacobian: computing a TCP Jacobian must not
        # modify the articulation's reusable world-frame body Jacobian.
        jacobian = self.jacobian_w.clone()
        inverse_base = math_utils.quat_inv(self._asset.data.root_quat_w.torch)
        base_matrix = math_utils.matrix_from_quat(inverse_base)
        jacobian[:, :3] = torch.bmm(base_matrix, jacobian[:, :3])
        jacobian[:, 3:] = torch.bmm(base_matrix, jacobian[:, 3:])
        if self.cfg.body_offset is not None:
            body_quat_base = math_utils.quat_mul(inverse_base, self._asset.data.body_quat_w.torch[:,self._body_idx])
            offset_base = math_utils.quat_apply(body_quat_base, self._offset_pos)
            jacobian[:, :3] += torch.bmm(-math_utils.skew_symmetric_matrix(offset_base), jacobian[:, 3:])
        return jacobian

    def apply_actions(self):
        ee_pos,ee_quat=self._compute_frame_pose()
        joint_pos=self._asset.data.joint_pos.torch[:,self._joint_ids]
        jacobian=self._compute_frame_jacobian()
        position_error,rotation_error=math_utils.compute_pose_error(
            ee_pos,ee_quat,self._ik_controller.ee_pos_des,self._ik_controller.ee_quat_des,
            rot_error_type='axis_angle')
        error=torch.cat((position_error,rotation_error),dim=-1)
        # Near a straight wrist, joints 4 and 6 have almost the same axis.
        # Penalize their opposing motion without changing the Cartesian command.
        damping=torch.full_like(joint_pos,1e-4)
        wrist_damping=.04*torch.exp(-(joint_pos[:,4]/.25)**2)
        damping[:,3]+=wrist_damping;damping[:,5]+=wrist_damping
        transpose=jacobian.transpose(-1,-2)
        delta=torch.linalg.solve(transpose@jacobian+torch.diag_embed(damping),
                                 (transpose@error.unsqueeze(-1))).squeeze(-1)
        self._asset.set_joint_position_target_index(target=joint_pos+delta,joint_ids=self._joint_ids)
