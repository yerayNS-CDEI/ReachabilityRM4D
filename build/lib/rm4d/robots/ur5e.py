import numpy as np
import pybullet as p
from .base import RobotBase
from .assets import UR5E_URDF


class UR5E(RobotBase):
    def __init__(self, simulator, base_pos=None, base_orn=None):
        super().__init__(simulator, base_pos, base_orn)
        self._robot_id = self.sim.bullet_client.loadURDF(UR5E_URDF,
                                                         self.base_pos, self.base_orn, useFixedBase=True)
        
        print(">>> Loading URDF from:", UR5E_URDF)

        # robot properties
        self._end_effector_link_id = 17
        self._arm_joint_ids = [0, 1, 2, 3, 4, 5]
        self.home_conf = [1.57, -1.57, 1.57, -1.57, -1.57, 0.0]

        # for i in range(self.sim.bullet_client.getNumJoints(self.robot_id)):
        #     print(i, self.sim.bullet_client.getJointInfo(self.robot_id, i)[12])

        # determine joint limits
        limits = []
        for i in self.arm_joint_ids:
            limits.append(self.sim.bullet_client.getJointInfo(self.robot_id, i)[8:10])
        self._joint_limits = np.asarray(limits)

        # set initial robot configuration
        self.reset_joint_pos(self.home_conf)

    @property
    def range_radius(self) -> float:
        return 1.10

    @property
    def range_z(self) -> float:
        return 1.25

    @property
    def end_effector_link_id(self):
        return self._end_effector_link_id

    @property
    def robot_id(self):
        return self._robot_id

    @property
    def arm_joint_ids(self):
        return self._arm_joint_ids

    @property
    def joint_limits(self):
        return self._joint_limits

    def in_self_collision(self):
        # check self-collision
        ignore_links = [11, 16, 17]  # they do not have a collision shape
        first_links = [0, 1, 2, 3, 4]  # 5 cannot collide with the fingers due to kinematics

        for first_link in first_links:
            # skip links that are next to each other (supposed to be in contact) plus all the ignore links
            check_links = [link for link in np.arange(first_link + 2, self.end_effector_link_id + 1) if
                           link not in ignore_links]
            for check_link in check_links:
                collision = self.sim.links_in_collision(self.robot_id, first_link, self.robot_id, check_link)
                if collision:
                    return True
        return False

    def _do_inverse_kinematics(self, pos, quat, start_q, rest_q, n_iterations, threshold):
        self.reset_joint_pos(start_q)

        # get joint ranges and limits
        lower_limits = self.joint_limits[:, 0]
        upper_limits = self.joint_limits[:, 1]
        joint_ranges = upper_limits - lower_limits

        # include finger joints
        lower_limits = lower_limits.tolist() + [0.0]*6
        upper_limits = upper_limits.tolist() + [0.0]*6
        joint_ranges = joint_ranges.tolist() + [0.0]*6
        rest_poses = rest_q.tolist() + [0.0]*6

        # execute IK
        full_joint_positions = self.sim.bullet_client.calculateInverseKinematics(
            self.robot_id, self.end_effector_link_id, pos, quat,
            lowerLimits=lower_limits, upperLimits=upper_limits, jointRanges=joint_ranges, restPoses=rest_poses,
            maxNumIterations=n_iterations, residualThreshold=threshold)

        return np.array(full_joint_positions)[self.arm_joint_ids]
