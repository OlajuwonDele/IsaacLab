import os

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import Articulation

robot_prim_path = "/AR4_Main_DH_SLDASM"
path_to_robot_usd = "C:/Isaaclab/source/isaaclab_assets/isaaclab_assets/robots/ar_mk3/ar_mk3.usd"
rmp_path = "C:/Isaaclab/source/isaaclab_assets/isaaclab_assets/robots/ar_mk3/ar4_rmp_config.yaml"
robot_desc = "C:/Isaaclab/source/isaaclab_assets/isaaclab_assets/robots/ar_mk3/robot_description.yaml"
urdf = "C:/Isaaclab/source/isaaclab_assets/isaaclab_assets/robots/ar_mk3/ar4_mk3.urdf"

class RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name: str, robot_articulation: Articulation, physics_dt: float = 1.0 / 60.0) -> None:

        self.rmpflow = mg.lula.motion_policies.RmpFlow(
            robot_description_path=robot_desc,
            rmpflow_config_path=rmp_path,
            urdf_path=urdf,
            end_effector_frame_name="gripper_base_link",
            maximum_substep_size=0.00334,
        )

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmpflow, physics_dt)

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )