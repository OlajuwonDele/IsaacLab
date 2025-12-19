import os
from typing import Optional

from isaacsim.core.prims import Articulation
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver

robot_prim_path = "/AR4_Main_DH_SLDASM"
path_to_robot_usd = "C:/Isaaclab/source/isaaclab_assets/isaaclab_assets/robots/ar_mk3/ar_mk3.usd"
rmp_path = "C:/Isaaclab/source/isaaclab_assets/isaaclab_assets/robots/ar_mk3/ar4_rmp_config.yaml"
robot_desc = "C:/Isaaclab/source/isaaclab_assets/isaaclab_assets/robots/ar_mk3/robot_description.yaml"
urdf = "C:/Isaaclab/source/isaaclab_assets/isaaclab_assets/robots/ar_mk3/ar4_mk3.urdf"

class KinematicsSolver(ArticulationKinematicsSolver):
    def __init__(self, robot_articulation: Articulation, end_effector_frame_name: Optional[str] = None) -> None:
        self._kinematics = LulaKinematicsSolver(
            robot_description_path=robot_desc,
            urdf_path=urdf,
            
        )
        
        print("Valid frame names at which to compute kinematics:", self._kinematics.get_all_frame_names())
        if end_effector_frame_name is None:
            end_effector_frame_name="gripper_base_link",
        ArticulationKinematicsSolver.__init__(self, robot_articulation, self._kinematics, end_effector_frame_name)
        return