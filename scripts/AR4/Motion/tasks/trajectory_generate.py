from typing import Optional

import isaacsim.core.api.tasks as tasks

import numpy as np
import pandas as pd
import carb

from isaacsim.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.prims import delete_prim, get_prim_at_path

from isaacsim.robot_motion.motion_generation import (
    LulaCSpaceTrajectoryGenerator,
    LulaTaskSpaceTrajectoryGenerator,
    LulaKinematicsSolver,
    ArticulationTrajectory
)

robot_prim_path = "/AR4_Main_DH_SLDASM"
path_to_robot_usd = "C:/Isaaclab/source/isaaclab_assets/isaaclab_assets/robots/ar_mk3/ar_mk3.usd"

robot_desc = "C:/Isaaclab/source/isaaclab_assets/isaaclab_assets/robots/ar_mk3/robot_description.yaml"
urdf = "C:/Isaaclab/source/isaaclab_assets/isaaclab_assets/robots/ar_mk3/ar4_mk3.urdf"

class TrajectoryGeneration():
    def __init__(self):
        self._c_space_trajectory_generator = None
        self._taskspace_trajectory_generator = None
        self._kinematics_solver = None

        self._action_sequence = []
        self._action_sequence_index = 0

        self._articulation = None

    def load_example_assets(self):
        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)
        print(f"Artiuclation created = {self._articulation}")
        return [self._articulation]

    def setup(self):
        self._c_space_trajectory_generator = LulaCSpaceTrajectoryGenerator(
            robot_description_path=robot_desc,
            urdf_path=urdf
        )

        self._taskspace_trajectory_generator = LulaTaskSpaceTrajectoryGenerator(
            robot_description_path=robot_desc,
            urdf_path=urdf
        )

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=robot_desc,
            urdf_path=urdf
        )

        self._end_effector_name = "gripper_base_link"

    def setup_taskspace_trajectory(self):
        df = pd.read_excel("C:/Users/juwon/Documents/Robotics/TCD Research/Annin package/annin_sample_trajectory_100_2.xlsx")
        task_space_position_targets = (df[['X', 'Y', 'Z']] / 100).to_numpy()

        # orientation is (w, x, y, z)
        task_space_orientation_targets = np.tile(np.array([1,0,0,0]),(task_space_position_targets.shape[0],1))

        trajectory = self._taskspace_trajectory_generator.compute_task_space_trajectory_from_points(
            task_space_position_targets, task_space_orientation_targets, self._end_effector_name
        )

        # Visualize target poses
        for i, (position, orientation) in enumerate(zip(task_space_position_targets, task_space_orientation_targets)):
            add_reference_to_stage(
                get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd",
                f"/visualized_frames/target_{i}"
            )
            frame = XFormPrim(
                f"/visualized_frames/target_{i}",
                positions=np.array([position]),
                orientations=np.array([orientation]),
                scales=np.array([[0.04, 0.04, 0.04]])
            )

        if trajectory is None:
            carb.log_warn("No trajectory could be computed")
            self._action_sequence = []
        else:
            physics_dt = 1 / 60
            articulation_trajectory = ArticulationTrajectory(self._articulation, trajectory, physics_dt)
            print(f"traj = {trajectory}")
            print(f"Art_traj = {articulation_trajectory}")
            self._action_sequence = articulation_trajectory.get_action_sequence()

    def update(self):
        if len(self._action_sequence) == 0:
            return

        if self._action_sequence_index >= len(self._action_sequence):
            self._action_sequence_index += 1
            self._action_sequence_index %= len(self._action_sequence) + 10
            return

        if self._action_sequence_index == 0:
            self._teleport_robot_to_position(self._action_sequence[0])

        self._articulation.apply_action(self._action_sequence[self._action_sequence_index])
        self._action_sequence_index += 1
        self._action_sequence_index %= len(self._action_sequence) + 10

    def reset(self):
        if get_prim_at_path("/visualized_frames"):
            delete_prim("/visualized_frames")

        self._action_sequence = []
        self._action_sequence_index = 0

    def _teleport_robot_to_position(self, articulation_action):
        initial_positions = np.zeros(self._articulation.num_dof)
        initial_positions[articulation_action.joint_indices] = articulation_action.joint_positions

        self._articulation.set_joint_positions(initial_positions)
        self._articulation.set_joint_velocities(np.zeros_like(initial_positions))