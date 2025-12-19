# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.osc_gain_tuner.osc_gain_tuner_env_cfg as osc_gain_tuner_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.ar4 import AR4_OSC_CFG  # isort: skip


@configclass
class AR4OSCGainTunerEnvCfg(osc_gain_tuner_env_cfg.OSCGainTunerEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We remove stiffness and damping for the shoulder and forearm joints for effort control
        self.scene.robot = AR4_OSC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.rigid_props.disable_gravity = True
        ee_body_name = "end_effector"  # Change to your AR4's EE link name if different
        
        # Update command configuration with AR4 end-effector
        self.commands.ee_pose.body_name = ee_body_name
        # Set a range for the pitch command (e.g., +/- 90 degrees)
        self.commands.ee_pose.ranges.pitch = (-1.57, 1.57) 

        # 2. Fix rewards configuration body names (using the end-effector link)
        # These fields were set to MISSING in the base class and must be defined here.
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ee_body_name
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ee_body_name
        
        # FIX: Removed the redundant 'rewards' attribute which caused the AttributeError
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ee_body_name
        
        # Check for the custom velocity penalty mentioned in the traceback (it must exist in your base config)
        if hasattr(self.rewards, "end_effector_velocity_penalty"):
            self.rewards.end_effector_velocity_penalty.params["asset_cfg"].body_names = ee_body_name
        # Check for the acceleration penalty (often paired with velocity for smoothness)
        if hasattr(self.rewards, "end_effector_acceleration_penalty"):
             self.rewards.end_effector_acceleration_penalty.params["asset_cfg"].body_names = ee_body_name

        # If closed-loop contact force control is desired, contact sensors should be enabled for the robot
        # self.scene.robot.spawn.activate_contact_sensors = True

        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=["Joint.*"],
            body_name="end_effector",
            # If a task frame different from articulation root/base is desired, a RigidObject, e.g., "task_frame",
            # can be added to the scene and its relative path could provided as task_frame_rel_path
            # task_frame_rel_path="task_frame",
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_abs"],
                impedance_mode="variable_kp",
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=True,
                gravity_compensation=True,
                motion_stiffness_task=100.0,
                motion_damping_ratio_task=1.0,
                motion_stiffness_limits_task=(10.0, 400.0),
                motion_damping_ratio_limits_task=(0.3, 4.0),
            ),
             # Scales control how controller interprets action channels:
            # - position_scale/orientation_scale scale pose commands
            # - stiffness_scale scales how an additional action channel maps to task stiffness (kp)
            # We set stiffness_scale so that the agent *must* output the stiffness/damping channels
            position_scale=1.0,
            orientation_scale=1.0,
            stiffness_scale=1.0,  # expects extra channels for stiffness in the action
            # damping_scale=1.0,    # ex
        )
        # Removing these observations as they are not needed for OSC and we want keep the observation space small
        self.observations.policy.joint_pos = None
        self.observations.policy.joint_vel = None


@configclass
class AR4OSCGainTunerEnvCfg_PLAY(AR4OSCGainTunerEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
