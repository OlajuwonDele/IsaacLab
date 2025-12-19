"""
Configuration for the ar4_mujoco MK3 robot.

Available configurations:

* ar4_mujoco_CFG: ar4_mujoco-MK3 robot with standard actuators
* ar4_mujoco_PD_CFG: ar4_mujoco-MK3 for Differential IK with position control
* ar4_mujoco_OSC_CFG: ar4_mujoco-MK3 optimized for Operational Space Control (torque control)
"""

import copy
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# -----------------------------
# Base ar4_mujoco Configuration
# -----------------------------
ar4_mujoco_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/juwon/Documents/Robotics/TCD Research/Annin package/Github URDF/Mujoco/ar4/ar4_mujoco_sim/urdf/mj_ar4/mj_ar4.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=6,
            solver_velocity_iteration_count=0
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=0.0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 0.0,
            "joint_4": 0.0,
            "joint_5": 0.0,
            "joint_6": 0.0,
        }
    ),
    actuators={
        "ar4_mujoco_base": ImplicitActuatorCfg(
            joint_names_expr=["joint_1"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "ar4_mujoco_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint_2"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "ar4_mujoco_bicep": ImplicitActuatorCfg(
            joint_names_expr=["joint_3"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "ar4_mujoco_forearm": ImplicitActuatorCfg(
            joint_names_expr=["joint_4"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "ar4_mujoco_wrist1": ImplicitActuatorCfg(
            joint_names_expr=["joint_5"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.2,
            stiffness=2000.0,
            damping=100.0,
        ),
        "ar4_mujoco_wrist2": ImplicitActuatorCfg(
            joint_names_expr=["joint_6"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.2,
            stiffness=2000.0,
            damping=100.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Base configuration of ar4_mujoco-MK3 robot."""

# -----------------------------
# PD Control Configuration
# -----------------------------
ar4_mujoco_PD_CFG = copy.deepcopy(ar4_mujoco_CFG)
ar4_mujoco_PD_CFG.spawn.rigid_props.disable_gravity = True
ar4_mujoco_PD_CFG.spawn.inertial_dynamics_decoupling = True
"""ar4_mujoco-MK3 configuration with stiffer PD control for Differential IK."""

# -----------------------------
# Operational Space Control (OSC)
# -----------------------------
ar4_mujoco_OSC_CFG = copy.deepcopy(ar4_mujoco_CFG)

# OSC-specific settings
ar4_mujoco_OSC_CFG.spawn.rigid_props.disable_gravity = False  # Gravity compensation needed
ar4_mujoco_OSC_CFG.spawn.inertial_dynamics_decoupling = False  # Use full dynamics
ar4_mujoco_OSC_CFG.spawn.articulation_props.solver_position_iteration_count = 12
ar4_mujoco_OSC_CFG.spawn.articulation_props.solver_velocity_iteration_count = 4

# Actuator stiffness/damping for torque control
ar4_mujoco_OSC_CFG.actuators = {
    "ar4_mujoco_base": ImplicitActuatorCfg(
        joint_names_expr=["joint_1"],
        effort_limit_sim=87.0,
        velocity_limit_sim=0.5,
        stiffness=0.0,
        damping=0.0,
    ),
    "ar4_mujoco_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["joint_2"],
        effort_limit_sim=87.0,
        velocity_limit_sim=0.5,
        stiffness=0.0,
        damping=0.0,
    ),
    "ar4_mujoco_bicep": ImplicitActuatorCfg(
        joint_names_expr=["joint_3"],
        effort_limit_sim=87.0,
        velocity_limit_sim=0.5,
        stiffness=0.0,
        damping=0.0,
    ),
    "ar4_mujoco_forearm": ImplicitActuatorCfg(
        joint_names_expr=["joint_4"],
        effort_limit_sim=12.0,
        velocity_limit_sim=0.5,
        stiffness=0.0,
        damping=0.0,
    ),
    "ar4_mujoco_wrist1": ImplicitActuatorCfg(
        joint_names_expr=["joint_5"],
        effort_limit_sim=200.0,
        velocity_limit_sim=1.0,
        stiffness=0.0,
        damping=0.0,
    ),
    "ar4_mujoco_wrist2": ImplicitActuatorCfg(
        joint_names_expr=["joint_6"],
        effort_limit_sim=200.0,
        velocity_limit_sim=1.0,
        stiffness=0.0,
        damping=0.0,
    ),
}
"""ar4_mujoco-MK3 configuration optimized for Operational Space Control (torque control)."""
