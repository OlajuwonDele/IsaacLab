"""
Configuration for the AR4 MK3 robot.

Available configurations:

* AR4_CFG: AR4-MK3 robot with standard actuators
* AR4_PD_CFG: AR4-MK3 for Differential IK with position control
* AR4_OSC_CFG: AR4-MK3 optimized for Operational Space Control (torque control)
"""

import copy
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# -----------------------------
# Base AR4 Configuration
# -----------------------------
AR4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/juwon/Documents/Robotics/TCD Research/Annin package/AR4 URDF JUWON DH2/AR4 Main DH.SLDASM/urdf/AR4 Main DH.SLDASM/AR4 Main DH.SLDASM.usd",
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
            "Joint1": 0.0,
            "Joint2": 0.0,
            "Joint3": 0.0,
            "Joint4": 0.0,
            "Joint5": 0.0,
            "Joint6": 0.0,
        }
    ),
    actuators={
        "ar4_base": ImplicitActuatorCfg(
            joint_names_expr=["Joint1"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "ar4_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["Joint2"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "ar4_bicep": ImplicitActuatorCfg(
            joint_names_expr=["Joint3"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "ar4_forearm": ImplicitActuatorCfg(
            joint_names_expr=["Joint4"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "ar4_wrist1": ImplicitActuatorCfg(
            joint_names_expr=["Joint5"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.2,
            stiffness=2000.0,
            damping=100.0,
        ),
        "ar4_wrist2": ImplicitActuatorCfg(
            joint_names_expr=["Joint6"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.2,
            stiffness=2000.0,
            damping=100.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Base configuration of AR4-MK3 robot."""

# -----------------------------
# PD Control Configuration
# -----------------------------
AR4_PD_CFG = copy.deepcopy(AR4_CFG)
AR4_PD_CFG.spawn.rigid_props.disable_gravity = True
AR4_PD_CFG.spawn.inertial_dynamics_decoupling = True
"""AR4-MK3 configuration with stiffer PD control for Differential IK."""

# -----------------------------
# Operational Space Control (OSC)
# -----------------------------
AR4_OSC_CFG = copy.deepcopy(AR4_CFG)

# OSC-specific settings
AR4_OSC_CFG.spawn.rigid_props.disable_gravity = False  # Gravity compensation needed
AR4_OSC_CFG.spawn.inertial_dynamics_decoupling = False  # Use full dynamics
AR4_OSC_CFG.spawn.articulation_props.solver_position_iteration_count = 32
AR4_OSC_CFG.spawn.articulation_props.solver_velocity_iteration_count = 16

# Actuator stiffness/damping for torque control
AR4_OSC_CFG.actuators = {
    "ar4_base": ImplicitActuatorCfg(
        joint_names_expr=["Joint1"],
        effort_limit_sim=200.0,
        velocity_limit_sim=2.175,
        stiffness=0.0,
        damping=0.0,
    ),
    "ar4_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["Joint2"],
        effort_limit_sim=200.0,
        velocity_limit_sim=2.175,
        stiffness=0.0,
        damping=0.0,
    ),
    "ar4_bicep": ImplicitActuatorCfg(
        joint_names_expr=["Joint3"],
        effort_limit_sim=200.0,
        velocity_limit_sim=2.175,
        stiffness=0.0,
        damping=0.0,
    ),
    "ar4_forearm": ImplicitActuatorCfg(
        joint_names_expr=["Joint4"],
        effort_limit_sim=200.0,
        velocity_limit_sim=2.61,
        stiffness=0.0,
        damping=0.0,
    ),
    "ar4_wrist1": ImplicitActuatorCfg(
        joint_names_expr=["Joint5"],
        effort_limit_sim=200.0,
        velocity_limit_sim=0.2,
        stiffness=0.0,
        damping=0.0,
    ),
    "ar4_wrist2": ImplicitActuatorCfg(
        joint_names_expr=["Joint6"],
        effort_limit_sim=200.0,
        velocity_limit_sim=0.2,
        stiffness=0.0,
        damping=0.0,
    ),
}
"""AR4-MK3 configuration optimized for Operational Space Control (torque control)."""
