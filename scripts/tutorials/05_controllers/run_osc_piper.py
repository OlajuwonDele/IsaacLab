
import argparse

from isaaclab.app import AppLauncher
from isaaclab.utils.math import subtract_frame_transforms

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="ar4", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

import torch
import pandas as pd
import numpy as np

##
# Pre-defined configs
##
from isaaclab_assets import ar4_mujoco_OSC_CFG
# -------------------------
# Scene config
# -------------------------
class AR4SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = AR4_OSC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    
    robot = scene["robot"]

    # --- Load trajectory waypoints ---
    df = pd.read_excel("C:/Users/juwon/Documents/Robotics/TCD Research/Annin package/annin_sample_trajectory_100_2.xlsx")
    # positions
    xyz = (df[['X', 'Y', 'Z']] / 100).to_numpy()

    # constant orientation for each waypoint (w, x, y, z)
    quat = np.tile(np.array([[0.7071, 0, 0.7071, 0]]), (xyz.shape[0], 1))
    waypoints = torch.tensor(np.hstack((xyz, quat)), dtype=torch.float32, device=sim.device)

    # Differential IK controller
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=True,
        gravity_compensation=True,
        motion_stiffness_task=[800, 800, 600, 500, 500, 300],
        motion_damping_ratio_task=1.5,
        motion_control_axes_task=[1, 1, 1, 1, 1, 1],
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=scene.num_envs, device=sim.device)


    # Resolve robot EE frame
    entity = SceneEntityCfg("robot", joint_names=["Joint.*"], body_names=["end_effector"])
    entity.resolve(scene)
    ee_body_id = entity.body_ids[0]
    joint_ids = entity.joint_ids

    ee_frame_name = "link_6"
    arm_joint_names = ["joint_.*"]
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]

    # Jacobian index for fixed-base arm
    ee_jacobian_index = ee_body_id - 1

    # Simulation loop
    wp_index = 0
    sim_dt = sim.get_physics_dt()

    N = scene.num_envs
    current_goal_idx = torch.zeros(N, dtype=torch.long, device=sim.device)

    print("✔ Starting IsaacLab trajectory following...")

    robot.update(dt=sim_dt)
    

    # Get the center of the robot soft joint limits
    joint_centers = torch.mean(robot.data.soft_joint_pos_limits[:, arm_joint_ids, :], dim=-1)

    # get the updated states
    (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        # ee_force_b,
        joint_pos,
        joint_vel,
    ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids)

    # Track the given target command
    command = torch.zeros(
        scene.num_envs, osc.action_dim, device=sim.device
    )  # Generic target command, which can be pose, position, force, etc.
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)  # Target pose in the body frame
    ee_target_pose_w = torch.zeros(scene.num_envs, 7, device=sim.device)  # Target pose in the world frame (for marker)

    # Set joint efforts to zero
    zero_joint_efforts = torch.zeros(scene.num_envs, robot.num_joints, device=sim.device)
    joint_efforts = torch.zeros(scene.num_envs, len(arm_joint_ids), device=sim.device)

    # Markers
    # Ensure instancer size matches number of envs:
    N = scene.num_envs

    # create marker objects (you already do this)
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Dummy arrays to initialize the instancer with N entries
    # Use numpy arrays of correct shapes: (N,3) translations, (N,4) orientations.
    dummy_trans = np.zeros((N, 3), dtype=np.float32)
    dummy_orient = np.tile(np.array([0.7071, 0, 0.7071, 0], dtype=np.float32), (N, 1))

    count = 0
    # default_kp = torch.tensor([300, 300, 300, 20, 20, 20], device=sim.device)
    # command[:, 7:] = default_kp.unsqueeze(0).expand(scene.num_envs, -1)

    goals = waypoints[current_goal_idx]        # (N,7)
    command, ee_target_pose_b,  ee_target_pose_w = update_target(
                sim, scene, osc, root_pose_w, goals)
    command, task_frame_pose_b = convert_to_task_frame(osc, command=command, ee_target_pose_b=ee_target_pose_b)
       
    osc.reset()
    osc.set_command(command=command, current_ee_pose_b=ee_pose_b, current_task_frame_pose_b=task_frame_pose_b)
    # osc.set_command(command=command, current_ee_pose_b=ee_pose_b)
    while simulation_app.is_running():

        count += 1
        (
            jacobian_b,
            mass_matrix,
            gravity,
            ee_pose_b,
            ee_vel_b,
            root_pose_w,
            ee_pose_w,
            joint_pos,
            joint_vel,
        ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids)
        # compute the joint commands
        joint_efforts = osc.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            mass_matrix=mass_matrix,
            gravity=gravity,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
            nullspace_joint_pos_target=joint_centers,
        )
        # apply actions
        robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
        robot.write_data_to_sim()
        robot.update(sim_dt)

        # scene.write_data_to_sim()
        sim.step()
        # scene.update(sim_dt)

        # Advance waypoint
        ee_pos = ee_pose_w[:, 0:3]
        goal_pos = goals[:, 0:3] + scene.env_origins
        pos_error = torch.norm(ee_pos - goal_pos,dim=1)
        POS_THRESH = 0.05 # 0.5 cm
        # print(pos_error)
        reached = pos_error < POS_THRESH

        # Update only envs that reached the goal
        if reached.any():
            # Increment goal index for reached environments
            current_goal_idx[reached] = (current_goal_idx[reached] + 1) % len(waypoints)
            
            # Update commands only for reached environments
            new_goals = waypoints[current_goal_idx[reached]]
            
            # Update command for reached environments
            command_update, ee_target_pose_b_update, ee_target_pose_w_update = update_target(
                sim, scene, osc, root_pose_w[reached], new_goals)
            command_update, task_frame_pose_b_update = convert_to_task_frame(
                osc, command=command_update, ee_target_pose_b=ee_target_pose_b_update)
            
            # Update only the reached environments' commands
            command[reached] = command_update
            ee_target_pose_b[reached] = ee_target_pose_b_update
            ee_target_pose_w[reached] = ee_target_pose_w_update
            
            # Set command for all (but only reached ones changed)
            osc.set_command(command=command, current_ee_pose_b=ee_pose_b,
                        current_task_frame_pose_b=task_frame_pose_b)
            # osc.set_command(command=command, current_ee_pose_b=ee_pose_b)
    
        ee_trans = ee_pos.cpu().numpy()
        ee_orient = ee_pose_w[:, 3:7].cpu().numpy()

        goal_trans = goal_pos.cpu().numpy()
        goal_orient = ee_target_pose_w[:, 3:7].cpu().numpy()             

        # if count % 100 == 0:  # Print every 100 steps
        #     print(f"Step {count}")
        #     print(f"  Position errors (m): min={pos_error.min():.4f}, max={pos_error.max():.4f}, mean={pos_error.mean():.4f}")
        #     print(f"  Joint efforts: min={joint_efforts.min():.2f}, max={joint_efforts.max():.2f}")
        #     print(f"  EE velocity: {torch.norm(ee_vel_b[:, :3], dim=1).mean():.4f} m/s")

        ee_marker.visualize(translations=ee_trans, orientations=ee_orient)
        goal_marker.visualize(translations=goal_trans, orientations=goal_orient)

def main():
    # Simulation
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.5, 1.0], [0, 0, 0])

    scene = InteractiveScene(AR4SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0))
    sim.reset()
    run_simulator(sim,scene)


def update_states(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    ee_frame_idx: int,
    arm_joint_ids: list[int],
):
    """Update the robot states.

    Args:
        sim: (SimulationContext) Simulation context.
        scene: (InteractiveScene) Interactive scene.
        robot: (Articulation) Robot articulation.
        ee_frame_idx: (int) End-effector frame index.
        arm_joint_ids: (list[int]) Arm joint indices.
        contact_forces: (ContactSensor) Contact sensor.

    Returns:
        jacobian_b (torch.tensor): Jacobian in the body frame.
        mass_matrix (torch.tensor): Mass matrix.
        gravity (torch.tensor): Gravity vector.
        ee_pose_b (torch.tensor): End-effector pose in the body frame.
        ee_vel_b (torch.tensor): End-effector velocity in the body frame.
        root_pose_w (torch.tensor): Root pose in the world frame.
        ee_pose_w (torch.tensor): End-effector pose in the world frame.
        ee_force_b (torch.tensor): End-effector force in the body frame.
        joint_pos (torch.tensor): The joint positions.
        joint_vel (torch.tensor): The joint velocities.

    Raises:
        ValueError: Undefined target_type.
    """
    # obtain dynamics related quantities from simulation
    ee_jacobi_idx = ee_frame_idx - 1
    jacobian_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
    gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]
    # Convert the Jacobian from world to root frame
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    # Compute current pose of the end-effector
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_w = robot.data.body_pos_w[:, ee_frame_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_frame_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=-1)
    ee_pose_w = torch.cat([ee_pos_w, ee_quat_w], dim=-1)
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    # Compute the current velocity of the end-effector
    ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]  # Extract end-effector velocity in the world frame
    root_vel_w = robot.data.root_vel_w  # Extract root velocity in the world frame
    relative_vel_w = ee_vel_w - root_vel_w  # Compute the relative velocity in the world frame
    ee_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])  # From world to root frame
    ee_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    # Calculate the contact force
    ee_force_w = torch.zeros(scene.num_envs, 3, device=sim.device)
    sim_dt = sim.get_physics_dt()
    # contact_forces.update(sim_dt)  # update contact sensor
    # Calculate the contact force by averaging over last four time steps (i.e., to smoothen) and
    # taking the max of three surfaces as only one should be the contact of interest
    # ee_force_w, _ = torch.max(torch.mean(contact_forces.data.net_forces_w_history, dim=1), dim=1)

    # This is a simplification, only for the sake of testing.
    # ee_force_b = ee_force_w

    # Get joint positions and velocities
    joint_pos = robot.data.joint_pos[:, arm_joint_ids]
    joint_vel = robot.data.joint_vel[:, arm_joint_ids]

    return (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        # ee_force_b,
        joint_pos,
        joint_vel,
    )


# Update the target commands
def update_target(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    osc: OperationalSpaceController,
    root_pose_w: torch.tensor,
    goals: torch.tensor,
):
    """Update the targets for the operational space controller.

    Args:
        sim: (SimulationContext) Simulation context.
        scene: (InteractiveScene) Interactive scene.
        osc: (OperationalSpaceController) Operational space controller.
        root_pose_w: (torch.tensor) Root pose in the world frame.
        ee_target_set: (torch.tensor) End-effector target set.
        current_goal_idx: (int) Current goal index.

    Returns:
        command (torch.tensor): Updated target command.
        ee_target_pose_b (torch.tensor): Updated target pose in the body frame.
        ee_target_pose_w (torch.tensor): Updated target pose in the world frame.
        next_goal_idx (int): Next goal index.

    Raises:
        ValueError: Undefined target_type.
    """

    # update the ee desired command
    command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    command[:, :7] = goals

    command[:, :7] = subtract_frame_transforms(
        root_pose_w[:, :3],      # base position in world
        root_pose_w[:, 3:],      # base rotation in world
        command [:,0:3],          # your xyz (meters)
        command [:,0:7]         # your chosen quaternion target
    )

    # update the ee desired pose
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            ee_target_pose_b[:] = command[:, :7]
        elif target_type == "wrench_abs":
            pass  # ee_target_pose_b could stay at the root frame for force control, what matters is ee_target_b
        else:
            raise ValueError("Undefined target_type within update_target().")

    # update the target desired pose in world frame (for marker)
    ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
    )
    ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)

    return command, ee_target_pose_b, ee_target_pose_w




# Convert the target commands to the task frame
def convert_to_task_frame(osc: OperationalSpaceController, command: torch.tensor, ee_target_pose_b: torch.tensor):
    """Converts the target commands to the task frame.

    Args:
        osc: OperationalSpaceController object.
        command: Command to be converted.
        ee_target_pose_b: Target pose in the body frame.

    Returns:
        command (torch.tensor): Target command in the task frame.
        task_frame_pose_b (torch.tensor): Target pose in the task frame.

    Raises:
        ValueError: Undefined target_type.
    """
    command = command.clone()
    task_frame_pose_b = ee_target_pose_b.clone()

    cmd_idx = 0
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            command[:, :3], command[:, 3:7] = subtract_frame_transforms(
                task_frame_pose_b[:, :3], task_frame_pose_b[:, 3:], command[:, :3], command[:, 3:7]
            )
            cmd_idx += 7
        elif target_type == "wrench_abs":
            # These are already defined in target frame for ee_goal_wrench_set_tilted_task (since it is
            # easier), so not transforming
            cmd_idx += 6
        else:
            raise ValueError("Undefined target_type within _convert_to_task_frame().")

    return command, task_frame_pose_b

if __name__ == "__main__":
    main()
    simulation_app.close()


