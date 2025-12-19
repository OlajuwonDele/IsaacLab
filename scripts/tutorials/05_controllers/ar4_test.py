# full_script_with_fixed_osc_tuner.py
import argparse
import math
import numpy as np
import torch
import pandas as pd
from isaaclab.app import AppLauncher
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms, matrix_from_quat, quat_apply_inverse, quat_inv
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# your AR4 config import
from isaaclab_assets import AR4_OSC_CFG

# CLI
parser = argparse.ArgumentParser(description="AR4 OSC with auto fixed gain tuner.")
parser.add_argument("--robot", type=str, default="ar4", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------- Scene config (kept from your script) ----------
class AR4SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot = AR4_OSC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

# ---------- helper functions (you already had these) ----------
def update_states(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    ee_frame_idx: int,
    arm_joint_ids: list[int],
):
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
        joint_pos,
        joint_vel,
    )

def update_target(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    osc: OperationalSpaceController,
    root_pose_w: torch.tensor,
    goals: torch.tensor,
):
    command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    command[:, :7] = goals
    command[:, :3], command[:, 3:] = subtract_frame_transforms(
        root_pose_w[:, :3],
        root_pose_w[:, 3:],
        command[:, 0:3],
        command[:, 3:7],
    )
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            ee_target_pose_b[:] = command[:, :7]
        elif target_type == "wrench_abs":
            pass
        else:
            raise ValueError("Undefined target_type within update_target().")
    ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
    )
    ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)
    return command, ee_target_pose_b, ee_target_pose_w

def convert_to_task_frame(osc: OperationalSpaceController, command: torch.tensor, ee_target_pose_b: torch.tensor):
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
            cmd_idx += 6
        else:
            raise ValueError("Undefined target_type within _convert_to_task_frame().")
    return command, task_frame_pose_b

# ---------- TUNER UTILITIES ----------
def compute_taskspace_lambda(J: torch.Tensor, M: torch.Tensor, eps=1e-6):
    """
    J: (6, n_j) torch tensor (single env)
    M: (n_j, n_j) torch tensor (single env)
    Returns Lambda (6x6), robust inversion using small regularization if needed.
    """
    # J @ M^-1 @ J^T
    Minv = torch.linalg.inv(M)
    Amat = J @ Minv @ J.T  # 6x6
    # regularize for numeric stability
    Amat_reg = Amat + eps * torch.eye(6, device=Amat.device, dtype=Amat.dtype)
    # Use pinv for robustness but prefer inv of regularized matrix
    try:
        Lambda = torch.linalg.inv(Amat_reg)
    except Exception:
        Lambda = torch.linalg.pinv(Amat_reg)
    return Lambda

def compute_effective_masses(Lambda: torch.Tensor):
    # Lambda is 6x6; translational axes are indices 0..2, rotational 3..5
    e_x = torch.tensor([1.,0.,0.,0.,0.,0.], device=Lambda.device, dtype=Lambda.dtype)
    e_y = torch.tensor([0.,1.,0.,0.,0.,0.], device=Lambda.device, dtype=Lambda.dtype)
    e_z = torch.tensor([0.,0.,1.,0.,0.,0.], device=Lambda.device, dtype=Lambda.dtype)
    e_rx = torch.tensor([0.,0.,0.,1.,0.,0.], device=Lambda.device, dtype=Lambda.dtype)
    e_ry = torch.tensor([0.,0.,0.,0.,1.,0.], device=Lambda.device, dtype=Lambda.dtype)
    e_rz = torch.tensor([0.,0.,0.,0.,0.,1.], device=Lambda.device, dtype=Lambda.dtype)
    m_x = float((e_x @ Lambda @ e_x).item())
    m_y = float((e_y @ Lambda @ e_y).item())
    m_z = float((e_z @ Lambda @ e_z).item())
    I_rx = float((e_rx @ Lambda @ e_rx).item())
    I_ry = float((e_ry @ Lambda @ e_ry).item())
    I_rz = float((e_rz @ Lambda @ e_rz).item())
    return np.array([m_x, m_y, m_z, I_rx, I_ry, I_rz], dtype=float)

def compute_Kp_from_meff(M_eff: np.ndarray, desired_hz=2.0):
    """
    Choose natural frequency wn = 2*pi*desired_hz.
    Compute Kp_task = M_eff * wn^2 (per axis).
    """
    wn = 2.0 * math.pi * desired_hz
    Kp = M_eff * (wn ** 2)
    # clamp Kp to avoid tiny or huge numbers (safety bounds)
    Kp = np.clip(Kp, a_min=0.1, a_max=2e5)
    return Kp

# ---------- Main runtime ----------
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]

    # load trajectory waypoints (user path)
    df = pd.read_excel("C:/Users/juwon/Documents/Robotics/TCD Research/Annin package/annin_sample_trajectory_100_2.xlsx")
    xyz = (df[['X', 'Y', 'Z']] / 100).to_numpy()
    quat = np.tile(np.array([[0.7071, 0, 0.7071, 0]]), (xyz.shape[0], 1))
    waypoints = torch.tensor(np.hstack((xyz, quat)), dtype=torch.float32, device=sim.device)

    # Resolve robot EE frame and joints
    entity = SceneEntityCfg("robot", joint_names=["Joint.*"], body_names=["end_effector"])
    entity.resolve(scene)
    ee_frame_idx = entity.body_ids[0][0]
    arm_joint_ids = entity.joint_ids[0]

    sim_dt = sim.get_physics_dt()
    N = scene.num_envs
    current_goal_idx = torch.zeros(N, dtype=torch.long, device=sim.device)

    # warm-up: step once so robot and physx readers are valid
    sim.reset()
    # update scene data to populate robot.data
    sim.step()

    # read states once to compute initial J and M for the tuner
    (jacobian_b,
     mass_matrix,
     gravity,
     ee_pose_b,
     ee_vel_b,
     root_pose_w,
     ee_pose_w,
     joint_pos,
     joint_vel) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids)

    # jacobian_b: (N,6,n_j); mass_matrix: (N,n_j,n_j)
    # take environment 0 as representative (fixed gains)
    J0 = jacobian_b[0]        # 6 x n_j
    M0 = mass_matrix[0]      # n_j x n_j

    # convert to double for more stable math
    J0 = J0.to(dtype=torch.float64)
    M0 = M0.to(dtype=torch.float64)

    # compute Lambda and effective masses
    Lambda0 = compute_taskspace_lambda(J0, M0, eps=1e-6)
    M_eff = compute_effective_masses(Lambda0)
    print("Effective task-space masses/inertias (x,y,z,rx,ry,rz):", np.round(M_eff, 6))

    # compute Kp from desired natural frequency (2 Hz default, adjust if you want faster/slower)
    DESIRED_HZ = 2.0
    Kp_task = compute_Kp_from_meff(M_eff, desired_hz=DESIRED_HZ)
    # choose damping ratio (critical damping default)
    damping_ratios = np.ones(6, dtype=float)  # zeta = 1.0 for critical damping

    print("Auto-computed task-space stiffness (Kp):", np.round(Kp_task, 6))
    print("Using motion_damping_ratio_task (zeta) =", damping_ratios.tolist())

    # sanity torque estimate for a small step (helps catch too-large Kp)
    # construct small step F_task = Kp * delta_pos, for rotation use small angle
    delta_pos = 0.03  # 3 cm step
    delta_rot = 0.06  # ~3.4 deg
    F_task = torch.tensor([Kp_task[0]*delta_pos, Kp_task[1]*delta_pos, Kp_task[2]*delta_pos,
                           Kp_task[3]*delta_rot, Kp_task[4]*delta_rot, Kp_task[5]*delta_rot],
                          device=J0.device, dtype=torch.float64)
    tau_est = (J0.T @ F_task).cpu().numpy()
    print("Estimated joint torques for small step (approx):", np.round(tau_est, 6))
    # (Optionally you could compare to motor limits here if you have them.)

    # Now create OSC with the tuned fixed gains
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=True,
        gravity_compensation=True,
        motion_stiffness_task=Kp_task.tolist(),
        motion_damping_ratio_task=damping_ratios.tolist(),
        motion_control_axes_task=[1, 1, 1, 1, 1, 1],
        nullspace_control="position",
        nullspace_stiffness=12.0,
        nullspace_damping_ratio=1.0,
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=scene.num_envs, device=sim.device)

    # Prepare for main loop
    robot.update(dt=sim_dt)
    joint_centers = torch.mean(robot.data.soft_joint_pos_limits[:, arm_joint_ids, :], dim=-1)

    # Initial command and set
    N = scene.num_envs
    command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    ee_target_pose_w = torch.zeros(scene.num_envs, 7, device=sim.device)
    goals = waypoints[current_goal_idx]
    command, ee_target_pose_b, ee_target_pose_w = update_target(sim, scene, osc, root_pose_w, goals)
    command, task_frame_pose_b = convert_to_task_frame(osc, command=command, ee_target_pose_b=ee_target_pose_b)
    osc.reset()
    osc.set_command(command=command, current_ee_pose_b=ee_pose_b, current_task_frame_pose_b=task_frame_pose_b)

    print("✔ Starting IsaacLab trajectory following with AUTO-TUNED fixed OSC gains...")

    count = 0
    # create markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    while simulation_app.is_running():
        count += 1
        (jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, root_pose_w, ee_pose_w, joint_pos, joint_vel) = \
            update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids)

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

        robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
        robot.write_data_to_sim()
        robot.update(sim_dt)
        sim.step()

        # Advance waypoint
        ee_pos = ee_pose_w[:, 0:3]
        goal_pos = goals[:, 0:3] + scene.env_origins
        pos_error = torch.norm(ee_pos - goal_pos, dim=1)
        POS_THRESH = 0.05
        reached = pos_error < POS_THRESH

        if reached.any():
            current_goal_idx[reached] = (current_goal_idx[reached] + 1) % len(waypoints)
            new_goals = waypoints[current_goal_idx[reached]]
            command_update, ee_target_pose_b_update, ee_target_pose_w_update = update_target(
                sim, scene, osc, root_pose_w[reached], new_goals
            )
            command_update, task_frame_pose_b_update = convert_to_task_frame(
                osc, command=command_update, ee_target_pose_b=ee_target_pose_b_update
            )
            command[reached] = command_update
            ee_target_pose_b[reached] = ee_target_pose_b_update
            ee_target_pose_w[reached] = ee_target_pose_w_update
            osc.set_command(command=command, current_ee_pose_b=ee_pose_b, current_task_frame_pose_b=task_frame_pose_b)

        ee_trans = ee_pos.cpu().numpy()
        ee_orient = ee_pose_w[:, 3:7].cpu().numpy()
        goal_trans = goal_pos.cpu().numpy()
        goal_orient = ee_target_pose_w[:, 3:7].cpu().numpy()

        ee_marker.visualize(translations=ee_trans, orientations=ee_orient)
        goal_marker.visualize(translations=goal_trans, orientations=goal_orient)

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.5, 1.0], [0, 0, 0])
    scene = InteractiveScene(AR4SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0))
    sim.reset()
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
