# -------------------------------------------------------------
# AR4 Task-Space Velocity Controller Following a Trajectory File
# -------------------------------------------------------------

import argparse
import torch
import numpy as np

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="AR4 velocity controller using Cartesian trajectory.")
parser.add_argument("--trajectory", type=str, default="C:/Users/juwon/Documents/Robotics/TCD Research/Annin package/sample_annin_cartesian_trajectory.txt",
                    help="Path to trajectory file generated from MATLAB.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------------------------------------------
# Isaac Lab Imports
# -------------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_assets import AR4_PD_CFG
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG


# -------------------------------------------------------------
# Scene Configuration

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    robot = AR4_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

# -------------------------------------------------------------
# Load MATLAB trajectory
# -------------------------------------------------------------
def load_matlab_cartesian_trajectory(path):
    # Define goals for the arm
    df = pd.read_excel("C:/Users/juwon/Documents/Robotics/TCD Research/Annin package/annin_sample_trajectory_100_2.xlsx")
    # positions
    xyz = (df[['X', 'Y', 'Z']] / 100).to_numpy()

    # constant orientation for each waypoint (w, x, y, z)
    quat = np.tile(np.array([[0, 0,0, 1]]), (xyz.shape[0], 1))
    ee_goals = np.hstack((xyz, quat))

    return ee_goals


# -------------------------------------------------------------
# Simulation Loop
# -------------------------------------------------------------
def run_simulator(sim, scene):

    robot = scene["robot"]

    # Load MATLAB trajectory
    ee_goals = load_matlab_cartesian_trajectory(args_cli.trajectory)
    print("[INFO] Loaded trajectory with", len(ee_goals), "samples.")

    # Convert to torch
    ee_goals= torch.tensor(ee_goals, device=sim.device, dtype=torch.float32)


    # Controller in velocity (relative pose) mode
    controller_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_rel"],  # use relative pose for velocity commands
        motion_stiffness_task=200.0,
        motion_damping_ratio_task=1.0
    )
    controller = OperationalSpaceController(controller_cfg, num_envs=scene.num_envs, device=sim.device)

    # Get entity for AR4
    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=["Joint.*"],
        body_names=["Link6"]
    )
    robot_entity_cfg.resolve(scene)
    ee_body_id = robot_entity_cfg.body_ids[0]

    # EE Jacobian index
    ee_jacobi_idx = ee_body_id if not robot.is_fixed_base else ee_body_id - 1

    # Markers
    markers = VisualizationMarkers(FRAME_MARKER_CFG.replace(prim_path="/Vis/ee"))
    target_marker = VisualizationMarkers(FRAME_MARKER_CFG.replace(prim_path="/Vis/target"))

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    traj_idx = 0

    print("[INFO] Starting simulation...")

    while simulation_app.is_running():

        # Select trajectory index by time
        traj_idx = int((sim_time / time_vec[-1]) * (len(time_vec)-1))
        traj_idx = min(traj_idx, len(time_vec)-1)

        # Desired EE velocity -> convert to delta pose
        delta_pose = torch.zeros(scene.num_envs, 6, device=sim.device, dtype=torch.float32)
        delta_pose[:, 0:3] = qd_torch[traj_idx] * sim_dt  # linear velocity -> position delta

        # Read robot state
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        # Express EE frame in base frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Get EE velocity in world frame
        ee_vel_w = robot.data.body_vel_w[:, ee_body_id, 0:6]  # linear + angular

        # Convert linear velocity to base frame
        root_rot_w = robot.data.root_pose_w[:, 3:7]  # root quaternion
        ee_linear_vel_b = quat_apply_inverse(root_rot_w, ee_vel_w[:, 0:3])
        ee_angular_vel_b = quat_apply_inverse(root_rot_w, ee_vel_w[:, 3:6])

        ee_vel_b = torch.cat([ee_linear_vel_b, ee_angular_vel_b], dim=1)

        # Send delta_pose as command with current EE pose & velocity
        controller.set_command(
            delta_pose,
            current_ee_pose_b=torch.cat([ee_pos_b, ee_quat_b], dim=1),
        )

        # Compute joint velocities
        joint_vel_cmd = controller.compute(
            jacobian_b=jacobian,
            current_ee_pose_b=torch.cat([ee_pos_b, ee_quat_b], dim=1),
            current_ee_vel_b=ee_vel_b,
            current_joint_pos=joint_pos
        )
        # Apply joint velocities
        robot.set_joint_velocity_target(joint_vel_cmd, joint_ids=robot_entity_cfg.joint_ids)
        
        # End-effector positions
        current_pos = ee_pos_b[0].cpu().numpy()          # (3,)
        desired_pos = q_torch[traj_idx].cpu().numpy()     # (3,)
        desired_orien = qd_torch[traj_idx].cpu().numpy()
        pos_error = desired_pos - current_pos

        # Body (base_link) pose in world frame
        body_pos = root_pose_w[0].cpu().numpy()               # (3,)

        print("\n========== ROBOT STATE DEBUG ==========")
        print(f"Robot Base Position (World, m): {body_pos}")

        print(f"\nCurrent EE Position   (m): {current_pos}")
        print(f"Desired EE Position   (m): {desired_pos}")
        print(f"EE Position Error      (m): {pos_error}")
        print("========================================\n")
        # Step simulation
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        sim_time += sim_dt

        # Visualize EE
        markers.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        desired_pos_t = torch.tensor(desired_pos, device=sim.device, dtype=torch.float32).unsqueeze(0)  # (1,3)
        desired_quat_t = ee_quat_b[0:1]  # (1,4) use current env orientation as a reasonable target orientation
        target_marker.visualize(desired_pos_t, desired_quat_t)
# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 1.2], [0, 0, 0])

    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    print("[INFO] Scene ready. Running trajectory follower...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
