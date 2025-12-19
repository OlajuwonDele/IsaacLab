
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
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

import torch
import pandas as pd
import numpy as np

from isaaclab_assets import AR4_MK3_PD_CFG

class AR4SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = AR4_MK3_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.1)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.5, 1.0], [0, 0, 0])

    scene = InteractiveScene(AR4SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    N = scene.num_envs

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    df = pd.read_excel("C:/ar4MP/source/ar4MP/ar4MP/tasks/manager_based/ar4mp/sample_trajectory.xlsx")

    xyz = (df[['X', 'Y', 'Z']]).to_numpy()
    quat = np.tile(np.array([[0.7071, 0.70711, 0, 0]]), (xyz.shape[0], 1))
    waypoints = torch.tensor(np.hstack((xyz, quat)), dtype=torch.float32, device=sim.device)

    # Differential IK controller
    ik_cfg = DifferentialIKControllerCfg(command_type="pose",use_relative_mode=False, ik_method="dls")
    ik = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Resolve robot EE frame
    entity = SceneEntityCfg("robot", joint_names=["joint_.*"], body_names=["gripper_base_link"])
    entity.resolve(scene)
    ee_body_id = entity.body_ids[0]
    joint_ids = entity.joint_ids

    # Jacobian index for fixed-base arm
    ee_jacobian_index = ee_body_id - 1
    sim_dt = sim.get_physics_dt()

    current_goal_idx = torch.zeros(N, dtype=torch.long, device=sim.device)

    print("✔ Starting IsaacLab trajectory following...")

    while simulation_app.is_running():
        # Waypoint pose
        goals = waypoints[current_goal_idx]      
        ik.set_command(goals)

        # Simulation state
        jac = robot.root_physx_view.get_jacobians()[:, ee_jacobian_index, :, joint_ids]
        ee_pose = robot.data.body_pose_w[:, ee_body_id]
        root_pose = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, joint_ids]

        # Transform to root frame
        pos_b, rot_b = subtract_frame_transforms(
            root_pose[:,0:3], root_pose[:,3:7],
            ee_pose[:,0:3], ee_pose[:,3:7]
        )

        # Differential IK solve
        joint_targets = ik.compute(pos_b, rot_b, jac, joint_pos)
        robot.set_joint_position_target(joint_targets, joint_ids=joint_ids)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        # Advance waypoint
        ee_pos = ee_pose[:, 0:3]
        goal_pos = goals[:, 0:3] + scene.env_origins

        pos_error = torch.norm(ee_pos - goal_pos,dim=1)
        POS_THRESH = 0.05 # 0.5 cm

        reached = pos_error < POS_THRESH

        # Update only envs that reached the goal
        if reached.any():
            reached_ids = torch.where(reached)[0]
            for env_id in reached_ids:
                current_goal_idx[env_id] = (current_goal_idx[env_id] + 1) % len(waypoints)

        ee_trans = ee_pos.cpu().numpy()
        ee_orient = ee_pose[:, 3:7].cpu().numpy()

        goal_trans = goal_pos.cpu().numpy()
        goal_orient = goals[:, 3:7].cpu().numpy()             


        ee_marker.visualize(translations=ee_trans, orientations=ee_orient)
        goal_marker.visualize(translations=goal_trans, orientations=goal_orient)



if __name__ == "__main__":
    main()
    simulation_app.close()
