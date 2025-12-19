import argparse
import torch
from isaaclab.app import AppLauncher
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.time import Rate
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)
from isaaclab_assets import AR4_OSC_CFG
import isaaclab.sim as sim_utils


# ----------------- CLI -----------------
parser = argparse.ArgumentParser(description="Tutorial on using the operational space controller.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------- Scene Config -----------------
@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a simple scene with a tilted wall."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = AR4_OSC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.rigid_props.disable_gravity = True


# ----------------- Simulator Loop -----------------
def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    contact_forces = scene["contact_forces"]

    # End-effector & joints
    ee_frame_name = "end_effector"
    arm_joint_names = ["Joint.*"]
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]

    # OSC
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs", "wrench_abs"],
        impedance_mode="variable_kp",
        inertial_dynamics_decoupling=True,
        motion_damping_ratio_task=1.0,
        contact_wrench_stiffness_task=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        motion_control_axes_task=[1, 1, 0, 1, 1, 1],
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=scene.num_envs, device=sim.device)

    sim_dt = sim.get_physics_dt()
    rate = Rate(200)
    iteration = 0
    count = 0

    target_pose = robot.data.default_root_state[:7].clone()
    k_values = [100, 200, 300, 400]
    d_values = [0.5, 0.7, 1.0, 1.2]

    zero_joint_efforts = torch.zeros(scene.num_envs, robot.num_joints, device=sim.device)

    while simulation_app.is_running():
        sim.step()

        # Sweep OSC gains
        if iteration % 400 == 0:
            k = k_values[(iteration // 400) % len(k_values)]
            d = d_values[(iteration // 400) % len(d_values)]
            osc_cfg.motion_stiffness_task = [k, k, k, 30, 30, 30]
            osc_cfg.motion_damping_ratio_task = d
            osc.update_config(osc_cfg)
            print(f"[TUNE] stiffness={k}, damping={d}")

        # Compute and apply OSC
        action = osc.compute(target_pose)
        robot.apply_action(action)

        # Debug info
        if iteration % 50 == 0:
            # These variables must be updated from update_states()
            # Example placeholders:
            ee_pos = torch.zeros(3)
            goal_pos = torch.zeros(3)
            pos_error = torch.norm(ee_pos - goal_pos)
            print(f"Position error [xyz] = {ee_pos - goal_pos}, pos_error = {pos_error}")
            # Mock Z-error
            z_error = 0.0
            print(f"Height Error: {z_error:.4f} m")
            shoulder_torque = 0.0
            print(f"Shoulder Torque: {shoulder_torque:.2f} Nm")

        iteration += 1
        rate.sleep()

        # Reset every 500 steps
        if count % 500 == 0:
            default_joint_pos = robot.data.default_joint_pos.clone()
            default_joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            robot.set_joint_effort_target(zero_joint_efforts)
            robot.write_data_to_sim()
            robot.reset()
            contact_forces.reset()

        count += 1


# ----------------- Main -----------------
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
