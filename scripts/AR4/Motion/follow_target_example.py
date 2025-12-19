from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from controller.ik_solver import KinematicsSolver
from isaacsim.core.api import World
from tasks.follow_target import FollowTarget

my_world = World(stage_units_in_meters=1.0)
# Initialize the Follow Target task with a target location for the cube to be followed by the end effector
my_task = FollowTarget(name="ar4_follow_target", target_position=np.array([0.3, 0.3, 0.2]))
my_world.add_task(my_task)
my_world.reset()
task_params = my_world.get_task("ar4_follow_target").get_params()
target_name = task_params["target_name"]["value"]
ar4_name = task_params["robot_name"]["value"]
my_ar4 = my_world.scene.get_object(ar4_name)
print(my_ar4)
# initialize the ik solver
ik_solver = KinematicsSolver(my_ar4)
articulation_controller = my_ar4.get_articulation_controller()

# run the simulation
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()

        observations = my_world.get_observations()
        actions, succ = ik_solver.compute_inverse_kinematics(
            target_position=observations[target_name]["position"],
            target_orientation=observations[target_name]["orientation"],
        )
        if succ:
            articulation_controller.apply_action(actions)
        else:
            print("IK did not converge to a solution.  No action is being taken.")
simulation_app.close()