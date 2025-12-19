from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux

import omni
import numpy as np
from isaacsim.core.api import World

# Import YOUR custom task
from tasks.trajectory_generate import TrajectoryGeneration

# Create the simulation world
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

stage = omni.usd.get_context().get_stage()
sphereLight = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/MySphereLight"))
sphereLight.CreateRadiusAttr(0.5)
sphereLight.CreateIntensityAttr(50000.0)
sphereLight.AddTranslateOp().Set(Gf.Vec3f(5.0, 5.0, 5.0))
# Create trajectory generator
tg = TrajectoryGeneration()

# Load robot
assets = tg.load_example_assets()

for asset in assets:
    world.scene.add(asset)

world.reset()  


tg.setup()
tg.setup_taskspace_trajectory()

print("Simulation starting...")

# Run simulation loop
while simulation_app.is_running():
    world.step(render=True)
    tg.update()

simulation_app.close()
