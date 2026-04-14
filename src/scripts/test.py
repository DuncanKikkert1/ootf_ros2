###
# This code creates a simple world in Isaac Sim imported from a .usd file
###

from pathlib import Path
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage

usd_path = Path(__file__).parent.parent / "Doosan-ROS2.usd"

world = World()

world.scene.add_default_ground_plane()

add_reference_to_stage(usd_path=str(usd_path), prim_path="/World/Robot")

world.reset()

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close