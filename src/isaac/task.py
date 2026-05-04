from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Waypoint:
    position:    np.ndarray
    orientation: np.ndarray
    gripper:     float
    n_steps:     int = 15


def _quat_eef_down() -> np.ndarray:
    xyzw = Rotation.from_euler('x', np.pi).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


class PickAndPlaceTask:
    """
    Scripted pick-and-place with per-episode domain randomisation.
    Call task.sample(rng) once per episode to get waypoints + instruction.
    """

    DEFAULT_INSTRUCTIONS = [
        "pick up the object from the pallet and place it on the conveyor",
        "grasp the object and move it to the conveyor belt",
        "pick up the item from the pallet and place it on the belt",
        "move the object from the pallet to the conveyor",
        "pick up the red object and place it on the conveyor belt",
        "place the mug next to the beaker",
        "move the mug next to the beaker",
    ]

    def __init__(
        self,
        pick_x:         tuple = (0.7,  1.2),
        pick_y:         tuple = (-0.1, 0.6),
        pick_z:         float = 0.6,
        place_x:        tuple = (-1.0,  0.0),
        place_y:        tuple = (-1.5, -1.0),
        place_z:        float = 0.806,
        lift_h:         float = 0.20,
        approach:       float = 0.05,
        min_separation: float = 0.10,
        max_reach_xy:   float = 1.65,
        instructions:   Optional[List[str]] = None,
    ):
        self.pick_x       = pick_x
        self.pick_y       = pick_y
        self.pick_z       = pick_z
        self.place_x      = place_x
        self.place_y      = place_y
        self.place_z      = place_z
        self.lift_h       = lift_h
        self.approach     = approach
        self.min_sep      = min_separation
        self.max_reach_xy = max_reach_xy
        self.instructions = instructions or self.DEFAULT_INSTRUCTIONS

    def _sample_xy(self, rng, x_range, y_range) -> np.ndarray:
        for _ in range(500):
            xy = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
            if np.linalg.norm(xy) <= self.max_reach_xy:
                return xy
        return np.array([(x_range[0] + x_range[1]) / 2,
                         (y_range[0] + y_range[1]) / 2])

    def sample(self, rng: np.random.Generator):
        pick_xy   = self._sample_xy(rng, self.pick_x,  self.pick_y)
        place_xy  = self._sample_xy(rng, self.place_x, self.place_y)
        pick_pos  = np.array([*pick_xy,  self.pick_z])
        place_pos = np.array([*place_xy, self.place_z])
        instruction = str(rng.choice(self.instructions))
        return self._build_waypoints(pick_pos, place_pos), instruction, pick_pos, place_pos

    def _build_waypoints(self, pick: np.ndarray, place: np.ndarray) -> List[Waypoint]:
        q = _quat_eef_down()
        pz, plz   = self.pick_z, self.place_z
        h, a      = self.lift_h, self.approach
        transit_z = max(pz, plz) + h
        return [
            Waypoint(np.array([pick[0],  pick[1],  pz + h]),    q, 0.0, 20),
            Waypoint(np.array([pick[0],  pick[1],  pz + a]),    q, 0.0, 15),
            Waypoint(np.array([pick[0],  pick[1],  pz]),         q, 0.0, 10),
            Waypoint(np.array([pick[0],  pick[1],  pz]),         q, 1.0, 15),
            Waypoint(np.array([pick[0],  pick[1],  transit_z]), q, 1.0, 20),
            Waypoint(np.array([place[0], place[1], transit_z]), q, 1.0, 25),
            Waypoint(np.array([place[0], place[1], plz + h]),   q, 1.0, 15),
            Waypoint(np.array([place[0], place[1], plz + a]),   q, 1.0, 15),
            Waypoint(np.array([place[0], place[1], plz]),        q, 1.0, 10),
            Waypoint(np.array([place[0], place[1], plz]),        q, 0.0, 15),
            Waypoint(np.array([place[0], place[1], plz + h]),   q, 0.0, 20),
        ]
