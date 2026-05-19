from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    Call task.sample(rng) once per episode to get:
        waypoints, instruction, pick_pos, place_pos, obj_type, obj_half_size

    surface_z  — Z of the pallet/table surface in world frame. The pick object
                 is placed with its bottom face at this height, so its centre is
                 at surface_z + obj_half_size.

    place_z    — Z of the conveyor surface in world frame. The object is released
                 with its bottom face at this height, so its centre is at
                 place_z + obj_half_size.

    Waypoints add (obj_half_size + eef_z_offset) to each surface-relative centre
    to obtain the link_6 FK target at which the cup tips contact the object top:

        cup_tip_world_Z = link6_world_Z − eef_z_offset
        contact_Z       = obj_centre_Z  + obj_half_size   (top surface)
        → link6_world_Z = obj_centre_Z  + obj_half_size + eef_z_offset
    """

    OBJ_TYPES = ['cube', 'sphere']

    DEFAULT_INSTRUCTIONS = [
        "pick up the object from the pallet and place it on the conveyor",
        "grasp the object and move it to the conveyor belt",
        "pick up the item from the pallet and place it on the belt",
        "move the object from the pallet to the conveyor",
        "pick and place the object onto the conveyor belt",
        "grab the item on the pallet and drop it on the conveyor",
        "transfer the object from the pallet to the belt",
    ]

    def __init__(
        self,
        pick_x:          Tuple[float, float] = (0.59, 1.26),
        pick_y:          Tuple[float, float] = (-0.3, 0.8),
        surface_z:       float = 0.43,    # pallet surface Z in world frame
        place_x:         Tuple[float, float] = (-0.95, 0.85),
        place_y:         Tuple[float, float] = (-1.55, -1.0),
        place_z:         float = 0.85,    # conveyor surface Z in world frame
        lift_h:          float = 0.20,
        approach:        float = 0.05,
        min_separation:  float = 0.10,
        max_reach_xy:    float = 1.65,
        eef_z_offset:    float = 0.185,   # link_6 → cup-tip distance along gripper -Z
        obj_half_size:   float = 0.05,    # object half-size (metres): cube half-edge / sphere radius
        instructions:    Optional[List[str]] = None,
    ):
        self.pick_x         = pick_x
        self.pick_y         = pick_y
        self.surface_z      = surface_z
        self.place_x        = place_x
        self.place_y        = place_y
        self.place_z        = place_z
        self.lift_h         = lift_h
        self.approach       = approach
        self.min_sep        = min_separation
        self.max_reach_xy   = max_reach_xy
        self.eef_z_offset   = eef_z_offset
        self.obj_half_size  = obj_half_size
        self.instructions   = instructions or self.DEFAULT_INSTRUCTIONS

    def _sample_xy(self, rng, x_range, y_range) -> np.ndarray:
        for _ in range(500):
            xy = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
            if np.linalg.norm(xy) <= self.max_reach_xy:
                return xy
        return np.array([(x_range[0] + x_range[1]) / 2,
                         (y_range[0] + y_range[1]) / 2])

    def sample(self, rng: np.random.Generator):
        obj_type      = str(rng.choice(self.OBJ_TYPES))
        obj_half_size = self.obj_half_size
        pick_xy       = self._sample_xy(rng, self.pick_x,  self.pick_y)
        place_xy      = self._sample_xy(rng, self.place_x, self.place_y)

        # Object centre is half-size above the surface
        pick_pos  = np.array([*pick_xy,  self.surface_z + obj_half_size])
        place_pos = np.array([*place_xy, self.place_z   + obj_half_size])

        instruction = str(rng.choice(self.instructions))
        waypoints   = self._build_waypoints(pick_pos, place_pos, obj_half_size)
        return waypoints, instruction, pick_pos, place_pos, obj_type, obj_half_size

    def _build_waypoints(
        self,
        pick:         np.ndarray,
        place:        np.ndarray,
        obj_half_size: float,
    ) -> List[Waypoint]:
        q   = _quat_eef_down()
        pz  = pick[2]    # cube/sphere centre Z at pick
        plz = place[2]   # cube/sphere centre Z at place
        h   = self.lift_h
        a   = self.approach

        # link_6 FK height at which cup tips contact the object's top surface
        contact_offset = obj_half_size + self.eef_z_offset
        pick_l6z  = pz  + contact_offset
        place_l6z = plz + contact_offset
        # Transit at pick-lift height or 10 cm above place contact — whichever
        # is higher. Avoids swinging near the ceiling when the conveyor is elevated.
        transit_z = max(pick_l6z + h, place_l6z + 0.10)

        return [
            # ── Approach pick ────────────────────────────────────────────────
            Waypoint(np.array([pick[0],  pick[1],  pick_l6z + h]),    q, 0.0, 40),
            Waypoint(np.array([pick[0],  pick[1],  pick_l6z + a]),    q, 0.0, 20),
            Waypoint(np.array([pick[0],  pick[1],  pick_l6z]),         q, 0.0, 15),
            # ── Grip: close and dwell 1 s (2 × RETRY_INTERVAL) ──────────────
            Waypoint(np.array([pick[0],  pick[1],  pick_l6z]),         q, 1.0, 60),
            # ── Lift and transit ─────────────────────────────────────────────
            Waypoint(np.array([pick[0],  pick[1],  transit_z]),       q, 1.0, 25),
            Waypoint(np.array([place[0], place[1], transit_z]),       q, 1.0, 30),
            # ── Approach place ───────────────────────────────────────────────
            Waypoint(np.array([place[0], place[1], place_l6z + a]),   q, 1.0, 25),
            Waypoint(np.array([place[0], place[1], place_l6z]),        q, 1.0, 15),
            # ── Release and retract ──────────────────────────────────────────
            Waypoint(np.array([place[0], place[1], place_l6z]),        q, 0.0, 20),
            Waypoint(np.array([place[0], place[1], transit_z]),       q, 0.0, 20),
        ]
