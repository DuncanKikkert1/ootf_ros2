# task.py — Scripted pick-and-place tasks with LULA IK waypoint generation.
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Waypoint:
    """One pose in a scripted joint-space trajectory."""
    position: np.ndarray  # 6-element joint angles in radians
    gripper:  float       # 0.0 = open, 1.0 = closed
    n_steps:  int = 15
    blend:    int = 0     # steps to curve into next waypoint (0 = hard stop)


def _quat_eef_down() -> np.ndarray:
    """Gripper +Z points world-down (standard straight-down pick orientation)."""
    xyzw = Rotation.from_euler('x', np.pi).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def _quat_eef_down_yaw(yaw: float) -> np.ndarray:
    """Straight-down EEF orientation rotated yaw radians around world Z."""
    xyzw = (Rotation.from_euler('z', yaw) * Rotation.from_euler('x', np.pi)).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def _quat_for_face_normal(n_outward: np.ndarray) -> np.ndarray:
    """wxyz quaternion aligning gripper +Z with the inward face normal (-n_outward).

    Builds a full orthonormal body frame to eliminate the unconstrained axial spin
    around the approach axis. Falls back to world-forward when n_outward is near-vertical.
    """
    z_body = -n_outward / np.linalg.norm(n_outward)

    world_ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z_body, world_ref)) > 0.9:
        world_ref = np.array([0.0, 1.0, 0.0])

    x_body = np.cross(world_ref, z_body)
    x_body /= np.linalg.norm(x_body)
    y_body = np.cross(z_body, x_body)

    R    = np.column_stack([x_body, y_body, z_body])
    xyzw = Rotation.from_matrix(R).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


# ── Fixed-geometry objects ─────────────────────────────────────────────────────
_OBJ_PARAMS: Dict[str, dict] = {
    'cube':     dict(height=0.10, grip_dwell=60),
    'cylinder': dict(height=0.08, grip_dwell=60),
}

# ── Pyramid pool ───────────────────────────────────────────────────────────────
# True apex pyramids (rectangular base L×W, apex at height H, no flat top).
# Pre-baked shapes covering the parameter space; one is selected per episode.
# Dynamic prim creation between episodes is not used because force_load_physics_
# from_usd() invalidates Isaac Sim's tensor simulation view.
#
# Face normals for base (±L/2, ±W/2, 0) and apex (0, 0, H):
#   ±X faces → outward normal ∝ (±2H, 0, L),  graspable width at centroid = 2W/3
#   ±Y faces → outward normal ∝ (0, ±2H, W),  graspable width at centroid = 2L/3
#   Best face = most-horizontal normal (largest Z-component) with enough width
#   for the suction footprint.  Since f(x)=x/√(4H²+x²) is monotone, the face
#   associated with the LARGER base dimension always wins if both fit.
#
# Tilt angle from vertical = arctan(2H / max(L, W)).  All shapes below ≤ 45°.
# Rows alternate L>W and W>L so both ±X and ±Y grabs are represented.
PYRAMID_POOL = [
    # idx  L      W      H      best-face  tilt
    dict(L=0.20, W=0.12, H=0.08),  #  0  ±X  ~39°
    dict(L=0.12, W=0.20, H=0.08),  #  1  ±Y  ~39°
    dict(L=0.24, W=0.14, H=0.10),  #  2  ±X  ~40°
    dict(L=0.14, W=0.24, H=0.10),  #  3  ±Y  ~40°
    dict(L=0.22, W=0.13, H=0.09),  #  4  ±X  ~39°
    dict(L=0.13, W=0.22, H=0.09),  #  5  ±Y  ~39°
    dict(L=0.26, W=0.15, H=0.11),  #  6  ±X  ~40°
    dict(L=0.15, W=0.26, H=0.11),  #  7  ±Y  ~40°
    dict(L=0.18, W=0.12, H=0.08),  #  8  ±X  ~41°
    dict(L=0.12, W=0.18, H=0.08),  #  9  ±Y  ~41°
    dict(L=0.25, W=0.16, H=0.09),  # 10  ±X  ~36°
    dict(L=0.16, W=0.25, H=0.09),  # 11  ±Y  ~36°
    dict(L=0.20, W=0.14, H=0.10),  # 12  ±X  ~45°
    dict(L=0.14, W=0.20, H=0.10),  # 13  ±Y  ~45°
    dict(L=0.24, W=0.16, H=0.12),  # 14  ±X  ~45°
    dict(L=0.16, W=0.24, H=0.12),  # 15  ±Y  ~45°
]

# Minimum graspable width (m): 2/3 × perpendicular_dim must exceed this.
# Slightly above the 4.4 cm cup span measured in gripper.py.
PYRAMID_GRIPPER_FP = 0.05
PYRAMID_APPROACH   = 0.03   # m — extra pre-contact clearance for slanted faces
PYRAMID_GRIP_DWELL = 45     # steps to dwell at contact before gripper fires

# Pyramid needs more steps because the EEF must swing from home (J6≈0) to the
# tilted face-normal orientation (J6≈90°) — ~1.5°/step at 60 steps causes PD overshoot.
PYRAMID_SAFE_STEPS     = 150  # home → pallet_safe (large J2 + J6 swing)
PYRAMID_APPROACH_STEPS = 75   # pallet_safe → pre_pick, pre_pick → contact
PYRAMID_TRANSFER_STEPS = 120  # retract → transfer and transfer → pre_place (large multi-joint swings)

# IK solving tolerances — kept in sync with DataCollector constants
_IK_POS_TOL = 0.005   # m
_IK_ORI_TOL = 0.01    # rad

# Home / rest pose (radians) — robot start position and final return waypoint.
_JP_HOME = np.array([0.0, 0.0, 1.57, 0.0, 1.57, 0.0])

# Shared step counts for cube/cylinder (straight-down EEF, smaller J6 swing than pyramid).
_SAFE_STEPS      = 90   # home → center_align (J2 swings ~90° from upright)
_CENTERING_STEPS = 90   # center_align → pallet_safe (mostly vertical descent)
_APPROACH_STEPS  = 60   # pre-pick → contact and retract
_TRANSFER_STEPS  = 90   # retract → transfer and transfer → pre-place
_HOME_STEPS      = 120  # retract-from-place → home (J1 ~90°, J2 ~80° in one arc)

# World-frame link_6 Z for the centering waypoint.  The arm swings from HOME to
# directly above the pick object at this height, so the wrist camera sees the object
# move toward the frame centre.  Descent to pallet_safe (z=1.00) follows vertically.
_CENTERING_Z = 1.50

# J2 ≈ 0 rad is near-vertical; arcs crossing this zone swing the arm through
# the [0,0,0,0,0,0] singularity region with unpredictable joint velocities.
_J2_SWING_THRESHOLD = np.radians(15.0)


def _arc_crosses_vertical(a: float, b: float) -> bool:
    """True when the J2 arc from a to b crosses the near-vertical zone in its interior.

    Arcs that start or end near vertical (home, J2=0) are intentional and are not flagged.
    """
    diff  = (b - a + np.pi) % (2 * np.pi) - np.pi
    b_end = a + diff
    if abs(b_end) < _J2_SWING_THRESHOLD:   # destination is near vertical — intentional (home)
        return False
    if abs(a) < _J2_SWING_THRESHOLD:       # source is near vertical — intentional (home)
        return False
    return a * b_end < 0                   # sign change → arc crosses zero


def _normalize_j6(waypoints: List[Waypoint]) -> List[Waypoint]:
    """Resolve each J6 to the ±π equivalent closest to the preceding waypoint.

    The SMC gripper has 180° rotational symmetry, so J6 and J6±π produce identical
    grasps. Anchoring each step to the previous eliminates 180° spins that arise
    when adjacent IK solutions land on opposite sides of the ±π boundary.
    """
    result = []
    prev_j6 = 0.0
    for wp in waypoints:
        q   = wp.position.copy()
        j6  = q[5]
        alt = j6 + (np.pi if j6 < 0 else -np.pi)
        if abs(alt - prev_j6) < abs(j6 - prev_j6):
            q[5] = alt
        prev_j6 = q[5]
        result.append(Waypoint(q, wp.gripper, wp.n_steps, wp.blend))
    return result


def _trajectory_safe(waypoints: List[Waypoint]) -> bool:
    """Return False if any consecutive pair would sweep J2 through the near-vertical zone."""
    for i in range(len(waypoints) - 1):
        if _arc_crosses_vertical(waypoints[i].position[1],
                                 waypoints[i + 1].position[1]):
            return False
    return True


# Semantic labels for the 12-waypoint trajectory — used by debug tooling.
WAYPOINT_NAMES = [
    "Center Above Pick",    # arm moves XY to directly above object at high Z
    "Pallet Safe",          # vertical descent to clearance height above pick
    "Pre-Pick",
    "Pick Contact (open)",
    "Pick Dwell (closed)",
    "Retract with object",
    "Transfer",
    "Pre-Place",
    "Place Contact (closed)",
    "Place Release (open)",
    "Retract from place",
    "Return Home",
]


class PickAndPlaceTask:
    """Scripted pick-and-place with per-episode domain randomisation for cube, cylinder, and pyramid objects."""

    OBJ_TYPES = list(_OBJ_PARAMS) + ['pyramid']

    # {obj} is filled with the episode's object type (cube/cylinder/pyramid) so
    # the language carries information: identical scenes with different
    # instructions demand different behaviour, which keeps the pretrained
    # language→action pathway load-bearing instead of learned-to-be-ignored.
    DEFAULT_INSTRUCTIONS = [
        "pick up the {obj} from the pallet and place it on the conveyor",
        "grasp the {obj} and move it to the conveyor belt",
        "pick up the {obj} from the pallet and place it on the belt",
        "move the {obj} from the pallet to the conveyor",
        "pick and place the {obj} onto the conveyor belt",
        "grab the {obj} on the pallet and drop it on the conveyor",
        "transfer the {obj} from the pallet to the belt",
        "take the {obj} and put it on the conveyor",
        "pick the {obj} off the pallet and set it on the belt",
        "move the {obj} from the pallet onto the belt",
    ]

    def __init__(
        self,
        pick_x:         Tuple[float, float] = (0.59, 1.26),
        pick_y:         Tuple[float, float] = (-0.3, 0.8),
        surface_z:      float = 0.43,
        place_x:        Tuple[float, float] = (-0.95, 0.85),
        place_y:        Tuple[float, float] = (-1.55, -1.0),
        place_z:        float = 0.85,
        lift_h:         float = 0.20,
        approach:       float = 0.05,
        min_separation: float = 0.10,
        min_reach_xy:   float = 0.65,
        max_reach_xy:   float = 1.65,
        eef_z_offset:   float = 0.215,
        instructions:   Optional[List[str]] = None,
        verbose:        bool  = False,
    ):
        """Configure workspace bounds and EEF geometry for episode sampling."""
        self.pick_x       = pick_x
        self.pick_y       = pick_y
        self.surface_z    = surface_z
        self.place_x      = place_x
        self.place_y      = place_y
        self.place_z      = place_z
        self.lift_h       = lift_h
        self.approach     = approach
        self.min_sep      = min_separation
        self.min_reach_xy = min_reach_xy
        self.max_reach_xy = max_reach_xy
        self.eef_z_offset = eef_z_offset
        self.instructions = instructions or self.DEFAULT_INSTRUCTIONS
        self.verbose      = verbose

    def _sample_xy(self, rng, x_range, y_range) -> np.ndarray:
        """Sample a random XY position within the given ranges and the reachable annulus."""
        for _ in range(500):
            xy = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
            r = np.linalg.norm(xy)
            if self.min_reach_xy <= r <= self.max_reach_xy:
                return xy
        return np.array([(x_range[0]+x_range[1])/2, (y_range[0]+y_range[1])/2])

    @staticmethod
    def _pyramid_best_face(L: float, W: float, H: float):
        """Return (face_axis, n_outward_pos_side) for the most-horizontal graspable face, or None."""
        candidates = []

        if 2.0 * W / 3.0 >= PYRAMID_GRIPPER_FP:   # ±X pair
            n = np.array([2.0*H, 0.0, L])
            n /= np.linalg.norm(n)
            candidates.append(('x', float(n[2]), n))

        if 2.0 * L / 3.0 >= PYRAMID_GRIPPER_FP:   # ±Y pair
            n = np.array([0.0, 2.0*H, W])
            n /= np.linalg.norm(n)
            candidates.append(('y', float(n[2]), n))

        if not candidates:
            return None
        best_axis, _, n_pos = max(candidates, key=lambda c: c[1])
        return best_axis, n_pos

    def _solve_ik(self, ik, pos_world: np.ndarray, ori_wxyz: np.ndarray,
                  robot_world_pos: np.ndarray, robot_R: np.ndarray,
                  ee_frame: str, warm_start: np.ndarray = None):
        """Resolve a world-space Cartesian target to joint angles; returns (q, ok).

        Output is wrapped to [-π, π] because the LULA YAML has no joint limits and
        can return values like 256.9° that ArticulationAction would try to reach physically.
        """
        pos_robot = robot_R.T @ (pos_world - robot_world_pos)
        q, ok = ik.compute_inverse_kinematics(
            ee_frame, pos_robot, ori_wxyz,
            position_tolerance=_IK_POS_TOL,
            orientation_tolerance=_IK_ORI_TOL,
            warm_start=warm_start,
        )
        if ok:
            q = (q + np.pi) % (2 * np.pi) - np.pi
        return q, ok

    def _pick_ik_branch(
        self,
        ik,
        pick_l6:     np.ndarray,
        pick_pre:    np.ndarray,
        pallet_safe: np.ndarray,
        q_pick:      np.ndarray,
        robot_world_pos: np.ndarray,
        robot_R:     np.ndarray,
        ee_frame:    str,
    ):
        """Try multiple IK seeds and return the (q_safe, q_p3, q_p4) triple closest to home.

        pick_l6 / pick_pre / pallet_safe are world-space link_6 targets; q_pick is the
        wxyz EEF orientation quaternion for the pick face.
        LULA finds different local minima from different seeds; for tilted-face pyramids the
        default seed (home) often yields large J4/J5 swings. Returns None if no seed
        yields a complete three-waypoint chain.
        """
        pick_robot = robot_R.T @ (pick_l6 - robot_world_pos)
        j1_est = float(np.arctan2(pick_robot[1], pick_robot[0]))

        seeds = [
            _JP_HOME.copy(),                                        # home (same as YAML default_q)
            np.array([j1_est,  0.0, 1.57,  0.0,  1.57, 0.0]),    # J1 toward pick, J5 up
            np.array([j1_est,  0.0, 1.57,  0.0, -1.57, 0.0]),    # J1 toward pick, J5 flipped
            np.array([0.0,     0.0, 1.57,  0.0, -1.57, 0.0]),    # home + J5 flipped
        ]

        best_triple = None
        best_dist   = np.inf
        best_seed_i = -1

        for i, seed in enumerate(seeds):
            q4, ok4 = self._solve_ik(ik, pick_l6,     q_pick, robot_world_pos, robot_R, ee_frame,
                                     warm_start=seed)
            if not ok4:
                continue
            q3, ok3 = self._solve_ik(ik, pick_pre,    q_pick, robot_world_pos, robot_R, ee_frame,
                                     warm_start=q4)
            if not ok3:
                continue
            qs, oks = self._solve_ik(ik, pallet_safe, q_pick, robot_world_pos, robot_R, ee_frame,
                                     warm_start=q3)
            if not oks:
                continue
            dist = float(np.sum((qs - _JP_HOME) ** 2))
            if dist < best_dist:
                best_dist  = dist
                best_triple = (qs, q3, q4)
                best_seed_i = i

        if best_triple is not None:
            qs, _, _ = best_triple
            if self.verbose:
                print(f"[TASK] pick branch: seed={best_seed_i}  "
                      f"q_safe={np.degrees(qs).round(1).tolist()}  "
                      f"dist_home={best_dist:.3f}", flush=True)
        return best_triple

    @staticmethod
    def _format_instruction(template: str, obj_type: str) -> str:
        """Fill the {obj} placeholder with the episode's object type.

        Uses str.replace so fixed instructions without a placeholder
        (e.g. a CLI --instruction override) pass through unchanged.
        """
        return template.replace("{obj}", obj_type)

    def sample(self, rng: np.random.Generator, ik,
               robot_world_pos: np.ndarray, robot_R: np.ndarray,
               ee_frame: str = 'link_6',
               force_obj_type: str = None,
               ik_place=None,
               pick_yaw: float = 0.0,
               ) -> Tuple[Optional[List[Waypoint]], str, np.ndarray, np.ndarray, str, dict]:
        """Sample one episode: returns (waypoints, instruction, pick_pos, place_pos, obj_type, meta)."""
        obj_type = force_obj_type if force_obj_type is not None \
                   else str(rng.choice(self.OBJ_TYPES))
        instruction = str(rng.choice(self.instructions))
        pick_xy     = self._sample_xy(rng, self.pick_x,  self.pick_y)
        place_xy    = self._sample_xy(rng, self.place_x, self.place_y)

        if obj_type == 'pyramid':
            pool_idx  = int(rng.integers(len(PYRAMID_POOL)))
            dims      = PYRAMID_POOL[pool_idx]
            L, W, H   = dims['L'], dims['W'], dims['H']
            face_info = self._pyramid_best_face(L, W, H)

            if face_info is None:
                obj_type = 'cube'   # shouldn't happen with current pool, but safe
            else:
                face_axis, n_pos = face_info
                side      = int(rng.choice([-1, 1]))
                n_outward = n_pos.copy()
                if side == -1:
                    n_outward[0 if face_axis == 'x' else 1] *= -1

                pick_pos  = np.array([*pick_xy,  self.surface_z])
                place_pos = np.array([*place_xy, self.place_z])
                waypoints = self._build_pyramid_waypoints(
                    pick_pos, place_pos, L, W, H, face_axis, side, n_outward,
                    ik, robot_world_pos, robot_R, ee_frame, ik_place=ik_place,
                )
                return waypoints, self._format_instruction(instruction, 'pyramid'), \
                       pick_pos, place_pos, 'pyramid', \
                       {'pool_idx': pool_idx, 'L': L, 'W': W, 'H': H,
                        'face_axis': face_axis, 'side': side,
                        'n_outward': n_outward}

        params    = _OBJ_PARAMS[obj_type]
        hh        = params['height'] / 2.0
        pick_pos  = np.array([*pick_xy,  self.surface_z + hh])
        place_pos = np.array([*place_xy, self.place_z   + hh])
        waypoints = self._build_waypoints(
            pick_pos, place_pos, params, ik, robot_world_pos, robot_R, ee_frame,
            ik_place=ik_place, pick_yaw=pick_yaw,
        )
        return waypoints, self._format_instruction(instruction, obj_type), \
               pick_pos, place_pos, obj_type, {}

    # ── waypoint builders ──────────────────────────────────────────────────────

    def _build_waypoints(self, pick, place, params,
                         ik, robot_world_pos: np.ndarray,
                         robot_R: np.ndarray, ee_frame: str,
                         ik_place=None, pick_yaw: float = 0.0) -> Optional[List[Waypoint]]:
        """Build an 11-waypoint trajectory for cube/cylinder with straight-down EEF.

        All waypoints are IK-solved with the previous solution as the warm-start seed
        so LULA stays in the same kinematic branch across the entire trajectory.
        Returns None if any IK solve fails so the episode can be resampled cleanly.
        """
        q_pick = _quat_eef_down_yaw(pick_yaw)
        q_down = _quat_eef_down()
        hh     = params['height'] / 2.0
        gd     = params['grip_dwell']

        # 5 mm clearance keeps cup tips above the object so the arm doesn't press
        # it down. Surface grippers attach by scan — 8 cm range is plenty.
        _PICK_Z_CLEARANCE = 0.005
        co       = hh + self.eef_z_offset + _PICK_Z_CLEARANCE
        # All positions are link_6 targets in world space (NOT TCP).
        # TCP_tip = link_6 + R @ [0,0,0.215]; arm pointing down → TCP_Z = link_6_Z - 0.215.
        pick_l6     = np.array([pick[0],  pick[1],  pick[2]  + co])
        place_l6    = np.array([place[0], place[1], place[2] + co])
        pallet_safe = np.array([pick_l6[0],  pick_l6[1],  1.00])
        pre_pick    = pick_l6  + np.array([0.0, 0.0, 0.20])
        transfer    = np.array([0.9, -0.5, 1.565])   # fixed mid-air waypoint; TCP ~0.215 m below
        pre_place   = place_l6 + np.array([0.0, 0.0, 0.20])

        # Centering waypoint: directly above pick at high Z so the arm moves XY first.
        # The wrist camera sees the object approach the frame centre during this arc.
        center_align = np.array([pick_l6[0], pick_l6[1], _CENTERING_Z])
        q_center,   ok = self._solve_ik(ik, center_align, q_pick, robot_world_pos, robot_R, ee_frame)
        if not ok: return None
        q_safe,     ok = self._solve_ik(ik, pallet_safe, q_pick, robot_world_pos, robot_R, ee_frame, warm_start=q_center)
        if not ok: return None
        q_p3,       ok = self._solve_ik(ik, pre_pick,    q_pick, robot_world_pos, robot_R, ee_frame, warm_start=q_safe)
        if not ok: return None
        q_p4,       ok = self._solve_ik(ik, pick_l6,     q_pick, robot_world_pos, robot_R, ee_frame, warm_start=q_p3)
        if not ok: return None
        _ik_pl      = ik_place if ik_place is not None else ik
        q_transfer, ok = self._solve_ik(_ik_pl, transfer,    q_down, robot_world_pos, robot_R, ee_frame)
        if not ok: return None
        q_p6,       ok = self._solve_ik(_ik_pl, pre_place,   q_down, robot_world_pos, robot_R, ee_frame, warm_start=q_transfer)
        if not ok: return None
        q_p7,       ok = self._solve_ik(_ik_pl, place_l6,    q_down, robot_world_pos, robot_R, ee_frame, warm_start=q_p6)
        if not ok: return None

        wps = [
            Waypoint(q_center,   0.0, _SAFE_STEPS),      # center above pick – XY alignment from home
            Waypoint(q_safe,     0.0, _CENTERING_STEPS), # pallet safe – vertical descent to clearance
            Waypoint(q_p3,       0.0, _APPROACH_STEPS),  # pre-pick
            Waypoint(q_p4,       0.0, _APPROACH_STEPS),  # pick contact (open)
            Waypoint(q_p4,       1.0, gd),               # pick dwell   (closed)
            Waypoint(q_p3,       1.0, _APPROACH_STEPS),  # retract with object
            Waypoint(q_transfer, 1.0, _TRANSFER_STEPS),  # transfer
            Waypoint(q_p6,       1.0, _TRANSFER_STEPS),  # pre-place
            Waypoint(q_p7,       1.0, 60),               # place contact (closed)
            Waypoint(q_p7,       0.0, 45),               # place release (open)
            Waypoint(q_p6,       0.0, 60),               # retract from place
            Waypoint(_JP_HOME,   0.0, _HOME_STEPS),      # return home
        ]
        # Exclude the final home waypoint from J6 normalization — home needs J6=0
        # exactly; normalizing it can flip it to ±π, causing a 180° wrist spin during settle.
        wps = _normalize_j6(wps[:-1]) + [wps[-1]]
        return wps if _trajectory_safe(wps) else None

    def _build_pyramid_waypoints(
        self,
        pick_base:  np.ndarray,
        place_base: np.ndarray,
        L: float, W: float, H: float,
        face_axis:  str,
        side:       int,
        n_outward:  np.ndarray,
        ik, robot_world_pos: np.ndarray,
        robot_R: np.ndarray, ee_frame: str,
        ik_place=None,
        ik_pre_place=None,
    ) -> Optional[List[Waypoint]]:
        """Build an 11-waypoint trajectory for pyramid pick-and-place with face-normal EEF.

        Pick approaches along the chosen face normal (tilted EEF). Place maintains the same
        EEF orientation so the gripped face is pressed flat against the belt.
        ik_place seeds toward the belt zone; ik_pre_place overrides the pre_place solve only
        (defaults to ik_place, or ik if ik_place is None).
        Returns None if any IK solve fails.
        """
        q_pick = _quat_for_face_normal(n_outward)
        a      = self.approach + PYRAMID_APPROACH
        ez     = self.eef_z_offset

        if face_axis == 'x':
            fc_off = np.array([side * L / 3.0, 0.0, H / 3.0])
        else:
            fc_off = np.array([0.0, side * W / 3.0, H / 3.0])

        _PICK_CLEARANCE = 0.005   # m — keep cup tips off face surface
        pick_fc  = pick_base + fc_off
        pick_l6  = pick_fc  + (ez + _PICK_CLEARANCE) * n_outward
        pick_pre = pick_l6  + np.array([0.0, 0.0, a])  # vertical final approach

        # Offset the place target by fc_off so the pyramid BASE lands centered at place_base.
        # The grip face centroid is at fc_off relative to the base (horizontal + vertical).
        # Without the horizontal component the base lands shifted off-center by side*L/3 or side*W/3.
        place_base_lifted = place_base + fc_off
        place_l6    = place_base_lifted + ez * n_outward
        pallet_safe = pick_l6   + 0.15 * n_outward
        transfer    = np.array([0.9, -0.5, 1.565])   # fixed mid-air waypoint, same as cube/cylinder
        pre_place   = place_l6  + np.array([0.0, 0.0, a])  # vertical final approach to place

        _pick_branch = self._pick_ik_branch(
            ik, pick_l6, pick_pre, pallet_safe, q_pick,
            robot_world_pos, robot_R, ee_frame,
        )
        if _pick_branch is None:
            return None
        q_safe, q_p3, q_p4 = _pick_branch

        # Normalize J6 for the pick approach now so all five pre-contact waypoints share
        # a consistent J6 branch. Without early normalization the arm spins 180° between
        # pick-contact and dwell — right as the gripper fires.
        _pre = _normalize_j6([
            Waypoint(q_safe, 0.0, 1),
            Waypoint(q_p3,   0.0, 1),
            Waypoint(q_p4,   0.0, 1),
        ])
        q_safe_n, q_p3_n, q_p4_n = _pre[0].position, _pre[1].position, _pre[2].position

        # Centering waypoint: directly above pick base at high Z with straight-down
        # EEF orientation.  The face-normal tilt is only needed from pallet_safe
        # onwards; using it here at high Z makes IK unreliable.
        center_align = np.array([pick_base[0], pick_base[1], _CENTERING_Z])
        q_center, ok = self._solve_ik(ik, center_align, _quat_eef_down(),
                                      robot_world_pos, robot_R, ee_frame)
        if not ok: return None

        _ik_pl = ik_place if ik_place is not None else ik
        _ik_pp = ik_pre_place if ik_pre_place is not None else _ik_pl
        q_transfer, ok = self._solve_ik(_ik_pl, transfer, q_pick, robot_world_pos, robot_R, ee_frame)
        if not ok: return None

        # Snap transfer J6 to the ±π equivalent closest to normalized pick J6 before
        # warm-starting place solves — the wrong J6 propagates into J1-J5.
        q_transfer = q_transfer.copy()
        _alt = q_transfer[5] + (np.pi if q_transfer[5] < 0 else -np.pi)
        if abs(_alt - q_p4_n[5]) < abs(q_transfer[5] - q_p4_n[5]):
            q_transfer[5] = _alt

        # Geometric seeds for place phase: J1 estimated from arctan2 toward platform,
        # J2/J3/J5 from observed belt-zone configs. Avoids near-singular J2≈90°
        # that the default warm-start (J1=-29° from q_transfer) converges to.
        _pp_xyz = robot_R.T @ (pre_place - robot_world_pos)
        _j1_est = float(np.arctan2(_pp_xyz[1], _pp_xyz[0]))
        _j6     = q_p4_n[5]
        _geo_seeds = [
            np.array([_j1_est,        np.radians(66.), np.radians(66.), 0., np.radians(48.), _j6]),
            np.array([_j1_est,        np.radians(77.), np.radians(41.), 0., np.radians(62.), _j6]),
            np.array([_j1_est,        np.radians(56.), np.radians(70.), 0., np.radians(50.), _j6]),
            np.array([_j1_est - 0.35, np.radians(66.), np.radians(66.), 0., np.radians(48.), _j6]),
            np.array([_j1_est + 0.35, np.radians(66.), np.radians(66.), 0., np.radians(48.), _j6]),
        ]
        if _ik_pp is _ik_pl:
            # Same solver: try q_transfer-based seed first (J6 continuity), then geo seeds,
            # then ik_place's default_q (J1=-130°) which may find a different branch.
            _place_seed    = q_transfer.copy()
            _place_seed[5] = q_p4_n[5]
            _pp_seeds = [_place_seed] + _geo_seeds + [None]
        else:
            _pp_seeds = _geo_seeds + [None]
        q_p6 = None
        for _seed in _pp_seeds:
            q_p6, ok = self._solve_ik(_ik_pp, pre_place, q_pick, robot_world_pos, robot_R, ee_frame,
                                      warm_start=_seed)
            if ok:
                break
        if not ok: return None
        q_p7, ok = self._solve_ik(_ik_pp, place_l6, q_pick, robot_world_pos, robot_R, ee_frame,
                                   warm_start=q_p6)
        if not ok:
            for _seed in _geo_seeds + [None]:
                q_p7, ok = self._solve_ik(_ik_pp, place_l6, q_pick, robot_world_pos, robot_R, ee_frame,
                                          warm_start=_seed)
                if ok:
                    break
        if not ok: return None

        # Snap place-side J6 to the same ±π branch as the pick J6.
        # Belt-zone IK can land on the π-opposite J6 solution, causing the gripped
        # pyramid to appear to rotate 180° as it is set down.
        q_p6 = q_p6.copy()
        q_p7 = q_p7.copy()
        for _q in (q_p6, q_p7):
            _alt = _q[5] + (np.pi if _q[5] < 0 else -np.pi)
            if abs(_alt - q_p4_n[5]) < abs(_q[5] - q_p4_n[5]):
                _q[5] = _alt

        wps = [
            Waypoint(q_center,   0.0, PYRAMID_SAFE_STEPS),      # center above pick – XY alignment from home
            Waypoint(q_safe_n,   0.0, _CENTERING_STEPS),        # pallet safe – vertical descent to clearance
            Waypoint(q_p3_n,     0.0, PYRAMID_APPROACH_STEPS),  # pre-pick (tilted)
            Waypoint(q_p4_n,     0.0, PYRAMID_APPROACH_STEPS),  # pick contact (open)
            Waypoint(q_p4_n,     1.0, PYRAMID_GRIP_DWELL),      # dwell (closed)
            Waypoint(q_p3_n,     1.0, PYRAMID_APPROACH_STEPS),  # retract with object
            Waypoint(q_transfer, 1.0, PYRAMID_TRANSFER_STEPS),  # transfer (arm settles before pre-place)
            Waypoint(q_p6,       1.0, PYRAMID_TRANSFER_STEPS),  # pre-place
            Waypoint(q_p7,       1.0, 90),                      # place contact (closed)
            Waypoint(q_p7,       0.0, 45),                      # place release (open)
            Waypoint(q_p6,       0.0, 60),                      # retract from place
            Waypoint(_JP_HOME,   0.0, _HOME_STEPS),             # return home
        ]
        wps = _normalize_j6(wps[:-1]) + [wps[-1]]
        return wps if _trajectory_safe(wps) else None


# ── Sorting task ───────────────────────────────────────────────────────────────

# Platform centre positions [x, y, z] in world frame. Slabs are 0.01 m thick;
# the top surface (where objects rest) is at centre_z + 0.005 = 0.860 m.
SORT_PLATFORM_CENTRES = {
    'cube':     np.array([-0.5, -1.28, 0.855]),
    'cylinder': np.array([ 0.0, -1.28, 0.855]),
    'pyramid':  np.array([ 0.5, -1.28, 0.855]),
}

# Platform L × W × H (metres).
SORT_PLATFORM_DIMS = {
    'cube':     np.array([0.10, 0.10, 0.01]),
    'cylinder': np.array([0.10, 0.10, 0.01]),
    'pyramid':  np.array([0.25, 0.25, 0.01]),
}

_SORT_SURFACE_Z = 0.860   # platform top surface = centre_z + half-thickness


class SortingTask(PickAndPlaceTask):
    """Pick-and-sort variant: each object type has a fixed designated platform on the belt."""

    # {obj} is filled with the episode's object type.  Naming the object is
    # essential here: the designated platform depends on the object type, so
    # without it the instruction never disambiguates the goal.
    SORT_INSTRUCTIONS = [
        "place the {obj} on its designated platform on the belt",
        "sort the {obj} to its correct platform on the conveyor",
        "move the {obj} to its matching target platform",
        "pick up the {obj} and put it on the right slot on the belt",
        "place the {obj} on its correct platform",
        "find the right platform and place the {obj} on it",
        "sort the {obj} to its designated spot on the conveyor",
        "deliver the {obj} to its assigned platform on the belt",
        "pick the {obj} and sort it to the correct platform",
        "move the {obj} to its designated platform on the conveyor",
    ]

    def __init__(self, **kwargs):
        """Override place_z to the platform surface and use sorting-specific instructions."""
        kwargs.setdefault('place_z', _SORT_SURFACE_Z)
        super().__init__(**kwargs)
        self.instructions = self.SORT_INSTRUCTIONS

    def sample(self, rng: np.random.Generator, ik,
               robot_world_pos: np.ndarray, robot_R: np.ndarray,
               ee_frame: str = 'link_6',
               force_obj_type: str = None,
               ik_place=None,
               pick_yaw: float = 0.0,
               ) -> Tuple[Optional[List[Waypoint]], str, np.ndarray, np.ndarray, str, dict]:
        """Sample one episode; place position is always the fixed platform for the sampled object type."""
        obj_type    = force_obj_type if force_obj_type is not None \
                      else str(rng.choice(self.OBJ_TYPES))
        instruction = str(rng.choice(self.instructions))
        pick_xy     = self._sample_xy(rng, self.pick_x, self.pick_y)

        place_xy = SORT_PLATFORM_CENTRES[
            obj_type if obj_type in SORT_PLATFORM_CENTRES else 'cube'
        ][:2].copy()

        if obj_type == 'pyramid':
            pool_idx  = int(rng.integers(len(PYRAMID_POOL)))
            dims      = PYRAMID_POOL[pool_idx]
            L, W, H   = dims['L'], dims['W'], dims['H']
            face_info = self._pyramid_best_face(L, W, H)

            if face_info is None:
                obj_type = 'cube'
            else:
                face_axis, n_pos = face_info
                side      = int(rng.choice([-1, 1]))
                n_outward = n_pos.copy()
                if side == -1:
                    n_outward[0 if face_axis == 'x' else 1] *= -1

                pick_pos  = np.array([*pick_xy,  self.surface_z])
                place_pos = np.array([*place_xy, self.place_z])

                # Cheap XY reach check: the face normal shifts link_6 by eef_z_offset
                # in XY. Reject face/side combos whose predicted place_l6 XY distance
                # exceeds the robot's reach before spending IK calls.
                _l6_xy  = place_xy + self.eef_z_offset * n_outward[:2]
                _l6_rxy = _l6_xy   - robot_world_pos[:2]
                if np.linalg.norm(_l6_rxy) > self.max_reach_xy:
                    return None, self._format_instruction(instruction, 'pyramid'), \
                           pick_pos, place_pos, 'pyramid', \
                           {'pool_idx': pool_idx, 'L': L, 'W': W, 'H': H,
                            'face_axis': face_axis, 'side': side,
                            'n_outward': n_outward}

                # Use ik_place for both transfer and pre_place/place_l6.
                # Geometric seeds (J1≈ arctan2 toward platform, J2=56-77°) avoid the
                # near-singular J2≈90° configuration that the default warm-start converges to.
                waypoints = self._build_pyramid_waypoints(
                    pick_pos, place_pos, L, W, H, face_axis, side, n_outward,
                    ik, robot_world_pos, robot_R, ee_frame,
                    ik_place=ik_place, ik_pre_place=ik_place,
                )
                return waypoints, self._format_instruction(instruction, 'pyramid'), \
                       pick_pos, place_pos, 'pyramid', \
                       {'pool_idx': pool_idx, 'L': L, 'W': W, 'H': H,
                        'face_axis': face_axis, 'side': side,
                        'n_outward': n_outward}

        params    = _OBJ_PARAMS[obj_type]
        hh        = params['height'] / 2.0
        pick_pos  = np.array([*pick_xy,  self.surface_z + hh])
        place_pos = np.array([*place_xy, self.place_z   + hh])
        waypoints = self._build_waypoints(
            pick_pos, place_pos, params, ik, robot_world_pos, robot_R, ee_frame,
            ik_place=ik_place, pick_yaw=pick_yaw,
        )
        return waypoints, self._format_instruction(instruction, obj_type), \
               pick_pos, place_pos, obj_type, {}
