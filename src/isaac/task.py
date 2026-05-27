from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Waypoint:
    position: np.ndarray  # 6-element joint angles in radians
    gripper:  float       # 0.0 = open, 1.0 = closed
    n_steps:  int = 15
    blend:    int = 0     # steps to curve into next waypoint (0 = hard stop)


def _quat_eef_down() -> np.ndarray:
    """Gripper +Z points world-down (standard straight-down pick orientation)."""
    xyzw = Rotation.from_euler('x', np.pi).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def _quat_eef_down_yaw(yaw: float) -> np.ndarray:
    """Gripper +Z pointing down, rotated yaw radians around world Z.
    Used for pick-side IK so J6 aligns with the cube's yaw orientation.
    yaw=0 gives the same result as _quat_eef_down()."""
    xyzw = (Rotation.from_euler('z', yaw) * Rotation.from_euler('x', np.pi)).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def _quat_for_face_normal(n_outward: np.ndarray) -> np.ndarray:
    """wxyz quaternion aligning gripper +Z with the inward face normal (-n_outward).

    Builds a full orthonormal body frame to eliminate the unconstrained axial
    spin around the approach axis that the single-axis-align approach leaves free:
      z_body = -n_outward          (cups press perpendicular into the surface)
      x_body = world_ref × z_body  (horizon-locked; removes the free yaw DOF)
      y_body = z_body × x_body     (completes the right-handed frame)
    Falls back to world-forward as the reference when n_outward is near-vertical.
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
PYRAMID_APPROACH   = 0.03   # extra pre-contact clearance for slanted faces
PYRAMID_GRIP_DWELL = 45     # dwell at contact before gripper fires (arm settled by then)

# Pyramid approach needs more steps than cube because the EEF must swing from
# home (J6≈0) to the tilted face-normal orientation (J6≈90°) — ~1.5°/step at
# 60 steps is too fast for the PD controller and causes visible oscillation.
PYRAMID_SAFE_STEPS     = 150  # home → pallet_safe (large J2 + J6 swing)
PYRAMID_APPROACH_STEPS = 75   # pallet_safe → pre_pick, pre_pick → contact
PYRAMID_TRANSFER_STEPS = 120  # retract → transfer and transfer → pre_place (large multi-joint swings)

# IK solving tolerances — kept in sync with DataCollector constants
_IK_POS_TOL = 0.005
_IK_ORI_TOL = 0.01

# Home / rest pose (radians) — robot start position and final return waypoint.
_JP_HOME = np.array([0.0, 0.0, 1.57, 0.0, 1.57, 0.0])

# Shared step counts for cube/cylinder (straight-down EEF, smaller J6 swing than pyramid).
_SAFE_STEPS     = 90   # home → pallet_safe (J2 swings ~90° from upright)
_APPROACH_STEPS = 60   # pre-pick → contact and retract
_TRANSFER_STEPS = 90   # retract → transfer and transfer → pre-place
_HOME_STEPS     = 120  # retract-from-place → home (J1 ~90°, J2 ~80° in one arc)

# Minimum absolute J2 value permitted at either end of any interpolated arc.
# J2 ≈ 0 rad means the upper arm is near-vertical; an arc that crosses from
# positive J2 to negative J2 (or vice versa) sweeps the arm upward through the
# [0,0,0,0,0,0] singularity region.
_J2_SWING_THRESHOLD = np.radians(15.0)


def _arc_crosses_vertical(a: float, b: float) -> bool:
    """True when the shortest-arc interpolation from J2=a to J2=b crosses through the
    near-vertical zone (-_J2_SWING_THRESHOLD, +_J2_SWING_THRESHOLD) in the interior.

    An arc whose DESTINATION is near zero (e.g., returning to home where J2=0) is NOT
    flagged — the arm is intentionally stopping at vertical, not swinging through it.
    An arc whose SOURCE is near zero (e.g., starting from home) is likewise allowed.
    """
    diff  = (b - a + np.pi) % (2 * np.pi) - np.pi
    b_end = a + diff
    if abs(b_end) < _J2_SWING_THRESHOLD:   # arm stops at/near vertical — intentional (home)
        return False
    if abs(a) < _J2_SWING_THRESHOLD:       # arm starts at/near vertical — intentional (home)
        return False
    return a * b_end < 0                   # sign change → arc crosses zero


def _normalize_j6(waypoints: list) -> list:
    """Resolve each J6 to the ±π equivalent closest to the preceding waypoint.

    The SMC two-cup suction gripper has 180° rotational symmetry, so J6 and
    J6±π produce identical grasps. Anchoring each step to the previous rather
    than independently to zero eliminates inter-waypoint 180° spins that arise
    when two adjacent IK solutions land on opposite sides of the ±π boundary.
    The first waypoint is anchored to 0 (home J6).
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


def _trajectory_safe(waypoints) -> bool:
    """Return False if any consecutive waypoint pair would sweep J2 through the
    near-vertical zone, causing the arm to fling upward through [0,0,0,0,0,0]."""
    for i in range(len(waypoints) - 1):
        if _arc_crosses_vertical(waypoints[i].position[1],
                                 waypoints[i + 1].position[1]):
            return False
    return True


# Semantic labels for the 10-waypoint trajectory — used by debug tooling.
WAYPOINT_NAMES = [
    "Pallet Safe",
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
    """
    Scripted pick-and-place with per-episode domain randomisation.

    Object types and EEF strategy:
      cube     — 10×10×10 cm, straight-down EEF.
      cylinder — 10 cm dia × 8 cm, straight-down EEF.
      pyramid  — one of 16 pre-baked apex pyramids (PYRAMID_POOL), selected
                 randomly each episode.  EEF tilts to the outward normal of
                 the most-horizontal graspable face.

    sample(rng, ik, robot_world_pos, robot_R, ee_frame) returns:
        waypoints, instruction, pick_pos, place_pos, obj_type, meta

    All Cartesian targets are resolved to joint angles (6-element arrays, radians)
    during sample() using the supplied LULA solver.  Returns waypoints=None when
    IK fails for any target so collect.py can resample the episode cleanly.

    pick_pos for cube/cylinder = geometric centre (DynamicCuboid/Cylinder convention).
    pick_pos for pyramid       = base centre at surface_z (USD local origin).
    meta = {} for cube/cylinder; {'pool_idx': int, 'L', 'W', 'H'} for pyramid.

    Instructions always use generic nouns ("object"/"item") — the model must
    learn the correct approach angle from the visual geometry.
    """

    OBJ_TYPES = list(_OBJ_PARAMS) + ['pyramid']

    DEFAULT_INSTRUCTIONS = [
        "pick up the object from the pallet and place it on the conveyor",
        "grasp the object and move it to the conveyor belt",
        "pick up the item from the pallet and place it on the belt",
        "move the object from the pallet to the conveyor",
        "pick and place the object onto the conveyor belt",
        "grab the item on the pallet and drop it on the conveyor",
        "transfer the object from the pallet to the belt",
        "take the item and put it on the conveyor",
        "pick the object off the pallet and set it on the belt",
        "move the item from the pallet onto the belt",
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
        eef_z_offset:   float = 0.195,
        instructions:   Optional[List[str]] = None,
    ):
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

    def _sample_xy(self, rng, x_range, y_range) -> np.ndarray:
        for _ in range(500):
            xy = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
            r = np.linalg.norm(xy)
            if self.min_reach_xy <= r <= self.max_reach_xy:
                return xy
        return np.array([(x_range[0]+x_range[1])/2, (y_range[0]+y_range[1])/2])

    @staticmethod
    def _pyramid_best_face(L: float, W: float, H: float):
        """
        Return (face_axis, n_outward_pos_side) for the most-horizontal face that
        fits the gripper footprint, or None if neither face pair qualifies.

        face_axis          ∈ {'x', 'y'}
        n_outward_pos_side = unit outward normal of the positive-side face;
                             caller flips sign for the −side.
        """
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
        """Resolve a world-space Cartesian target to joint angles. Returns (q, ok).

        All returned joint angles are wrapped to [-π, π].  The generated LULA YAML
        has no joint position limits, so LULA can return angles such as 256.9° for
        a joint whose hardware range is ±180°.  Wrapping here prevents those values
        from reaching the waypoint list and the Isaac Sim ArticulationAction, which
        would attempt to drive the joint to a physically unreachable position.
        _normalize_j6 runs after IK and handles the J6 ±π symmetry separately.
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
        """Return (q_safe, q_p3, q_p4) for the pick approach whose pallet-safe
        joint configuration is closest to home in joint space, by trying several
        IK seeds.  LULA finds different local minima depending on the seed; for
        tilted-face pyramids the default-q (home) seed produces large J4/J5
        swings.  Trying seeds with J5 negated, J1 pre-rotated toward the pick
        position, and combinations thereof often reveals a branch where the arm
        stays much closer to its rest pose.  Returns None if no seed yields a
        complete three-waypoint chain.
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
            print(f"[TASK] pick branch: seed={best_seed_i}  "
                  f"q_safe={np.degrees(qs).round(1).tolist()}  "
                  f"dist_home={best_dist:.3f}", flush=True)
        return best_triple

    def sample(self, rng: np.random.Generator, ik,
               robot_world_pos: np.ndarray, robot_R: np.ndarray,
               ee_frame: str = 'link_6',
               force_obj_type: str = None,
               ik_place=None,
               pick_yaw: float = 0.0):
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
                return waypoints, instruction, pick_pos, place_pos, 'pyramid', \
                       {'pool_idx': pool_idx, 'L': L, 'W': W, 'H': H,
                        'face_axis': face_axis, 'side': side,
                        'n_outward': n_outward}

        # cube / cylinder (or fallback)
        params    = _OBJ_PARAMS[obj_type]
        hh        = params['height'] / 2.0
        pick_pos  = np.array([*pick_xy,  self.surface_z + hh])
        place_pos = np.array([*place_xy, self.place_z   + hh])
        waypoints = self._build_waypoints(
            pick_pos, place_pos, params, ik, robot_world_pos, robot_R, ee_frame,
            ik_place=ik_place, pick_yaw=pick_yaw,
        )
        return waypoints, instruction, pick_pos, place_pos, obj_type, {}

    # ── waypoint builders ──────────────────────────────────────────────────────

    def _build_waypoints(self, pick, place, params,
                         ik, robot_world_pos: np.ndarray,
                         robot_R: np.ndarray, ee_frame: str,
                         ik_place=None, pick_yaw: float = 0.0) -> Optional[List[Waypoint]]:
        """Straight-down EEF for cube / cylinder — 11-waypoint fully-chained sequence.

        All waypoints are IK-solved with the previous solution as the warm-start seed
        so LULA stays in the same kinematic branch across the entire trajectory.
        Returns None if any IK solve fails so the episode can be resampled cleanly.
        """
        q_pick = _quat_eef_down_yaw(pick_yaw)  # pick orientation rotated to match cube yaw
        q_down = _quat_eef_down()               # straight down for place zone
        hh     = params['height'] / 2.0
        gd     = params['grip_dwell']

        # 1 cm clearance keeps the cup tips above the object so the arm doesn't
        # press the cube.  Surface grippers attach by scan — 8 cm range is plenty.
        _PICK_Z_CLEARANCE = 0.005
        co          = hh + self.eef_z_offset + _PICK_Z_CLEARANCE
        # All positions are link_6 targets in world space (NOT TCP).
        # TCP_tip = link_6 + R @ [0,0,0.215]; arm pointing down → TCP_Z = link_6_Z - 0.215.
        pick_l6     = np.array([pick[0],  pick[1],  pick[2]  + co])
        place_l6    = np.array([place[0], place[1], place[2] + co])
        pallet_safe = np.array([pick_l6[0],  pick_l6[1],  1.00])
        pre_pick    = pick_l6  + np.array([0.0, 0.0, 0.20])
        transfer    = np.array([0.9, -0.5, 1.565])   # link_6 target; TCP ~0.215 m below
        pre_place   = place_l6 + np.array([0.0, 0.0, 0.20])

        q_safe,     ok = self._solve_ik(ik, pallet_safe, q_pick, robot_world_pos, robot_R, ee_frame)
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
            Waypoint(q_safe,     0.0, _SAFE_STEPS),      # pallet safe – clearance above pick
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
        # Normalize all waypoints except the final home waypoint.  Home always
        # needs J6=0 exactly; including it in normalization can flip it to ±π,
        # causing the wrist to lerp to 180° and then snap back during settle.
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
        """Pure joint-space 12-step sequence for pyramid pick-and-place.

        Pick phase (p3/p4) approaches along the chosen face normal with a tilted
        EEF so the suction cups press perpendicular into the sloped surface.
        Place phase (p6/p7) maintains the same EEF orientation (q_pick) so the
        gripped face is pressed flat against the conveyor belt.
        All Cartesian targets are pre-solved to joint angles during episode setup.
        Returns None if any IK solve fails.
        """
        q_pick = _quat_for_face_normal(n_outward)
        a      = self.approach + PYRAMID_APPROACH
        ez     = self.eef_z_offset

        if face_axis == 'x':
            fc_off = np.array([side * L / 3.0, 0.0, H / 3.0])
        else:
            fc_off = np.array([0.0, side * W / 3.0, H / 3.0])

        _PICK_CLEARANCE = 0.005                  # keep cup tips off face surface (same as cube)
        pick_fc  = pick_base + fc_off
        pick_l6  = pick_fc  + (ez + _PICK_CLEARANCE) * n_outward
        pick_pre = pick_l6  + a  * n_outward   # pre-pick along face normal

        # Raise the place target so the pyramid BASE lands at place_z, not the grip face.
        # The grip face centroid is H/3 above the base; without this lift the base would
        # collide with the conveyor before the arm reaches its target.
        place_base_lifted = place_base + np.array([0.0, 0.0, H / 3.0])
        place_l6    = place_base_lifted + ez * n_outward
        pallet_safe = pick_l6   + 0.15 * n_outward              # clearance along face normal
        transfer    = np.array([0.9, -0.5, 1.565])              # fixed mid-air waypoint, same as cube/cylinder
        pre_place   = place_l6  + a  * n_outward               # approach belt along face normal

        # Solve pick approach with multi-seed branch selection: try several IK seeds
        # and keep the (q_safe, q_p3, q_p4) triple whose pallet-safe config is closest
        # to home in joint space.  LULA finds different local minima from different seeds;
        # the default seed (home) often yields large J4/J5 swings for tilted-face pyramids.
        _pick_branch = self._pick_ik_branch(
            ik, pick_l6, pick_pre, pallet_safe, q_pick,
            robot_world_pos, robot_R, ee_frame,
        )
        if _pick_branch is None:
            return None
        q_safe, q_p3, q_p4 = _pick_branch

        # Normalize J6 for the approach sequence now, so all five pre-contact waypoints
        # (safe, pre-pick, pick-contact, dwell, retract) share a consistent J6 branch.
        # Dwell and retract reuse q_p4/q_p3 arrays; without early normalization the arm
        # would spin 180° between pick-contact and dwell — right as the gripper fires.
        _pre = _normalize_j6([
            Waypoint(q_safe, 0.0, 1),
            Waypoint(q_p3,   0.0, 1),
            Waypoint(q_p4,   0.0, 1),
        ])
        q_safe_n, q_p3_n, q_p4_n = _pre[0].position, _pre[1].position, _pre[2].position

        _ik_pl = ik_place if ik_place is not None else ik
        _ik_pp = ik_pre_place if ik_pre_place is not None else _ik_pl
        q_transfer, ok = self._solve_ik(_ik_pl, transfer, q_pick, robot_world_pos, robot_R, ee_frame)
        if not ok: return None
        # Snap transfer J6 to the ±π equivalent closest to normalized pick J6 before
        # warm-starting place solves — the wrong J6 would propagate into J1-J5.
        q_transfer = q_transfer.copy()
        _alt = q_transfer[5] + (np.pi if q_transfer[5] < 0 else -np.pi)
        if abs(_alt - q_p4_n[5]) < abs(q_transfer[5] - q_p4_n[5]):
            q_transfer[5] = _alt
        # Compute geometric seeds for the place phase.  The belt zone (y≈-1.28)
        # needs J1 pointing toward the platform and J2 well below 90°.
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
            # Same solver: try q_transfer-based seed first (J6 continuity), then geo,
            # then ik_place's own default_q (J1=-130°) which may find a different
            # solution branch that avoids J2=90°.
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
        # Try to solve place_l6 from q_p6, with geo seed fallbacks if that fails.
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
        # The IK for the belt zone (different arm config) can land on the π-opposite
        # J6 solution; without this snap the gripper rotates ~180° around the
        # approach axis between pick and place, which appears as a 180° rotation of
        # the placed pyramid.
        q_p6 = q_p6.copy()
        q_p7 = q_p7.copy()
        for _q in (q_p6, q_p7):
            _alt = _q[5] + (np.pi if _q[5] < 0 else -np.pi)
            if abs(_alt - q_p4_n[5]) < abs(_q[5] - q_p4_n[5]):
                _q[5] = _alt

        wps = [
            Waypoint(q_safe_n,   0.0, PYRAMID_SAFE_STEPS),      # pallet safe – clearance above pick
            Waypoint(q_p3_n,     0.0, PYRAMID_APPROACH_STEPS),  # pre-pick (tilted)
            Waypoint(q_p4_n,     0.0, PYRAMID_APPROACH_STEPS),  # pick contact (open)
            Waypoint(q_p4_n,     1.0, PYRAMID_GRIP_DWELL),      # dwell (closed)
            Waypoint(q_p3_n,     1.0, PYRAMID_APPROACH_STEPS),  # retract with object
            Waypoint(q_transfer, 1.0, PYRAMID_TRANSFER_STEPS),  # transfer (no blend: arm settles before pre-place)
            Waypoint(q_p6,       1.0, PYRAMID_TRANSFER_STEPS),  # pre-place
            Waypoint(q_p7,       1.0, 90),              # place contact (closed)
            Waypoint(q_p7,       0.0, 45),              # place release (open)
            Waypoint(q_p6,       0.0, 60),              # retract from place
            Waypoint(_JP_HOME,   0.0, _HOME_STEPS),     # return home
        ]
        wps = _normalize_j6(wps[:-1]) + [wps[-1]]
        return wps if _trajectory_safe(wps) else None


# ── Sorting task ───────────────────────────────────────────────────────────────

# Platform centre positions [x, y, z] in world frame.  Slabs are 0.01 m thick;
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
    """Pick-and-sort variant: each object type has a fixed designated platform.

    The pick side is identical to PickAndPlaceTask (random position on the pallet).
    The place side always targets the platform assigned to the sampled object type:
        cube     → [0.312, -1.28]
        cylinder → [0.725, -1.28]
        pyramid  → [1.17,  -1.28]
    All placements target z = 0.860 m (top surface of 0.01 m thick platform slabs).
    """

    SORT_INSTRUCTIONS = [
        "place the object on its designated platform on the belt",
        "sort the object to its correct platform on the conveyor",
        "move the object to its matching target platform",
        "pick up the object and put it on the right slot on the belt",
        "place the item on its correct platform",
        "find the right platform and place the object on it",
        "sort the item to its designated spot on the conveyor",
        "deliver the object to its assigned platform on the belt",
        "pick the object and sort it to the correct platform",
        "move the item to its designated platform on the conveyor",
    ]

    def __init__(self, **kwargs):
        kwargs.setdefault('place_z', _SORT_SURFACE_Z)
        super().__init__(**kwargs)
        self.instructions = self.SORT_INSTRUCTIONS

    def sample(self, rng: np.random.Generator, ik,
               robot_world_pos: np.ndarray, robot_R: np.ndarray,
               ee_frame: str = 'link_6',
               force_obj_type: str = None,
               ik_place=None,
               pick_yaw: float = 0.0):
        obj_type    = force_obj_type if force_obj_type is not None \
                      else str(rng.choice(self.OBJ_TYPES))
        instruction = str(rng.choice(self.instructions))
        pick_xy     = self._sample_xy(rng, self.pick_x, self.pick_y)

        # Fixed place xy for this object type.
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

                # Cheap XY reach check: the pyramid platform is at a fixed x that
                # can be far from the robot.  The face normal shifts link_6 by
                # eef_z_offset * n_outward in XY.  Reject face/side combos whose
                # predicted place_l6 XY distance exceeds the robot's reach before
                # spending IK calls — collect.py will resample with a new face/side.
                _l6_xy  = place_xy + self.eef_z_offset * n_outward[:2]
                _l6_rxy = _l6_xy   - robot_world_pos[:2]
                if np.linalg.norm(_l6_rxy) > self.max_reach_xy:
                    return None, instruction, pick_pos, place_pos, 'pyramid', \
                           {'pool_idx': pool_idx, 'L': L, 'W': W, 'H': H,
                            'face_axis': face_axis, 'side': side,
                            'n_outward': n_outward}

                # Use ik_place for both transfer and pre_place/place_l6.
                # The previous failure of ik_place for pyramid came from a bad warm
                # start (J1=-29° from q_transfer).  Geometric seeds (J1≈-47° biased
                # toward the platform, J2=56-77°) avoid the near-singular J2≈90°
                # configuration that ik converges to in the belt zone.
                waypoints = self._build_pyramid_waypoints(
                    pick_pos, place_pos, L, W, H, face_axis, side, n_outward,
                    ik, robot_world_pos, robot_R, ee_frame,
                    ik_place=ik_place, ik_pre_place=ik_place,
                )
                return waypoints, instruction, pick_pos, place_pos, 'pyramid', \
                       {'pool_idx': pool_idx, 'L': L, 'W': W, 'H': H,
                        'face_axis': face_axis, 'side': side,
                        'n_outward': n_outward}

        # cube / cylinder (or pyramid fallback)
        params    = _OBJ_PARAMS[obj_type]
        hh        = params['height'] / 2.0
        pick_pos  = np.array([*pick_xy,  self.surface_z + hh])
        place_pos = np.array([*place_xy, self.place_z   + hh])
        waypoints = self._build_waypoints(
            pick_pos, place_pos, params, ik, robot_world_pos, robot_R, ee_frame,
            ik_place=ik_place, pick_yaw=pick_yaw,
        )
        return waypoints, instruction, pick_pos, place_pos, obj_type, {}

