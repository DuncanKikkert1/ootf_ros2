"""SurfaceGripperController — wraps isaacsim.robot.surface_gripper for the SMC
two-cup suction gripper mounted at /World/h2017/link_6/SMC_gripper.

Initialisation must happen in two phases so that physics prims exist before
world.reset() compiles the articulation:

    # --- BEFORE world.reset() ---
    enable_extension("isaacsim.robot.surface_gripper")
    gripper = SurfaceGripperController()
    gripper.create_prims(stage)

    # --- AFTER world.reset() ---
    gripper.acquire_interface()

Architecture:

    /World/SMCGripperBody        — kinematic rigid body (non-articulation).
                                   Posed every physics step via sync_to_link6()
                                   so it tracks link_6 without constraint jitter.
    /World/SurfaceGripperJoints  — D6 attachment point joints.
                                   body0 = SMCGripperBody
                                   body1 = link_6  (extension swaps to cube at grip time)
    /World/SurfaceGripper        — IsaacSurfaceGripper prim.
    /World/SMCGripperBody/Marker_* — red sphere markers that track the gripper.

Tune _CUP_TIPS (in SMCGripperBody / link_6 local frame) until the red viewport
spheres sit at the tip of each suction cone.
"""

from pxr import Gf, Sdf, UsdGeom, UsdPhysics

import isaacsim.robot.surface_gripper._surface_gripper as _sg
import usd.schema.isaac.robot_schema as robot_schema

# ── prim paths ───────────────────────────────────────────────────────────────
_LINK6_PATH        = "/World/h2017/link_6"
_ASSEMBLY_PATH     = "/World/SMCGripperBody"        # kinematic rigid body (body0)
_GRIPPER_PRIM_PATH = "/World/SurfaceGripper"
_JOINTS_SCOPE_PATH = "/World/SurfaceGripperJoints"

# Scan origins in the SMCGripperBody (= link_6) local frame (metres).
# link_6 +Z points DOWNWARD in world, so positive Z = toward the cube.
# clearanceOffset=0 so the scan starts at the joint origin.  maxGripDistance=80mm
# gives plenty of margin.  Joint origin at Z=0.180 in link_6 frame → world Z
# = link6_Z-0.180 = ~0.565m.  Cube top at ~0.529m is 3.6cm inside the window.
# retryInterval=0 so every close_gripper() call fires a scan ray (no minimum wait).
_CUP_TIPS = [
    Gf.Vec3f(-0.022, -0.0189, 0.215),   # cup A
    Gf.Vec3f( 0.020,  0.0235, 0.215),   # cup B
]

_MARKER_TIPS = [
    Gf.Vec3f(-0.022, -0.0189, 0.215),
    Gf.Vec3f( 0.020,  0.0235, 0.215),
]

# Identity → scan in +Z of body0 = downward in world (toward the cube).
_JOINT_ROT = Gf.Quatf(1.0, 0.0, 0.0, 0.0)

# Surface-gripper physics parameters
_MAX_GRIP_DISTANCE   = 0.08   # m  — 8 cm scan range from joint origin
_COAXIAL_FORCE_LIMIT = 500.0   # N  — axial pull-off limit
_SHEAR_FORCE_LIMIT   = 2000.0  # N  — lateral force limit (high: arm moves fast)
_RETRY_INTERVAL      = 0.016  # s  — ≈ one physics frame at 60 Hz; scan retries every step

# Visual marker appearance
_MARKER_RADIUS = 0.005
_MARKER_COLOR  = Gf.Vec3f(1.0, 0.0, 0.0)


class SurfaceGripperController:
    """Two-phase open/close controller for the SMC two-cup surface gripper."""

    def __init__(self, verbose: bool = True):
        self._interface  = None
        self._body_prim  = None   # SingleRigidPrim set in acquire_interface()
        self._verbose    = verbose

    # ── public API ───────────────────────────────────────────────────────────

    def create_prims(self, stage):
        """Phase 1 — write all USD prims.  Call BEFORE world.reset()."""
        self._create_assembly_body(stage)
        joint_paths = self._create_attachment_joints(stage)
        self._create_visual_markers(stage)
        self._create_surface_gripper_prim(stage, joint_paths)
        if self._verbose:
            print(f"[GRIPPER] Prims created")
            print(f"[GRIPPER]   assembly body : {_ASSEMBLY_PATH}")
            print(f"[GRIPPER]   gripper prim  : {_GRIPPER_PRIM_PATH}")
            print(f"[GRIPPER]   cup tips (local frame): {_CUP_TIPS}")

    def acquire_interface(self):
        """Phase 2 — bind to the PhysX runtime.  Call AFTER world.reset()."""
        from isaacsim.core.prims import SingleRigidPrim
        self._body_prim = SingleRigidPrim(prim_path=_ASSEMBLY_PATH, name="gripper_body")
        self._interface = _sg.acquire_surface_gripper_interface()
        self._interface.set_write_to_usd(True)
        if self._verbose:
            print("[GRIPPER] Interface acquired — ready")
            self._diagnose()

    def open(self):
        if self._interface:
            self._interface.open_gripper(_GRIPPER_PRIM_PATH)
            if self._verbose:
                print(f"[GRIPPER] open() → status={self._interface.get_gripper_status(_GRIPPER_PRIM_PATH)}")

    def close(self):
        if self._interface:
            self._interface.close_gripper(_GRIPPER_PRIM_PATH)

    def status(self) -> str:
        """Return 'open', 'closing', or 'closed'."""
        if not self._interface:
            return "open"
        s = self._interface.get_gripper_status(_GRIPPER_PRIM_PATH)
        if s == _sg.GripperStatus.Closed:
            return "closed"
        if s == _sg.GripperStatus.Closing:
            return "closing"
        return "open"

    def is_closed(self) -> bool:
        return self.status() == "closed"

    def sync_to_link6(self, world_pos, world_quat_wxyz):
        """Pose the kinematic assembly body to match link_6's world transform.

        Call every physics step BEFORE world.step() so the surface-gripper scan
        origin is current.  world_pos is a (3,) array; world_quat_wxyz is (4,)
        in w-x-y-z order (Isaac Sim convention).
        """
        if self._body_prim is not None:
            self._body_prim.set_world_pose(position=world_pos, orientation=world_quat_wxyz)

    # ── private helpers ──────────────────────────────────────────────────────

    def _diagnose(self):
        """Print USD prim state so we can tell why close_gripper() is silent."""
        import omni.usd
        stage = omni.usd.get_context().get_stage()

        for path in (_ASSEMBLY_PATH, _JOINTS_SCOPE_PATH, _GRIPPER_PRIM_PATH):
            p = stage.GetPrimAtPath(path)
            tname = p.GetTypeName() if p.IsValid() else "N/A"
            print(f"[GRIPPER-DIAG] {path}: valid={p.IsValid()}, type={tname!r}")

        gp = stage.GetPrimAtPath(_GRIPPER_PRIM_PATH)
        if gp.IsValid():
            rel = gp.GetRelationship("isaac:attachmentPoints")
            targets = rel.GetTargets() if rel.IsValid() else []
            print(f"[GRIPPER-DIAG] attachmentPoints ({len(targets)}): {targets}")
            for t in targets:
                jp = stage.GetPrimAtPath(t)
                if not jp.IsValid():
                    print(f"[GRIPPER-DIAG]   {t}: INVALID")
                    continue
                schemas = jp.GetAppliedSchemas()
                b0rel = jp.GetRelationship("physics:body0")
                b0 = b0rel.GetTargets() if b0rel.IsValid() else "NO body0 rel"
                b1rel = jp.GetRelationship("physics:body1")
                b1 = b1rel.GetTargets() if b1rel.IsValid() else "NO body1 rel"
                print(f"[GRIPPER-DIAG]   {t}: schemas={schemas}, body0={b0}, body1={b1}")

        p = stage.GetPrimAtPath(_ASSEMBLY_PATH)
        if p.IsValid():
            has_rb      = p.HasAPI("PhysicsRigidBodyAPI")
            has_kin     = p.GetAttribute("physics:kinematicEnabled").Get()
            has_art     = p.HasAPI("PhysicsArticulationRootAPI")
            print(f"[GRIPPER-DIAG] SMCGripperBody: RigidBodyAPI={has_rb}, kinematic={has_kin}, ArticulationRootAPI={has_art}")

        raw_status = self._interface.get_gripper_status(_GRIPPER_PRIM_PATH)
        print(f"[GRIPPER-DIAG] Initial gripper status (raw): {raw_status}")

    def _create_assembly_body(self, stage):
        """Kinematic rigid body — posed each step via sync_to_link6(), no constraint jitter."""
        UsdGeom.Xform.Define(stage, _ASSEMBLY_PATH)
        prim = stage.GetPrimAtPath(_ASSEMBLY_PATH)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        prim.GetAttribute("physics:kinematicEnabled").Set(True)
        UsdPhysics.MassAPI.Apply(prim)
        prim.GetAttribute("physics:mass").Set(0.001)

    def _create_attachment_joints(self, stage) -> list:
        """One D6 attachment point joint per suction cup tip.

        Mirrors SurfaceGripper_gantry.usda exactly:
          body0 = SMCGripperBody (gripper body)
          body1 = link_6        (arm body — extension swaps this to the gripped object)
          transX/Y: free (lateral alignment)
          transZ:   [0, 0.01 m] with spring drive (suction compression)
          rotX/Y/Z: ±3° with spring drives (angular compliance)
        The scan goes in +Z of body0 (downward for H2017).
        """
        stage.DefinePrim(_JOINTS_SCOPE_PATH, "Scope")
        joint_paths = []

        for i, tip_pos in enumerate(_CUP_TIPS):
            path  = f"{_JOINTS_SCOPE_PATH}/D6Joint_{i:02d}"
            joint = UsdPhysics.Joint.Define(stage, path)
            prim  = stage.GetPrimAtPath(path)

            prim.AddAppliedSchema("IsaacAttachmentPointAPI")

            # All translational axes: high-stiffness drives so the cube tracks
            # the gripper rigidly.  transX/Y were previously stiffness=0 (damping
            # only) which caused the cube to drift in XY during retract.
            # transZ upper limit is 0.06 m to accommodate the 3 cm pick clearance
            # (cup tip is ~3 cm above cube top at attachment time).
            for axis in ("transX", "transY"):
                UsdPhysics.LimitAPI.Apply(prim, axis)
                prim.GetAttribute(f"limit:{axis}:physics:high").Set(-1.0)
                prim.GetAttribute(f"limit:{axis}:physics:low").Set(1.0)
                _dl = UsdPhysics.DriveAPI.Apply(prim, axis)
                _dl.GetStiffnessAttr().Set(1e6)
                _dl.GetDampingAttr().Set(1e4)

            UsdPhysics.LimitAPI.Apply(prim, "transZ")
            prim.GetAttribute("limit:transZ:physics:high").Set(0.06)
            prim.GetAttribute("limit:transZ:physics:low").Set(0.0)
            _dz = UsdPhysics.DriveAPI.Apply(prim, "transZ")
            _dz.GetStiffnessAttr().Set(1e6)
            _dz.GetDampingAttr().Set(1e4)

            # rotX/Y/Z: stiff springs, ±10° range
            for axis in ("rotX", "rotY", "rotZ"):
                UsdPhysics.LimitAPI.Apply(prim, axis)
                prim.GetAttribute(f"limit:{axis}:physics:high").Set(10.0)
                prim.GetAttribute(f"limit:{axis}:physics:low").Set(-10.0)
                _dr = UsdPhysics.DriveAPI.Apply(prim, axis)
                _dr.GetStiffnessAttr().Set(1e6)
                _dr.GetDampingAttr().Set(1e4)

            try:
                from pxr import PhysxSchema
                PhysxSchema.PhysxLimitAPI.Apply(prim, "transX")
                PhysxSchema.PhysxLimitAPI.Apply(prim, "transY")
            except Exception:
                pass

            prim.GetAttribute("isaac:forwardAxis").Set("Z")
            prim.GetAttribute("isaac:clearanceOffset").Set(0.0)

            joint.GetBody0Rel().SetTargets([Sdf.Path(_ASSEMBLY_PATH)])
            # body1 = link_6.  The extension swaps this to the gripped cube at runtime.
            joint.GetBody1Rel().SetTargets([Sdf.Path(_LINK6_PATH)])
            joint.GetBreakForceAttr().Set(3.4028235e38)
            joint.GetBreakTorqueAttr().Set(3.4028235e38)
            joint.GetExcludeFromArticulationAttr().Set(True)
            joint.GetJointEnabledAttr().Set(True)

            joint.GetLocalPos0Attr().Set(tip_pos)
            joint.GetLocalRot0Attr().Set(_JOINT_ROT)
            joint.GetLocalPos1Attr().Set(tip_pos)   # same frame as body0 (both pinned at link_6 origin)
            joint.GetLocalRot1Attr().Set(_JOINT_ROT)

            joint_paths.append(Sdf.Path(path))

        return joint_paths

    def _create_visual_markers(self, stage):
        """Red spheres parented under SMCGripperBody — they track the robot."""
        for i, tip_pos in enumerate(_MARKER_TIPS):
            path   = f"{_LINK6_PATH}/Marker_{i:02d}"
            sphere = UsdGeom.Sphere.Define(stage, path)
            sphere.GetRadiusAttr().Set(_MARKER_RADIUS)
            sphere.GetDisplayColorAttr().Set([_MARKER_COLOR])
            sphere.AddTranslateOp().Set(Gf.Vec3d(tip_pos[0], tip_pos[1], tip_pos[2]))
            # Explicitly disable collision: Isaac Sim auto-assigns collision shapes to
            # geometry children of RigidBodyAPI prims.  A kinematic body sweeping
            # through the scene with active colliders pushes dynamic objects (like the
            # pick cube) with unlimited force.  Applying CollisionAPI and setting
            # collisionEnabled=False prevents this without removing the visual spheres.
            prim = stage.GetPrimAtPath(path)
            UsdPhysics.CollisionAPI.Apply(prim)
            prim.GetAttribute("physics:collisionEnabled").Set(False)

    def _create_surface_gripper_prim(self, stage, joint_paths: list):
        robot_schema.CreateSurfaceGripper(stage, _GRIPPER_PRIM_PATH)
        prim = stage.GetPrimAtPath(_GRIPPER_PRIM_PATH)

        prim.GetRelationship(
            robot_schema.Relations.ATTACHMENT_POINTS.name
        ).SetTargets(joint_paths)

        prim.GetAttribute(robot_schema.Attributes.MAX_GRIP_DISTANCE.name).Set(_MAX_GRIP_DISTANCE)
        prim.GetAttribute(robot_schema.Attributes.COAXIAL_FORCE_LIMIT.name).Set(_COAXIAL_FORCE_LIMIT)
        prim.GetAttribute(robot_schema.Attributes.SHEAR_FORCE_LIMIT.name).Set(_SHEAR_FORCE_LIMIT)
        prim.GetAttribute(robot_schema.Attributes.RETRY_INTERVAL.name).Set(_RETRY_INTERVAL)
