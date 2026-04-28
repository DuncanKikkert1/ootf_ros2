# =============================================================================
# Name        : collect_and_train.py
# Author      : Duncan Kikkert
# Created     : 24/4/2026
# Description : Synthetic data collection + Octo finetuning pipeline.
#
# Usage:
#   bash launch/launch_collect.sh --output-dir data/exp_01 --n-episodes 500
#   bash launch/launch_collect.sh --output-dir data/exp_01 --collect-only
#   bash launch/launch_collect.sh --output-dir data/exp_01 --finetune-only
#
# Sweep example (call directly from another script):
#   import sys; sys.path.insert(0, "src")
#   from collect_and_train import SyntheticTrainingPipeline, PipelineConfig
#   for n in [200, 500, 1000]:
#       SyntheticTrainingPipeline(PipelineConfig(
#           output_dir=f"data/sweep_n{n}", n_episodes=n,
#       )).run()
# =============================================================================

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

# ─────────────────────────────────────────────────────────────────────────────
# Task definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Waypoint:
    position:    np.ndarray   # [x, y, z] world frame (metres)
    orientation: np.ndarray   # wxyz quaternion
    gripper:     float        # 0.0 = open, 1.0 = closed
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
        "pick up the object and place it at the target",
        "grasp the block and move it to the goal position",
        "pick up the item and set it at the marked location",
        "move the object to the target zone",
        "pick up the red cube and place it at the goal",
    ]

    def __init__(
        self,
        pick_x:         tuple = (0.35, 0.55),
        pick_y:         tuple = (-0.20,  0.20),
        place_x:        tuple = (0.35, 0.55),
        place_y:        tuple = (-0.20,  0.20),
        table_z:        float = 1.07,
        lift_h:         float = 0.15,
        approach:       float = 0.05,
        min_separation: float = 0.10,
        instructions:   Optional[List[str]] = None,
    ):
        self.pick_x   = pick_x
        self.pick_y   = pick_y
        self.place_x  = place_x
        self.place_y  = place_y
        self.table_z  = table_z
        self.lift_h   = lift_h
        self.approach = approach
        self.min_sep  = min_separation
        self.instructions = instructions or self.DEFAULT_INSTRUCTIONS

    def sample(self, rng: np.random.Generator):
        pick_pos = np.array([
            rng.uniform(*self.pick_x),
            rng.uniform(*self.pick_y),
            self.table_z,
        ])
        for _ in range(200):
            place_pos = np.array([
                rng.uniform(*self.place_x),
                rng.uniform(*self.place_y),
                self.table_z,
            ])
            if np.linalg.norm(place_pos[:2] - pick_pos[:2]) >= self.min_sep:
                break
        instruction = str(rng.choice(self.instructions))
        return self._build_waypoints(pick_pos, place_pos), instruction, pick_pos, place_pos

    def _build_waypoints(self, pick: np.ndarray, place: np.ndarray) -> List[Waypoint]:
        q = _quat_eef_down()
        z, h, a = self.table_z, self.lift_h, self.approach
        return [
            Waypoint(np.array([pick[0],  pick[1],  z + h]), q, 0.0, 20),
            Waypoint(np.array([pick[0],  pick[1],  z + a]), q, 0.0, 15),
            Waypoint(np.array([pick[0],  pick[1],  z    ]), q, 0.0, 10),
            Waypoint(np.array([pick[0],  pick[1],  z    ]), q, 1.0, 15),
            Waypoint(np.array([pick[0],  pick[1],  z + h]), q, 1.0, 20),
            Waypoint(np.array([place[0], place[1], z + h]), q, 1.0, 25),
            Waypoint(np.array([place[0], place[1], z + a]), q, 1.0, 15),
            Waypoint(np.array([place[0], place[1], z    ]), q, 1.0, 10),
            Waypoint(np.array([place[0], place[1], z    ]), q, 0.0, 15),
            Waypoint(np.array([place[0], place[1], z + h]), q, 0.0, 20),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Episode buffer
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeBuffer:
    def __init__(self):
        self._images:  list = []
        self._actions: list = []
        self._instr:   str  = ""

    def add_step(self, image: np.ndarray, action: np.ndarray, instruction: str):
        self._images.append(image)
        self._actions.append(action.astype(np.float32))
        self._instr = instruction

    def save(self, path: Path):
        np.savez_compressed(
            path,
            images      = np.stack(self._images,  axis=0),
            actions     = np.stack(self._actions, axis=0),
            instruction = self._instr,
        )

    def __len__(self):
        return len(self._images)


# ─────────────────────────────────────────────────────────────────────────────
# Isaac Sim data collector
# ─────────────────────────────────────────────────────────────────────────────

class DataCollector:
    """
    Runs scripted pick-and-place in Isaac Sim and writes one .npz per episode.
    Supports resuming: episodes already in raw_dir are skipped.

    Tuning notes:
      CAM_TRANSLATION / CAM_ORIENTATION — wrist camera offset from link_6.
        Run --collect-only with 3 episodes to verify the camera view first.
    """

    ROBOT_PRIM  = "/World/h2017"
    EE_FRAME    = "link_6"
    OBJ_PRIM    = "/World/pick_object"
    OBJ_SIZE    = 0.04
    HOME_POS    = [0.0, 0.0, 1.57, 0.0, 1.57, 0.0]
    HOME_SETTLE = 60
    CAM_TRANSLATION = np.array([0.05, 0.0,  0.03])
    CAM_ORIENTATION = np.array([0.0,  0.0,  0.0,  1.0])
    IK_POS_TOL  = 0.005
    IK_ORI_TOL  = 0.01

    def __init__(
        self,
        raw_dir:    Path,
        task:       PickAndPlaceTask,
        n_episodes: int,
        image_size: int = 128,
        seed:       int = 42,
        scene_usd:  str = None,
        urdf_path:  str = None,
        lula_desc:  str = None,
    ):
        root = Path(__file__).parent.parent
        self.raw_dir    = Path(raw_dir)
        self.task       = task
        self.n_episodes = n_episodes
        self.image_size = image_size
        self.rng        = np.random.default_rng(seed)
        self.scene_usd  = scene_usd or str(root / "scenes/usd/doosan_BIC.usd")
        self.urdf_path  = urdf_path or str(root / "scenes/h2017/urdf/h2017.urdf")
        self.lula_desc  = lula_desc or str(root / "scenes/h2017/urdf/h2017_lula.yaml")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.api.objects import DynamicCuboid
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
        from isaacsim.sensors.camera import Camera

        world = World(physics_dt=1/60, rendering_dt=1/60, stage_units_in_meters=1.0)
        omni.usd.get_context().get_stage().GetRootLayer().subLayerPaths.insert(
            0, self.scene_usd
        )
        robot  = world.scene.add(Robot(prim_path=self.ROBOT_PRIM, name="robot"))
        camera = world.scene.add(Camera(
            prim_path   = f"{self.ROBOT_PRIM}/{self.EE_FRAME}/wrist_camera",
            translation = self.CAM_TRANSLATION,
            orientation = self.CAM_ORIENTATION,
            resolution  = (self.image_size, self.image_size),
        ))
        obj = world.scene.add(DynamicCuboid(
            prim_path = self.OBJ_PRIM,
            name      = "pick_object",
            position  = np.array([0.45, 0.0, self.task.table_z]),
            scale     = np.array([self.OBJ_SIZE] * 3),
            color     = np.array([0.85, 0.2, 0.2]),
        ))

        world.reset()
        camera.initialize()

        n_dof = robot.num_dof
        robot.get_articulation_controller().set_gains(
            kps=np.full(n_dof, 1e6), kds=np.full(n_dof, 1e5),
        )

        if not Path(self.lula_desc).exists():
            self._generate_lula(self.urdf_path, self.lula_desc)

        ik = LulaKinematicsSolver(
            robot_description_path=self.lula_desc,
            urdf_path=self.urdf_path,
        )

        ep_idx = len(list(self.raw_dir.glob("*.npz")))
        print(f"[COLLECT] {self.n_episodes} episodes (resuming from {ep_idx})")

        while ep_idx < self.n_episodes:
            waypoints, instruction, pick_pos, _ = self.task.sample(self.rng)
            obj.set_world_pose(position=pick_pos)

            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=self.HOME_POS)
            )
            for _ in range(self.HOME_SETTLE):
                world.step(render=False)

            buf, prev_pos, prev_rot, ok = EpisodeBuffer(), None, None, True

            for wp in waypoints:
                q_target, ik_ok = ik.compute_inverse_kinematics(
                    self.EE_FRAME, wp.position, wp.orientation,
                    position_tolerance=self.IK_POS_TOL,
                    orientation_tolerance=self.IK_ORI_TOL,
                )
                if not ik_ok:
                    print(f"[COLLECT] ep {ep_idx}: IK failed — skipping")
                    ok = False
                    break

                q_start = robot.get_joint_positions().copy()
                for s in range(wp.n_steps):
                    alpha = (s + 1) / wp.n_steps
                    robot.get_articulation_controller().apply_action(
                        ArticulationAction(
                            joint_positions=(1 - alpha) * q_start + alpha * q_target
                        )
                    )
                    world.step(render=True)

                    rgba = camera.get_rgba()
                    if rgba is None:
                        continue
                    rgb = (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)

                    cur_pos, cur_rot = ik.compute_forward_kinematics(
                        self.EE_FRAME, robot.get_joint_positions()
                    )
                    if prev_pos is None:
                        action = np.zeros(7, dtype=np.float32)
                    else:
                        d_pos   = cur_pos - prev_pos
                        d_euler = Rotation.from_matrix(cur_rot @ prev_rot.T).as_euler('xyz')
                        action  = np.concatenate([d_pos, d_euler, [wp.gripper]])

                    buf.add_step(rgb, action.astype(np.float32), instruction)
                    prev_pos, prev_rot = cur_pos, cur_rot.copy()

            if ok and len(buf) > 0:
                buf.save(self.raw_dir / f"episode_{ep_idx:06d}.npz")
                ep_idx += 1
                print(f"[COLLECT] {ep_idx}/{self.n_episodes}  "
                      f"steps={len(buf)}  '{instruction}'")

    @staticmethod
    def _generate_lula(urdf_path: str, output_path: str):
        import xml.etree.ElementTree as ET
        root      = ET.parse(urdf_path).getroot()
        joints    = [j.get('name') for j in root.findall('joint')
                     if j.get('type') == 'revolute']
        parent    = root.find("joint/parent")
        root_link = parent.get('link') if parent is not None else 'base_link'
        n = len(joints)
        Path(output_path).write_text(
            "api_version: 1.0\n\ncspace:\n"
            + "".join(f"    - {j}\n" for j in joints)
            + f"\nroot_link: {root_link}\n\n"
            f"default_q: [{', '.join(['0.0']*n)}]\n\n"
            f"acceleration_limits: [{', '.join(['40.0']*n)}]\n"
            f"jerk_limits: [{', '.join(['500.0']*n)}]\n"
        )
        print(f"[COLLECT] Generated LULA description → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset converter
# ─────────────────────────────────────────────────────────────────────────────

class DatasetConverter:
    def __init__(self, raw_dir: Path, tfds_dir: Path, image_size: int = 128):
        self.raw_dir    = Path(raw_dir)
        self.tfds_dir   = Path(tfds_dir)
        self.image_size = image_size

    def run(self):
        sys.path.insert(0, str(Path(__file__).parent))
        from synthetic.tfds_builder import OotfSyntheticBuilder

        npz_files = sorted(self.raw_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz episodes found in {self.raw_dir}")

        print(f"[CONVERT] {len(npz_files)} episodes → TFDS at {self.tfds_dir}")
        OotfSyntheticBuilder(
            raw_data_dir=self.raw_dir,
            data_dir=str(self.tfds_dir),
        ).download_and_prepare()
        print("[CONVERT] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Octo finetuner
# ─────────────────────────────────────────────────────────────────────────────

class OctoFinetuner:
    DEFAULT_OCTO_DIR = Path.home() / "Documents" / "octo"

    def __init__(
        self,
        tfds_dir:        Path,
        checkpoint_dir:  Path,
        pretrained_path: str  = "hf://rail-berkeley/octo-small-1.5",
        finetune_mode:   str  = "head_mlp_only",
        n_steps:         int  = 20_000,
        batch_size:      int  = 128,
        octo_dir:        Path = None,
    ):
        self.tfds_dir        = Path(tfds_dir)
        self.checkpoint_dir  = Path(checkpoint_dir)
        self.pretrained_path = pretrained_path
        self.finetune_mode   = finetune_mode
        self.n_steps         = n_steps
        self.batch_size      = batch_size
        self.octo_dir        = Path(octo_dir) if octo_dir else self.DEFAULT_OCTO_DIR

    def run(self):
        script = self.octo_dir / "scripts" / "finetune.py"
        if not script.exists():
            raise FileNotFoundError(
                f"Octo finetune script not found: {script}\n"
                f"Clone with:  git clone https://github.com/octo-models/octo.git {self.octo_dir}"
            )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        src_dir = Path(__file__).parent
        env = {
            **os.environ,
            "TFDS_MODULES_IMPORT": "synthetic.tfds_builder",
            "PYTHONPATH": f"{src_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        }
        cmd = [
            sys.executable, str(script),
            f"--config={self.octo_dir}/scripts/configs/finetune_config.py",
            f"--config.pretrained_path={self.pretrained_path}",
            f"--config.dataset_kwargs.name=ootf_synthetic",
            f"--config.dataset_kwargs.data_dir={self.tfds_dir}",
            f"--config.dataset_kwargs.image_obs_keys.wrist=image_wrist",
            f"--config.dataset_kwargs.language_key=language_instruction",
            f"--config.finetuning_mode={self.finetune_mode}",
            f"--config.num_steps={self.n_steps}",
            f"--config.batch_size={self.batch_size}",
            f"--config.save_dir={self.checkpoint_dir}",
        ]
        print(f"[FINETUNE] mode={self.finetune_mode}  steps={self.n_steps}")
        subprocess.run(cmd, env=env, check=True)
        print(f"\n[FINETUNE] Done — checkpoint at {self.checkpoint_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    output_dir:       str
    n_episodes:       int              = 500
    task:             PickAndPlaceTask = field(default_factory=PickAndPlaceTask)
    image_size:       int              = 128
    seed:             int              = 42
    scene_usd:        Optional[str]    = None
    urdf_path:        Optional[str]    = None
    lula_desc:        Optional[str]    = None
    pretrained_path:  str              = "hf://rail-berkeley/octo-small-1.5"
    finetune_mode:    str              = "head_mlp_only"
    n_finetune_steps: int              = 20_000
    batch_size:       int              = 128
    octo_dir:         Optional[str]    = None


class SyntheticTrainingPipeline:
    """
    Orchestrates: collect → convert → finetune.

      raw/        .npz episode files
      tfds/       TFDS tfrecords for Octo
      checkpoint/ finetuned checkpoint
    """

    def __init__(self, config: PipelineConfig):
        self.config    = config
        self._out      = Path(config.output_dir)
        self._raw_dir  = self._out / "raw"
        self._tfds_dir = self._out / "tfds"
        self._ckpt_dir = self._out / "checkpoint"

    def collect(self):
        c = self.config
        DataCollector(
            raw_dir=self._raw_dir, task=c.task, n_episodes=c.n_episodes,
            image_size=c.image_size, seed=c.seed, scene_usd=c.scene_usd,
            urdf_path=c.urdf_path, lula_desc=c.lula_desc,
        ).run()

    def convert(self):
        DatasetConverter(self._raw_dir, self._tfds_dir, self.config.image_size).run()

    def finetune(self):
        c = self.config
        OctoFinetuner(
            tfds_dir=self._tfds_dir, checkpoint_dir=self._ckpt_dir,
            pretrained_path=c.pretrained_path, finetune_mode=c.finetune_mode,
            n_steps=c.n_finetune_steps, batch_size=c.batch_size, octo_dir=c.octo_dir,
        ).run()

    def run(self):
        self.collect()
        self.convert()
        self.finetune()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Collect synthetic data in Isaac Sim and finetune Octo."
    )
    ap.add_argument("--output-dir",       required=True)
    ap.add_argument("--n-episodes",       type=int,   default=500)
    ap.add_argument("--image-size",       type=int,   default=128)
    ap.add_argument("--seed",             type=int,   default=42)
    ap.add_argument("--table-z",          type=float, default=1.07,
                    help="Table surface height in world Z (metres)")
    ap.add_argument("--pick-x",           type=float, nargs=2, default=[0.35, 1.5],
                    metavar=("MIN", "MAX"))
    ap.add_argument("--pick-y",           type=float, nargs=2, default=[-1.5, 1.5],
                    metavar=("MIN", "MAX"))
    ap.add_argument("--place-x",          type=float, nargs=2, default=[-1.5, 1.5],
                    metavar=("MIN", "MAX"))
    ap.add_argument("--place-y",          type=float, nargs=2, default=[-1.5, 1.5],
                    metavar=("MIN", "MAX"))
    ap.add_argument("--pretrained-path",  default="hf://rail-berkeley/octo-small-1.5")
    ap.add_argument("--finetune-mode",    default="head_mlp_only",
                    choices=["full", "head_only", "head_mlp_only"])
    ap.add_argument("--n-finetune-steps", type=int, default=20_000)
    ap.add_argument("--batch-size",       type=int, default=128)
    phase = ap.add_mutually_exclusive_group()
    phase.add_argument("--collect-only",  action="store_true")
    phase.add_argument("--convert-only",  action="store_true")
    phase.add_argument("--finetune-only", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    config = PipelineConfig(
        output_dir       = args.output_dir,
        n_episodes       = args.n_episodes,
        image_size       = args.image_size,
        seed             = args.seed,
        pretrained_path  = args.pretrained_path,
        finetune_mode    = args.finetune_mode,
        n_finetune_steps = args.n_finetune_steps,
        batch_size       = args.batch_size,
        task             = PickAndPlaceTask(
            pick_x  = tuple(args.pick_x),
            pick_y  = tuple(args.pick_y),
            place_x = tuple(args.place_x),
            place_y = tuple(args.place_y),
            table_z = args.table_z,
        ),
    )
    pipeline = SyntheticTrainingPipeline(config)

    if args.convert_only:
        pipeline.convert()
    elif args.finetune_only:
        pipeline.finetune()
    else:
        # Collect phase requires Isaac Sim — start it now.
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": True})
        try:
            pipeline.collect()
        finally:
            simulation_app.close()
        if not args.collect_only:
            pipeline.convert()
            pipeline.finetune()


if __name__ == "__main__":
    main()
