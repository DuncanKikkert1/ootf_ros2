# =============================================================================
# pipeline.py — Training pipeline: .npz episodes → TFDS → Octo finetune.
#
# Usage:
#   python src/training/pipeline.py --raw-dir data/exp_01/raw --output-dir data/exp_01
#   python src/training/pipeline.py --raw-dir ... --output-dir ... --convert-only
#   python src/training/pipeline.py --raw-dir ... --output-dir ... --finetune-only
# =============================================================================

import argparse
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path


class DatasetConverter:
    """Converts a directory of .npz episodes to a TFDS dataset."""

    def __init__(self, raw_dir: Path, tfds_dir: Path):
        self.raw_dir  = Path(raw_dir)
        self.tfds_dir = Path(tfds_dir)

    def run(self) -> None:
        """Build the TFDS dataset from the raw .npz files."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from training.tfds_builder import OotfSyntheticBuilder

        npz_files = sorted(self.raw_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz episodes found in {self.raw_dir}")

        print(f"[CONVERT] {len(npz_files)} episodes → TFDS at {self.tfds_dir}")
        OotfSyntheticBuilder(
            raw_data_dir=self.raw_dir,
            data_dir=str(self.tfds_dir),
        ).download_and_prepare()
        print("[CONVERT] Done.")


class OctoFinetuner:
    """Launches Octo's finetune.py script against a TFDS dataset."""

    DEFAULT_OCTO_DIR = Path.home() / "Documents" / "octo"

    def __init__(
        self,
        tfds_dir:        Path,
        checkpoint_dir:  Path,
        pretrained_path: str  = "hf://rail-berkeley/octo-base-1.5",
        finetune_mode:   str  = "head_mlp_only",
        n_steps:         int  = 50_000,
        batch_size:      int  = 32,
        octo_dir:        Path = None,
    ):
        self.tfds_dir        = Path(tfds_dir)
        self.checkpoint_dir  = Path(checkpoint_dir)
        self.pretrained_path = pretrained_path
        self.finetune_mode   = finetune_mode
        self.n_steps         = n_steps
        self.batch_size      = batch_size
        self.octo_dir        = Path(octo_dir) if octo_dir else self.DEFAULT_OCTO_DIR

    def run(self) -> None:
        """Launch the Octo finetune script as a subprocess."""
        script = self.octo_dir / "scripts" / "finetune.py"
        if not script.exists():
            raise FileNotFoundError(
                f"Octo finetune script not found: {script}\n"
                f"Clone with:  git clone https://github.com/octo-models/octo.git {self.octo_dir}"
            )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        src_dir = Path(__file__).parent.parent   # src/
        env = {
            **os.environ,
            "TFDS_MODULES_IMPORT": "training.tfds_builder",
            "PYTHONPATH": f"{src_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
            "WANDB_MODE": "disabled",
        }
        config = Path(__file__).parent / "finetune_config.py"
        cmd = [
            sys.executable, str(script),
            f"--config={config}:{self.finetune_mode},text_conditioned",
            f"--config.pretrained_path={self.pretrained_path}",
            f"--config.dataset_kwargs.data_dir={self.tfds_dir}",
            f"--config.num_steps={self.n_steps}",
            f"--config.batch_size={self.batch_size}",
            f"--config.save_dir={self.checkpoint_dir}",
            "--config.eval_interval=999999999",
            "--config.viz_kwargs.trajs_for_metrics=0",
        ]
        print(f"[FINETUNE] mode={self.finetune_mode}  steps={self.n_steps}")
        subprocess.run(cmd, env=env, check=True)
        print(f"\n[FINETUNE] Done — checkpoint at {self.checkpoint_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> Namespace:
    ap = argparse.ArgumentParser(
        description="Convert .npz episodes to TFDS and finetune Octo."
    )
    ap.add_argument("--raw-dir",          required=True,
                    help="Directory containing .npz episode files")
    ap.add_argument("--output-dir",       required=True,
                    help="Root output dir (tfds/ and checkpoint/ created inside)")
    ap.add_argument("--pretrained-path",  default="hf://rail-berkeley/octo-base-1.5")
    ap.add_argument("--finetune-mode",    default="head_mlp_only",
                    choices=["full", "head_only", "head_mlp_only"])
    ap.add_argument("--n-finetune-steps", type=int, default=50_000)
    ap.add_argument("--batch-size",       type=int, default=32)
    phase = ap.add_mutually_exclusive_group()
    phase.add_argument("--convert-only",  action="store_true")
    phase.add_argument("--finetune-only", action="store_true")
    return ap.parse_args()


def main() -> None:
    args     = parse_args()
    out      = Path(args.output_dir)
    tfds_dir = out / "tfds"
    ckpt_dir = out / "checkpoint"

    if not args.finetune_only:
        DatasetConverter(raw_dir=args.raw_dir, tfds_dir=tfds_dir).run()

    if not args.convert_only:
        # Octo's 95/5 train/val split requires at least 20 episodes — the 5%
        # slice resolves to 0 records below that, crashing validation setup.
        n_episodes = len(list(Path(args.raw_dir).glob("*.npz")))
        if n_episodes < 20:
            print(
                f"[FINETUNE] Skipping: only {n_episodes} episode(s) in {args.raw_dir}.\n"
                f"           Octo's 95/5 train/val split requires at least 20 episodes.\n"
                f"           Re-run with --n-episodes 200 (or at least 20) to finetune."
            )
        else:
            OctoFinetuner(
                tfds_dir        = tfds_dir,
                checkpoint_dir  = ckpt_dir,
                pretrained_path = args.pretrained_path,
                finetune_mode   = args.finetune_mode,
                n_steps         = args.n_finetune_steps,
                batch_size      = args.batch_size,
            ).run()


if __name__ == "__main__":
    main()
