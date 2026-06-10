# tfds_builder.py — TFDS GeneratorBasedBuilder for the ootf_synthetic dataset.
#
# Registered via TFDS_MODULES_IMPORT env var (set automatically by OctoFinetuner).
# Schema uses Octo's bridge_dataset observation keys so the pretrained image
# encoder transfers without remapping.

import os
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class OotfSyntheticBuilder(tfds.core.GeneratorBasedBuilder):
    """Reads .npz episodes from DataCollector and serves them as 'ootf_synthetic'."""

    name    = "ootf_synthetic"
    VERSION = tfds.core.Version("1.0.0")

    def __init__(self, raw_data_dir: Path = None, **kwargs):
        """Accept raw_data_dir in addition to standard TFDS kwargs.

        raw_data_dir is only needed when building the dataset for the first time.
        TFDS/Octo instantiate the builder by name when reading an already-built
        dataset and will not pass raw_data_dir, so it must be optional.
        """
        self._raw_data_dir = Path(raw_data_dir) if raw_data_dir is not None else None
        super().__init__(**kwargs)

    def _info(self):
        """Declare the dataset schema."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image_wrist": tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                        ),
                        "proprio": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    }),
                    "action":               tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    "language_instruction": tfds.features.Text(),
                    "is_first":             tf.bool,
                    "is_last":              tf.bool,
                    "is_terminal":          tf.bool,
                    "reward":               tf.float32,
                    "discount":             tf.float32,
                }),
            }),
            homepage="",
            description="Synthetic pick-and-place episodes collected in Isaac Sim.",
        )

    def _split_generators(self, dl_manager) -> dict:
        """Return the single training split pointing at the raw episode directory."""
        if self._raw_data_dir is None:
            raise RuntimeError(
                "raw_data_dir is required when building the dataset from scratch. "
                "It is not needed when reading an already-built TFDS dataset."
            )
        return {"train": self._generate_examples(self._raw_data_dir)}

    # Keep 1-in-N dwell frames to prevent near-zero hold actions from
    # dominating training loss. Gripper transitions are always kept.
    # OOTF_OVERFIT=1 keeps every dwell frame so the model sees the full
    # trajectory; normal training uses 1-in-5 to avoid dwell dominance.
    _DWELL_SUBSAMPLE = 1 if os.environ.get("OOTF_OVERFIT", "0") == "1" else 5
    _DWELL_THRESHOLD = 1e-4   # action norm below this = dwell/hold frame

    @classmethod
    def _active_indices(cls, actions: np.ndarray) -> np.ndarray:
        """Return indices to keep: all motion frames + every Nth dwell frame."""
        motion_norm = np.linalg.norm(actions[:, :6], axis=1)
        gripper     = actions[:, 6]

        # Gripper transition: gripper value changed vs previous step
        grip_change        = np.zeros(len(actions), dtype=bool)
        grip_change[1:]    = np.abs(np.diff(gripper)) > 0.1
        grip_change[0]     = True   # always keep first frame

        is_dwell = motion_norm < cls._DWELL_THRESHOLD
        keep     = (~is_dwell) | grip_change | (np.arange(len(actions)) % cls._DWELL_SUBSAMPLE == 0)
        return np.where(keep)[0]

    def _generate_examples(self, raw_dir: Path) -> Generator[Tuple[str, dict], None, None]:
        """Yield (key, episode) pairs from .npz files in raw_dir."""
        for npz_path in sorted(raw_dir.glob("*.npz")):
            data        = np.load(npz_path, allow_pickle=True)
            images      = data["images"]           # (T, H, W, 3) uint8
            actions     = data["actions"]          # (T, 7) float32
            instruction = str(data["instruction"])
            n_steps     = len(actions)
            proprios    = (data["proprios"] if "proprios" in data
                           else np.zeros((n_steps, 3), dtype=np.float32))

            indices = self._active_indices(actions)

            yield npz_path.stem, {
                "steps": [
                    {
                        "observation": {
                            "image_wrist": images[i],
                            "proprio":     proprios[i].astype(np.float32),
                        },
                        "action":      actions[i].astype(np.float32),
                        "language_instruction": instruction,
                        "is_first":    idx == 0,
                        "is_last":     idx == len(indices) - 1,
                        "is_terminal": idx == len(indices) - 1,
                        "reward":      float(idx == len(indices) - 1),
                        "discount":    0.0 if idx == len(indices) - 1 else 1.0,
                    }
                    for idx, i in enumerate(indices)
                ],
            }
