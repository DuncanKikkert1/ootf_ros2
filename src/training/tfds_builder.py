# =============================================================================
# Name        : tfds_builder.py
# Description : TFDS GeneratorBasedBuilder for the ootf_synthetic dataset.
#
#               Registered with TFDS via the TFDS_MODULES_IMPORT env var,
#               which OctoFinetuner sets automatically before launching
#               finetune.py.  You do not need to install this as a package.
#
#               Schema matches Octo's bridge_dataset observation keys so
#               the pretrained image encoder can be reused without changes.
# =============================================================================

from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class OotfSyntheticBuilder(tfds.core.GeneratorBasedBuilder):
    """
    Reads .npz episode files written by DataCollector and serves them as
    a TFDS dataset named 'ootf_synthetic'.

    Observation key 'image_wrist' matches Octo's bridge_dataset wrist key
    so the existing image encoder weights transfer without remapping.
    """

    name    = "ootf_synthetic"
    VERSION = tfds.core.Version("1.0.0")

    def __init__(self, raw_data_dir: Path, **kwargs):
        self._raw_data_dir = Path(raw_data_dir)
        super().__init__(**kwargs)

    def _info(self):
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image_wrist": tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                        ),
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

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples(self._raw_data_dir)}

    def _generate_examples(self, raw_dir: Path):
        for npz_path in sorted(raw_dir.glob("*.npz")):
            data        = np.load(npz_path, allow_pickle=True)
            images      = data["images"]           # (T, H, W, 3) uint8
            actions     = data["actions"]          # (T, 7) float32
            instruction = str(data["instruction"])
            n_steps     = len(actions)

            yield npz_path.stem, {
                "steps": [
                    {
                        "observation": {"image_wrist": images[i]},
                        "action":      actions[i].astype(np.float32),
                        "language_instruction": instruction,
                        "is_first":    i == 0,
                        "is_last":     i == n_steps - 1,
                        "is_terminal": i == n_steps - 1,
                        "reward":      float(i == n_steps - 1),
                        "discount":    0.0 if i == n_steps - 1 else 1.0,
                    }
                    for i in range(n_steps)
                ],
            }
