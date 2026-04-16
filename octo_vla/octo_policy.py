# =============================================================================
# Name        : octo_policy.py
# Author      : Duncan Kikkert
# Created     : 16/4/2026
# Description : Wrapper around the Octo VLA model.
#               Manages image history, task creation (language or goal image),
#               and inference.  Returns a 7-value EEF delta action:
#                   [dx, dy, dz, drx, dry, drz, gripper]
#
# Requires    : JAX (+ cuda if GPU), flax, octo (~/Documents/octo)
#               pip install -e ~/Documents/octo
# =============================================================================

import cv2
import numpy as np
import jax
import jax.numpy as jnp
from octo.model.octo_model import OctoModel

# Octo was pretrained on 256×256 RGB images
IMAGE_SIZE = 256

# The pretrained checkpoint's action statistics come from bridge_dataset.
# This is used to unnormalize Octo's normalised action output back to
# real-world EEF delta units (metres / radians).
DEFAULT_DATASET = "bridge_dataset"


class OctoPolicy:
    """
    Wraps OctoModel for live single-camera inference.

    Typical usage:
        policy = OctoPolicy()
        policy.set_task_text("pick up the circle and place it in the mold")

        # inside control loop:
        action = policy.step(rgb_image)   # → np.ndarray shape (7,)
        # [dx, dy, dz, drx, dry, drz, gripper]
    """

    def __init__(
        self,
        model_path: str = "hf://rail-berkeley/octo-small-1.5",
        dataset_name: str = DEFAULT_DATASET,
        window_size: int = 2,
    ):
        """
        Args:
            model_path:   HuggingFace path or local directory of an Octo checkpoint.
            dataset_name: Dataset key used to look up unnormalisation statistics.
                          Must match what the checkpoint was trained / finetuned on.
            window_size:  Number of consecutive frames passed to the model.
                          2 gives the model minimal temporal context.
        """
        print(f"[OCTO] Loading model from: {model_path}")
        self.model       = OctoModel.load_pretrained(model_path)
        self.dataset     = dataset_name
        self.window_size = window_size

        self._rng           = jax.random.PRNGKey(0)
        self._image_history = []   # rolling list of preprocessed (256,256,3) frames
        self.task           = None

        print("[OCTO] Model loaded.")
        print(self.model.get_pretty_spec())

    # ------------------------------------------------------------------
    # Task creation  (call once before the control loop, or per episode)
    # ------------------------------------------------------------------

    def set_task_text(self, instruction: str):
        """Condition Octo on a natural-language instruction.

        Example: "pick up the circle and place it in the mold"
        """
        self.task = self.model.create_tasks(texts=[instruction])
        print(f"[OCTO] Task (language): '{instruction}'")

    def set_task_goal_image(self, goal_rgb: np.ndarray):
        """Condition Octo on a goal image (HxWx3 RGB uint8).

        The model will try to reproduce the scene shown in goal_rgb.
        """
        goal = cv2.resize(goal_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.uint8)
        # create_tasks expects (batch, H, W, C) for goal images
        self.task = self.model.create_tasks(
            goals={"image_primary": goal[None]}
        )
        print("[OCTO] Task (goal image) set.")

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------

    def reset(self):
        """Clear the image history at the start of a new episode."""
        self._image_history = []

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def step(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run one inference step.

        Args:
            image_rgb: HxWx3 uint8 RGB image from the camera.

        Returns:
            action: np.ndarray shape (7,) = [dx, dy, dz, drx, dry, drz, gripper]
                    Units follow the bridge_dataset convention:
                      dx/dy/dz   → metres
                      drx/dry/drz → radians
                      gripper    → 0.0 (open) … 1.0 (closed)
        """
        if self.task is None:
            raise RuntimeError(
                "No task set. Call set_task_text() or set_task_goal_image() first."
            )

        # ---- preprocess frame ----
        frame = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.uint8)

        # ---- update rolling history ----
        self._image_history.append(frame)
        if len(self._image_history) > self.window_size:
            self._image_history.pop(0)

        # Pad with the oldest frame if we don't have a full window yet
        n_real   = len(self._image_history)
        n_pad    = self.window_size - n_real
        padded   = [self._image_history[0]] * n_pad + self._image_history

        # ---- build observation dict ----
        # image_primary:        (batch=1, window_size, H, W, C)
        # timestep_pad_mask:    (batch=1, window_size) — False for padded frames
        obs_images = np.stack(padded, axis=0)[None]          # (1, ws, 256, 256, 3)
        pad_mask   = np.array([[False] * n_pad + [True] * n_real])  # (1, ws)

        observation = {
            "image_primary":    obs_images,
            "timestep_pad_mask": pad_mask,
        }

        # ---- sample action ----
        self._rng, key = jax.random.split(self._rng)

        actions = self.model.sample_actions(
            observation,
            self.task,
            unnormalization_statistics=(
                self.model.dataset_statistics[self.dataset]["action"]
            ),
            rng=key,
        )
        # actions: (batch=1, action_horizon, action_dim=7)
        # Take the first step of the predicted action chunk.
        action = np.array(actions[0, 0], dtype=np.float64)
        return action
