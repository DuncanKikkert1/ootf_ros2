# octo_policy.py — Octo VLA model wrapper for live single-camera inference.
#
# Manages rolling image history, task creation (language or goal image),
# and inference.  Returns a 7-value EEF delta: [dx, dy, dz, drx, dry, drz, gripper].

import cv2
import numpy as np
import jax
import jax.numpy as jnp
from octo.model.octo_model import OctoModel

IMAGE_SIZE = 128   # Octo wrist camera input resolution (primary is 256×256)

# The pretrained checkpoint's action statistics come from bridge_dataset.
# Used to unnormalize Octo's output back to real-world EEF delta units.
DEFAULT_DATASET = "bridge_dataset"


class OctoPolicy:
    """Wraps OctoModel for live wrist-camera inference with a rolling image history."""

    def __init__(
        self,
        model_path: str = "hf://rail-berkeley/octo-small-1.5",
        #model_path: str = "hf://rail-berkeley/octo-base-1.5",
        dataset_name: str = DEFAULT_DATASET,
        window_size: int = 2,
        step: int = None,
    ):
        """Load a checkpoint and initialise the rolling image buffer."""
        print(f"[OCTO] Loading model from: {model_path}" + (f" (step {step})" if step else ""))
        self.model       = OctoModel.load_pretrained(model_path, step=step)
        self.dataset     = dataset_name
        self.window_size = window_size

        self._rng           = jax.random.PRNGKey(0)
        self._image_history = []   # rolling list of preprocessed frames
        self.task           = None

        print("[OCTO] Model loaded.")
        print(self.model.get_pretty_spec())

    def set_task_text(self, instruction: str):
        """Condition the model on a natural-language instruction."""
        self.task = self.model.create_tasks(texts=[instruction])
        print(f"[OCTO] Task (language): '{instruction}'")

    def set_task_goal_image(self, goal_rgb: np.ndarray):
        """Condition the model on a goal image (HxWx3 RGB uint8)."""
        goal = cv2.resize(goal_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.uint8)
        # create_tasks expects (batch, H, W, C) for goal images
        self.task = self.model.create_tasks(
            goals={"image_wrist": goal[None]}
        )
        print("[OCTO] Task (goal image) set.")

    def reset(self):
        """Clear the image history at the start of a new episode."""
        self._image_history = []

    def step(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run one inference step and return a 7-value EEF delta action."""
        if self.task is None:
            raise RuntimeError(
                "No task set. Call set_task_text() or set_task_goal_image() first."
            )

        frame = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.uint8)

        self._image_history.append(frame)
        if len(self._image_history) > self.window_size:
            self._image_history.pop(0)

        # Pad with the oldest frame if we don't have a full window yet
        n_real   = len(self._image_history)
        n_pad    = self.window_size - n_real
        padded   = [self._image_history[0]] * n_pad + self._image_history

        obs_images = np.stack(padded, axis=0)[None]          # (1, ws, H, W, C)
        pad_mask   = np.array([[False] * n_pad + [True] * n_real])  # (1, ws)

        dummy_primary = np.zeros((1, self.window_size, 1, 1, 3), dtype=np.uint8)
        observation = {
            "image_primary":     dummy_primary,
            "image_wrist":       obs_images,
            "timestep_pad_mask": pad_mask,
        }

        self._rng, key = jax.random.split(self._rng)

        # Finetuned checkpoints may store stats as a flat dict ({"action": ...})
        # while pretrained ones are nested ({"bridge_dataset": {"action": ...}}).
        stats = self.model.dataset_statistics
        if self.dataset in stats:
            unnorm = stats[self.dataset]["action"]
        elif "action" in stats:
            unnorm = stats["action"]
        else:
            unnorm = None

        actions = self.model.sample_actions(
            observation,
            self.task,
            unnormalization_statistics=unnorm,
            rng=key,
        )
        # actions: (batch=1, action_horizon, action_dim=7) — take the first step
        action = np.array(actions[0, 0], dtype=np.float64)
        return action
