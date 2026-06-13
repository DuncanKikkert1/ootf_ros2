# octo_policy.py — Octo VLA model wrappers for live single-camera inference.
#
# OctoPolicy      — original diffusion-head path (model.sample_actions).
# LinearHeadPolicy — frozen backbone + trained linear regression head
#                    (model.run_transformer → pooled tokens → W·x + b).
#
# Both return (action_horizon, 7) EEF deltas: [dx, dy, dz, drx, dry, drz, gripper].

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

        self._rng            = jax.random.PRNGKey(0)
        self._image_history  = []   # rolling list of preprocessed frames
        self._proprio_history = []  # rolling list of (3,) float32 EEF positions
        self.task            = None

        # Read the primary camera resolution from the model's example_batch so
        # the observation dict matches regardless of how the checkpoint was trained.
        _ex = self.model.example_batch.get(
            "observation", self.model.example_batch.get("observations", {})
        )
        _ps = _ex["image_primary"].shape   # (B, ws, H, W, C)
        self._primary_h = int(_ps[2])
        self._primary_w = int(_ps[3])
        self._has_wrist = "image_wrist" in _ex   # False for primary-only finetuned checkpoints
        # Store the full example observation so step() inherits pad_mask_dict,
        # timestep, task_completed etc. — same pattern as LinearHeadPolicy.
        self._ex_obs = _ex

        print("[OCTO] Model loaded.")
        print(self.model.get_pretty_spec())

    def set_task_text(self, instruction: str):
        """Condition the model on a natural-language instruction."""
        self.task = self.model.create_tasks(texts=[instruction])
        print(f"[OCTO] Task (language): '{instruction}'")

    def set_task_goal_image(self, goal_rgb: np.ndarray):
        """Condition the model on a goal image (HxWx3 RGB uint8)."""
        # The goal must be supplied under the same image keys the model reads.
        # This checkpoint feeds the wrist stream into the primary slot, so the
        # goal image goes under image_primary at the primary resolution —
        # putting it under image_wrist (the old code) lands it in a slot the
        # model doesn't read, silently disabling goal conditioning.  image_wrist
        # is only set when the checkpoint actually kept a wrist tokenizer.
        primary = cv2.resize(goal_rgb, (self._primary_w, self._primary_h)).astype(np.uint8)
        goals   = {"image_primary": primary[None]}   # (batch, H, W, C)
        if self._has_wrist:
            goals["image_wrist"] = cv2.resize(
                goal_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.uint8)[None]
        self.task = self.model.create_tasks(goals=goals)
        print("[OCTO] Task (goal image) set.")

    def reset(self):
        """Clear the image and proprio history at the start of a new episode."""
        self._image_history   = []
        self._proprio_history = []

    def step(self, image_rgb: np.ndarray, proprio: np.ndarray = None) -> np.ndarray:
        """Run one inference step and return a (action_horizon, 7) EEF delta chunk.

        proprio: optional (3,) float32 EEF xyz position in robot local frame.
                 When the checkpoint was trained with proprio, omitting it degrades performance.
        """
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

        # Feed wrist image to primary slot at the resolution the checkpoint expects.
        primary_frames = np.stack([
            cv2.resize(f, (self._primary_w, self._primary_h))
            for f in padded
        ], axis=0)[None].astype(np.uint8)   # (1, ws, PH, PW, 3)

        # Start from the example_batch observation so pad_mask_dict, timestep,
        # task_completed etc. are present — their absence silences all attention
        # masking and produces garbage embeddings.
        observation = dict(self._ex_obs)
        observation["image_primary"]     = primary_frames
        observation["timestep_pad_mask"] = pad_mask
        if self._has_wrist:
            observation["image_wrist"] = obs_images
        elif "image_wrist" in observation:
            del observation["image_wrist"]

        # ── Proprioception ────────────────────────────────────────────────────
        if proprio is not None:
            self._proprio_history.append(proprio.astype(np.float32))
            if len(self._proprio_history) > self.window_size:
                self._proprio_history.pop(0)

            n_real_p = len(self._proprio_history)
            n_pad_p  = self.window_size - n_real_p
            padded_p = [self._proprio_history[0]] * n_pad_p + self._proprio_history
            prop_arr = np.stack(padded_p, axis=0)[None]   # (1, ws, 3)

            # Normalize using dataset stats if they include proprio
            stats_all = self.model.dataset_statistics
            _ds = stats_all.get(self.dataset, stats_all)
            prop_stats = _ds.get("proprio") if isinstance(_ds, dict) else None
            if prop_stats is not None:
                mean = np.array(prop_stats["mean"])
                std  = np.array(prop_stats["std"])
                prop_arr = (prop_arr - mean) / (std + 1e-8)

            observation["proprio"] = prop_arr

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
        # actions: (batch=1, action_horizon, action_dim=7)
        return np.array(actions[0], dtype=np.float64)   # (action_horizon, 7)


# ── token helpers shared by LinearHeadPolicy and train_linear_head ───────────

def _extract_tokens(readout):
    """Pull the raw token array out of whatever object run_transformer returns."""
    if isinstance(readout, dict):
        for k in ("tokens", "token_embeddings", "x", "y", "data", "value"):
            if k in readout:
                return jnp.asarray(readout[k])
        return jnp.asarray(next(iter(readout.values())))
    for attr in ("tokens", "token_embeddings", "x", "y", "data", "value"):
        if hasattr(readout, attr):
            return jnp.asarray(getattr(readout, attr))
    if hasattr(readout, "as_dict"):
        d = readout.as_dict()
        for k in ("tokens", "token_embeddings", "x", "y", "data", "value"):
            if k in d:
                return jnp.asarray(d[k])
    raise TypeError(f"Cannot extract tokens from {type(readout)}")


def _pool_tokens(tok):
    """(B, H, [1,] D) → (B, D) by averaging over the history window."""
    x = jnp.asarray(tok)
    if x.ndim == 4:          # (B, H, 1, D)
        x = x[:, :, 0, :]
    if x.ndim == 3:          # (B, H, D)
        return jnp.mean(x, axis=1)
    return x                 # already (B, D)


# ── LinearHeadPolicy ──────────────────────────────────────────────────────────

class LinearHeadPolicy:
    """Frozen Octo transformer backbone with a trained linear regression head.

    At inference:
        embedding = pool(run_transformer(obs, task).readout_action)   # (D,)
        pred_norm = embedding @ W + b                                  # (H*A,)
        pred      = pred_norm * y_std + y_mean                        # (H*A,)
        → reshape to (action_horizon, 7)

    Train the head with src/training/train_linear_head.py.
    """

    def __init__(
        self,
        model_path: str = None,
        head_path:  str = None,
        window_size: int = 2,
    ):
        if head_path is None:
            raise ValueError("head_path must point to a trained linear_head.npz")

        print(f"[LINEAR] Loading head: {head_path}")
        ck = np.load(head_path)

        # Always use the backbone path baked into the .npz at training time so
        # the embedding space at inference is guaranteed to match training.
        # If model_path was explicitly supplied and differs, warn but honour the npz.
        saved_backbone = str(ck["model_path"])
        if model_path is not None and model_path != saved_backbone:
            print(f"[LINEAR] WARNING: ignoring requested backbone '{model_path}' — "
                  f"using training backbone '{saved_backbone}' to match embedding space.")
        backbone = saved_backbone
        print(f"[LINEAR] Loading backbone: {backbone}")
        self.model       = OctoModel.load_pretrained(backbone)
        self.window_size = window_size
        self.W      = jnp.asarray(ck["W"])           # (D, H*A)
        self.b      = jnp.asarray(ck["b"])           # (H*A,)
        self.y_mean = np.array(ck["y_mean"],  dtype=np.float64)   # (H*A,)
        self.y_std  = np.array(ck["y_std"],   dtype=np.float64)   # (H*A,)
        self.act_h   = int(ck["act_h"])
        self.act_dim = int(ck["act_dim"])

        # Store the full example observation so step() can copy it and only
        # replace the image arrays — same approach as Bahar's infer_linear_head_v2.
        # This preserves pad_mask_dict, timestep, task_completed etc. which Octo
        # uses for attention masking; omitting them produces noisier embeddings.
        _ex = self.model.example_batch.get(
            "observation", self.model.example_batch.get("observations", {})
        )
        self._ex_obs    = _ex
        _ps = _ex["image_primary"].shape   # (B, ws, H, W, C)
        self._primary_h = int(_ps[2])
        self._primary_w = int(_ps[3])
        self._has_wrist = "image_wrist" in _ex

        self._image_history = []
        self.task = None
        print(f"[LINEAR] Ready  W={self.W.shape}  act_h={self.act_h}  act_dim={self.act_dim}")

    def set_task_text(self, instruction: str):
        self.task = self.model.create_tasks(texts=[instruction])
        print(f"[LINEAR] Task: '{instruction}'")

    def set_task_goal_image(self, goal_rgb: np.ndarray):
        goal = cv2.resize(goal_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.uint8)
        self.task = self.model.create_tasks(goals={"image_wrist": goal[None]})
        print("[LINEAR] Task (goal image) set.")

    def reset(self):
        self._image_history = []

    def step(self, image_rgb: np.ndarray, proprio: np.ndarray = None) -> np.ndarray:
        """Run one step and return (action_horizon, 7) EEF delta chunk."""
        if self.task is None:
            raise RuntimeError("Call set_task_text() or set_task_goal_image() first.")

        frame = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.uint8)
        self._image_history.append(frame)
        if len(self._image_history) > self.window_size:
            self._image_history.pop(0)

        n_real = len(self._image_history)
        n_pad  = self.window_size - n_real
        padded = [self._image_history[0]] * n_pad + self._image_history

        obs_images = np.stack(padded, axis=0)[None]              # (1, ws, H, W, C)
        pad_mask   = np.array([[False] * n_pad + [True] * n_real])  # (1, ws)

        # Feed the wrist image to both slots (Bahar pattern): the pretrained
        # backbone attends most to image_primary, so zeros there wastes the
        # channel it was trained to use most.
        primary_frames = np.stack([
            cv2.resize(f, (self._primary_w, self._primary_h))
            for f in padded
        ], axis=0)[None].astype(np.uint8)   # (1, ws, PH, PW, 3)

        obs = dict(self._ex_obs)
        obs["image_primary"] = primary_frames
        if self._has_wrist:
            obs["image_wrist"] = obs_images

        out = self.model.run_transformer(
            observations=obs,
            tasks=self.task,
            timestep_pad_mask=pad_mask,
            train=False,
        )

        vec = np.array(_pool_tokens(_extract_tokens(out["readout_action"])),
                       dtype=np.float64)   # (1, D)

        pred_norm = vec @ np.array(self.W) + np.array(self.b)   # (1, H*A)
        pred      = pred_norm[0] * self.y_std + self.y_mean     # (H*A,)

        return pred.reshape(self.act_h, self.act_dim)
