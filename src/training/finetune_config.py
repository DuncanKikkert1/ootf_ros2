# finetune_config.py — Octo finetune configuration for the ootf_synthetic dataset.
#
# Config string format: "<mode>,<task>[,overfit]"
#   mode:   full | head_only | head_mlp_only
#   task:   image_conditioned | text_conditioned | multimodal
#   overfit (optional): disables augmentation, shrinks batch/intervals/warmup,
#                       zeros weight decay — use for initial end-to-end debugging.

import os

from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder


def get_config(config_string: str = "head_mlp_only,text_conditioned") -> ConfigDict:
    """Build the ml_collections config for the given finetune mode and task type."""
    parts  = config_string.split(",")
    mode   = parts[0]
    task   = parts[1]
    overfit = "overfit" in parts[2:]

    assert task in ["image_conditioned", "text_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    # ── PROPRIO toggle ───────────────────────────────────────────────────────
    # Default is set here; an env var OOTF_PROPRIO=1/0 OVERRIDES it per-run, so
    # you can train a with-proprio and a no-proprio model on the SAME data
    # without editing this file (e.g. OOTF_PROPRIO=1 ./run.sh … --finetune-only).
    # With proprio the model can predict the action from its own arm pose and
    # under-use the cube (goes to the mean pose → ~100 mm overshoot); without it
    # (image+language only, like bridge Octo) it must servo on the magenta cube.
    _PROPRIO_DEFAULT = True    # proprio ON (the no-proprio run flailed + rammed; proprio is needed)
    _PROPRIO = {"1": True, "0": False}.get(
        os.environ.get("OOTF_PROPRIO", ""), _PROPRIO_DEFAULT)

    FINETUNING_KWARGS = {
        "name": "ootf_synthetic",
        "data_dir": placeholder(str),
        # Single wrist camera → primary slot only.  Mapping the same image into
        # both primary and wrist causes independent augmentation during training
        # (two different random crops of the same frame) while inference feeds
        # identical resized frames, creating a train/eval mismatch.
        "image_obs_keys": {"primary": "image_wrist"},
        # Proprio on/off via _PROPRIO (default + OOTF_PROPRIO env override above).
        # The proprio observation tokenizer below follows this automatically.
        "proprio_obs_key": "proprio" if _PROPRIO else None,
        "language_key": "language_instruction",
        "action_proprio_normalization_type": "normal",
        # Don't normalize the gripper dimension (index 6)
        "action_normalization_mask": [True, True, True, True, True, True, False],
    }

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    elif mode == "head_mlp_only":
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )
    else:
        raise ValueError("Invalid mode")

    max_steps   = FieldReference(50000)
    window_size = FieldReference(default=2)

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        # Overfit/full: small batch to fit GPU memory (full mode OOMs at 32 on RTX A5000).
        batch_size            = 8     if (overfit or mode == "full") else 32,
        # Overfit: small shuffle buffer — large buffers dilute rare frames in tiny datasets.
        shuffle_buffer_size   = 1000  if overfit else 50_000,
        num_steps=max_steps,
        log_interval          = 25    if overfit else 100,
        eval_interval         = 250   if overfit else 5000,
        save_interval         = 250   if overfit else 5000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="octo_finetune", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs=FINETUNING_KWARGS,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,
                # Overfit: short warmup so peak LR is reached within 1 k-step runs.
                warmup_steps  = 50   if overfit else 2000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            # Overfit: zero weight decay — regularisation fights memorisation.
            weight_decay          = 0.0  if overfit else 0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
        viz_kwargs=dict(
            eval_batch_size=128,
            trajs_for_metrics=100,
            trajs_for_viz=8,
            samples_per_state=8,
        ),
    )

    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "text_conditioned":
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    traj_transform_kwargs = dict(
        window_size=window_size,
        action_horizon=4,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(keep_image_prob=keep_image_prob),
    )

    # Overfit: no augmentation — the model needs to memorise the exact scene.
    # Normal training: random crop + colour jitter for robustness.
    if overfit:
        wrist_augment_kwargs = {}
    else:
        wrist_augment_kwargs = dict(
            random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
            random_brightness=[0.1],
            random_contrast=[0.9, 1.1],
            random_saturation=[0.9, 1.1],
            random_hue=[0.05],
            augment_order=[
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ],
        )

    frame_transform_kwargs = dict(
        resize_size={"primary": (256, 256)},
        image_augment_kwargs={"primary": wrist_augment_kwargs} if wrist_augment_kwargs else {},
    )

    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs

    # Remove the pretrained wrist tokenizer — we have no wrist camera stream.
    # Without this, it stays in the model, wastes parameters, and logs a warning
    # every step. finetune.py uses config_delete_keys to drop entries before
    # building the model.
    # Keep the pretrained DiffusionActionHead (exp_16 — the run that actually got
    # 2/10 — used diffusion; the L1 head regressed to deterministic-to-the-mean).
    config["config_delete_keys"] = {
        "model": {"observation_tokenizers": {"wrist": True}}
    }

    # Inject a LowdimObsTokenizer for 7D proprio [x, y, z, rx, ry, rz, gripper].
    # bin_type="normal" uses Gaussian quantile bins, correct for z-scored inputs.
    # Octo's finetune.py deep-merges update_config into the pretrained model config
    # so the new tokenizer is added without touching the pretrained weights.
    # Keep the pretrained DiffusionActionHead (no head override) — only add the
    # proprio tokenizer below when proprio is enabled.
    config["update_config"] = {
        "model": {
            "observation_tokenizers": {},
        }
    }

    # Proprio tokenizer is injected ONLY when proprio is enabled (PROPRIO TOGGLE
    # above), so toggling that one line turns proprio fully on/off — data key and
    # model tokenizer stay in sync.  LowdimObsTokenizer over the z-scored 7D proprio.
    if FINETUNING_KWARGS["proprio_obs_key"] is not None:
        config["update_config"]["model"]["observation_tokenizers"]["proprio"] = {
            "module": "octo.model.components.tokenizers",
            "name": "LowdimObsTokenizer",
            "args": [],
            "kwargs": {"n_bins": 256, "bin_type": "normal", "obs_keys": ("proprio",)},
        }

    return ConfigDict(config)
