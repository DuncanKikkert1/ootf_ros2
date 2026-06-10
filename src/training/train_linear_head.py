# train_linear_head.py — Fit a linear regression head on a frozen Octo backbone.
#
# For every frame in the raw .npz episodes, runs the frozen pretrained Octo
# transformer to produce a readout-action embedding, then solves ridge regression:
#
#   action_chunk_normalised = embedding @ W + b
#
# The result is saved as a small .npz that LinearHeadPolicy loads at inference.
#
# Usage:
#   python src/training/train_linear_head.py \
#       --raw-dir data/exp_07/raw \
#       --output  data/exp_07/linear_head.npz
#
# Pipeline: collect.py → pipeline.py (optional finetune) → THIS → run_octo_live.py

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import jax
import jax.numpy as jnp
from octo.model.octo_model import OctoModel

IMAGE_SIZE   = 128
WINDOW_SIZE  = 2    # must match pretrained model window size
ACTION_DIM   = 7    # [dx, dy, dz, drx, dry, drz, gripper]
ACTION_HORIZ = 4    # action steps predicted per observation

# Normalise all action dims except gripper (index 6 of each step).
_NORM_MASK = np.array(
    [i % ACTION_DIM != 6 for i in range(ACTION_HORIZ * ACTION_DIM)],
    dtype=bool,
)


# ── token helpers (same pattern as Bahar/infer_linear_head_v2) ───────────────

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


def _pool(tok):
    """(B, H, [1,] D) → (B, D) by averaging over history window."""
    x = jnp.asarray(tok)
    if x.ndim == 4:          # (B, H, 1, D)
        x = x[:, :, 0, :]
    if x.ndim == 3:          # (B, H, D)
        return jnp.mean(x, axis=1)
    return x                 # (B, D)


def _tile_tasks(task, B: int):
    """Repeat a batch-1 task dict to batch size B."""
    def _t(x):
        x = jnp.asarray(x)
        if x.ndim >= 1 and x.shape[0] == 1 and B > 1:
            return jnp.tile(x, [B] + [1] * (x.ndim - 1))
        return x
    return jax.tree_util.tree_map(_t, task)


# ── embedding extraction ──────────────────────────────────────────────────────

def extract_embeddings(model: OctoModel, raw_dir: Path, batch_size: int):
    """
    Walk every .npz episode in raw_dir.
    For each step t build a (window_size=2) observation, run the frozen
    transformer, and collect the pooled readout-action embedding.

    Returns
    -------
    embeddings : (N, D) float32
    chunks     : (N, ACTION_HORIZ * ACTION_DIM) float32
    """
    npz_files = sorted(raw_dir.glob("*.npz"))
    if not npz_files:
        sys.exit(f"[ERR] No .npz files found in {raw_dir}")

    # Store the full example observation so we can copy it per batch and only
    # replace the image arrays — preserves pad_mask_dict, timestep etc. (Bahar pattern).
    ex_obs  = model.example_batch.get(
        "observation", model.example_batch.get("observations", {})
    )
    _pshape = ex_obs["image_primary"].shape   # (B, ws, H, W, C)
    PRIMARY_H, PRIMARY_W = int(_pshape[2]), int(_pshape[3])
    print(f"[TRAIN] primary placeholder: {PRIMARY_H}×{PRIMARY_W}  wrist: {IMAGE_SIZE}×{IMAGE_SIZE}")

    print(f"[TRAIN] {len(npz_files)} episodes  batch_size={batch_size}")

    all_emb   = []
    all_chunk = []
    t_start   = time.monotonic()
    t_compile_done = None   # set after first episode completes

    for ep_idx, path in enumerate(npz_files):
        data        = np.load(path, allow_pickle=True)
        images_raw  = data["images"]           # (T, H, W, C) uint8
        actions     = data["actions"].astype(np.float32)  # (T, 7)
        instruction = str(data["instruction"])
        T           = len(actions)

        # Resize all frames once per episode
        frames = np.stack([
            cv2.resize(images_raw[t], (IMAGE_SIZE, IMAGE_SIZE))
            for t in range(T)
        ])  # (T, 128, 128, 3)

        task_single = model.create_tasks(texts=[instruction])

        for b_start in range(0, T, batch_size):
            b_end = min(b_start + batch_size, T)
            B     = b_end - b_start

            # Rolling-window observation for each step in the batch.
            # Feed wrist image to both slots (Bahar pattern): the pretrained
            # backbone attends most to image_primary, so zeros there wastes the
            # channel it was trained to use most.
            wrist    = np.zeros((B, WINDOW_SIZE, IMAGE_SIZE,  IMAGE_SIZE,  3), dtype=np.uint8)
            primary  = np.zeros((B, WINDOW_SIZE, PRIMARY_H,   PRIMARY_W,   3), dtype=np.uint8)
            pad_mask = np.ones((B, WINDOW_SIZE), dtype=bool)

            for i, t in enumerate(range(b_start, b_end)):
                if t == 0:
                    wrist[i, 0]    = frames[0]
                    primary[i, 0]  = cv2.resize(frames[0], (PRIMARY_W, PRIMARY_H))
                    pad_mask[i, 0] = False
                else:
                    wrist[i, 0]   = frames[t - 1]
                    primary[i, 0] = cv2.resize(frames[t - 1], (PRIMARY_W, PRIMARY_H))
                wrist[i, 1]   = frames[t]
                primary[i, 1] = cv2.resize(frames[t], (PRIMARY_W, PRIMARY_H))

            # Copy full example observation and replace only images (Bahar pattern).
            obs = dict(ex_obs)
            obs["image_primary"] = primary
            obs["image_wrist"]   = wrist

            out = model.run_transformer(
                observations=obs,
                tasks=_tile_tasks(task_single, B),
                timestep_pad_mask=pad_mask,
                train=False,
            )

            vecs = np.array(_pool(_extract_tokens(out["readout_action"])),
                            dtype=np.float32)   # (B, D)
            all_emb.append(vecs)

            # Action target: ACTION_HORIZ-step chunk starting at t
            for i, t in enumerate(range(b_start, b_end)):
                chunk = np.stack([
                    actions[min(t + h, T - 1)] for h in range(ACTION_HORIZ)
                ]).reshape(-1)   # (ACTION_HORIZ * ACTION_DIM,)
                all_chunk.append(chunk)

        t_now = time.monotonic()

        if ep_idx == 0:
            t_compile_done = t_now
            compile_s = t_compile_done - t_start
            print(f"  [ep 1/{len(npz_files)}] JAX compilation done in {compile_s:.0f}s  "
                  f"steps={T}")
        else:
            elapsed   = t_now - t_compile_done
            done_eps  = ep_idx          # episodes finished after compile (ep_idx 1..N-1)
            rate      = done_eps / elapsed if elapsed > 0 else 0
            remaining = len(npz_files) - (ep_idx + 1)
            eta_s     = remaining / rate if rate > 0 else 0
            eta_str   = (f"{int(eta_s//60)}m{int(eta_s%60):02d}s"
                         if eta_s < 3600
                         else f"{int(eta_s//3600)}h{int((eta_s%3600)//60):02d}m")
            n_so_far  = sum(len(e) for e in all_emb)
            print(f"  [ep {ep_idx + 1}/{len(npz_files)}] "
                  f"elapsed={elapsed:.0f}s  "
                  f"rate={rate:.1f} ep/s  "
                  f"ETA={eta_str}  "
                  f"samples={n_so_far}")

    embeddings = np.vstack(all_emb)                  # (N, D)
    chunks     = np.array(all_chunk, dtype=np.float32)  # (N, H*A)
    print(f"[TRAIN] Total: {len(embeddings)} samples  embed_dim={embeddings.shape[1]}")
    return embeddings, chunks


# ── ridge regression ──────────────────────────────────────────────────────────

def fit_linear_head(embeddings: np.ndarray, chunks: np.ndarray, lam: float = 1e-4):
    """
    Fit W, b via ridge regression on normalised targets.

    pred_norm = embedding @ W + b
    pred      = pred_norm * y_std + y_mean
    """
    N, D = embeddings.shape
    A    = chunks.shape[1]

    # Compute statistics and normalise
    y_mean          = chunks.mean(axis=0)
    y_std           = chunks.std(axis=0)
    y_std[~_NORM_MASK] = 1.0          # keep gripper dims un-scaled
    y_std[y_std < 1e-8] = 1.0        # prevent division by zero
    Y_norm = (chunks - y_mean) / y_std

    print(f"[TRAIN] Fitting ridge regression  N={N}  D={D}  A={A}  λ={lam}")

    # Augment embeddings with a bias column
    X = np.hstack([embeddings, np.ones((N, 1), dtype=np.float32)])  # (N, D+1)

    # Normal equations: (X^T X + λI) w = X^T Y  (don't regularise bias)
    XtX      = X.T @ X                   # (D+1, D+1)
    reg      = lam * np.eye(D + 1, dtype=np.float32)
    reg[-1, -1] = 0.0                    # no regularisation on bias
    XtX     += reg
    XtY      = X.T @ Y_norm              # (D+1, A)

    sol = np.linalg.solve(XtX, XtY).astype(np.float32)   # (D+1, A)
    W   = sol[:-1]    # (D, A)
    b   = sol[-1]     # (A,)

    # Training residual
    pred_norm = X @ sol
    mse = float(np.mean((pred_norm - Y_norm) ** 2))
    print(f"[TRAIN] Train MSE (normalised): {mse:.6f}")

    return W, b, y_mean.astype(np.float32), y_std.astype(np.float32)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Train a linear head on frozen Octo embeddings."
    )
    ap.add_argument("--raw-dir",      required=True,
                    help="Directory of .npz training episodes")
    ap.add_argument("--output",       required=True,
                    help="Path for the output .npz (e.g. data/exp_07/linear_head.npz)")
    ap.add_argument("--model-path",   default="hf://rail-berkeley/octo-base-1.5",
                    help="Pretrained Octo checkpoint used as backbone")
    ap.add_argument("--batch-size",   type=int, default=32,
                    help="Frames per transformer forward pass (default 32)")
    ap.add_argument("--lam",          type=float, default=1e-4,
                    help="Ridge regularisation coefficient (default 1e-4)")
    return ap.parse_args()


def main():
    args    = parse_args()
    raw_dir = Path(args.raw_dir)
    out     = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[TRAIN] Loading backbone: {args.model_path}")
    model = OctoModel.load_pretrained(args.model_path)
    print("[TRAIN] Backbone ready.")

    embeddings, chunks = extract_embeddings(model, raw_dir, args.batch_size)
    W, b, y_mean, y_std = fit_linear_head(embeddings, chunks, lam=args.lam)

    np.savez(
        out,
        W          = W,
        b          = b,
        y_mean     = y_mean,
        y_std      = y_std,
        act_h      = np.array(ACTION_HORIZ),
        act_dim    = np.array(ACTION_DIM),
        embed_dim  = np.array(W.shape[0]),
        model_path = np.array(args.model_path),
    )
    print(f"[TRAIN] Saved → {out}")
    print(f"        W={W.shape}  b={b.shape}")


if __name__ == "__main__":
    main()
