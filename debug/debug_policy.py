#!/usr/bin/env python3
# =============================================================================
# debug_policy.py — Diagnose a finetuned Octo checkpoint against ground-truth
#                   .npz episodes.
#
# Output saved to debug/policy_diagnosis/ with names encoding data + checkpoint:
#   <raw_exp>__<ckpt_experiment>.png / .txt
#   <raw_exp>__<ckpt_experiment>__step<N>.png / .txt
#   <raw_exp>__pretrained.png / .txt
#
# Usage:
#   ./run.sh debug policy
#   ./run.sh debug policy --n-episodes 10
#   ./run.sh debug policy --pretrained
#   ./run.sh debug policy --checkpoint data/exp_02/checkpoint/octo_finetune/experiment_XXX
#   ./run.sh debug policy --step 10000
# =============================================================================

import argparse
import sys
from pathlib import Path

import numpy as np

# ── resolve paths ────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
_DATA_ROOT  = ROOT / "data"
_DIAG_DIR   = ROOT / "debug" / "policy_diagnosis"

sys.path.insert(0, str(ROOT / "src" / "vla"))
from octo_policy import OctoPolicy

DOF_LABELS = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]


def _latest_checkpoint() -> Path | None:
    """Return the most recently modified local finetune experiment dir, or None."""
    candidates = [
        d for d in _DATA_ROOT.glob("*/checkpoint/octo_finetune/experiment_*")
        if d.is_dir() and any(c.name.isdigit() for c in d.iterdir() if c.is_dir())
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _latest_raw_dir() -> Path | None:
    """Return the most recently modified data/*/raw directory, or None."""
    candidates = [d for d in _DATA_ROOT.glob("*/raw") if d.is_dir()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _output_name(raw_dir: Path, model_path: str, step: int | None, pretrained: bool) -> Path:
    """Build a descriptive filename encoding the data and checkpoint used."""
    raw_exp = raw_dir.parent.name   # e.g. "exp_02"  from  data/exp_02/raw

    if pretrained:
        ckpt_part = "pretrained"
    else:
        # Last component of the checkpoint path, e.g. "experiment_20260504_104838"
        ckpt_part = Path(model_path).name

    step_part = f"__step{step}" if step is not None else ""
    filename  = f"{raw_exp}__{ckpt_part}{step_part}.png"
    return _DIAG_DIR / filename


def load_episodes(raw_dir: Path, n: int) -> list[dict]:
    """Load up to n episodes from .npz files in raw_dir."""
    files = sorted(raw_dir.glob("*.npz"))[:n]
    if not files:
        sys.exit(f"[ERR] No .npz files found in {raw_dir}")
    episodes = []
    for f in files:
        d    = np.load(f, allow_pickle=True)
        lang = str(d["language_instruction"]) if "language_instruction" in d.files else "pick up the object"
        episodes.append({"images": d["images"], "actions": d["actions"], "instruction": lang})
    return episodes


def run_inference(policy: OctoPolicy, episodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Run the policy on all episodes and return (ground_truth, predictions) arrays."""
    all_gt, all_pred = [], []
    for ep_idx, ep in enumerate(episodes):
        images, gt_actions = ep["images"], ep["actions"]
        policy.set_task_text(ep["instruction"])
        policy.reset()
        for t in range(len(images)):
            all_pred.append(policy.step(images[t]))
            all_gt.append(gt_actions[t])
        print(f"  episode {ep_idx+1}/{len(episodes)} done ({len(images)} steps)")
    return np.array(all_gt), np.array(all_pred)


def _build_stats_table(gt: np.ndarray, pred: np.ndarray) -> str:
    """Format a per-DOF statistics table comparing gt and pred arrays."""
    lines = []
    lines.append("── Action statistics ────────────────────────────────────────────")
    header = f"{'DOF':<8}  {'GT mean':>9}  {'GT std':>9}  {'Pred mean':>10}  {'Pred std':>9}  {'MAE':>8}  {'Corr':>7}"
    lines.append(header)
    lines.append("─" * len(header))
    for i, label in enumerate(DOF_LABELS):
        gt_m, gt_s = gt[:, i].mean(),   gt[:, i].std()
        pr_m, pr_s = pred[:, i].mean(), pred[:, i].std()
        mae        = np.abs(gt[:, i] - pred[:, i]).mean()
        corr_mat   = np.corrcoef(gt[:, i], pred[:, i])
        corr       = corr_mat[0, 1] if not np.isnan(corr_mat[0, 1]) else 0.0
        flag = ""
        if abs(gt_m - pr_m) > 3 * gt_s:    flag += "  ← large mean shift"
        if gt_s > 0 and pr_s / gt_s < 0.1: flag += "  ← pred too small"
        if gt_s > 0 and pr_s / gt_s > 10:  flag += "  ← pred too large"
        if abs(corr) < 0.1:                 flag += "  ← no correlation"
        lines.append(f"{label:<8}  {gt_m:>+9.4f}  {gt_s:>9.4f}  {pr_m:>+10.4f}  {pr_s:>9.4f}  {mae:>8.4f}  {corr:>+7.3f}{flag}")
    return "\n".join(lines)


def print_stats(gt: np.ndarray, pred: np.ndarray):
    """Print the per-DOF statistics table to stdout."""
    print("\n" + _build_stats_table(gt, pred) + "\n")


def save_stats(
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
    checkpoint_label: str,
    raw_dir: Path,
    n_episodes: int,
    n_steps: int,
):
    """Write the statistics table and metadata to a .txt file alongside out_path."""
    import datetime
    lines = [
        "=" * 68,
        "Octo policy diagnosis",
        "=" * 68,
        f"Date       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Checkpoint : {checkpoint_label}",
        f"Raw data   : {raw_dir}",
        f"Episodes   : {n_episodes}",
        f"Steps      : {n_steps}",
        f"Plot       : {out_path.with_suffix('.png').name}",
        "",
        _build_stats_table(gt, pred),
        "",
    ]
    txt_path = out_path.with_suffix(".txt")
    txt_path.write_text("\n".join(lines))
    print(f"[VIZ] Saved table → {txt_path}")


def make_plot(gt: np.ndarray, pred: np.ndarray, out_path: Path, n_episodes: int, checkpoint_label: str):
    """Save a 4-row diagnostic plot (time series, scatter, distributions, MAE/corr bars)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[WARN] matplotlib not available — skipping plot. pip install matplotlib")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 22))
    fig.suptitle(
        f"Octo action diagnosis\n{checkpoint_label}\n({n_episodes} episodes, {len(gt)} steps)",
        fontsize=13,
    )
    gs = gridspec.GridSpec(4, 7, figure=fig, hspace=0.55, wspace=0.35)

    # Row 0 — time series (first 300 steps)
    T_SHOW = min(300, len(gt))
    for i, label in enumerate(DOF_LABELS):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(gt[:T_SHOW, i],   label="GT",   color="#1f77b4", lw=1.2, alpha=0.8)
        ax.plot(pred[:T_SHOW, i], label="Pred", color="#ff7f0e", lw=1.2, alpha=0.8, linestyle="--")
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_ylabel("value", fontsize=7)
            ax.legend(fontsize=6)

    # Row 1 — scatter GT vs Pred
    for i, label in enumerate(DOF_LABELS):
        ax = fig.add_subplot(gs[1, i])
        ax.scatter(gt[:, i], pred[:, i], s=2, alpha=0.3, color="#2ca02c")
        lo = min(gt[:, i].min(), pred[:, i].min())
        hi = max(gt[:, i].max(), pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_title(f"{label} scatter", fontsize=9)
        ax.set_xlabel("GT", fontsize=7)
        ax.set_ylabel("Pred", fontsize=7)
        ax.tick_params(labelsize=7)

    # Row 2 — distributions
    for i, label in enumerate(DOF_LABELS):
        ax  = fig.add_subplot(gs[2, i])
        lo  = min(gt[:, i].min(), pred[:, i].min())
        hi  = max(gt[:, i].max(), pred[:, i].max())
        ax.hist(gt[:, i],   bins=40, range=(lo, hi), alpha=0.6, label="GT",   density=True, color="#1f77b4")
        ax.hist(pred[:, i], bins=40, range=(lo, hi), alpha=0.6, label="Pred", density=True, color="#ff7f0e")
        ax.set_title(f"{label} dist", fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6)

    # Row 3 — MAE bar + correlation bar
    ax_mae  = fig.add_subplot(gs[3, :3])
    ax_corr = fig.add_subplot(gs[3, 4:])

    maes  = [np.abs(gt[:, i] - pred[:, i]).mean() for i in range(7)]
    corrs = [float(np.corrcoef(gt[:, i], pred[:, i])[0, 1]) if gt[:, i].std() > 0 else 0.0 for i in range(7)]

    ax_mae.bar(DOF_LABELS, maes,  color=["#d62728" if m > 0.05 else "#2ca02c" for m in maes])
    ax_mae.set_title("Mean Absolute Error per DOF  (red = > 0.05)", fontsize=9)
    ax_mae.set_ylabel("MAE", fontsize=8)
    ax_mae.tick_params(labelsize=8)

    ax_corr.bar(DOF_LABELS, corrs, color=["#d62728" if abs(c) < 0.3 else "#2ca02c" for c in corrs])
    ax_corr.axhline(0, color="black", lw=0.8)
    ax_corr.set_ylim(-1.1, 1.1)
    ax_corr.set_title("Pearson correlation GT vs Pred  (red = |r| < 0.3)", fontsize=9)
    ax_corr.set_ylabel("r", fontsize=8)
    ax_corr.tick_params(labelsize=8)

    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    print(f"[VIZ] Saved plot → {out_path}")
    plt.close(fig)


def parse_args():
    """Parse CLI arguments for checkpoint, data directory, and evaluation settings."""
    ap = argparse.ArgumentParser(description="Visualize Octo policy predictions vs ground truth.")
    ap.add_argument("--checkpoint",   type=str, default=None,
                    help="Path to an octo_finetune experiment dir (default: latest)")
    ap.add_argument("--step",         type=int, default=None,
                    help="Checkpoint step to load (default: latest)")
    ap.add_argument("--dataset-name", type=str, default="ootf_synthetic",
                    help="Dataset key for unnorm stats (default: ootf_synthetic)")
    ap.add_argument("--pretrained",   action="store_true",
                    help="Use the pretrained base model instead of a local checkpoint")
    ap.add_argument("--raw-dir",      type=str, default=None,
                    help="Directory with .npz episodes (default: most recently modified data/*/raw)")
    ap.add_argument("--n-episodes",   type=int, default=5,
                    help="Number of episodes to evaluate (default: 5)")
    ap.add_argument("--window-size",  type=int, default=1,
                    help="Observation history length passed to Octo (default: 1)")
    return ap.parse_args()


def main():
    """Resolve checkpoint and data, run inference, and save plot + stats table."""
    args = parse_args()

    if args.pretrained:
        model_path   = "hf://rail-berkeley/octo-small-1.5"
        dataset_name = "bridge_dataset"
        ckpt_label   = "pretrained octo-small-1.5"
    else:
        if args.checkpoint:
            model_path = args.checkpoint
        else:
            ckpt = _latest_checkpoint()
            if ckpt is None:
                sys.exit(f"[ERR] No finetune checkpoint found under {_DATA_ROOT}. "
                         "Run the pipeline first, or pass --checkpoint.")
            model_path = str(ckpt)
            print(f"[INFO] Using latest checkpoint: {ckpt}")
        dataset_name = args.dataset_name
        ckpt_label   = model_path

    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
    else:
        raw_dir = _latest_raw_dir()
        if raw_dir is None:
            sys.exit(f"[ERR] No raw episode directories found under {_DATA_ROOT}")
        print(f"[INFO] Using raw dir: {raw_dir}")

    out_path = _output_name(raw_dir, model_path, args.step, args.pretrained)
    print(f"[INFO] Output → {out_path}")

    print(f"\n[INFO] Loading model …")
    policy = OctoPolicy(
        model_path   = model_path,
        dataset_name = dataset_name,
        window_size  = args.window_size,
        step         = args.step,
    )

    stats = policy.model.dataset_statistics
    print(f"\n[INFO] dataset_statistics keys: {list(stats.keys())}")
    if dataset_name not in stats and "action" not in stats:
        print(f"[WARN] Neither '{dataset_name}' nor 'action' found in dataset_statistics.")
        print(f"       Actions will NOT be unnormalized — predictions may be wrong scale.")
        print(f"       Available keys: {list(stats.keys())}")

    episodes = load_episodes(raw_dir, args.n_episodes)
    print(f"\n[INFO] Running inference on {len(episodes)} episodes …")
    gt, pred = run_inference(policy, episodes)
    print(f"\n[INFO] Total steps evaluated: {len(gt)}")

    print_stats(gt, pred)
    save_stats(gt, pred, out_path, ckpt_label, raw_dir, len(episodes), len(gt))
    make_plot(gt, pred, out_path, len(episodes), ckpt_label)


if __name__ == "__main__":
    main()
