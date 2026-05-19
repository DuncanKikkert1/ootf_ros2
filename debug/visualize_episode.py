"""
visualize_episode.py — Render a collected .npz episode as an MP4 video.

Usage:
    python debug/visualize_episode.py data/exp_01/raw/episode_000000.npz
    python debug/visualize_episode.py data/exp_01/raw/episode_000000.npz --output /tmp/ep.mp4
    python debug/visualize_episode.py data/exp_01/raw/episode_000000.npz --fps 15
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


# Panel dimensions
FRAME_SIZE   = 384          # wrist cam rendered at this size (upscaled from 128)
PANEL_W      = FRAME_SIZE
PANEL_H      = FRAME_SIZE
INFO_H       = 110          # height of the info strip below the frame
TIMELINE_H   = 28           # height of the gripper timeline bar
TOTAL_H      = PANEL_H + INFO_H + TIMELINE_H
TOTAL_W      = PANEL_W

# Colours (BGR)
C_BG         = (30,  30,  30)
C_WHITE      = (240, 240, 240)
C_GREY       = (140, 140, 140)
C_GREEN      = (80,  200, 80)
C_RED        = (80,  80,  200)
C_YELLOW     = (40,  210, 210)
C_OPEN       = (80,  180, 80)
C_CLOSED     = (60,  60,  210)


def _bar(canvas, x, y, w, h, frac, col_fill, col_bg=(60, 60, 60)):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), col_bg, -1)
    cv2.rectangle(canvas, (x, y), (x + int(w * frac), y + h), col_fill, -1)


def _text(canvas, txt, x, y, scale=0.45, col=C_WHITE, thickness=1):
    cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, col, thickness, cv2.LINE_AA)


def render_frame(images, actions, step: int) -> np.ndarray:
    n = len(actions)
    img   = images[step]
    act   = actions[step]
    grip  = act[6]
    d_pos = act[:3]
    d_eul = act[3:6]

    canvas = np.full((TOTAL_H, TOTAL_W, 3), C_BG, dtype=np.uint8)

    # ── Wrist camera ─────────────────────────────────────────────────────────
    frame_up = cv2.resize(img[:, :, ::-1], (FRAME_SIZE, FRAME_SIZE),
                          interpolation=cv2.INTER_NEAREST)
    canvas[:PANEL_H, :PANEL_W] = frame_up

    # Gripper state badge (top-left corner of frame)
    badge_col  = C_CLOSED if grip > 0.5 else C_OPEN
    badge_text = "GRIP: CLOSED" if grip > 0.5 else "GRIP:  OPEN"
    cv2.rectangle(canvas, (0, 0), (150, 22), badge_col, -1)
    _text(canvas, badge_text, 6, 15, scale=0.48, col=(240, 240, 240), thickness=1)

    # Step counter (top-right corner)
    step_str = f"{step + 1:3d}/{n}"
    _text(canvas, step_str, PANEL_W - 68, 15, scale=0.48, col=C_WHITE)

    # ── Info strip ───────────────────────────────────────────────────────────
    y0 = PANEL_H + 10

    _text(canvas, "dX", 10, y0 + 14, col=C_GREY)
    _text(canvas, "dY", 10, y0 + 34, col=C_GREY)
    _text(canvas, "dZ", 10, y0 + 54, col=C_GREY)
    _text(canvas, "rX", 10, y0 + 74, col=C_GREY)
    _text(canvas, "rY", 10, y0 + 94, col=C_GREY)

    MAX_POS, MAX_ROT = 0.05, 0.20
    bar_x, bar_w = 35, 160

    for i, (v, vmax, label) in enumerate([
        (d_pos[0], MAX_POS, None),
        (d_pos[1], MAX_POS, None),
        (d_pos[2], MAX_POS, None),
        (d_eul[0], MAX_ROT, None),
        (d_eul[1], MAX_ROT, None),
    ]):
        by   = y0 + i * 20
        frac = np.clip(abs(v) / vmax, 0.0, 1.0)
        col  = C_RED if v < 0 else C_GREEN
        _bar(canvas, bar_x, by, bar_w, 14, frac, col)
        _text(canvas, f"{v:+.4f}", bar_x + bar_w + 6, by + 11,
              scale=0.40, col=C_WHITE)

    # rZ on right column
    v    = d_eul[2]
    frac = np.clip(abs(v) / MAX_ROT, 0.0, 1.0)
    col  = C_RED if v < 0 else C_GREEN
    _text(canvas, "rZ", PANEL_W // 2 + 10, y0 + 14, col=C_GREY)
    _bar(canvas, PANEL_W // 2 + 30, y0, 120, 14, frac, col)
    _text(canvas, f"{v:+.4f}", PANEL_W // 2 + 156, y0 + 11,
          scale=0.40, col=C_WHITE)

    # ── Gripper timeline ─────────────────────────────────────────────────────
    ty = PANEL_H + INFO_H
    cv2.rectangle(canvas, (0, ty), (TOTAL_W, TOTAL_H), (20, 20, 20), -1)

    for s in range(n):
        x1 = int(s * TOTAL_W / n)
        x2 = int((s + 1) * TOTAL_W / n)
        col = C_CLOSED if actions[s, 6] > 0.5 else C_OPEN
        cv2.rectangle(canvas, (x1, ty + 4), (x2, TOTAL_H - 4), col, -1)

    # Playhead
    px = int(step * TOTAL_W / n)
    cv2.line(canvas, (px, ty), (px, TOTAL_H), C_WHITE, 2)

    return canvas


def visualize(npz_path: Path, output: Path, fps: int):
    data    = np.load(npz_path, allow_pickle=True)
    images  = data["images"]
    actions = data["actions"]
    instr   = str(data["instruction"])
    n       = len(actions)

    print(f"Episode : {npz_path.name}")
    print(f"Steps   : {n}")
    print(f"Instr   : {instr}")
    print(f"Output  : {output}  ({fps} fps)")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (TOTAL_W, TOTAL_H))

    for s in range(n):
        frame = render_frame(images, actions, s)
        writer.write(frame)

    writer.release()
    print(f"Done — {n} frames written.")


def main():
    ap = argparse.ArgumentParser(description="Visualise a collected .npz episode.")
    ap.add_argument("npz", type=Path, help="Path to episode_XXXXXX.npz")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output .mp4 path (default: same dir as npz)")
    ap.add_argument("--fps", type=int, default=20,
                    help="Playback frames per second (default: 20)")
    args = ap.parse_args()

    out = args.output or args.npz.with_suffix(".avi")
    visualize(args.npz, out, args.fps)


if __name__ == "__main__":
    main()
