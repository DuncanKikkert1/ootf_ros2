#!/usr/bin/env python3
"""Encode a sequence of PNG frames into an mp4 with OpenCV — no ffmpeg binary.

Frames are taken in sorted order, so the zero-padded names the recorder writes
(viewport_000123.png / wrist_000123.png) play in capture order.

Usage:
  python3 debug/encode_video.py <frame_prefix> <out.mp4> [fps] [scale]
    <frame_prefix>  e.g. /tmp/eval_frames/viewport_   (globs <prefix>*.png)
    [fps]           default 20
    [scale]         optional integer upscale factor (e.g. 4 for the tiny wrist POV)

Examples:
  python3 debug/encode_video.py /tmp/eval_frames/viewport_ /tmp/grasp_scene.mp4 20
  python3 debug/encode_video.py /tmp/eval_frames/wrist_    /tmp/grasp_pov.mp4 20 4
"""
import sys
import glob

import cv2


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    prefix = sys.argv[1]
    out    = sys.argv[2]
    fps    = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
    scale  = int(sys.argv[4])   if len(sys.argv) > 4 else 1

    files = sorted(glob.glob(prefix + "*.png"))
    if not files:
        sys.exit(f"no PNGs matching {prefix}*.png")

    h, w = cv2.imread(files[0]).shape[:2]
    w, h = w * scale, h * scale
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        sys.exit(f"could not open VideoWriter for {out}")

    for f in files:
        img = cv2.imread(f)              # BGR; VideoWriter also wants BGR → correct
        if img is None:
            continue
        if (img.shape[1], img.shape[0]) != (w, h):
            interp = cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA
            img = cv2.resize(img, (w, h), interpolation=interp)
        vw.write(img)
    vw.release()
    print(f"wrote {out}  ({len(files)} frames, {w}x{h}, {fps}fps)")


if __name__ == "__main__":
    main()
