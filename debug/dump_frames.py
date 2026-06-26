#!/usr/bin/env python3
"""Dump the camera frames SAVED in a collected episode .npz to a montage PNG.

This shows EXACTLY what was written into the training data, so you can lay it
next to the Isaac viewport and confirm the saved images match what you see
(e.g. is the cube the same colour on disk as in the sim?).

Usage:
  python3 debug/dump_frames.py data/exp_diag/raw/episode_000000_00_pick.npz
  python3 debug/dump_frames.py data/exp_diag/raw        # newest *_pick.npz in dir
  python3 debug/dump_frames.py data/exp_diag/raw out.png
"""
import os
import sys
import glob

import numpy as np
from PIL import Image


def _resolve(path: str) -> str:
    if os.path.isdir(path):
        fs = sorted(glob.glob(os.path.join(path, "*_pick.npz"))) \
            or sorted(glob.glob(os.path.join(path, "*.npz")))
        if not fs:
            sys.exit(f"no .npz found in {path}")
        return fs[-1]
    return path


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    f = _resolve(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/frames.png"

    d = np.load(f, allow_pickle=True)
    imgs = d["images"]
    n = len(imgs)
    k = min(8, n)
    idxs = [round(i * (n - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]

    tiles = [Image.fromarray(imgs[i].astype("uint8")).resize((160, 160)) for i in idxs]
    montage = Image.new("RGB", (sum(t.width for t in tiles), max(t.height for t in tiles)))
    x = 0
    for t in tiles:
        montage.paste(t, (x, 0))
        x += t.width
    montage.save(out)

    last = imgs[-1].astype(int).reshape(-1, 3)
    # crude magenta vs green pixel tally so the colour question is answered in text too
    R, G, B = last[:, 0], last[:, 1], last[:, 2]
    mag = int(((R > 110) & (B > 110) & (G < 90)).sum())
    grn = int(((G > 110) & (R < 90) & (B < 90)).sum())
    print(f"episode      : {os.path.basename(f)}")
    print(f"frames       : {n}  (montage shows idx {idxs})")
    print(f"image shape  : {imgs[0].shape}  dtype={imgs[0].dtype}")
    print(f"last frame   : mean RGB={last.mean(0).round(1)}  magenta-px={mag}  green-px={grn}")
    print(f"montage      : {out}  (open it to compare against the viewport)")


if __name__ == "__main__":
    main()
