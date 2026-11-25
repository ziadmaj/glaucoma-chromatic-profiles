#!/usr/bin/env python3
"""Visual + numeric QC of segmentation masks (0 bg, 1 disc, 2 cup). Labels are NOT used.

    python qc_masks.py --manifest manifest_hygd.csv --root . --out qc_hygd.png [--n 24]

Prints: share of images with no disc / no cup, disc-area and vCDR distributions, disc-centre spread.
Writes a montage with disc (green) and cup (blue) contours on random images - inspect it by eye.
"""
import argparse

import cv2
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="qc.png")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    m = pd.read_csv(a.manifest)
    stats = []
    for r in m.to_dict("records"):
        mk = cv2.imread(f"{a.root}/{r['mask_path']}", cv2.IMREAD_UNCHANGED)
        if mk is None:
            stats.append((np.nan,) * 4)
            continue
        disc, cup = mk >= 1, mk == 2
        ys, xs = np.nonzero(disc)
        ext = lambda b: (np.ptp(np.nonzero(b)[0]) + 1) if b.any() else 0
        stats.append((disc.mean(), ext(cup) / ext(disc) if disc.any() and cup.any() else np.nan,
                      xs.mean() / mk.shape[1] if xs.size else np.nan, ys.mean() / mk.shape[0] if ys.size else np.nan))
    s = pd.DataFrame(stats, columns=["disc_area_frac", "vcdr", "cx", "cy"])
    print(f"images: {len(m)}  | no disc: {s.disc_area_frac.isna().mean()+(s.disc_area_frac==0).mean():.1%}"
          f"  | no cup: {s.vcdr.isna().mean():.1%}")
    print(s.describe().round(3).loc[["mean", "std", "min", "25%", "50%", "75%", "max"]].to_string())
    idx = np.random.default_rng(a.seed).choice(len(m), min(a.n, len(m)), replace=False)
    tiles = []
    for i in idx:
        r = m.iloc[i]
        im = cv2.imread(f"{a.root}/{r['image_path']}")
        mk = cv2.imread(f"{a.root}/{r['mask_path']}", cv2.IMREAD_UNCHANGED)
        if im is None or mk is None:
            continue
        mk = cv2.resize(mk, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        for val, col in ((1, (0, 255, 0)), (2, (255, 80, 0))):
            cs, _ = cv2.findContours((mk >= val).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, cs, -1, col, max(2, im.shape[1] // 250))
        tiles.append(cv2.resize(im, (320, 320)))
    while len(tiles) % a.cols:
        tiles.append(np.zeros((320, 320, 3), np.uint8))
    rows = [np.hstack(tiles[k:k + a.cols]) for k in range(0, len(tiles), a.cols)]
    cv2.imwrite(a.out, np.vstack(rows))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
