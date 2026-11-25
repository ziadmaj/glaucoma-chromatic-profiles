#!/usr/bin/env python3
"""Classical (network-free) optic-disc masks for a dataset with no OD/OC annotation (HYGD).

    python auto_masks.py --hygd-dir hygd --out-dir hygd_auto --limit 30 --manifest-out manifest_hygd_auto_pilot.csv
    python qc_masks.py   --manifest manifest_hygd_auto_pilot.csv --root . --out qc_auto.png --n 30

Masks written as 0 = background, 1 = disc (no cup: the brightness proxy is not a cup).
Labels are carried to the manifest but never used here. --limit uses the same random sample
(seed 0) as segment_hygd.py run --limit, so both pilots show the same images.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import chromatic_features as cf


def detect_disc_v2(img):
    """Contrast-blob disc detector. Green channel (retina is dark in G, disc bright):
    1. difference of Gaussians at the expected disc scale -> compact bright blob, not a hazy bright area
    2. candidates closer than 7% of the FOV diameter to the FOV edge are ignored (glare crescents)
    3. boundary = local threshold between the surrounding ring and the blob core, clamped to 1.7 r0
    """
    fov = cf.fov_mask(img)
    h, w = fov.shape
    D = 2.0 * np.sqrt(fov.sum() / np.pi)
    r0 = 0.062 * D
    ff = fov.astype(np.float32)
    g = img[..., 1]

    def nblur(x, s):  # normalised (FOV-aware) Gaussian blur
        return cv2.GaussianBlur(x * ff, (0, 0), s) / np.maximum(cv2.GaussianBlur(ff, (0, 0), s), 1e-3)

    dog = nblur(g, r0 / 1.4) - nblur(g, r0 * 2.2)
    dist = cv2.distanceTransform(fov.astype(np.uint8), cv2.DIST_L2, 5)
    dog[dist < 0.07 * D] = -np.inf
    py, px = np.unravel_index(np.argmax(dog), dog.shape)
    yy, xx = np.ogrid[:h, :w]
    rr = np.hypot(yy - py, xx - px)
    gs = cv2.GaussianBlur(g, (0, 0), max(1.0, r0 / 6))
    ring = (rr > 1.8 * r0) & (rr < 2.5 * r0) & fov
    inner = rr < 0.8 * r0
    if ring.sum() < 50:
        return np.zeros((h, w), bool)
    thr = 0.5 * (np.median(gs[ring]) + np.percentile(gs[inner], 85))
    cand = (gs > thr) & (rr < 1.7 * r0)
    n, lab, _, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros((h, w), bool)
    k = lab[py, px] if lab[py, px] > 0 else 1 + int(np.argmin([np.hypot(*(np.argwhere(lab == i).mean(0) - (py, px))) for i in range(1, n)]))
    blob = lab == k
    pts = np.column_stack(np.where(blob)[::-1]).astype(np.int32)
    if len(pts) < 20:
        return np.zeros((h, w), bool)
    out = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hygd-dir", required=True)
    ap.add_argument("--out-dir", default="hygd_auto")
    ap.add_argument("--manifest-out", default="manifest_hygd_auto.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--method", default="v2", choices=["v1", "v2"])
    ap.add_argument("--seed", type=int, default=0, help="sampling seed for --limit (0 = same images as the first pilot)")
    a = ap.parse_args()
    d = Path(a.hygd_dir)
    labels = pd.read_csv(next(d.rglob("Labels.csv")))
    labels.columns = [c.strip() for c in labels.columns]
    if a.limit:
        labels = labels.sample(a.limit, random_state=a.seed).reset_index(drop=True)
    img_dir = d / "Images" if (d / "Images").is_dir() else next(p for p in d.rglob("Images") if p.is_dir())
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(labels.to_dict("records")):
        stem = Path(str(r["Image Name"])).stem
        ip, mp = img_dir / f"{stem}.jpg", out / f"{stem}.png"
        if not mp.exists():
            img = cf.read_rgb(ip)
            h, w = img.shape[:2]
            f = min(1.0, a.max_side / max(h, w))
            small = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA) if f < 1 else img
            disc = detect_disc_v2(small) if a.method == "v2" else cf.auto_disc_cup(small)[0]
            m = cv2.resize(disc.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(mp), m)
        rows.append(dict(id=f"HYGD_{stem}", dataset="HYGD",
                         label=1 if str(r["Label"]).strip().upper() == "GON+" else 0,
                         patient_id=f"HYGD_{r['Patient']}", quality=r.get("Quality Score", np.nan),
                         image_path=ip.as_posix(), mask_path=mp.as_posix()))
        if i % 50 == 0:
            print(f"{i}/{len(labels)}", flush=True)
    pd.DataFrame(rows).to_csv(a.manifest_out, index=False)
    print("wrote", a.manifest_out, len(rows), "rows")


if __name__ == "__main__":
    main()
