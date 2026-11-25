#!/usr/bin/env python3
"""
chromatic_profiles.py  (v2)  -  spatial chromatic profiles of the optic nerve head.

v1 reduced each region to means/histograms and lost the spatial order. The book's method
is built for ORDERED data (signals, trajectories). Here the disc is unrolled into
N_DIR angular sectors, and in each sector colour is read along the normalised radius
t = r / R_disc(theta):   bins  [0-.25, .25-.5, .5-.75, .75-1, 1-1.25, 1.25-1.5]
(4 bins inside the disc, 2 peripapillary bins outside). Each sector is therefore an
ordered signal of 6 colour samples.

Feature groups (all rotation-invariant: sectors are aggregated by mean/std[/min/max], so
left/right eye and image orientation do not matter):
  rimg   rim-width profile = (R_disc - R_cup)/R_disc per direction: mean/std/min/max.
         A purely GEOMETRIC competitor (ISNT-like); it is the fair baseline to beat.
  ptraj  H,L,S trajectory: per radial bin, Hx,Hy,L,S of the sector colour (Eq.1.2-1.7)
  pgen2  second-generation chromaticity (Ch.2): the G-channel, L and S trajectories of a
         sector are passed through 3 triangular processors -> Hx,Hy,L,S of the trajectory
  pcon   chromatic distances dHS,dL,dHLS between radial bins (centre vs rim, rim vs outside)

Usage:
  python chromatic_profiles.py --manifest manifest_labeled.csv --root . --out features_prof.csv
Then:
  python lodo_eval.py --features features.csv --extra features_prof.csv ...
"""
import argparse
import warnings
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import chromatic_features as cf

N_DIR = 16
T_EDGES = np.array([0, .25, .5, .75, 1.0, 1.25, 1.5])
K = len(T_EDGES) - 1
warnings.filterwarnings("ignore")


def ray_extents(mask, cx, cy, thetas, rmax):
    """Distance from (cx,cy) to the last True pixel of `mask` along each direction."""
    r = np.arange(0, rmax, 0.5)
    xs = np.clip(np.round(cx + r[None, :] * np.cos(thetas)[:, None]).astype(int), 0, mask.shape[1] - 1)
    ys = np.clip(np.round(cy + r[None, :] * np.sin(thetas)[:, None]).astype(int), 0, mask.shape[0] - 1)
    hit = mask[ys, xs]
    last = hit.shape[1] - 1 - np.argmax(hit[:, ::-1], axis=1)
    return np.where(hit.any(1), r[last], 0.0)


def agg(out, name, arr, extra=False):
    arr = np.asarray(arr, float)
    ok = np.isfinite(arr)
    if ok.sum() < 3:
        vals = [np.nan] * (4 if extra else 2)
    else:
        a = arr[ok]
        vals = [a.mean(), a.std()] + ([a.min(), a.max()] if extra else [])
    for suf, v in zip(("mean", "std", "min", "max"), vals):
        out[f"{name}_{suf}"] = float(v)


def profile_features(img_n, disc, cup, P):
    out = {}
    ys, xs = np.nonzero(disc)
    cx, cy = xs.mean(), ys.mean()
    thetas = (np.arange(N_DIR) + 0.5) * 2 * np.pi / N_DIR
    rmax = 3.0 * np.sqrt(disc.sum() / np.pi)
    Rd = ray_extents(disc, cx, cy, thetas, rmax)
    Rc = ray_extents(cup, cx, cy, thetas, rmax) if cup.sum() >= 30 else np.zeros(N_DIR)
    Rd = np.maximum(Rd, 2.0)
    agg(out, "rimg__rimwidth", (Rd - np.minimum(Rc, Rd)) / Rd, extra=True)
    # disc radius per direction relative to its mean: the std over sectors is the radius variability (eccentricity proxy)
    agg(out, "rimg__discradius_cv", Rd / Rd.mean(), extra=False)

    # pixel -> (sector, radial bin)
    h, w = disc.shape
    win = int(rmax * 0.5 * 1.6)
    x0, x1 = max(0, int(cx) - win), min(w, int(cx) + win)
    y0, y1 = max(0, int(cy) - win), min(h, int(cy) + win)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - cx, yy - cy
    sector = (np.mod(np.arctan2(dy, dx), 2 * np.pi) / (2 * np.pi) * N_DIR).astype(int) % N_DIR
    t = np.hypot(dx, dy) / Rd[sector]
    ok = t < T_EDGES[-1]
    kbin = np.clip(np.digitize(t, T_EDGES) - 1, 0, K - 1)
    idx = (sector * K + kbin)[ok]
    cnt = np.bincount(idx, minlength=N_DIR * K).astype(float)
    traj = np.stack([np.bincount(idx, weights=img_n[y0:y1, x0:x1, c][ok], minlength=N_DIR * K)
                     for c in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    traj[cnt < 8] = np.nan
    traj = traj.reshape(N_DIR, K, 3)  # sector, bin, RGB

    H, L, S = cf.chromatic_hls(traj[..., 0], traj[..., 1], traj[..., 2])
    Hx, Hy = cf.hs_cartesian(H, S)
    for k in range(K):
        for nm, v in (("L", L[:, k]), ("S", S[:, k]), ("Hx", Hx[:, k]), ("Hy", Hy[:, k])):
            agg(out, f"ptraj__{nm}_b{k}", v)

    # second generation: each sector trajectory (6 ordered samples) -> 3 triangular processors
    g2 = {nm: np.full((N_DIR, 4), np.nan) for nm in ("G", "L", "S")}
    for j in range(N_DIR):
        for nm, sig in (("G", traj[j, :, 1]), ("L", L[j]), ("S", S[j])):
            if np.isfinite(sig).all() and sig.max() > 0:
                hh, ll, ss = cf.chromatic_hls(*(P @ (sig / sig.max())))
                hx, hy = cf.hs_cartesian(hh, ss)
                g2[nm][j] = [hx, hy, ll, ss]
    for nm, a in g2.items():
        for i, comp in enumerate(("Hx", "Hy", "L", "S")):
            agg(out, f"pgen2__{nm}_{comp}", a[:, i], extra=True)

    # chromatic distances between radial bins (Sec 8.4-style: suspect vs reference zone)
    for a, b, nm in ((0, 2, "c02"), (2, 4, "c24"), (1, 3, "c13")):
        dHS, dL, dHLS = cf.chromatic_distance(H[:, a], L[:, a], S[:, a], H[:, b], L[:, b], S[:, b])
        for comp, v in (("dHS", dHS), ("dL", dL), ("dHLS", dHLS)):
            agg(out, f"pcon__{nm}_{comp}", v, extra=True)
    return out


def extract_one(row, cfg):
    root = Path(cfg["root"])
    img = cf.read_rgb(root / row["image_path"])
    ms = cfg["max_side"]
    if ms and max(img.shape[:2]) > ms:
        f = ms / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1] * f), round(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    if isinstance(row.get("mask_path"), str) and row["mask_path"]:
        disc, cup = cf.split_masks(cf.read_mask(root / row["mask_path"], (h, w)), cfg["convention"])
    else:
        disc, cup = cf.auto_disc_cup(img)
    cup = cup & disc
    if int(disc.sum()) < 100:
        raise ValueError("no optic disc")
    rim = disc & ~cup
    ref = cf.retina_reference(img, disc, cf.fov_mask(img), rim)
    img_n = np.clip(img / (2.0 * ref[None, None, :] + cf.EPS), 0, 1) if cfg["ref_norm"] == "retina" else img
    P = cf.make_processors(K, "triangular")
    out = {k: row.get(k) for k in ("id", "dataset", "label", "patient_id") if k in row}
    out.update(profile_features(img_n, disc, cup, P))
    return out


def _work(a):
    try:
        return extract_one(*a)
    except Exception as e:
        return {"id": a[0].get("id"), "error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--convention", default="auto", choices=["auto", "refuge", "refuge_inv", "labels"])
    ap.add_argument("--ref-norm", dest="ref_norm", default="retina", choices=["none", "retina"])
    ap.add_argument("--max-side", dest="max_side", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    cfg = dict(root=a.root, convention=a.convention, ref_norm=a.ref_norm, max_side=a.max_side)
    jobs = [(r, cfg) for r in pd.read_csv(a.manifest).to_dict("records")]
    res = Pool(a.workers).map(_work, jobs, chunksize=8) if a.workers > 1 else [_work(j) for j in jobs]
    ok = [r for r in res if "error" not in r]
    bad = [r for r in res if "error" in r]
    pd.DataFrame(ok).to_csv(a.out, index=False)
    print(f"wrote {len(ok)} rows, {a.out}; {len(bad)} failed", *bad[:5], sep="\n  ")


if __name__ == "__main__":
    main()
