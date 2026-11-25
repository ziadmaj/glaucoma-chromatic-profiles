#!/usr/bin/env python3
"""
chromatic_features.py
=====================
Chromatic (H, L, S) signatures of the optic disc / cup / rim in fundus images,
following Jones, Deakin & Spencer, "Chromatic Monitoring of Complex Conditions"
(CRC Press, 2008):

  * Ch.1  Eq. 1.2-1.7   H, L, S transform of three non-orthogonal processor outputs
          Eq. 1.8-1.10  chromatic distance dHS, dL, dHLS
          Eq. 1.13      chromatic vector C_T
  * Ch.2  Sec. 2.5      discrete data processed by triangular R,G,B processors
  * Ch.3  Eq. 3.4-3.6   x,y,z   and   Eq. 3.26-3.28  Lab with a reference illuminant
  * Ch.8  Sec. 8.4      ratio spectra (suspect / healthy)  ->  cup / rim here

Notes on equations whose text is garbled in the extracted book:
  * Eq. 1.11 (distance when a hue is undefined) is implemented as dHS = |Si - Sr|.
  * Eq. 1.13 is implemented as
        C_T = sqrt( aH*HN^2 + aL*(bL-LN)^2 + aS*(bS-SN)^2 ),
    with bL=1, bS=0 (the setting the book uses when dH/dL<0 and dH/dS>0).
    Check both against the printed book before relying on them in a paper.

Sub-commands
------------
  inspect   print a folder summary and the pixel values found in mask images
  extract   manifest CSV -> features CSV
  selftest  generate a synthetic 3-"camera" data set and run the whole pipeline

Manifest CSV columns
--------------------
  image_path                       required
  mask_path                        one mask with disc+cup (see --convention), OR
  disc_mask_path, cup_mask_path    two binary masks
  dataset                          source name (ORIGA / REFUGE / G1020 ...)
  label                            0 = normal, 1 = glaucoma (optional)
  patient_id, id                   optional

Feature columns are named  <group>__<region>_<name>  with groups
  geom   vCDR / area CDR from the masks (structural baseline)
  stats  plain mean/std of R,G,B per region (colour baseline)
  direct H,L,S of the region's mean R,G,B (Eq. 1.2-1.7)
  xyz    x,y,z chromaticity of the region mean (Eq. 3.4-3.6)
  lab    Lab with the retina reference as illuminant (Eq. 3.26-3.28)
  hist   H,L,S of the intensity histogram of each channel through R,G,B
         triangular processors (Sec. 2.5)
  rel    cup-vs-rim relations: dHS, dL, dHLS, Lab distance, mean ratios and
         chromatic ratio spectra (Sec. 8.4)
Columns ending in _H hold the raw hue (NaN where undefined); the models should
use the continuous _Hx = S cos(H), _Hy = S sin(H) instead.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

EPS = 1e-9
H_UNDEFINED = -199.0  # sentinel used in the book (Sec. 1.3.2a)


# --------------------------------------------------------------------------
# 1. Chromatic processors and transforms
# --------------------------------------------------------------------------
def make_processors(n_bins: int, profile: str = "triangular") -> np.ndarray:
    """Three overlapping processors R,G,B over n_bins discrete data points.

    Centres at the first, middle and last bin; neighbouring processors cross
    at half height. Triangular processors sum to 1 everywhere (linear H scale,
    Sec. 2.2); Gaussian ones give a non-uniform, tunable sensitivity.
    """
    if n_bins < 3:
        raise ValueError("need at least 3 bins")
    x = np.arange(n_bins, dtype=np.float64)
    half = (n_bins - 1) / 2.0
    centres = np.array([0.0, half, 2 * half])
    if profile == "triangular":
        return np.clip(1.0 - np.abs(x[None, :] - centres[:, None]) / half, 0.0, 1.0)
    if profile == "gaussian":
        sigma = (half / 2.0) / np.sqrt(2.0 * np.log(2.0))
        return np.exp(-0.5 * ((x[None, :] - centres[:, None]) / sigma) ** 2)
    raise ValueError(f"unknown profile {profile!r}")


def chromatic_hls(R, G, B, s_min: float = 1e-6):
    """Eq. 1.2-1.7. Returns H (degrees, H_UNDEFINED if S~0), L, S."""
    R, G, B = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (R, G, B)))
    mn = np.minimum(np.minimum(R, G), B)
    mx = np.maximum(np.maximum(R, G), B)
    r, g, b = R - mn, G - mn, B - mn
    L = (R + G + B) / 3.0
    S = (mx - mn) / (mx + mn + EPS)
    H = np.where(
        r == 0,
        240.0 - 120.0 * g / (g + b + EPS),
        np.where(g == 0, 360.0 - 120.0 * b / (b + r + EPS), 120.0 - 120.0 * r / (r + g + EPS)),
    )
    H = np.where(S < s_min, H_UNDEFINED, H)
    return H, L, S


def hs_cartesian(H, S):
    """Continuous (x, y) = (S cos H, S sin H); undefined hue maps to the origin."""
    H = np.asarray(H, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    ok = H > H_UNDEFINED + 1
    rad = np.deg2rad(np.where(ok, H, 0.0))
    return np.where(ok, S * np.cos(rad), 0.0), np.where(ok, S * np.sin(rad), 0.0)


def chromatic_distance(Hr, Lr, Sr, Hi, Li, Si):
    """Eq. 1.8-1.10 (dHS, dL, dHLS) with the undefined-hue rule of Eq. 1.11/1.12."""
    xr, yr = hs_cartesian(Hr, Sr)
    xi, yi = hs_cartesian(Hi, Si)
    dHS = np.hypot(xr - xi, yr - yi)
    undefined = (np.asarray(Hr) <= H_UNDEFINED + 1) | (np.asarray(Hi) <= H_UNDEFINED + 1)
    dHS = np.where(undefined, np.abs(np.asarray(Si) - np.asarray(Sr)), dHS)
    dL = np.abs(np.asarray(Li) - np.asarray(Lr))
    return dHS, dL, np.hypot(dHS, dL)


def chromatic_vector_CT(H, L, S, H0=360.0, L0=1.0, S0=1.0, aH=1.0, aL=1.0, aS=1.0, bL=1.0, bS=0.0):
    """Eq. 1.13 (see module docstring for the interpretation used)."""
    H = np.asarray(H, dtype=np.float64)
    Hn = np.where(H > H_UNDEFINED + 1, H / H0, 0.0)
    return np.sqrt(aH * Hn**2 + aL * (bL - np.asarray(L) / L0) ** 2 + aS * (bS - np.asarray(S) / S0) ** 2)


def xyz_chromaticity(R, G, B):
    """Eq. 3.4-3.6."""
    tot = R + G + B + EPS
    return R / tot, G / tot, B / tot


def lab_with_reference(R, G, B, ref):
    """Eq. 3.26-3.28 with the CIE linear branch below 0.008856."""

    def f(t):
        return np.where(t > 0.008856, np.cbrt(np.maximum(t, 0.0)), 7.787 * t + 16.0 / 116.0)

    fr, fg, fb = f(R / (ref[0] + EPS)), f(G / (ref[1] + EPS)), f(B / (ref[2] + EPS))
    return 116.0 * fg - 16.0, 500.0 * (fr - fg), 200.0 * (fg - fb)


# --------------------------------------------------------------------------
# 2. Image / mask handling
# --------------------------------------------------------------------------
def read_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def read_mask(path, shape_hw) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"cannot read mask {path}")
    if m.ndim == 3:
        m = m[..., :3].mean(-1)
    h, w = shape_hw
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return m


def split_masks(m: np.ndarray, convention: str):
    """Return (disc, cup) boolean masks (disc includes the cup)."""
    if convention == "auto":
        # background = the corner pixel; of the other two values the extreme one next to the
        # background direction is the cup (bg=255: cup is the lowest; bg=0: cup is the highest)
        bg = m[0, 0]
        others = sorted(int(v) for v in np.unique(m) if v != bg)
        disc = m != bg
        if len(others) >= 2:
            cupv = others[0] if bg > others[-1] else others[-1]
            return disc, m == cupv
        return disc, np.zeros_like(disc)
    if convention == "refuge":  # cup=0, disc=128, background=255
        cup, disc = m < 64, m < 192
    elif convention == "refuge_inv":  # cup=255, disc=128, background=0
        cup, disc = m >= 192, m >= 64
    elif convention == "labels":  # background=0, disc=1, cup=2
        cup, disc = m == 2, m >= 1
    else:
        raise ValueError(f"unknown mask convention {convention!r}")
    return disc, cup


def fov_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(img.mean(-1), (0, 0), 3)
    fov = (gray > 0.08).astype(np.uint8)
    k = max(3, int(0.03 * min(img.shape[:2])) | 1)
    return cv2.erode(fov, np.ones((k, k), np.uint8)) > 0


def retina_reference(img, disc, fov, rim) -> np.ndarray:
    """Per-channel median colour of the retina outside the disc (illuminant proxy).
    Fallback chain for disc-centred crops: rim -> whole image."""
    k = max(5, int(0.08 * min(img.shape[:2])) | 1)
    near_disc = cv2.dilate(disc.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    retina = fov & ~near_disc
    for region in (retina, rim, np.ones(disc.shape, bool)):
        if region.sum() >= 500:
            return np.median(img[region], axis=0)
    return img.reshape(-1, 3).mean(0)


def _largest_cc(m: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(m, bool)
    return lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def auto_disc_cup(img: np.ndarray):
    """Mask-free fallback (e.g. HYGD has no OD/OC masks). Heuristic, NOT a segmenter:
      disc = Otsu-bright region around the brightest blob (top 1.5% of blurred R+G), convex hull
      cup  = brightest 30% of the disc (largest component)
    Disc location is usually fine on disc-centred images; the cup is only a
    brightness proxy, so its geometry (vCDR) must not be reported as a CDR.
    """
    fov = fov_mask(img)
    s = max(3.0, 0.015 * min(img.shape[:2]))
    lum = cv2.GaussianBlur((img[..., 0] + img[..., 1]) / 2.0, (0, 0), s)
    lum = np.where(fov, lum, 0)
    thr = np.quantile(lum[fov], 0.985)
    blob = _largest_cc(lum >= thr)
    if blob.sum() < 50:
        return np.zeros_like(fov), np.zeros_like(fov)
    cy, cx = np.argwhere(blob).mean(0)
    rad = int(0.16 * min(img.shape[:2]))
    yy, xx = np.ogrid[: lum.shape[0], : lum.shape[1]]
    win = fov & ((yy - cy) ** 2 + (xx - cx) ** 2 <= rad**2)
    vals = np.clip(lum[win] * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = _largest_cc(win & (lum * 255 >= t))
    if bright.sum() < 100:
        bright = blob
    hull = cv2.convexHull(np.column_stack(np.where(bright)[::-1]).astype(np.int32))
    disc = np.zeros(blob.shape, np.uint8)
    cv2.fillConvexPoly(disc, hull, 1)
    disc = disc > 0
    lum_d = lum[disc]
    cup = _largest_cc(disc & (lum >= np.quantile(lum_d, 0.70)))
    return disc, cup


def extent(mask: np.ndarray, axis: int) -> int:
    idx = np.where(mask.any(axis=1 - axis))[0]
    return int(idx.max() - idx.min() + 1) if idx.size else 0


# --------------------------------------------------------------------------
# 3. Feature extraction
# --------------------------------------------------------------------------
def _region_signature(pix_n, pix_raw, ref, P, n_bins, region):
    """Per-region features. pix_n: (possibly reference-normalised) pixels in [0,1];
    pix_raw: original pixels (used by Lab, which has its own reference)."""
    d, internal = {}, {}
    mean = pix_n.mean(0)
    std = pix_n.std(0)
    for i, c in enumerate("RGB"):
        d[f"stats__{region}_mean{c}"] = mean[i]
        d[f"stats__{region}_std{c}"] = std[i]

    H, L, S = chromatic_hls(*mean)
    Hx, Hy = hs_cartesian(H, S)
    d.update({f"direct__{region}_H": float(H) if H > H_UNDEFINED + 1 else np.nan,
              f"direct__{region}_Hx": float(Hx), f"direct__{region}_Hy": float(Hy),
              f"direct__{region}_L": float(L), f"direct__{region}_S": float(S),
              f"direct__{region}_CT": float(chromatic_vector_CT(H, L, S))})
    internal["direct"] = (H, L, S)

    x, y, z = xyz_chromaticity(*mean)
    d.update({f"xyz__{region}_x": float(x), f"xyz__{region}_y": float(y)})

    Ll, a, b = lab_with_reference(*pix_raw.mean(0), ref)
    d.update({f"lab__{region}_L": float(Ll), f"lab__{region}_a": float(a),
              f"lab__{region}_b": float(b), f"lab__{region}_C": float(np.hypot(a, b))})
    internal["lab"] = (Ll, a, b)

    dens = {}
    for i, c in enumerate("RGB"):
        hist, _ = np.histogram(np.clip(pix_n[:, i], 0, 1), bins=n_bins, range=(0, 1))
        hist = hist.astype(np.float64)
        dens[c] = hist / (hist.sum() + EPS)
        peak_norm = hist / (hist.max() + EPS)  # discrete data amplitude scaled to 1 (Sec. 2.5)
        h, l, s = chromatic_hls(*(P @ peak_norm))
        hx, hy = hs_cartesian(h, s)
        d.update({f"hist__{region}_{c}_H": float(h) if h > H_UNDEFINED + 1 else np.nan,
                  f"hist__{region}_{c}_Hx": float(hx), f"hist__{region}_{c}_Hy": float(hy),
                  f"hist__{region}_{c}_L": float(l), f"hist__{region}_{c}_S": float(s)})
    internal["dens"] = dens
    internal["mean"] = mean
    return d, internal


def extract_one(row: dict, cfg: dict) -> dict:
    root = Path(cfg["root"]) if cfg.get("root") else Path(".")
    img = read_rgb(root / row["image_path"])
    ms = cfg.get("max_side") or 0
    if ms and max(img.shape[:2]) > ms:
        f = ms / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1] * f), round(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    mask_source = "given"

    def has(key):
        return isinstance(row.get(key), str) and row[key] != ""

    if has("disc_mask_path"):
        disc = read_mask(root / row["disc_mask_path"], (h, w)) > 127
        cup = read_mask(root / row["cup_mask_path"], (h, w)) > 127 if has("cup_mask_path") else np.zeros_like(disc)
    elif has("mask_path"):
        disc, cup = split_masks(read_mask(root / row["mask_path"], (h, w)), cfg["convention"])
    else:
        disc, cup = auto_disc_cup(img)
        mask_source = "auto"
    cup = cup & disc
    if int(disc.sum()) < 100:
        raise ValueError("no optic disc found")
    rim = disc & ~cup

    ref = retina_reference(img, disc, fov_mask(img), rim)
    img_n = np.clip(img / (2.0 * ref[None, None, :] + EPS), 0, 1) if cfg["ref_norm"] == "retina" else img
    P = make_processors(cfg["bins"], cfg["profile"])

    out = {k: row.get(k) for k in ("id", "dataset", "label", "patient_id", "image_path") if k in row}
    cup_ok = int(cup.sum()) >= 30
    dv, cv_ = extent(disc, 0), extent(cup, 0)
    dh, ch = extent(disc, 1), extent(cup, 1)
    out.update({
        "mask_source": mask_source,
        "geom__cup_present": int(cup_ok),
        "geom__vcdr": cv_ / dv if dv and cup_ok else np.nan,
        "geom__hcdr": ch / dh if dh and cup_ok else np.nan,
        "geom__area_cdr": cup.sum() / max(1, disc.sum()) if cup_ok else np.nan,
    })

    internals = {}
    dummy = np.random.default_rng(0).random((100, 3))
    for name, m in (("disc", disc), ("cup", cup), ("rim", rim)):
        if int(m.sum()) >= 30:
            d, it = _region_signature(img_n[m], img[m], ref, P, cfg["bins"], name)
            internals[name] = it
        else:
            d, _ = _region_signature(dummy, dummy, np.ones(3), P, cfg["bins"], name)
            d = {k: np.nan for k in d}
        out.update(d)

    # cup-vs-rim relations (Sec. 8.4: ratio of the suspect spectrum to the healthy one)
    rel_names = ["dHS", "dL", "dHLS", "dLab", "ratioMeanR", "ratioMeanG", "ratioMeanB"]
    rel_names += [f"rs{c}_{k}" for c in "RGB" for k in ("Hx", "Hy", "L", "S")]
    rel = {f"rel__cup_vs_rim_{k}": np.nan for k in rel_names}
    if "cup" in internals and "rim" in internals:
        c_, r_ = internals["cup"], internals["rim"]
        dHS, dL, dHLS = chromatic_distance(*r_["direct"], *c_["direct"])
        rel["rel__cup_vs_rim_dHS"], rel["rel__cup_vs_rim_dL"], rel["rel__cup_vs_rim_dHLS"] = map(float, (dHS, dL, dHLS))
        rel["rel__cup_vs_rim_dLab"] = float(np.linalg.norm(np.array(c_["lab"]) - np.array(r_["lab"])))
        for i, c in enumerate("RGB"):
            rel[f"rel__cup_vs_rim_ratioMean{c}"] = float(c_["mean"][i] / (r_["mean"][i] + EPS))
            eps = 1.0 / cfg["bins"] * 1e-2
            rs = (c_["dens"][c] + eps) / (r_["dens"][c] + eps)
            hh, ll, ss = chromatic_hls(*(P @ (rs / rs.max())))
            hx, hy = hs_cartesian(hh, ss)
            rel.update({f"rel__cup_vs_rim_rs{c}_Hx": float(hx), f"rel__cup_vs_rim_rs{c}_Hy": float(hy),
                        f"rel__cup_vs_rim_rs{c}_L": float(ll), f"rel__cup_vs_rim_rs{c}_S": float(ss)})
    out.update(rel)
    out["ref_R"], out["ref_G"], out["ref_B"] = (float(v) for v in ref)  # useful as a camera fingerprint
    return out


def _work(args):
    row, cfg = args
    try:
        return extract_one(row, cfg)
    except Exception as e:  # keep going, report at the end
        return {"id": row.get("id"), "image_path": row.get("image_path"), "error": repr(e)}


def cmd_extract(a):
    man = pd.read_csv(a.manifest)
    cfg = dict(root=a.root, max_side=a.max_side, convention=a.convention, bins=a.bins, profile=a.profile, ref_norm=a.ref_norm)
    jobs = [(r, cfg) for r in man.to_dict("records")]
    if a.workers > 1:
        with Pool(a.workers) as p:
            res = p.map(_work, jobs, chunksize=8)
    else:
        res = [_work(j) for j in jobs]
    ok = [r for r in res if "error" not in r]
    bad = [r for r in res if "error" in r]
    pd.DataFrame(ok).to_csv(a.out, index=False)
    print(f"wrote {len(ok)} rows to {a.out}; {len(bad)} failed")
    for r in bad[:5]:
        print("  failed:", r)
    if bad:
        pd.DataFrame(bad).to_csv(str(a.out) + ".errors.csv", index=False)


# --------------------------------------------------------------------------
# 4. inspect
# --------------------------------------------------------------------------
def cmd_inspect(a):
    root = Path(a.root)
    by_dir = defaultdict(lambda: defaultdict(int))
    files = [p for p in root.rglob("*") if p.is_file()]
    for p in files:
        by_dir[str(p.parent.relative_to(root))][p.suffix.lower()] += 1
    print(f"{len(files)} files under {root}")
    for d, ex in sorted(by_dir.items())[:60]:
        print(f"  {d or '.':60s} {dict(ex)}")
    for p in [p for p in files if p.suffix.lower() == ".csv"][:5]:
        print(f"\nCSV {p.relative_to(root)}:")
        print(pd.read_csv(p, nrows=5).to_string())
    keys = ("mask", "seg", "gt", "label", "cup", "disc", "od", "oc")
    cand = [p for p in files if p.suffix.lower() in (".png", ".jpg", ".bmp", ".tif", ".tiff")
            and any(k in str(p.parent).lower() or k in p.stem.lower() for k in keys)]
    print(f"\n{len(cand)} mask-like images; pixel values in the first {min(5, len(cand))}:")
    for p in cand[:5]:
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        vals = np.unique(m) if m is not None else None
        print(f"  {p.relative_to(root)} shape={None if m is None else m.shape} "
              f"values={None if vals is None else vals[:12].tolist()}")
    print("\nPick --convention: refuge (0/128/255), refuge_inv (255/128/0), labels (0/1/2), "
          "or use disc_mask_path + cup_mask_path columns for separate binary masks.")


# --------------------------------------------------------------------------
# 5. synthetic self-test (checks the code runs; says nothing about real data)
# --------------------------------------------------------------------------
def _synthetic(workdir: Path, n_per_set: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    cams = {"CamA": ((1.0, 1.0, 1.0), 1.0), "CamB": ((1.25, 0.95, 0.65), 0.9), "CamC": ((0.8, 1.0, 1.3), 1.1)}
    S = 256
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    rows = []
    (workdir / "img").mkdir(parents=True, exist_ok=True)
    (workdir / "msk").mkdir(parents=True, exist_ok=True)
    for ds, (gain, gamma) in cams.items():
        for i in range(n_per_set):
            label = int(rng.random() < 0.4)
            fov = (xx - S / 2) ** 2 + (yy - S / 2) ** 2 < 120**2
            vign = 1 - 0.3 * (((xx - S / 2) ** 2 + (yy - S / 2) ** 2) / 120**2)
            img = np.zeros((S, S, 3), np.float32)
            img[:] = np.array([0.70, 0.30, 0.12]) * vign[..., None]
            cx, cy = rng.uniform(80, 176), rng.uniform(80, 176)
            ra = rng.uniform(26, 34)
            disc = ((xx - cx) / ra) ** 2 + ((yy - cy) / (ra * 1.1)) ** 2 < 1
            q = rng.uniform(0.55, 0.8) if label else rng.uniform(0.3, 0.5)
            cup = ((xx - cx) / (ra * q * 0.9)) ** 2 + ((yy - cy) / (ra * 1.1 * q)) ** 2 < 1
            img[disc] = np.array([0.92, 0.62 + 0.04 * label, 0.35])
            img[cup] = np.array([0.98, 0.86, 0.62])
            img += rng.normal(0, 0.02, img.shape).astype(np.float32)
            img = np.clip(np.clip(img * np.array(gain, np.float32), 0, 1) ** gamma, 0, 1) * fov[..., None]
            m = np.full((S, S), 255, np.uint8)
            m[disc] = 128
            m[cup] = 0
            fid = f"{ds}_{i:04d}"
            cv2.imwrite(str(workdir / "img" / f"{fid}.png"), cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(workdir / "msk" / f"{fid}.png"), m)
            rows.append(dict(id=fid, dataset=ds, label=label, patient_id=fid,
                             image_path=f"img/{fid}.png", mask_path=f"msk/{fid}.png"))
    pd.DataFrame(rows).to_csv(workdir / "manifest.csv", index=False)
    return workdir / "manifest.csv"


def cmd_selftest(a):
    wd = Path(a.workdir)
    man = _synthetic(wd)
    for rn in ("none", "retina"):
        ns = argparse.Namespace(manifest=man, out=wd / f"features_refnorm_{rn}.csv", root=str(wd),
                                convention="refuge", max_side=0, bins=16, profile="triangular", ref_norm=rn, workers=1)
        cmd_extract(ns)
    df = pd.read_csv(wd / "features_refnorm_retina.csv")
    print(f"{df.shape[1]} columns; groups:",
          sorted({c.split('__')[0] for c in df.columns if '__' in c}))
    print("Now run:  python lodo_eval.py --features", wd / "features_refnorm_none.csv",
          "   and   --features", wd / "features_refnorm_retina.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect"); p.add_argument("--root", required=True); p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("extract")
    p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--root", default=None, help="prefix for the paths in the manifest")
    p.add_argument("--convention", default="auto", choices=["auto", "refuge", "refuge_inv", "labels"])
    p.add_argument("--bins", type=int, default=16)
    p.add_argument("--profile", default="triangular", choices=["triangular", "gaussian"])
    p.add_argument("--ref-norm", dest="ref_norm", default="retina", choices=["none", "retina"],
                   help="divide pixels by the retina median colour before stats/direct/xyz/hist")
    p.add_argument("--max-side", dest="max_side", type=int, default=1024,
                   help="downscale images whose longer side exceeds this (0 = off); G1020 is ~3000 px")
    p.add_argument("--workers", type=int, default=1); p.set_defaults(fn=cmd_extract)
    p = sub.add_parser("selftest"); p.add_argument("--workdir", required=True); p.set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
