#!/usr/bin/env python3
"""Build manifest.csv for the Kaggle 'glaucoma-datasets' folder (G1020, ORIGA, REFUGE).

Run from the folder that contains G1020/, ORIGA/, REFUGE/ :
    python build_manifest.py --root . --out manifest.csv

It also prints diagnostics (mask values per dataset, REFUGE label source, class counts);
send that output back if anything looks wrong.
Images and masks are paired by file stem: Images/<stem>.jpg <-> Masks/<stem>.png.
Full-size Images/Masks are used (not *_Cropped / *_Square), so the retina outside the
disc is available as the colour reference.
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def pair(img_dir: Path, mask_dir: Path):
    masks = {p.stem: p for p in mask_dir.glob("*.png")}
    for p in sorted(img_dir.glob("*.jpg")):
        if p.stem in masks:
            yield p.stem, p, masks[p.stem]


def rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()


def g1020(root: Path):
    d = root / "G1020"
    lab = pd.read_csv(d / "G1020.csv")
    lab["stem"] = lab["imageID"].str.replace(r"\.jpg$", "", regex=True)
    lab = dict(zip(lab["stem"], lab["binaryLabels"]))
    rows = []
    for stem, ip, mp in pair(d / "Images", d / "Masks"):
        if stem in lab:
            rows.append(dict(id=f"G1020_{stem}", dataset="G1020", label=int(lab[stem]), patient_id=f"G1020_{stem}",
                             image_path=rel(ip, root), mask_path=rel(mp, root)))
    return rows


def origa(root: Path):
    d = root / "ORIGA"
    lab = pd.read_csv(d / "OrigaList.csv")
    lab.columns = [c.strip() for c in lab.columns]
    lab["stem"] = lab["Filename"].astype(str).str.strip().str.replace(r"\.jpg$", "", regex=True)
    info = dict(zip(lab["stem"], zip(lab["Glaucoma"], lab["ExpCDR"])))
    rows = []
    for stem, ip, mp in pair(d / "Images", d / "Masks"):
        if stem in info:
            rows.append(dict(id=f"ORIGA_{stem}", dataset="ORIGA", label=int(info[stem][0]), patient_id=f"ORIGA_{stem}",
                             expert_cdr=float(info[stem][1]), image_path=rel(ip, root), mask_path=rel(mp, root)))
    return rows


def refuge_labels(root: Path):
    """Try to read labels from the json files; return {stem: label} and print what was found."""
    labels = {}
    for split in ("train", "val", "test"):
        for jp in (root / "REFUGE" / split).glob("*.json"):
            txt = jp.read_text(encoding="utf-8", errors="ignore")
            print(f"[REFUGE/{split}] {jp.name} head: {txt[:300]!r}")
            try:
                obj = json.loads(txt)
            except Exception:
                continue
            items = obj.items() if isinstance(obj, dict) else enumerate(obj if isinstance(obj, list) else [])
            for k, v in items:
                name, val = None, None
                if isinstance(v, dict):
                    name = v.get("ImgName") or v.get("image") or v.get("name") or v.get("ImageName") or (k if isinstance(k, str) else None)
                    val = v.get("Label") if "Label" in v else v.get("label", v.get("Glaucoma", v.get("glaucoma")))
                elif isinstance(v, (int, float)) and isinstance(k, str):
                    name, val = k, v
                if name is not None and val is not None:
                    try:
                        labels[Path(str(name)).stem] = int(val)
                    except (TypeError, ValueError):
                        pass
    return labels


def refuge(root: Path):
    d = root / "REFUGE"
    labels = refuge_labels(root)
    print(f"[REFUGE] labels parsed from json: {len(labels)}")
    rows = []
    for split in ("train", "val", "test"):
        for stem, ip, mp in pair(d / split / "Images", d / split / "Masks"):
            lab = labels.get(stem)
            if lab is None:  # original REFUGE train naming: g0001 = glaucoma, n0001 = normal
                m = re.match(r"^([gn])\d+", stem, re.I)
                lab = (1 if m.group(1).lower() == "g" else 0) if m else np.nan
            rows.append(dict(id=f"REFUGE_{split}_{stem}", dataset="REFUGE", label=lab, patient_id=f"REFUGE_{split}_{stem}",
                             split=split, image_path=rel(ip, root), mask_path=rel(mp, root)))
    return rows


def mask_report(df, root, n=5):
    for ds, g in df.groupby("dataset"):
        out = []
        for p in g["mask_path"].head(n):
            m = cv2.imread(str(root / p), cv2.IMREAD_UNCHANGED)
            out.append((None if m is None else np.unique(m).tolist(), None if m is None else int(m[0, 0])))
        print(f"[{ds}] mask unique values / corner pixel (first {n}): {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="manifest.csv")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    rows = []
    for fn in (g1020, origa, refuge):
        try:
            r = fn(root)
            print(f"{fn.__name__}: {len(r)} paired images")
            rows += r
        except Exception as e:
            print(f"{fn.__name__} FAILED: {e!r}")
    df = pd.DataFrame(rows)
    mask_report(df, root)
    print("\nlabel counts (NaN = unknown):")
    print(df.groupby("dataset")["label"].value_counts(dropna=False).to_string())
    df.to_csv(a.out, index=False)
    print(f"\nwrote {len(df)} rows to {a.out}")


if __name__ == "__main__":
    main()
