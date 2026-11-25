#!/usr/bin/env python3
"""Optic disc / cup segmentation with models/refuge_segmentation.pth (U-Net, 6 down / 6 up, 2 outputs).

Step 1 - find the right pre-processing on images WITH ground truth (REFUGE val):
    python segment_hygd.py check --manifest manifest_labeled.csv --root . --weights models/refuge_segmentation.pth
  Tries input size {256,512} x normalisation {/255, imagenet} x channel order {disc,cup | cup,disc}
  and prints Dice. This is only a pre-processing sanity check (the network was trained on REFUGE).

Step 2 - segment HYGD (never seen by the network):
    python segment_hygd.py run --hygd-dir <HYGD folder> --out-dir hygd_masks --size 512 --norm 255 --order disc_cup
  Writes masks (0 bg, 1 disc, 2 cup) + manifest_hygd.csv; resumable (skips existing masks).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- architecture (names match the checkpoint) ------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, i, o, mid=None):
        super().__init__()
        mid = mid or o
        self.double_conv = nn.Sequential(
            nn.Conv2d(i, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(i, o))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(i, o, i // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy, dx = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = nn.Conv2d(i, o, 1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1, self.down2, self.down3 = Down(64, 128), Down(128, 256), Down(256, 512)
        self.down4, self.down5, self.down6 = Down(512, 1024), Down(1024, 2048), Down(2048, 2048)
        self.up1, self.up2, self.up3 = Up(4096, 1024), Up(2048, 512), Up(1024, 256)
        self.up4, self.up5, self.up6 = Up(512, 128), Up(256, 64), Up(128, 64)
        self.output_layer = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x5 = self.down4(x4); x6 = self.down5(x5); x7 = self.down6(x6)
        x = self.up1(x7, x6); x = self.up2(x, x5); x = self.up3(x, x4)
        x = self.up4(x, x3); x = self.up5(x, x2); x = self.up6(x, x1)
        return self.output_layer(x)


def load_model(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m = UNet()
    m.load_state_dict(sd, strict=True)  # any naming/shape mismatch raises here
    return m.eval()


# ---- inference ---------------------------------------------------------------------------
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def predict_probs(model, img_rgb01, size, norm):
    h, w = img_rgb01.shape[:2]
    x = cv2.resize(img_rgb01, (size, size), interpolation=cv2.INTER_AREA)
    if norm.endswith("bgr"):
        x = x[..., ::-1]
    if norm.startswith("imagenet"):
        x = (x - MEAN) / STD
    elif norm == "half":
        x = (x - 0.5) / 0.5
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].float()
    with torch.no_grad():
        p = torch.sigmoid(model(t))[0].numpy()  # (2, size, size)
    return np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in p])


def largest_cc(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    return (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))) if n > 1 else np.zeros_like(m, bool)


def to_masks(probs, order, cup_thr=0.5, cup_alone=False, disc_thr=0.5):
    d, c = (probs[0], probs[1]) if order == "disc_cup" else (probs[1], probs[0])
    disc = largest_cc(d > disc_thr)
    if not disc.any():
        return disc, np.zeros_like(disc)
    cup = (c > cup_thr) if cup_alone else ((c > cup_thr) & disc)
    return disc, largest_cc(cup)


def dice(a, b):
    return 2.0 * (a & b).sum() / max(1, a.sum() + b.sum())


def read_rgb01(p):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


# ---- commands ----------------------------------------------------------------------------
def cmd_check(a):
    m = pd.read_csv(a.manifest)
    m = m[m["dataset"] == "REFUGE"]
    if "split" in m:
        m = m[m["split"] == a.split]
    m = m.head(a.n).copy()
    root = Path(a.root)
    sub = {"full": ("Images", "Masks"), "cropped": ("Images_Cropped", "Masks_Cropped"),
           "square": ("Images_Square", "Masks_Square")}[a.variant]
    if a.variant == "square":
        m["image_path"] = m["image_path"].str.replace(r"REFUGE/(train|val|test)/Images/", "REFUGE/Images_Square/", regex=True)
        m["mask_path"] = m["mask_path"].str.replace(r"REFUGE/(train|val|test)/Masks/", "REFUGE/Masks_Square/", regex=True)
    elif a.variant != "full":
        m["image_path"] = m["image_path"].str.replace("/Images/", f"/{sub[0]}/", regex=False)
        m["mask_path"] = m["mask_path"].str.replace("/Masks/", f"/{sub[1]}/", regex=False)
    model = load_model(a.weights)
    cup_opts = [(o, dt, ct) for o in a.orders for dt in a.disc_thrs for ct in a.cup_thrs]
    rows, ratios = [], []
    for size in a.sizes:
        for norm in a.norms:
            res = {}
            for r in m.to_dict("records"):
                img = read_rgb01(root / r["image_path"])
                gt = cv2.imread(str(root / r["mask_path"]), cv2.IMREAD_UNCHANGED)
                if gt.shape[:2] != img.shape[:2]:
                    gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                if size == a.sizes[0] and norm == a.norms[0]:
                    xs = np.nonzero((gt >= 1).any(0))[0]
                    ratios.append((xs.max() - xs.min() + 1) / gt.shape[1] if xs.size else np.nan)
                probs = predict_probs(model, img, size, norm)
                for o, dt, ct in cup_opts:
                    d, c = to_masks(probs, o, ct, False, dt)
                    res.setdefault((o, dt, ct), []).append((dice(d, gt >= 1), dice(c, gt == 2)))
            for (o, dt, ct), v in res.items():
                v = np.array(v)
                rows.append(dict(size=size, norm=norm, order=o, disc_thr=dt, cup_thr=ct,
                                 dice_disc=v[:, 0].mean(), dice_cup=v[:, 1].mean()))
            print(size, norm, "done", flush=True)
    print(f"variant={a.variant}: disc width / image width: mean {np.nanmean(ratios):.2f}, "
          f"min {np.nanmin(ratios):.2f}, max {np.nanmax(ratios):.2f}")
    df = pd.DataFrame(rows)
    df["score"] = df.dice_disc + df.dice_cup
    df = df.sort_values("score", ascending=False)
    print(df.round(3).head(15).to_string(index=False))
    print("\nUse the top row for 'run'. Target: disc Dice > 0.9, cup > 0.8.")


def cmd_run(a):
    d = Path(a.hygd_dir)
    labels = pd.read_csv(next(d.rglob("Labels.csv")))
    labels.columns = [c.strip() for c in labels.columns]
    print("Labels.csv columns:", list(labels.columns))
    if a.limit:
        labels = labels.sample(a.limit, random_state=0).reset_index(drop=True)
        print(f"pilot: {len(labels)} random images")
    img_dir = next(p for p in d.rglob("Images") if p.is_dir()) if not (d / "Images").is_dir() else d / "Images"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(a.weights)
    name_col = next(c for c in labels.columns if "image" in c.lower())
    pat_col = next((c for c in labels.columns if "patient" in c.lower()), None)
    lab_col = next(c for c in labels.columns if c.lower() == "label")
    qual_col = next((c for c in labels.columns if "quality" in c.lower()), None)
    rows = []
    for i, r in enumerate(labels.to_dict("records")):
        name = str(r[name_col])
        stem = Path(name).stem
        ip = img_dir / (name if Path(name).suffix else name + ".jpg")
        mp = out / f"{stem}.png"
        if not mp.exists():
            probs = predict_probs(model, read_rgb01(ip), a.size, a.norm)
            disc, cup = to_masks(probs, a.order, a.cup_thr, a.cup_alone, a.disc_thr)
            m = np.zeros(disc.shape, np.uint8)
            m[disc] = 1
            m[cup] = 2
            cv2.imwrite(str(mp), m)
        lab = 1 if str(r[lab_col]).strip().upper() in ("GON+", "1", "TRUE") else 0
        rows.append(dict(id=f"HYGD_{stem}", dataset="HYGD", label=lab,
                         patient_id=f"HYGD_{r[pat_col]}" if pat_col else f"HYGD_{stem}",
                         quality=r[qual_col] if qual_col else np.nan,
                         image_path=ip.as_posix(), mask_path=mp.as_posix()))
        if i % 25 == 0:
            print(f"{i}/{len(labels)}", flush=True)
    pd.DataFrame(rows).to_csv(a.manifest_out, index=False)
    print("wrote", a.manifest_out, len(rows), "rows")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--manifest", required=True); p.add_argument("--root", default=".")
    p.add_argument("--weights", default="models/refuge_segmentation.pth")
    p.add_argument("--variant", default="cropped", choices=["full", "cropped", "square"])
    p.add_argument("--split", default="val"); p.add_argument("--n", type=int, default=16)
    p.add_argument("--sizes", type=int, nargs="+", default=[256])
    p.add_argument("--cup-thrs", type=float, nargs="+", default=[0.5, 0.8, 0.95, 0.99])
    p.add_argument("--disc-thrs", type=float, nargs="+", default=[0.5, 0.8, 0.95, 0.99])
    p.add_argument("--orders", nargs="+", default=["disc_cup", "cup_disc"], choices=["disc_cup", "cup_disc"])
    p.add_argument("--norms", nargs="+", default=["imagenet"],
                   choices=["255", "imagenet", "imagenet_bgr", "half"])
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("run")
    p.add_argument("--hygd-dir", required=True); p.add_argument("--out-dir", default="hygd_masks")
    p.add_argument("--weights", default="models/refuge_segmentation.pth")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--norm", default="imagenet", choices=["255", "imagenet", "imagenet_bgr", "half"])
    p.add_argument("--cup-thr", type=float, default=0.5); p.add_argument("--cup-alone", action="store_true")
    p.add_argument("--disc-thr", type=float, default=0.5)
    p.add_argument("--order", default="disc_cup", choices=["disc_cup", "cup_disc"])
    p.add_argument("--manifest-out", default="manifest_hygd.csv")
    p.add_argument("--limit", type=int, default=0, help="process only N random images (pilot)")
    p.set_defaults(fn=cmd_run)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
