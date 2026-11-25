#!/usr/bin/env python3
"""Paired DeLong tests between a baseline feature set and every other one, on saved predictions.

    python lodo_eval.py --features features.csv --extra features_prof.csv --save-preds preds_within.csv --within ...
    python delong_compare.py --preds preds_within.csv --baseline "CDR only (geom)" --out delong_within.csv

Holm correction is applied across ALL comparisons in the file (both models, all held-out sets).
Images are treated as independent (patient ids are not available for all sets).
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats


def midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_components(y, preds):
    """preds: (k, n). Returns aucs (k,), covariance (k,k) per Sun & Xu (2014)."""
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    tx, ty, tz = (np.empty((k, m)), np.empty((k, n)), np.empty((k, m + n)))
    for r in range(k):
        tx[r], ty[r] = midrank(pos[r]), midrank(neg[r])
        tz[r] = midrank(np.concatenate([pos[r], neg[r]]))
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def delong_p(y, pa, pb):
    aucs, cov = delong_components(y, np.vstack([pa, pb]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], np.nan
    z = (aucs[1] - aucs[0]) / np.sqrt(var)
    return aucs[0], aucs[1], 2 * stats.norm.sf(abs(z))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(np.where(np.isnan(p), 2, p))
    adj = np.empty_like(p)
    m = np.isfinite(p).sum()
    run = 0.0
    for rank, i in enumerate(order[:m]):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    adj[order[m:]] = np.nan
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--baseline", default="CDR only (geom)")
    ap.add_argument("--out", default="delong.csv")
    a = ap.parse_args()
    d = pd.read_csv(a.preds)
    rows = []
    for (held, model), g in d.groupby(["held_out", "model"]):
        base = g[g.features == a.baseline].set_index("id")
        if base.empty:
            continue
        for feat, gf in g.groupby("features"):
            if feat == a.baseline:
                continue
            gf = gf.set_index("id").loc[base.index]
            y = base["y"].values.astype(int)
            auc_a, auc_b, p = delong_p(y, base["p"].values, gf["p"].values)
            rows.append(dict(held_out=held, model=model, baseline=a.baseline, features=feat,
                             auc_base=auc_a, auc_feat=auc_b, delta=auc_b - auc_a, p=p))
    r = pd.DataFrame(rows)
    r["p_holm"] = holm(r["p"].values)
    r = r.sort_values(["held_out", "model", "delta"], ascending=[True, True, False])
    r.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
