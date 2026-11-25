#!/usr/bin/env python3
"""Leave-one-dataset-out (LODO) evaluation of chromatic feature groups.

Usage: python lodo_eval.py --features features.csv [--out results.csv] [--boot 1000]

For every held-out dataset: train on the others, report AUC with a bootstrap 95% CI
(patient-level resampling when patient_id exists). Feature groups are the prefixes
before '__' (geom, stats, direct, xyz, lab, hist, rel); combos are listed in COMBOS.
Datasets with a single class (or fewer than 30 images) are skipped.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

COMBOS = {
    "CDR only (geom)": ["geom"],
    "RGB stats": ["stats"],
    "chromatic direct": ["direct"],
    "chromatic hist (3 processors)": ["hist"],
    "chromatic cup/rim relations": ["rel"],
    "Lab": ["lab"],
    "chromatic all (direct+hist+rel)": ["direct", "hist", "rel"],
    "geom + chromatic all": ["geom", "direct", "hist", "rel"],
    # v2 (need --extra features_prof.csv)
    "CDR + rim profile (geom2)": ["geom", "rimg"],
    "profile chromatic": ["ptraj", "pgen2", "pcon"],
    "CDR + profile chromatic": ["geom", "ptraj", "pgen2", "pcon"],
    "CDR + rim + profile chromatic": ["geom", "rimg", "ptraj", "pgen2", "pcon"],
}


def cols_for(df, groups):
    c = [x for x in df.columns if x.split("__")[0] in groups and not x.endswith("_H")]
    return [x for x in c if df[x].notna().mean() > 0.5]


def models():
    return {
        "LR": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(C=0.1, max_iter=2000)),
        "GBM": make_pipeline(SimpleImputer(strategy="median"),
                             HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05)),
    }


def boot_ci(y, p, groups, n, rng):
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if len(np.unique(y[s])) == 2:
            vals.append(roc_auc_score(y[s], p[s]))
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="lodo_results.csv")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--exclude-train", nargs="*", default=[], help="datasets never used for training (e.g. G1020)")
    ap.add_argument("--extra", default=None, help="second features CSV (features_prof.csv), merged on id")
    ap.add_argument("--save-preds", default=None, help="write out-of-sample predictions for delong_compare.py")
    ap.add_argument("--within", action="store_true",
                    help="within-dataset 5-fold CV (train and test on the same dataset) instead of LODO")
    a = ap.parse_args()
    df = pd.read_csv(a.features).dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    if a.extra:
        ex = pd.read_csv(a.extra)
        keep = ["id"] + [c for c in ex.columns if "__" in c]
        df = df.merge(ex[keep], on="id", how="inner")
        print(f"merged {a.extra}: {len(df)} rows")
    preds = []
    pid = df["patient_id"].astype(str).values if "patient_id" in df else df.index.astype(str).values
    rng = np.random.default_rng(0)
    rows = []
    for held in sorted(df["dataset"].unique()):
        te = (df["dataset"] == held).values
        if te.sum() < 30 or df.loc[te, "label"].nunique() < 2:
            print(f"skip {held}: too small or single-class")
            continue
        tr = ~te & ~df["dataset"].isin(a.exclude_train).values
        for name, groups in COMBOS.items():
            cols = cols_for(df, groups)
            if not cols:
                continue
            for mname, m in models().items():
                y = df.loc[te, "label"].values
                if a.within:
                    from sklearn.model_selection import StratifiedGroupKFold
                    from sklearn.base import clone
                    Xd = df.loc[te, cols]
                    p = np.zeros(te.sum())
                    for f_tr, f_te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xd, y, pid[te]):
                        mm = clone(m).fit(Xd.iloc[f_tr], y[f_tr])
                        p[f_te] = mm.predict_proba(Xd.iloc[f_te])[:, 1]
                else:
                    m.fit(df.loc[tr, cols], df.loc[tr, "label"])
                    p = m.predict_proba(df.loc[te, cols])[:, 1]
                if a.save_preds:
                    preds.append(pd.DataFrame(dict(held_out=held, features=name, model=mname,
                                                   id=df.loc[te, "id"].values, y=y, p=p)))
                lo, hi = boot_ci(y, p, pid[te], a.boot, rng)
                rows.append(dict(held_out=held, features=name, model=mname, n_feat=len(cols),
                                 auc=roc_auc_score(y, p), ci_lo=lo, ci_hi=hi))
    if a.save_preds and preds:
        pd.concat(preds).to_csv(a.save_preds, index=False)
    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)
    piv = res.pivot_table(index=["features", "model"], columns="held_out", values="auc").round(3)
    piv["mean"] = piv.mean(axis=1).round(3)
    print(piv.to_string())


if __name__ == "__main__":
    main()
