# Spatial chromatic profiles of the optic nerve head for glaucoma detection

Code for an exploratory study: does a chromatic-monitoring description of the optic disc
(Jones, Deakin & Spencer, *Chromatic Monitoring of Complex Conditions*, CRC Press, 2008)
add to the cup-to-disc ratio (CDR), and does it survive a change of dataset?

**Status: exploratory, negative-leaning.** Spatially ordered colour profiles beat global colour
features, but no colour feature set significantly beat CDR after multiple-comparison correction,
and colour transferred between datasets worse than CDR did. Read the Limitations before reusing anything.

## Results in brief (logistic regression, AUC)

| Feature set | ORIGA within | REFUGE within | ORIGA across | REFUGE across |
| --- | --- | --- | --- | --- |
| CDR | 0.820 | 0.910 | 0.822 | 0.918 |
| All v1 chromatic features (global) | 0.706 | 0.863 | 0.600 | 0.744 |
| Spatial chromatic profile (v2) | 0.762 | 0.936 | 0.695 | 0.848 |
| CDR + spatial profile | 0.818 | 0.950 | 0.791 | 0.899 |

"Within" = five-fold, patient-grouped. "Across" = trained on the other dataset. Full tables, gradient-boosting
results and paired DeLong tests are in `results/`. The best nominal gain over CDR (REFUGE within, +0.040 AUC,
p = 0.012) does not survive Holm correction over 66 comparisons (adjusted p = 0.52) and is absent on ORIGA.

## What is here

| File | Purpose |
| --- | --- |
| `build_manifest.py` | Pairs images, masks and labels for G1020, ORIGA, REFUGE into `manifest.csv` |
| `chromatic_features.py` | Version 1 (global) features: region H/L/S, Lab, histogram processors, cup-vs-rim relations |
| `chromatic_profiles.py` | Version 2: 16 angular sectors x 6 radial bins, H/L/S trajectories, second-generation features, rim-width profile |
| `lodo_eval.py` | Within-dataset CV and across-dataset evaluation, bootstrap CIs, saved predictions |
| `delong_compare.py` | Paired DeLong tests with Holm correction |
| `segment_hygd.py`, `auto_masks.py`, `qc_masks.py` | Attempts to obtain disc/cup masks for HYGD and visual QC (unsuccessful, see below) |
| `tests/` | Smoke tests on synthetic data |
| `results/` | Result tables from the study |

## Usage

```
pip install -r requirements.txt
python -m unittest discover -s tests -v      # synthetic smoke test, ~30 s
```

Data are not included. Download the Kaggle collection
[glaucoma-datasets](https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets) (ORIGA, REFUGE, G1020) and run from its folder:

```
python build_manifest.py --root . --out manifest.csv          # prints mask values and label counts: check them
python chromatic_features.py extract --manifest manifest_labeled.csv --root . --out features.csv
python chromatic_profiles.py --manifest manifest_labeled.csv --root . --out features_prof.csv
python lodo_eval.py --features features.csv --extra features_prof.csv --within --exclude-train G1020 --save-preds preds_within.csv --out r_within.csv
python lodo_eval.py --features features.csv --extra features_prof.csv --exclude-train G1020 --save-preds preds_across.csv --out r_across.csv
python delong_compare.py --preds preds_within.csv --out delong_within.csv
```

`manifest_labeled.csv` is `manifest.csv` without rows whose label is missing (REFUGE test images have none).
Masks use 0 = background, 1 = disc, 2 = cup; `--convention auto` detects other layouts.

## Limitations

- Features use the datasets' ground-truth masks, an upper bound for any real pipeline.
- G1020 labels were unrelated to CDR in this pipeline (AUC about 0.5), so G1020 was never used for training.
- REFUGE has 80 glaucoma cases among 800 labelled images. ORIGA and REFUGE have no patient identifiers, so DeLong tests treat images as independent.
- The planned external test on [HYGD](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.1.0/) could not be run:
  a REFUGE-trained U-Net found no disc in 83% of 30 pilot images, and a classical detector was correct in roughly 77-80% (visual count),
  below the 85% threshold set in advance. HYGD has no OD/OC masks.
- The text of the book's equations for undefined-hue distances (1.11) and the chromatic vector (1.13) was garbled in the copy used;
  the implementation is our reading and should be checked against the printed book.
- No deep network or retinal foundation model was evaluated, and no hyper-parameters were tuned.

## Licence and citation

No licence has been chosen yet; add one before making the repository public. Citation details will be added with the paper.
