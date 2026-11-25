"""Smoke tests on synthetic data (no real dataset needed):  python -m unittest discover -s tests -v"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chromatic_features as cf  # noqa: E402
import delong_compare as dl  # noqa: E402


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


class TestTransform(unittest.TestCase):
    def test_pure_red_and_grey(self):
        H, L, S = cf.chromatic_hls(1.0, 0.0, 0.0)
        self.assertAlmostEqual(float(H), 360.0, places=3)
        self.assertAlmostEqual(float(S), 1.0, places=6)
        self.assertAlmostEqual(float(L), 1 / 3, places=6)
        H, _, S = cf.chromatic_hls(0.5, 0.5, 0.5)
        self.assertEqual(float(H), cf.H_UNDEFINED)  # hue undefined when S = 0
        self.assertAlmostEqual(float(S), 0.0, places=6)

    def test_processors_triangular_sum_to_one(self):
        P = cf.make_processors(16, "triangular")
        np.testing.assert_allclose(P.sum(0), 1.0, atol=1e-9)


class TestDeLong(unittest.TestCase):
    def test_matches_sklearn_auc(self):
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(0)
        y = (rng.random(300) < 0.3).astype(int)
        a = y * 0.8 + rng.normal(size=300)
        b = y * 0.2 + rng.normal(size=300)
        auc_a, auc_b, p = dl.delong_p(y, a, b)
        self.assertAlmostEqual(auc_a, roc_auc_score(y, a), places=8)
        self.assertAlmostEqual(auc_b, roc_auc_score(y, b), places=8)
        self.assertLess(p, 0.05)  # b is clearly worse than a


class TestPipeline(unittest.TestCase):
    def test_end_to_end_on_synthetic(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            run("chromatic_features.py", "selftest", "--workdir", t)
            run("chromatic_profiles.py", "--manifest", t / "manifest.csv", "--root", t, "--out", t / "prof.csv", "--max-side", 0)
            run("lodo_eval.py", "--features", t / "features_refnorm_retina.csv", "--extra", t / "prof.csv",
                "--within", "--boot", 10, "--save-preds", t / "p.csv", "--out", t / "w.csv")
            run("delong_compare.py", "--preds", t / "p.csv", "--out", t / "d.csv")
            for f in ("features_refnorm_retina.csv", "prof.csv", "w.csv", "p.csv", "d.csv"):
                self.assertTrue((t / f).stat().st_size > 0, f)


if __name__ == "__main__":
    unittest.main()
