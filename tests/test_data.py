"""Checks for runs.py and data.py.   python3 -m unittest discover -s tests -v

File-based tests use $PINN_DATA_ROOT/les/<run>.h5 and are skipped when no extracted file exists
(a 3-frame test file is enough: scripts/extract.py --run stable_N --frames 3 --out-dir ...).
"""

import unittest

import numpy as np

from methane_pinn import data as D
from methane_pinn import runs as R


class TestRuns(unittest.TestCase):
    def test_split_is_half_per_class_and_alternates(self):
        self.assertEqual(len(R.ALL_RUNS), 24)
        for stab in R.TREES:
            train = {r.bearing for r in R.runs("train") if r.stab == stab}
            self.assertEqual(len(train), 4)
        self.assertNotEqual(R.TRAIN_BEARINGS["daytime"], R.TRAIN_BEARINGS["convective"])
        trained_somewhere = set().union(*R.TRAIN_BEARINGS.values())
        self.assertEqual(trained_somewhere, set(R.RUN_DIRS))

    def test_snapshot_times(self):
        t = R.snapshot_times()
        self.assertEqual(len(t), 31)
        self.assertEqual((t[0], t[-1]), (3600.0, 7200.0))
        self.assertEqual(R.step_of(7200.0), 360000)


class TestCombinations(unittest.TestCase):
    def test_sampler_respects_limits(self):
        q = D.sample_combinations(5000, np.random.default_rng(0))
        self.assertTrue((q >= 0).all())
        total = q.sum(axis=1)
        self.assertTrue((total > 0).all() and (total <= D.Q_TOTAL_MAX_KGPH + 1e-9).all())
        k = (q > 0).sum(axis=1)
        self.assertEqual((k.min(), k.max()), (1, D.N_SRC))

    def test_tables_reproducible_and_disjoint(self):
        run = R.runs("train")[0]
        a, b = D.combination_table(run, "train"), D.combination_table(run, "train")
        np.testing.assert_array_equal(a, b)
        v = D.combination_table(run, "val")
        self.assertEqual((len(a), len(v)), (2000, 200))
        self.assertFalse(np.isin(v.round(6), a.round(6)).all(axis=1).any())
        with self.assertRaises(ValueError):
            D.combination_table(run, "test")

    def test_superpose_matches_explicit_sum(self):
        rng = np.random.default_rng(1)
        basis = rng.random((D.N_SRC, 4, 5, 6)).astype(np.float32)
        q = np.zeros(D.N_SRC)
        q[2], q[9] = 500.0, 1000.0
        expect = 15.0 * basis[2] + 30.0 * basis[9]
        np.testing.assert_allclose(D.superpose(basis, q, axis=0), expect, rtol=1e-6)


def _any_extracted():
    files = sorted(D.les_dir().glob("*.h5")) if D.les_dir().exists() else []
    return files[0] if files else None


@unittest.skipIf(_any_extracted() is None, "no extracted LES file under $PINN_DATA_ROOT/les")
class TestExtractedRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.les = D.LESRun(_any_extracted())

    def test_first_frame_is_tracer_free(self):
        self.assertEqual(float(np.abs(self.les.basis(0)).max()), 0.0)

    def test_field_superposition(self):
        it = min(2, self.les.shape[0] - 1)
        q = np.zeros(D.N_SRC)
        q[3], q[7] = 500.0, 1000.0
        b = self.les.basis(it).astype(np.float64)
        expect = 15.0 * b[3] + 30.0 * b[7]
        np.testing.assert_allclose(self.les.field(it, q), expect, rtol=1e-5, atol=1e-12)

    def test_sensor_layout(self):
        lay = D.sensor_layout(self.les)
        self.assertEqual(len(lay.cols), 20)
        self.assertEqual(len({tuple(c) for c in lay.cols}), 20)
        src = {tuple(c) for c in self.les.source_cells()}
        self.assertFalse(src & {tuple(c) for c in lay.cols} or lay.lidar_col in src)
        g = self.les.grid
        d = np.min(np.hypot(lay.xy[:, :1] - g["src_xy"][:, 0], lay.xy[:, 1:] - g["src_xy"][:, 1]), axis=1)
        self.assertTrue((d[lay.near] <= D.NEAR_RADIUS_M).all() and (d[~lay.near] > D.NEAR_RADIUS_M).all())
        self.assertEqual(lay.n_inputs, 113)
        np.testing.assert_array_equal(lay.cols, D.sensor_layout(self.les).cols)   # deterministic

    def test_sensor_series_shape(self):
        s = D.sensor_series(self.les, D.sensor_layout(self.les))
        self.assertEqual(s["met"].shape, (self.les.shape[0], 113))
        self.assertTrue(np.isfinite(s["met"]).all())
        theta = s["met"][:, 3:80:4]
        self.assertTrue(((theta > 250) & (theta < 330)).all())


if __name__ == "__main__":
    unittest.main()
