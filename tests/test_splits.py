"""Leakage checks: run split, combination split, training/evaluation guards.

    python3 -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from methane_pinn import data as D
from methane_pinn import runs as R
from methane_pinn.evaluate import check_split
from methane_pinn.sampling import TrainSet


class TestRunSplit(unittest.TestCase):
    def test_disjoint_and_complete(self):
        train = {r.name for r in R.runs("train")}
        test = {r.name for r in R.runs("test")}
        self.assertEqual(len(train), 12)
        self.assertEqual(len(test), 12)
        self.assertFalse(train & test)
        self.assertEqual(train | test, {r.name for r in R.ALL_RUNS})

    def test_trainset_rejects_test_runs(self):
        with self.assertRaises(ValueError):
            TrainSet([R.runs("train")[0], R.runs("test")[0]])

    def test_evaluate_guard(self):
        tr, te = R.runs("train")[0], R.runs("test")[0]
        check_split(te, "test", train_runs=[tr.name])           # fine
        check_split(tr, "val", train_runs=[tr.name])            # fine
        with self.assertRaises(ValueError):
            check_split(tr, "test", train_runs=[tr.name])
        with self.assertRaises(ValueError):
            check_split(te, "val")
        with self.assertRaises(ValueError):                      # a test run that was (wrongly) trained on
            check_split(te, "test", train_runs=[te.name])
        check_split(tr, "scenario", train_runs=[tr.name])       # stage 1 scenarios: training runs only
        with self.assertRaises(ValueError):
            check_split(te, "scenario")


class TestCombinationSplit(unittest.TestCase):
    def test_val_disjoint_from_train_for_every_training_run(self):
        for run in R.runs("train"):
            a = D.combination_table(run, "train")
            v = D.combination_table(run, "val")
            rows = {tuple(r) for r in a.round(9)}
            self.assertFalse(any(tuple(r) in rows for r in v.round(9)), run.name)

    def test_tables_differ_between_runs(self):
        a = D.combination_table(R.runs("train")[0], "train")
        b = D.combination_table(R.runs("train")[1], "train")
        self.assertFalse(np.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
