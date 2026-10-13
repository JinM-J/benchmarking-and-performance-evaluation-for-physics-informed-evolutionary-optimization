"""Reporting must preserve raw results/FR and clip only verified tolerance cases."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.metrics.objective_reporting import annotate_run, objective_summary, sample_moments
from experiments.aggregate import aggregate_rows, summarize_run

spec = importlib.util.spec_from_file_location("reproduce_optimization", ROOT / "reproduction/reproduce_optimization.py")
reproduction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduction)


class ObjectiveReportingTests(unittest.TestCase):
    def test_confirmed_f5_cohort_keeps_raw_values_and_has_exact_zero_spread(self):
        with (ROOT / "reproduction/metadata/f5_algebra_reporting.csv").open(newline="") as stream:
            confirmed = list(csv.DictReader(stream))
        rows = [dict(r, best_f=float(r["F_raw"]), best_cv=float(r["CV_raw"]), feasible=True) for r in confirmed]
        original = [r["best_f"] for r in rows]
        reported = [annotate_run(r) for r in rows]
        result = objective_summary(rows)
        self.assertEqual(result["clipped_runs"], 30)
        self.assertEqual(result["F_reported_n"], 30)
        self.assertEqual(result["F_reported_mean"], -5.508013271595)
        self.assertEqual(result["F_reported_variance_ddof1"], 0.)
        self.assertEqual(result["F_reported_std_ddof1"], 0.)
        self.assertGreater(result["F_raw_std_ddof1"], 0.)
        self.assertEqual([r["best_f"] for r in reported], original)
        self.assertTrue(all(r["feasible"] for r in reported))

    def test_positive_cv_alone_never_confirms_a_new_f5_run(self):
        row = dict(problem="f05", method="gp_de", seed=450, feasible=True,
                   best_f=-5.50806, best_cv=5e-5, history_sha256="unconfirmed")
        reported = annotate_run(row)
        self.assertFalse(reported["was_clipped"])
        self.assertEqual(reported["F_reported"], row["best_f"])
        self.assertEqual(reported["reporting_status"], "below_reference_unverified")

    def test_both_aggregation_paths_use_per_run_reported_values_and_preserve_fr(self):
        x, state = np.sqrt(.5), .5004
        raw = x*x+(state-1)**2
        common = dict(problem="f08", paper_problem="F8", group="main", method="gp_de",
                      protocol="paper/f08", dataset_file="f08.npz", hf=145, runtime=1.,
                      train_t=.1, infer_t=.1, mse=.2)
        rows = [dict(common, seed=450, feasible=True, best_f=raw, best_cv=0., best_x=x, best_state=state),
                dict(common, seed=451, feasible=True, best_f=1., best_cv=0.),
                dict(common, seed=452, feasible=False, best_f=-100., best_cv=.1)]
        aggregate = aggregate_rows(rows)[0]
        reproduced = reproduction.summarize(rows)[0]
        for result in (aggregate, reproduced):
            self.assertEqual(result["feasible_runs"], 2)
            self.assertEqual(result["best_f_n"], 2)
            self.assertEqual(result["best_f_mean"], .875)
            self.assertEqual(result["best_f_variance_ddof1"], .03125)
            self.assertAlmostEqual(result["best_f_std_ddof1"], np.std([.75, 1.], ddof=1))
            self.assertEqual(result["clipped_runs"], 1)
        self.assertEqual(aggregate["feasible_rate"], 2/3)
        self.assertEqual(rows[0]["best_f"], raw)

    def test_unverified_archive_keeps_selected_decision_and_cv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.npz"
            manifest = dict(problem="f05", method="gp_de", protocol="paper/f05", seed=450,
                            dataset=dict(file="f05.npz"), budget=dict(n_state_queries=2,
                            surrogate_train_time_sec=.1, surrogate_infer_time_sec=.1),
                            metrics=dict(wall_time_total_sec=1., mse200_final=.2))
            np.savez(path, manifest_json=json.dumps(manifest), result_selection="best_feasible_hf_archive",
                     archive_real_obj=[-5.50806, -100.], archive_real_vio=[5e-5, 1.],
                     archive_dec=[[2., 3.], [9., 9.]])
            before = path.read_bytes()
            row = summarize_run(path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(json.loads(row["decision_raw"]), [2., 3.])
            self.assertEqual(row["CV_raw"], 5e-5)
            self.assertEqual(row["F_reported"], -5.50806)
            self.assertFalse(row["was_clipped"])
            self.assertEqual(row["reporting_status"], "below_reference_unverified")

    def test_valid_n_and_undefined_singleton_variance(self):
        self.assertEqual(sample_moments([1., None, np.nan, 3.])["n"], 2)
        self.assertEqual(sample_moments([1., None, np.nan, 3.])["variance_ddof1"], 2.)
        self.assertTrue(np.isnan(sample_moments([1.])["std_ddof1"]))
        self.assertTrue(np.isnan(sample_moments([])["mean"]))


if __name__ == "__main__":
    unittest.main()
