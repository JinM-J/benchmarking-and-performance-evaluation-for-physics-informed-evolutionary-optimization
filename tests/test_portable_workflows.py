"""Test data import and HF-archive summaries without an archived file identity."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.import_data import import_data
from experiments.aggregate import summarize_run, aggregate_rows


class PortableWorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_regenerated_npz_import_and_existing_destination(self):
        source, target = self.root / "source", self.root / "target"
        source.mkdir()
        np.savez_compressed(source / "f06.npz", u=np.arange(12).reshape(3, 4))
        catalog = [dict(paper_problem="F6", file="f06.npz", legacy_file="bench12_data.npz",
                        reference_file_bytes=1)]
        result = import_data(source, target, catalog)
        self.assertEqual(result[0]["status"], "copied")
        self.assertEqual((source / "f06.npz").read_bytes(), (target / "f06.npz").read_bytes())
        np.savez(target / "f06.npz", u=np.zeros((3, 4)))
        before = (target / "f06.npz").read_bytes()
        result = import_data(source, target, catalog)
        self.assertEqual(result[0]["status"], "kept_existing")
        self.assertEqual((target / "f06.npz").read_bytes(), before)

    def test_missing_state_array_still_has_a_useful_error(self):
        source = self.root / "source"
        source.mkdir()
        np.savez(source / "f06.npz", x=np.arange(3))
        catalog = [dict(paper_problem="F6", file="f06.npz")]
        with self.assertRaisesRegex(ValueError, "no u array"):
            import_data(source, self.root / "target", catalog)
        self.assertFalse((self.root / "target").exists())

    def test_summary_uses_consumed_feasible_hf_archive_without_data_digest(self):
        rows = []
        for seed, objective in ((450, 5.), (451, 7.)):
            manifest = dict(problem="f06", method="gp_de", protocol="paper/f06",
                            dataset=dict(file="f06.npz"), seed=seed,
                            budget=dict(n_state_queries=2, surrogate_train_time_sec=1.,
                                        surrogate_infer_time_sec=.1),
                            metrics=dict(wall_time_total_sec=2., mse200_final=.01))
            path = self.root / f"seed{seed}.npz"
            np.savez(path, manifest_json=json.dumps(manifest),
                     result_selection="best_feasible_hf_archive",
                     archive_real_obj=[objective, -1000.], archive_real_vio=[0., 1.],
                     final_real_obj=[-2000.], final_real_vio=[0.])
            rows.append(summarize_run(path))
        summary = aggregate_rows(rows)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["feasible_runs"], 2)
        self.assertEqual(summary[0]["best_f_mean"], 6.)
        self.assertEqual(summary[0]["best_f_std_ddof0"], 1.)


if __name__ == "__main__":
    unittest.main()
