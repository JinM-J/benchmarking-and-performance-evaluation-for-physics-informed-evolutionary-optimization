#!/usr/bin/env python3
"""Recompute a discrete PDE residual from one trusted saved model snapshot.

This repeats the archived offline finite-difference evaluation, not model
training or PDE solving. Checkpoint/probe identities and probe predictions are
checked first. Documented source compatibility is checked explicitly; unknown
source drift is rejected. Default grids are PINO=64, other methods=128.

Only f07 (paper F7) attaches its recorded NPZ to recover the Poisson source,
matching the original offline evaluator. Other problems use constructor physics,
not dataset-attached physics. In particular, f11 uses nominal D=0.05 here.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from core.registry import make_problem
from core.numbering import PAPER_NAMESPACE, LEGACY_NAMESPACE, paper_problem_id
from evaluation.offline_residual import pde_metrics, condition_metrics
from experiments.surrogate_checkpoint import load_checkpoint

PROBLEMS = tuple(f"f{i:02}" for i in range(1, 12))
PAPER = {name: f"F{i}" for i, name in enumerate(PROBLEMS, 1)}
METHODS = ("gp_de", "pigp_de", "pinn_de", "rbfn_de", "mlp_de", "pino_de")
OFFLINE_SOURCE_SHA256 = "7866eeafc3a38b8bd78f3c1123d367449edb0b7dd44a1e351870ecdbb10d637f"
OFFLINE_IMPLEMENTATION_SHA256 = "2798dbd8e2f9e7bd876f499038574de07676e17c7a74c9b84ad56ee893195b5b"
# Accepted differences are documented for exact archived/current source pairs.
SOURCE_DIFFERENCES = {
    "documentation", "diagnostic_messages", "problem_identifiers", "file_paths",
    "component_registry", "command_line_interface", "checkpoint_serialization",
    "test_coverage", "configuration_keys",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_checks(audit):
    provenance_path = ROOT / "reproduction/metadata/source_provenance.json"
    document = json.loads(provenance_path.read_text())
    if document.get("schema") != "source-compatibility-v1":
        raise ValueError("Unsupported source compatibility manifest")
    provenance = {r["file"]: r for r in document["source_files"]}
    results = []
    namespace = audit.get("problem_namespace", LEGACY_NAMESPACE)
    for name, expected in audit["source_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Checkpoint source key must be repository-relative: {name}")
        compatible = provenance.get(name, {}) if namespace == LEGACY_NAMESPACE else {}
        current_file = compatible.get("current_file", name)
        current_path = Path(current_file)
        if current_path.is_absolute() or ".." in current_path.parts:
            raise ValueError(f"Source path must be repository-relative: {current_file}")
        current = sha(ROOT / current_path)
        row = dict(file=name, current_file=current_file,
                   checkpoint_source_sha256=expected, current_sha256=current)
        if current == expected:
            row.update(status="byte_identical", differences=[])
        else:
            if (compatible.get("source_sha256") != expected
                    or compatible.get("current_sha256") != current):
                raise ValueError(f"Source fingerprint mismatch: {name}")
            differences = compatible.get("differences")
            if (not isinstance(differences, list) or not differences
                    or any(item not in SOURCE_DIFFERENCES for item in differences)):
                raise ValueError(f"Unsupported source differences: {name}")
            row.update(status="compatible_source_pair", differences=differences)
            print(f"[SOURCE] {name}: {row['status']} ({', '.join(differences)})", flush=True)
        results.append(row)
    offline = sha(ROOT / "evaluation/offline_residual.py")
    if offline != OFFLINE_IMPLEMENTATION_SHA256:
        raise ValueError("Offline residual implementation fingerprint mismatch")
    return dict(checkpoint_sources=results, provenance_sha256=sha(provenance_path),
                offline_residual_sha256=offline,
                offline_residual_original_sha256=OFFLINE_SOURCE_SHA256,
                offline_residual_byte_identical_to_archived_evaluator=offline == OFFLINE_SOURCE_SHA256,
                verification_scope="Exact source fingerprints and declared compatibility; model predictions are checked independently against saved probes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path, help="Trusted model snapshot directory")
    parser.add_argument("--data-dir", type=Path, help="Dataset directory; required only for f07 (paper F7). Archived filenames are read from checkpoint metadata.")
    parser.add_argument("--grid", type=int, choices=(32, 64, 128, 256), help="Default PINO=64; other methods=128")
    parser.add_argument("--device", default="snapshot", choices=("snapshot", "cpu", "cuda"))
    parser.add_argument("--conditions-64", action="store_true", help="Also compute separately labelled IC/BC quantities with n=64")
    parser.add_argument("--output", required=True, type=Path, help="New JSON file; existing output is refused")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    torch.set_num_threads(1)
    directory = args.snapshot.resolve()
    audit = json.loads((directory / "checkpoint.json").read_text())
    if audit.get("status") != "PASS":
        raise ValueError("Snapshot audit must have status PASS")
    artifact_problem = audit["problem"]
    namespace = audit.get("problem_namespace", LEGACY_NAMESPACE)
    problem_name = paper_problem_id(artifact_problem, namespace)
    if namespace == PAPER_NAMESPACE and audit.get("checkpoint_version") != 3:
        raise ValueError("Public-numbered snapshot audit requires checkpoint_version=3")
    method, seed = audit["method"], int(audit["seed"])
    if problem_name not in PAPER or method not in METHODS:
        raise ValueError("Snapshot is outside the F1–F11 / six-surrogate scope")
    hashes = {}
    for filename, key in (("final_surrogate.pkl", "checkpoint_sha256"),
                          ("checkpoint_probes.npz", "probes_sha256"),
                          ("resolved_config.yaml", "resolved_config_sha256")):
        actual = sha(directory / filename)
        if actual != audit[key]:
            raise ValueError(f"Snapshot audit hash mismatch: {filename}")
        hashes[filename] = actual
    sources = source_checks(audit)
    device = audit["replay_device"] if args.device == "snapshot" else args.device
    problem = make_problem(problem_name)
    dataset_check = dict(required=False, attached=False,
                         semantics="Original offline evaluator constructor physics; no reference-field loading.")
    if problem_name == "f07":
        if args.data_dir is None:
            parser.error("F7 requires --data-dir for its recorded Poisson source field")
        filename = Path(audit["dataset"]["file"])
        if filename.is_absolute() or len(filename.parts) != 1:
            raise ValueError("Checkpoint dataset must identify a filename, not an external path")
        dataset = args.data_dir / filename
        digest = sha(dataset)
        if digest != audit["dataset"]["sha256"]:
            raise ValueError("F7 Poisson source dataset hash mismatch")
        problem.attach_data(str(dataset))
        dataset_check = dict(required=True, attached=True, filename=dataset.name, sha256=digest,
                             semantics="Recorded Poisson source-interpolator attachment; no PDE solve.")
    started = time.perf_counter()
    model = load_checkpoint(directory / "final_surrogate.pkl", device=device, problem=problem)
    with np.load(directory / "checkpoint_probes.npz", allow_pickle=False) as probes:
        replay = np.asarray(model.predict(probes["queries"])).reshape(-1)
        expected = np.asarray(probes["predictions"]).reshape(-1)
        if not np.isfinite(replay).all() or not np.isfinite(expected).all():
            raise FloatingPointError("Nonfinite probe prediction")
        np.testing.assert_allclose(replay, expected, rtol=1e-5, atol=2e-6)
        probe_error = float(np.max(np.abs(replay - expected)))
    grid = args.grid or (64 if method == "pino_de" else 128)
    metrics = pde_metrics(problem, model, grid)
    result = dict(problem=problem_name, paper_problem=PAPER[problem_name], method=method, seed=seed,
                  problem_namespace=PAPER_NAMESPACE, artifact_problem=artifact_problem,
                  artifact_problem_namespace=namespace,
                  snapshot=directory.name, status="complete", prediction_device=device,
                  checkpoint_sha256=hashes["final_surrogate.pkl"], probe_replay_max_abs_diff=probe_error,
                  probe_tolerance=dict(rtol=1e-5, atol=2e-6), **metrics,
                  input_sha256=hashes, source_verification=sources, dataset_verification=dataset_check,
                  additional_hf_queries=0, additional_train_calls=0,
                  original_environment=audit.get("environment", {}),
                  current_environment={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "torch")},
                  evidence_limits=["Re-evaluates frozen predictions and the original common finite-difference residual; does not prove continuum convergence or model training reproducibility.",
                                   "Probe agreement checks saved finite query points; it is not a full-field identity proof.",
                                   "Source checks cover the archived audit paths and copied residual implementation; exact versions and numerical devices are recorded.",
                                   "Physics attachment follows the archived evaluator: F7 attaches source data; other problems retain constructor defaults (F11 diffusivity 0.05)."])
    if args.conditions_64:
        result["conditions_64"] = dict(evaluation_grid="64x64", **condition_metrics(problem, model, 64))
    result["elapsed_seconds"] = time.perf_counter()-started
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(json.dumps(dict(status=result["status"], paper_problem=result["paper_problem"], method=method,
                          seed=seed, grid_size=grid, pde_rmse=result["pde_rmse"], pde_points=result["pde_points"],
                          probe_replay_max_abs_diff=probe_error, output=str(args.output)), indent=2))


if __name__ == "__main__":
    main()
