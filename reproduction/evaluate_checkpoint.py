#!/usr/bin/env python3
"""Compute discrete PDE residuals from a saved model without fitting or PDE solving.

Default grids are PINO=64, other methods=128. F7 also needs its Poisson source
dataset. Other problems use constructor physics (F11 diffusivity D=0.05).
"""
from pathlib import Path
import argparse
import importlib.metadata
import json
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from core.registry import make_problem
from core.numbering import PAPER_NAMESPACE, LEGACY_NAMESPACE, paper_problem_id
from evaluation.offline_residual import pde_metrics, condition_metrics
from experiments.surrogate_checkpoint import load_checkpoint

PROBLEMS = tuple(f"f{i:02}" for i in range(1, 12))
PAPER = {name: f"F{i}" for i, name in enumerate(PROBLEMS, 1)}
CLASS_METHOD = dict(GPSurrogate="gp_de", PIGPSurrogate="pigp_de", PINNSurrogate="pinn_de",
                    RBFNSurrogate="rbfn_de", MLPSurrogate="mlp_de", PINOSurrogate="pino_de")


def replay_problem(problem_name, physics_profile):
    problem = make_problem(problem_name)
    if physics_profile == "ks_value_first_derivative":
        if problem_name != "f08":
            raise ValueError("KS replay profile cannot be applied to another problem")
        from problems.f08 import KuramotoSivashinskyPhysics
        problem._physics = KuramotoSivashinskyPhysics(
            problem.alpha, problem.beta, problem.gamma,
            problem.xmin, problem.xmax, problem.tmin, problem.tmax)
    elif physics_profile != "native":
        raise ValueError(f"Unknown offline physics profile: {physics_profile}")
    return problem


def snapshot_metadata(directory):
    """Read model metadata; companion probes, configuration and hashes are not needed."""
    with (directory / "final_surrogate.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    metadata = dict(payload.get("metadata", {}))
    metadata.setdefault("problem", payload["problem"])
    metadata.setdefault("problem_namespace", payload.get("problem_namespace", LEGACY_NAMESPACE))
    metadata.setdefault("method", CLASS_METHOD[payload["class_name"]])
    metadata.setdefault("seed", None)
    metadata.setdefault("dataset", dict(file=Path(metadata["data_path"]).name
                                        if metadata.get("data_path") else None))
    companion = directory / "checkpoint.json"
    if companion.is_file():
        metadata.update(json.loads(companion.read_text()))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path, help="Saved model directory")
    parser.add_argument("--data-dir", type=Path, help="Dataset directory; required only for F7")
    parser.add_argument("--grid", type=int, choices=(32, 64, 128, 256), help="Default PINO=64; others=128")
    parser.add_argument("--device", default="cpu", choices=("snapshot", "cpu", "cuda"))
    parser.add_argument("--physics-profile", choices=("native", "ks_value_first_derivative"),
                        help="Override the boundary-condition profile recorded with the model")
    parser.add_argument("--conditions-64", action="store_true", help="Also compute IC/BC quantities with n=64")
    parser.add_argument("--output", required=True, type=Path, help="New JSON output file")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    torch.set_num_threads(1)
    directory = args.snapshot.resolve()
    metadata = snapshot_metadata(directory)
    artifact_problem = metadata["problem"]
    namespace = metadata.get("problem_namespace", LEGACY_NAMESPACE)
    problem_name = paper_problem_id(artifact_problem, namespace)
    method, seed = metadata["method"], metadata["seed"]
    default_profile = ("ks_value_first_derivative"
                       if problem_name == "f08" and namespace == LEGACY_NAMESPACE else "native")
    profile = args.physics_profile or metadata.get("offline_physics_profile", default_profile)
    device = metadata.get("replay_device", "cpu") if args.device == "snapshot" else args.device
    problem = replay_problem(problem_name, profile)
    dataset_info = dict(attached=False)
    if problem_name == "f07":
        if args.data_dir is None:
            parser.error("F7 requires --data-dir for its Poisson source field")
        filename = Path(metadata.get("dataset", {}).get("file") or "f07.npz")
        if filename.is_absolute() or len(filename.parts) != 1:
            raise ValueError("Checkpoint dataset must identify a filename")
        dataset = args.data_dir / filename
        problem.attach_data(str(dataset))
        dataset_info = dict(attached=True, filename=dataset.name)
    started = time.perf_counter()
    model = load_checkpoint(directory / "final_surrogate.pkl", device=device, problem=problem)
    grid = args.grid or (64 if method == "pino_de" else 128)
    result = dict(problem=problem_name, paper_problem=PAPER[problem_name], method=method, seed=seed,
                  problem_namespace=PAPER_NAMESPACE, artifact_problem=artifact_problem,
                  artifact_problem_namespace=namespace, offline_physics_profile=profile,
                  snapshot=directory.name, status="complete", prediction_device=device,
                  **pde_metrics(problem, model, grid), dataset=dataset_info,
                  additional_hf_queries=0, additional_train_calls=0,
                  environment={name: importlib.metadata.version(name)
                               for name in ("numpy", "scipy", "scikit-learn", "torch")})
    if args.conditions_64:
        result["conditions_64"] = dict(evaluation_grid="64x64", **condition_metrics(problem, model, 64))
    result["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], paper_problem=result["paper_problem"], method=method,
                          seed=seed, grid_size=grid, pde_rmse=result["pde_rmse"],
                          pde_points=result["pde_points"], output=str(args.output)), indent=2))


if __name__ == "__main__":
    main()
