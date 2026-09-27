#!/usr/bin/env python3
"""Rebuild PDE-residual tables from archived scalar JSON records only.

No model loading, differentiation, training or PDE solving is performed.
PINO uses a 64x64 evaluation grid; the other five methods use
128x128. The optional condition tables use separate baseline 64-grid records.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.numbering import PAPER_TO_LEGACY

PROBLEMS = tuple(f"f{i:02}" for i in range(1, 12))
PAPER = {problem: f"F{index}" for index, problem in enumerate(PROBLEMS, 1)}
METHODS = ("gp_de", "pigp_de", "pinn_de", "rbfn_de", "mlp_de", "pino_de")
LABELS = dict(zip(METHODS, ("GP", "PIGP", "PINN", "RBFN", "MLP", "PINO")))
SEEDS = tuple(range(481, 491))
BASE = Path("results/pde_residual_evaluation")
CONDITIONS = ("ic_u_rmse", "ic_dt_rmse", "bc_u_rmse", "bc_dx_rmse")


def record_path(problem, method, seed, conditions=False):
    grid = 64 if conditions or method == "pino_de" else 128
    directory = BASE / "runs" if grid == 64 else BASE / "refined_grid/runs"
    return directory / PAPER_TO_LEGACY[problem] / f"{method}_seed{seed}_n{grid}.json", grid


def load_record(artifacts, relative, problem, method, seed, grid):
    path = artifacts / relative
    raw = path.read_bytes()
    record = json.loads(raw)
    expected = dict(problem=PAPER_TO_LEGACY[problem], paper_problem=PAPER[problem], method=method,
                    seed=seed, grid_size=grid, status="complete")
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"{relative}: {key}={record.get(key)!r}, expected {value!r}")
    if not np.isfinite(record["pde_rmse"]) or record["pde_rmse"] < 0:
        raise ValueError(f"Invalid PDE RMSE: {relative}")
    if not record.get("test_points_sha256") or int(record["pde_points"]) <= 0:
        raise ValueError(f"Missing point-set identity/count: {relative}")
    return record, hashlib.sha256(raw).hexdigest()


def csv_out(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows):
    lookup = {(r["problem"], r["method"]): r for r in rows}
    lines = ["# PDE residual RMSE", "", "Final-model evaluation, seeds 481–490. Each run supplies one PDE RMSE; entries are mean ± population standard deviation (ddof=0).", "",
             "GP, PIGP, PINN, RBFN and MLP use 128×128 test grids; PINO uses 64×64. Parametric problems use the original three fixed parameter slices. These are discrete residuals at stated resolutions, not evidence of continuous residual convergence. No cross-problem average is computed.", ""]
    for problems in (PROBLEMS[:4], PROBLEMS[4:8], PROBLEMS[8:]):
        lines += ["| Method (test grid) | " + " | ".join(PAPER[p] for p in problems) + " |",
                  "| :--- | " + " | ".join("---:" for _ in problems) + " |"]
        for method in METHODS:
            grid = 64 if method == "pino_de" else 128
            cells = [f"{lookup[p,method]['mean']:.3e} ± {lookup[p,method]['std']:.3e}" for p in problems]
            lines.append(f"| {LABELS[method]} ({grid}×{grid}) | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["IC/BC quantities, when requested with --include-conditions, are rebuilt separately from baseline 64-grid records. They are never attached to the 128-grid PDE values.", ""]
    return "\n".join(lines)


def compare_expected(rows, expected_file):
    with expected_file.open(encoding="utf-8-sig", newline="") as stream:
        expected_rows = list(csv.DictReader(stream))
    expected = {(r["problem"], r["method"]): r for r in expected_rows}
    actual = {(r["problem"], r["method"]): r for r in rows}
    if len(expected) != len(expected_rows) or set(expected) != set(actual):
        raise ValueError("Expected CSV must contain exactly the 66 unique problem/method groups")
    differences = []
    for key, row in actual.items():
        previous = expected[key]
        for field in ("paper_problem", "evaluation_grid"):
            if row[field] != previous[field]:
                raise ValueError(f"Expected CSV mismatch: {key}, {field}")
        for field in ("n", "ddof"):
            if row[field] != int(previous[field]):
                raise ValueError(f"Expected CSV mismatch: {key}, {field}")
        for field in ("mean", "std"):
            value = float(previous[field])
            if not np.isfinite(value) or not np.isclose(row[field], value, rtol=1e-12, atol=1e-14):
                raise ValueError(f"Expected CSV mismatch: {key}, {field}: {row[field]} vs {value}")
            differences.append(abs(row[field] - value))
    return dict(groups=len(rows), numeric_values=len(differences), rtol=1e-12, atol=1e-14,
                max_abs_difference=max(differences), expected_sha256=hashlib.sha256(expected_file.read_bytes()).hexdigest())


def rebuild_conditions(artifacts):
    per_run, summary, components, inputs = [], [], [], []
    for problem in PROBLEMS:
        for method in METHODS:
            records = []
            for seed in SEEDS:
                relative, grid = record_path(problem, method, seed, conditions=True)
                record, digest = load_record(artifacts, relative, problem, method, seed, grid)
                records.append(record)
                row = dict(problem=problem, paper_problem=PAPER[problem], method=method, seed=seed,
                           evaluation_grid="64x64", source_record=relative.as_posix())
                for metric in CONDITIONS:
                    value = record.get(metric, "")
                    if value != "" and (not np.isfinite(value) or value < 0):
                        raise ValueError(f"Invalid {metric}: {relative}")
                    row[metric] = value
                per_run.append(row)
                for component in record.get("condition_components", []):
                    components.append(dict(problem=problem, paper_problem=PAPER[problem], method=method, seed=seed,
                                           evaluation_grid="64x64", source_record=relative.as_posix(), **component))
                inputs.append(dict(path=relative.as_posix(), sha256=digest))
            for metric in CONDITIONS:
                values = [r[metric] for r in records if metric in r]
                if len(values) not in (0, 10):
                    raise ValueError(f"Partially missing condition quantity: {problem}/{method}/{metric}")
                summary.append(dict(problem=problem, paper_problem=PAPER[problem], method=method, metric=metric,
                                    evaluation_grid="64x64", n=len(values), expected=10,
                                    mean=float(np.mean(values)) if values else "",
                                    std=float(np.std(values, ddof=0)) if values else "", ddof=0))
    return per_run, summary, components, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path, help="Unpacked public artifact root")
    parser.add_argument("--out", required=True, type=Path, help="New output directory (existing directories are refused)")
    parser.add_argument("--expected", type=Path, default=ROOT / "reproduction/metadata/residual_expected.csv")
    parser.add_argument("--include-conditions", action="store_true", help="Also require all baseline 64-grid JSONs and rebuild their IC/BC quantities separately")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing existing output directory: {args.out}")
    rows, summary, inputs, point_sets = [], [], [], {}
    for problem in PROBLEMS:
        for method in METHODS:
            values = []
            for seed in SEEDS:
                relative, grid = record_path(problem, method, seed)
                record, digest = load_record(args.artifacts, relative, problem, method, seed, grid)
                values.append(record["pde_rmse"])
                point_sets.setdefault((problem, grid), set()).add(record["test_points_sha256"])
                rows.append(dict(problem=problem, paper_problem=PAPER[problem], method=method, seed=seed,
                                 grid_size=grid, evaluation_grid=f"{grid}x{grid}", status="complete",
                                 pde_rmse=record["pde_rmse"], pde_points=record["pde_points"],
                                 test_points_sha256=record["test_points_sha256"],
                                 checkpoint_sha256=record.get("checkpoint_sha256", ""),
                                 source_record=relative.as_posix()))
                inputs.append(dict(path=relative.as_posix(), sha256=digest))
            summary.append(dict(problem=problem, paper_problem=PAPER[problem], method=method,
                                evaluation_grid=f"{grid}x{grid}", n=len(values),
                                mean=float(np.mean(values)), std=float(np.std(values, ddof=0)), ddof=0))
    if len(rows) != 660 or any(len(hashes) != 1 for hashes in point_sets.values()):
        raise ValueError("Wrong run count or inconsistent test points within a problem/grid")
    comparison = compare_expected(summary, args.expected)
    conditions = rebuild_conditions(args.artifacts) if args.include_conditions else None
    args.out.mkdir(parents=True, exist_ok=False)
    csv_out(args.out / "per_run.csv", rows)
    csv_out(args.out / "summary.csv", summary)
    (args.out / "pde_rmse.md").write_text(markdown_table(summary), encoding="utf-8")
    if conditions is not None:
        per_run, condition_summary, components, condition_inputs = conditions
        csv_out(args.out / "conditions_64_per_run.csv", per_run)
        csv_out(args.out / "conditions_64_summary.csv", condition_summary)
        if components:
            fields = list(dict.fromkeys(key for row in components for key in row))
            csv_out(args.out / "conditions_64_components.csv", components, fields)
        (args.out / "conditions_64_README.md").write_text(
            "# Initial/boundary errors from baseline 64-grid records\n\n"
            "Seeds 481–490. The IC/BC point sets are separately defined in the original protocol; the 64-grid label identifies those baseline evaluations. Per-condition counts and hashes are preserved in conditions_64_components.csv. Each quantity is summarized across runs with ddof=0. No 128-grid condition values are implied. Missing quantities remain blank, not zero. The corresponding PDE table uses its separately declared 128/64 resolutions.\n", encoding="utf-8")
    verification = dict(status="PASS", paper_problems=list(PAPER.values()), methods=list(METHODS), seeds=list(SEEDS),
                        n_runs=len(rows), n_groups=len(summary),
                        standard_deviation_ddof=0, expected_comparison=comparison,
                        point_set_identity_consistent=True, input_records=inputs,
                        conditions_64_included=conditions is not None,
                        condition_input_records=conditions[3] if conditions else [],
                        scope="Aggregation of archived scalar JSON only; models, raw predictions and continuous residual accuracy are not revalidated.")
    (args.out / "verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(status="PASS", n_runs=len(rows), n_groups=len(summary),
                          comparison=comparison, conditions_64_included=conditions is not None), indent=2))


if __name__ == "__main__":
    main()
