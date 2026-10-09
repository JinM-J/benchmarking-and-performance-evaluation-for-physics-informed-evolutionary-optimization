"""Find and verify paper F7 attaining points on a supplied stored PDE field.

The Branin lower bound is 5/(4*pi); equality within
x1 in [0,5] requires x1=pi and u=2.275. Search the complete stored x2 axis
at fixed x1=pi, then cross-check each bracket by independent bilinear arithmetic.
The check evaluates the supplied field without solving the PDE.

Example, from the repository root:
    python reference/f07.py --data-dir /path/to/dataset \
        --output /path/to/f7_reference.json
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import mpmath as mp
import numpy as np
import scipy
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.f07 import F07


class ReferenceState:
    """Adapt the official reference query to the objective state-provider API."""

    def __init__(self, reference):
        self.reference = reference

    def evaluate(self, queries):
        return self.reference.query(queries)


def independent_bilinear(x, y, values, xq, yq):
    """Interpolate four raw finite corners without RegularGridInterpolator."""
    i = int(np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2))
    j = int(np.clip(np.searchsorted(y, yq, side="right") - 1, 0, len(y) - 2))
    corners = values[i:i + 2, j:j + 2]
    if not np.isfinite(corners).all():
        raise ValueError("Independent verification requires four finite raw corners")
    a = (xq - x[i]) / (x[i + 1] - x[i])
    b = (yq - y[j]) / (y[j + 1] - y[j])
    value = ((1 - a) * (1 - b) * corners[0, 0]
             + a * (1 - b) * corners[1, 0]
             + (1 - a) * b * corners[0, 1]
             + a * b * corners[1, 1])
    return float(value), [i, j], corners.tolist()


def verify(data_path):
    if not data_path.is_file():
        raise FileNotFoundError(f"Reference dataset is required: {data_path}")
    problem = F07()
    reference = problem.load_reference(data_path)
    provider = ReferenceState(reference)
    x1 = float(np.pi)
    target_u = 2.275
    # Independent raw read uses the documented stored (x2,x1) layout.
    with np.load(data_path) as data:
        x = np.asarray(data["x"], dtype=np.float64)
        y = np.asarray(data["t"], dtype=np.float64)
        raw_u = np.asarray(data["u"], dtype=np.float64).T
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ValueError("This verifier requires ascending F7 reference axes")
    if raw_u.shape != (len(x), len(y)):
        raise ValueError("Unexpected F7 reference array shape")
    if not np.array_equal(x, reference.grids[0]) or not np.array_equal(y, reference.grids[1]):
        raise ValueError("Independent and official coordinate axes differ")

    line_queries = np.column_stack((np.full(len(y), x1), y))
    line_valid = problem.is_valid_query(line_queries)
    if not line_valid.all():
        raise ValueError("The fixed x1=pi search line intersects excluded geometry")
    line_u = np.asarray(reference.query(line_queries), dtype=np.float64)
    differences = line_u - target_u
    if not np.isfinite(differences).all():
        raise ValueError("Nonfinite state on the target search line")

    mp.mp.dps = 80
    exact_lower = 5 / (4 * mp.pi)
    lower_float = float(exact_lower)
    candidates = []
    seen = []
    for j in range(len(y) - 1):
        left, right = float(y[j]), float(y[j + 1])
        fl, fr = float(differences[j]), float(differences[j + 1])
        if fl * fr > 0:
            continue
        root, info = brentq(
            lambda z: float(reference.query([[x1, z]])[0]) - target_u,
            left, right, xtol=1e-14, rtol=1e-14, maxiter=100, full_output=True,
        )
        if any(abs(root - previous) <= 1e-12 for previous in seen):
            continue
        seen.append(root)
        independent_left, _, _ = independent_bilinear(x, y, raw_u, x1, left)
        independent_right, _, _ = independent_bilinear(x, y, raw_u, x1, right)
        if independent_right == independent_left:
            if independent_left != target_u:
                raise ValueError("Flat independent bracket does not attain the target")
            independent_root = left
        else:
            independent_root = left + ((target_u - independent_left)
                                       / (independent_right - independent_left)) * (right - left)
        independent_u, cell, corners = independent_bilinear(x, y, raw_u, x1, root)
        decision = np.array([x1, root])
        u = float(provider.evaluate([decision])[0])
        objective = float(problem.evaluate_fitness(decision, provider))
        cv = float(problem.violation(decision, provider))
        hole_margins = [float(np.hypot(x1 - cx, root - cy) - radius)
                        for cx, cy, radius in problem.holes]
        in_bounds = bool(np.all(decision >= problem.decision_bounds[:, 0])
                         and np.all(decision <= problem.decision_bounds[:, 1]))
        valid = bool(problem.is_valid_query(decision[None, :])[0])
        checks = {
            "root_solver_converged": bool(info.converged),
            "in_decision_bounds": in_bounds,
            "valid_geometry": valid,
            "state_target_abs_error": abs(u - target_u),
            "objective_lower_bound_abs_error": abs(objective - lower_float),
            "independent_state_abs_difference": abs(independent_u - u),
            "independent_root_abs_difference": abs(float(independent_root) - root),
        }
        passed = (info.converged and in_bounds and valid and cv == 0.0
                  and min(hole_margins) > 0.0
                  and checks["state_target_abs_error"] <= 1e-10
                  and checks["objective_lower_bound_abs_error"] <= 1e-12
                  and checks["independent_state_abs_difference"] <= 1e-12
                  and checks["independent_root_abs_difference"] <= 1e-12)
        candidates.append({
            "decision": decision.tolist(), "state": u,
            "objective": objective, "constraint_violation": cv,
            "distance_to_hole_walls": hole_margins,
            "bracket": [left, right], "bracket_state_residuals": [fl, fr],
            "brentq_iterations": int(info.iterations),
            "independent_piecewise_linear_root": float(independent_root),
            "independent_bilinear_state": independent_u,
            "raw_cell_index_x1_x2": cell, "raw_cell_corners": corners,
            "checks": checks, "status": "PASS" if passed else "FAIL",
        })
    branches = []
    for multiple in (-1, 1, 3):
        coordinate = multiple * np.pi
        branches.append({
            "x1": float(coordinate), "multiple_of_pi": multiple,
            "in_decision_bounds": bool(0 <= coordinate <= 5),
            "target_state": float(5.1 * coordinate ** 2 / (4 * np.pi ** 2)
                                  - 5 * coordinate / np.pi + 6),
        })
    return {
        "paper_problem": "F7", "code_problem": "f07",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "Independent reference-point verification on the supplied stored PDE field.",
        "dataset": {
            "file": data_path.name,
            "bytes": data_path.stat().st_size,
            "axis_keys": ["x", "t"], "axis_semantics": ["x1", "x2"],
            "shape_x1_x2": list(raw_u.shape), "stored_layout": "(x2,x1); transposed on loading",
            "bounds": np.column_stack((reference.axes_min, reference.axes_max)).tolist(),
            "interpolation": "Official ReferenceDataset linear interpolation with median NaN filling; all independently checked cells are finite before filling.",
        },
        "decision_bounds": problem.decision_bounds.tolist(),
        "holes_center_x1_x2_radius": [list(hole) for hole in problem.holes],
        "algebraic_bound": {
            "formula": "F=S^2 + 10*(1-1/(8*pi))*cos(x1)+10 >= 5/(4*pi)",
            "equality_condition": "cos(x1)=-1 and S=0; on [0,5], x1=pi and u=2.275.",
            "reference_value_80_digit_working_precision": mp.nstr(exact_lower, 75),
            "classic_branin_branches": branches,
            "unique_in_bounds_x1_branch": True,
        },
        "method": {
            "search": "Scan all adjacent stored x2 nodes at fixed x1=pi; bracket sign changes or exact zero endpoints.",
            "search_range_x2": [float(y[0]), float(y[-1])],
            "segments_scanned": len(y) - 1,
            "brentq": {"xtol": 1e-14, "rtol": 1e-14, "maxiter": 100},
            "independent_check": "Raw four-corner bilinear interpolation and closed-form linear root within each bracket.",
            "pde_solves_launched": 0, "training_runs_launched": 0,
        },
        "representative": candidates[0] if candidates else None,
        "attaining_points": candidates, "attaining_point_count": len(candidates),
        "precision_scope": [
            "The analytic objective lower bound holds for every state; finite stored-field cells attain it numerically at the listed decisions.",
            "Root and interpolation checks use float64; reported coordinates are not certified exact-real PDE coordinates.",
            "Many digits of the algebraic lower bound do not establish the PDE field's discretization accuracy.",
            "No independent PDE solve, grid refinement, or continuous-PDE residual certification is performed by this script.",
            "Geometric exclusion is checked explicitly because the problem's algebraic violation() returns zero even inside holes.",
        ],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "mpmath": mp.__version__},
        "status": "PASS" if candidates and all(r["status"] == "PASS" for r in candidates) else "FAIL",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output", type=Path, help="New JSON path; existing output is never overwritten")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    result = verify(args.data_dir / "f07.npz")
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
        print(f"{result['status']}: {result['attaining_point_count']} attaining points; {args.output}")
        print(json.dumps(result["representative"], indent=2))
    else:
        print(payload, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
