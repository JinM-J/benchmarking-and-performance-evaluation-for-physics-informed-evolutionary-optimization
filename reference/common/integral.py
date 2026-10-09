"""Compute attainable reference points for paper F9--F11 on stored PDE fields.

First solve the scalar equation J(parameter)=J_target, then solve
u(x,0,parameter)=u_target. These checks use the official stored-field
interpolator and quadrature. Reference fields and configurations are read only.
"""
import gc
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.experiment import _PointProvider

TARGETS = {"F9": ("f09", 0.0, 0.7, 3.0),
           "F10": ("f10", 0.4, 0.4, 0.0),
           "F11": ("f11", 0.0, 0.5, 0.0)}


class Provider:
    def __init__(self, reference):
        self.reference = reference
        self.calls = 0
        self.points = 0

    def evaluate(self, queries):
        self.calls += 1
        self.points += len(queries)
        return self.reference.query(queries)


def scan_roots(function, axis):
    values = np.asarray([function(float(x)) for x in axis])
    roots = []
    for i in range(len(axis) - 1):
        a, b = float(axis[i]), float(axis[i + 1])
        if values[i] == 0:
            roots.append({"root": a, "bracket": [a, a], "residual": 0.0})
        elif values[i] * values[i + 1] < 0:
            root = brentq(function, a, b, xtol=1e-14, rtol=1e-14)
            roots.append({"root": root, "bracket": [a, b],
                          "residual": float(function(root))})
    if values[-1] == 0:
        roots.append({"root": float(axis[-1]), "bracket": [float(axis[-1])] * 2,
                      "residual": 0.0})
    return roots, {"samples": len(axis), "residual_min": float(values.min()),
                   "residual_max": float(values.max())}


def independent_state(reference, decision):
    """Eight-corner trilinear interpolation, independent of scipy RGI."""
    q = np.clip(np.asarray(decision, dtype=float), reference.axes_min, reference.axes_max)
    indices, weights = [], []
    for axis, value in zip(reference.grids, q):
        index = min(len(axis) - 2, max(0, int(np.searchsorted(axis, value) - 1)))
        indices.append(index)
        weights.append((value - axis[index]) / (axis[index + 1] - axis[index]))
    total = 0.0
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                bits = (a, b, c)
                weight = np.prod([w if bit else 1 - w for w, bit in zip(weights, bits)])
                total += weight * float(reference.u_grid[tuple(i + bit for i, bit in zip(indices, bits))])
    return float(total)


def components(problem, provider, decision, code):
    qj = np.column_stack((problem._j_xt, np.full(len(problem._j_xt), decision[2])))
    u = float(provider.evaluate(np.asarray(decision).reshape(1, 3))[0])
    field = provider.evaluate(qj)
    if code == "f09":
        j = float(field.sum() * problem._j_dx * problem._j_dt)
    elif code == "f10":
        j = problem._dissipation_j(field, decision[2])
    else:
        j = problem._work_j(field, decision[2])
    return u, j, qj


def solve(paper, data_dir):
    code, target_u, target_j, lower = TARGETS[paper]
    path = data_dir / f"{code}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    print(paper, "loading reference field", flush=True)
    module = importlib.import_module("problems." + code)
    problem = getattr(module, code.upper())()
    problem.attach_data(str(path))
    reference = problem.load_reference(str(path))
    provider = Provider(reference)
    bounds = problem.decision_bounds

    def j_residual(parameter):
        return components(problem, provider, [0.0, 0.0, parameter], code)[1] - target_j

    parameter_roots, parameter_scan = scan_roots(j_residual, np.linspace(*bounds[2], 1001))
    candidates = []
    for parameter_record in parameter_roots:
        parameter = parameter_record["root"]
        def u_residual(x):
            return float(provider.evaluate(np.array([[x, 0.0, parameter]]))[0]) - target_u
        state_roots, state_scan = scan_roots(u_residual, np.linspace(*bounds[0], 2001))
        for state_record in state_roots:
            decision = np.array([state_record["root"], 0.0, parameter])
            u, j, qj = components(problem, provider, decision, code)
            objective = problem.evaluate_fitness(decision, provider)
            direct_u = independent_state(reference, decision)
            direct_grid = np.array([independent_state(reference, q) for q in qj]).reshape(problem.j_nx, problem.j_nt)
            # Explicit sums independently check the quadrature functions.
            if code == "f09":
                direct_j = sum(float(v) for v in direct_grid.flat) * problem._j_dx * problem._j_dt
            elif code == "f10":
                direct_j = np.exp(np.clip(parameter, reference.axes_min[2], reference.axes_max[2])) * sum(
                    ((direct_grid[i + 1, k] - direct_grid[i - 1, k]) / (2 * problem._j_dx))**2
                    for i in range(1, problem.j_nx - 1) for k in range(problem.j_nt)) * problem._j_dx * problem._j_dt
            else:
                xs = np.linspace(problem.xmin, problem.xmax, problem.j_nx)
                ts = np.linspace(problem.tmin, problem.tmax, problem.j_nt, endpoint=False)
                direct_j = sum(direct_grid[i, k] * (1 + parameter) * 2 * np.sin(np.pi * xs[i])
                               * np.cos(2 * np.pi * ts[k]) for i in range(problem.j_nx)
                               for k in range(problem.j_nt)) * problem._j_dx * problem._j_dt
            # HF archive arithmetic rounds the anchor label, not all J labels.
            pool_provider = _PointProvider(decision, np.asarray(u).astype(reference.label_dtype), reference)
            pool_objective = problem.evaluate_fitness_for_pool(decision, pool_provider)
            decision32 = decision.astype(np.float32).astype(float)
            u32 = reference.query(decision32)[0].astype(reference.label_dtype)
            pool32 = problem.evaluate_fitness_for_pool(decision32, _PointProvider(decision32, u32, reference))
            candidates.append({"decision": decision.tolist(), "u": u, "J": j,
                "u_target_error": abs(u - target_u), "J_target_error": abs(j - target_j),
                "objective": objective, "objective_error_from_lower_bound": abs(objective - lower),
                "violation": problem.violation(decision, provider),
                "inside_decision_bounds": bool(np.all((decision >= bounds[:, 0]) & (decision <= bounds[:, 1]))),
                "parameter_root": parameter_record, "state_root": state_record, "state_scan": state_scan,
                "independent_trilinear_u_error": abs(u - direct_u),
                "independent_trilinear_and_quadrature_J_error": abs(j - float(direct_j)),
                "pool_objective_at_float64_decision": float(pool_objective),
                "float32_decision": decision32.tolist(), "pool_objective_at_float32_decision": float(pool32)})
    if not candidates:
        raise RuntimeError(f"No jointly attainable point found for {paper}")
    candidates.sort(key=lambda row: row["objective_error_from_lower_bound"])
    best = candidates[0]
    assert best["u_target_error"] < 1e-9 and best["J_target_error"] < 1e-9
    assert best["independent_trilinear_u_error"] < 1e-12
    assert best["independent_trilinear_and_quadrature_J_error"] < 1e-12
    result = {"paper_problem": paper, "code_problem": code, "status": "PASS",
        "scope": "Numerical attainment on the deposited interpolated field and implemented quadrature; not a continuous-PDE error certificate.",
        "data_file": path.name, "data_bytes": path.stat().st_size,
        "label_dtype": str(reference.label_dtype), "decision_bounds": bounds.tolist(),
        "target_u": target_u, "target_J": target_j, "algebraic_lower_bound": lower,
        "method": "1001-parameter sign scan, brentq; t=0, 2001-x sign scan, brentq; independent eight-corner interpolation and explicit quadrature",
        "brentq_xtol": 1e-14, "brentq_rtol": 1e-14, "random_seed": None,
        "parameter_scan": parameter_scan, "parameter_roots": parameter_roots,
        "candidate_count": len(candidates), "best": best, "candidates": candidates,
        "diagnostic_reference_calls": provider.calls, "diagnostic_reference_points": provider.points,
        "experiment_FE_consumed": 0, "data_regenerated": False}
    print(paper, json.dumps(best), flush=True, file=sys.stderr)
    del provider, reference, problem
    gc.collect()
    return result
